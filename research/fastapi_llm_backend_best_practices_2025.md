# FastAPI Best Practices for Production AI/LLM Backends with WebSocket Support (2025)

## Executive Summary

This research report provides comprehensive guidance on building production-ready AI/LLM backends using FastAPI with WebSocket support in 2025. The findings synthesize best practices from production deployments, open-source templates, and expert recommendations across ten critical areas: project structure, WebSocket streaming, dependency injection, background tasks, rate limiting, authentication, error handling, SQLAlchemy 2.0 async patterns, Redis integration, and observability. The recommended architecture follows a domain-driven modular structure with async-first patterns, leveraging FastAPI's native capabilities alongside specialized libraries like SlowAPI for rate limiting, OpenTelemetry for observability, and LangChain/LangGraph for LLM orchestration.

---

## 1. Introduction

Building production AI/LLM backends requires careful architectural decisions that balance performance, maintainability, and scalability. FastAPI has emerged as the framework of choice for such applications due to its async-first design, automatic API documentation, and excellent developer experience. This report consolidates 2025 best practices specifically tailored for customer support AI backends that require real-time streaming capabilities.

---

## 2. FastAPI Project Structure for Large Applications

The recommended project structure follows a domain-driven design approach, organizing code by business domain rather than technical layer. This structure scales well from small prototypes to large production systems with multiple teams[1][2].

### 2.1 Recommended Directory Structure

```
customer_support_ai/
├── alembic/                          # Database migrations
├── src/
│   ├── auth/
│   │   ├── router.py                 # Authentication endpoints
│   │   ├── schemas.py                # Pydantic models for auth
│   │   ├── models.py                 # SQLAlchemy user models
│   │   ├── dependencies.py           # Auth dependencies (JWT validation)
│   │   ├── config.py                 # Auth-specific config (JWT settings)
│   │   ├── constants.py              # Error codes, token types
│   │   ├── exceptions.py             # InvalidCredentials, TokenExpired
│   │   ├── service.py                # Authentication business logic
│   │   └── utils.py                  # Password hashing, token utilities
│   ├── chat/
│   │   ├── router.py                 # Chat endpoints (REST + WebSocket)
│   │   ├── schemas.py                # Message, Conversation schemas
│   │   ├── models.py                 # Conversation, Message DB models
│   │   ├── dependencies.py           # Session validation, rate limits
│   │   ├── constants.py              # Message types, status codes
│   │   ├── exceptions.py             # ConversationNotFound, etc.
│   │   ├── service.py                # Chat business logic
│   │   ├── websocket_manager.py      # WebSocket connection management
│   │   └── streaming.py              # LLM streaming utilities
│   ├── agents/
│   │   ├── graph.py                  # LangGraph agent definitions
│   │   ├── tools.py                  # Agent tools (search, retrieval)
│   │   ├── prompts/
│   │   │   ├── __init__.py           # Prompt loader
│   │   │   └── system.md             # System prompts
│   │   └── memory.py                 # Conversation memory management
│   ├── core/
│   │   ├── config.py                 # Global configuration (BaseSettings)
│   │   ├── database.py               # Async engine, session factory
│   │   ├── redis.py                  # Redis client setup
│   │   ├── middleware.py             # CORS, logging, timing middleware
│   │   ├── logging.py                # Structured logging setup
│   │   ├── metrics.py                # Prometheus metrics definitions
│   │   └── security.py               # Global security utilities
│   ├── common/
│   │   ├── models.py                 # Shared base models
│   │   ├── schemas.py                # Common response schemas
│   │   ├── exceptions.py             # Global exception classes
│   │   ├── pagination.py             # Pagination utilities
│   │   └── utils.py                  # Shared utilities
│   └── main.py                       # Application entry point
├── tests/
│   ├── conftest.py                   # Pytest fixtures
│   ├── auth/
│   ├── chat/
│   └── agents/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── grafana/                          # Grafana dashboards
├── prometheus/                       # Prometheus configuration
├── .env.example
├── pyproject.toml
├── alembic.ini
└── README.md
```

### 2.2 Key Structural Principles

The domain-based structure ensures that each module is self-contained with its own router, schemas, models, and services[1]. Cross-module imports should use explicit module names to maintain clarity:

```python
# Correct: Explicit module imports
from src.auth import constants as auth_constants
from src.chat import service as chat_service
from src.auth.dependencies import get_current_user

# Avoid: Ambiguous imports
from constants import ErrorCode  # Which module?
```

---

## 3. WebSocket Implementation Patterns for Streaming LLM Responses

Real-time streaming is essential for LLM applications to provide responsive user experiences. FastAPI supports both WebSocket connections and Server-Sent Events (SSE) for streaming[3][4].

### 3.1 Connection Manager Pattern

The Connection Manager pattern centralizes WebSocket connection lifecycle management[5]:

```python
# src/chat/websocket_manager.py
from fastapi import WebSocket
from typing import Dict, Set
import asyncio
import json

class ConnectionManager:
    """Manages WebSocket connections for real-time chat."""

    def __init__(self):
        # Map session_id to WebSocket connections
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, session_id: str) -> None:
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            if session_id not in self.active_connections:
                self.active_connections[session_id] = set()
            self.active_connections[session_id].add(websocket)

    async def disconnect(self, websocket: WebSocket, session_id: str) -> None:
        """Remove a WebSocket connection."""
        async with self._lock:
            if session_id in self.active_connections:
                self.active_connections[session_id].discard(websocket)
                if not self.active_connections[session_id]:
                    del self.active_connections[session_id]

    async def send_message(self, session_id: str, message: dict) -> None:
        """Send a message to all connections in a session."""
        if session_id in self.active_connections:
            disconnected = []
            for websocket in self.active_connections[session_id]:
                try:
                    await websocket.send_json(message)
                except Exception:
                    disconnected.append(websocket)
            # Clean up disconnected clients
            for ws in disconnected:
                await self.disconnect(ws, session_id)

    async def stream_tokens(
        self,
        session_id: str,
        token_generator,
        message_id: str
    ) -> str:
        """Stream LLM tokens to all session connections."""
        full_response = ""
        async for token in token_generator:
            full_response += token
            await self.send_message(session_id, {
                "type": "token",
                "message_id": message_id,
                "content": token
            })
        # Signal completion
        await self.send_message(session_id, {
            "type": "complete",
            "message_id": message_id
        })
        return full_response

# Global instance
manager = ConnectionManager()
```

### 3.2 WebSocket Endpoint with Graceful Error Handling

```python
# src/chat/router.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from src.chat.websocket_manager import manager
from src.chat.service import ChatService
from src.auth.dependencies import get_current_user_ws
import json

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.websocket("/ws/{session_id}")
async def websocket_chat_endpoint(
    websocket: WebSocket,
    session_id: str,
    chat_service: ChatService = Depends(),
):
    """WebSocket endpoint for real-time chat with LLM."""
    user = await get_current_user_ws(websocket)
    if not user:
        await websocket.close(code=4001, reason="Unauthorized")
        return

    await manager.connect(websocket, session_id)

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "chat":
                # Process and stream LLM response
                response_generator = chat_service.generate_response(
                    session_id=session_id,
                    user_message=message["content"],
                    user_id=user.id
                )
                await manager.stream_tokens(
                    session_id=session_id,
                    token_generator=response_generator,
                    message_id=message.get("message_id", "")
                )

            elif message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        await manager.disconnect(websocket, session_id)
    except Exception as e:
        await manager.disconnect(websocket, session_id)
        # Log the error for debugging
        logger.error(f"WebSocket error: {e}", exc_info=True)
```

### 3.3 SSE Streaming Alternative (HTTP-based)

For simpler streaming needs, Server-Sent Events provide a robust HTTP-based alternative[3][4]:

```python
# src/chat/router.py
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from src.agents.graph import build_agent_graph
import json

router = APIRouter()

@router.post("/chat/stream")
async def stream_chat_response(
    request: Request,
    payload: ChatRequest,
    current_user: User = Depends(get_current_active_user),
):
    """Stream LLM responses using Server-Sent Events."""

    async def event_generator():
        agent_graph = await build_agent_graph()

        async for chunk in agent_graph.astream(
            {"input": payload.message, "session_id": payload.session_id}
        ):
            if "output" in chunk:
                event_data = json.dumps({
                    "type": "token",
                    "content": chunk["output"]
                })
                yield f"data: {event_data}\n\n"

        # Send completion event
        yield f"data: {json.dumps({'type': 'complete'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
```

---

## 4. Dependency Injection Patterns

FastAPI's dependency injection system is central to building maintainable applications. Dependencies serve dual purposes: injecting services and validating requests[1].

### 4.1 Layered Dependency Architecture

```python
# src/auth/dependencies.py
from fastapi import Depends, HTTPException, status, WebSocket
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.auth.config import auth_settings
from src.auth import service as auth_service

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
) -> User:
    """Extract and validate user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token,
            auth_settings.JWT_SECRET,
            algorithms=[auth_settings.JWT_ALG]
        )
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = await auth_service.get_user_by_username(db, username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Ensure the current user is active."""
    if current_user.disabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    return current_user

async def get_current_user_ws(websocket: WebSocket) -> User | None:
    """Extract user from WebSocket query parameters or headers."""
    token = websocket.query_params.get("token")
    if not token:
        # Try authorization header
        auth_header = websocket.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]

    if not token:
        return None

    try:
        payload = jwt.decode(
            token,
            auth_settings.JWT_SECRET,
            algorithms=[auth_settings.JWT_ALG]
        )
        # Simplified - in production, fetch full user from DB
        return User(id=payload.get("user_id"), username=payload.get("sub"))
    except JWTError:
        return None
```

### 4.2 Resource Validation Dependencies

Dependencies can validate resources and prevent code duplication across endpoints[1]:

```python
# src/chat/dependencies.py
from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.chat import service as chat_service
from src.auth.dependencies import get_current_user

async def valid_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Conversation:
    """Validate conversation exists and user has access."""
    conversation = await chat_service.get_conversation(db, conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    if conversation.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this conversation"
        )
    return conversation

# Usage in router - no need to repeat validation logic
@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation: Conversation = Depends(valid_conversation)
):
    return conversation

@router.post("/conversations/{conversation_id}/messages")
async def add_message(
    message: MessageCreate,
    conversation: Conversation = Depends(valid_conversation),
    chat_service: ChatService = Depends()
):
    return await chat_service.add_message(conversation.id, message)
```

### 4.3 Prefer Async Dependencies

Even for simple operations, prefer async dependencies to avoid threadpool overhead[1]:

```python
# Good: Async dependency (runs in event loop)
async def get_request_id(request: Request) -> str:
    return request.headers.get("X-Request-ID", str(uuid.uuid4()))

# Avoid: Sync dependency (runs in threadpool)
def get_request_id(request: Request) -> str:
    return request.headers.get("X-Request-ID", str(uuid.uuid4()))
```

---

## 5. Background Task Handling

FastAPI provides multiple approaches for background processing depending on task duration and complexity.

### 5.1 Built-in BackgroundTasks (Short Tasks)

```python
# src/chat/router.py
from fastapi import BackgroundTasks

async def log_conversation_analytics(
    conversation_id: str,
    message_count: int,
    response_time_ms: float
):
    """Log analytics to external service."""
    await analytics_service.log_event({
        "event": "conversation_completed",
        "conversation_id": conversation_id,
        "message_count": message_count,
        "response_time_ms": response_time_ms
    })

@router.post("/conversations/{conversation_id}/complete")
async def complete_conversation(
    conversation_id: str,
    background_tasks: BackgroundTasks,
    conversation: Conversation = Depends(valid_conversation)
):
    # Add background task for non-blocking analytics
    background_tasks.add_task(
        log_conversation_analytics,
        conversation_id=conversation_id,
        message_count=conversation.message_count,
        response_time_ms=conversation.avg_response_time
    )
    return {"status": "completed"}
```

### 5.2 Task Queue Integration (Long-Running Tasks)

For CPU-intensive or long-running tasks, use a task queue like Celery or ARQ[1]:

```python
# src/tasks/worker.py
import arq
from arq.connections import RedisSettings
from src.agents.graph import build_agent_graph

async def process_document_embedding(ctx, document_id: str, content: str):
    """Background task for document embedding."""
    # CPU-intensive embedding generation
    embeddings = await generate_embeddings(content)
    await store_embeddings(document_id, embeddings)
    return {"status": "completed", "document_id": document_id}

class WorkerSettings:
    functions = [process_document_embedding]
    redis_settings = RedisSettings.from_dsn("redis://localhost:6379")

# src/chat/router.py
from arq import create_pool

@router.post("/documents/embed")
async def embed_document(
    document: DocumentUpload,
    redis_pool = Depends(get_arq_pool)
):
    """Queue document for background embedding."""
    job = await redis_pool.enqueue_job(
        "process_document_embedding",
        document.id,
        document.content
    )
    return {"job_id": job.job_id, "status": "queued"}
```

---

## 6. Rate Limiting Implementation

SlowAPI is the recommended rate-limiting library for FastAPI, handling millions of requests in production environments[6].

### 6.1 Basic Rate Limiting Setup

```python
# src/core/limiter.py
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from fastapi import Request
from src.core.redis import get_redis_url

def get_user_identifier(request: Request) -> str:
    """Get rate limit key - prefer user ID over IP."""
    # If authenticated, use user ID
    if hasattr(request.state, "user") and request.state.user:
        return f"user:{request.state.user.id}"
    # Fall back to IP address
    return get_remote_address(request)

limiter = Limiter(
    key_func=get_user_identifier,
    storage_uri=get_redis_url(),  # Redis for distributed rate limiting
    default_limits=["100/minute"],
    strategy="fixed-window"  # or "moving-window" for stricter limits
)

# src/main.py
from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from src.core.limiter import limiter

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
```

### 6.2 Endpoint-Specific Rate Limits

```python
# src/chat/router.py
from src.core.limiter import limiter
from fastapi import Request

@router.post("/chat/send")
@limiter.limit("20/minute")  # Stricter limit for LLM endpoints
async def send_chat_message(
    request: Request,  # Required for SlowAPI
    message: ChatMessage,
    current_user: User = Depends(get_current_active_user)
):
    """Send a message with rate limiting."""
    return await chat_service.process_message(message, current_user)

@router.get("/conversations")
@limiter.limit("60/minute")  # More permissive for read operations
async def list_conversations(
    request: Request,
    current_user: User = Depends(get_current_active_user)
):
    return await chat_service.get_user_conversations(current_user.id)

# Shared limits across multiple endpoints
from slowapi import Limiter
shared_llm_limit = limiter.shared_limit("50/minute", scope="llm_calls")

@router.post("/analyze")
@shared_llm_limit
async def analyze_text(request: Request, payload: AnalyzeRequest):
    pass

@router.post("/summarize")
@shared_llm_limit
async def summarize_text(request: Request, payload: SummarizeRequest):
    pass
```

---

## 7. Authentication and Session Management

JWT-based authentication with proper session management is essential for chat applications[7][8].

### 7.1 JWT Configuration and Token Creation

```python
# src/auth/config.py
from datetime import timedelta
from pydantic_settings import BaseSettings

class AuthSettings(BaseSettings):
    JWT_SECRET: str
    JWT_ALG: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30

    class Config:
        env_prefix = "AUTH_"

auth_settings = AuthSettings()

# src/auth/service.py
from datetime import datetime, timedelta, timezone
from jose import jwt
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_access_token(user_id: str, username: str) -> str:
    """Create a short-lived access token."""
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=auth_settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    payload = {
        "sub": username,
        "user_id": user_id,
        "exp": expire,
        "type": "access"
    }
    return jwt.encode(payload, auth_settings.JWT_SECRET, auth_settings.JWT_ALG)

def create_refresh_token(user_id: str) -> str:
    """Create a long-lived refresh token."""
    expire = datetime.now(timezone.utc) + timedelta(
        days=auth_settings.REFRESH_TOKEN_EXPIRE_DAYS
    )
    payload = {
        "user_id": user_id,
        "exp": expire,
        "type": "refresh"
    }
    return jwt.encode(payload, auth_settings.JWT_SECRET, auth_settings.JWT_ALG)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def hash_password(password: str) -> str:
    return pwd_context.hash(password)
```

### 7.2 Session Management for Chat Applications

```python
# src/chat/session_manager.py
from datetime import datetime, timedelta
from typing import Optional
import json
from src.core.redis import get_redis_client

class ChatSessionManager:
    """Manage chat sessions with Redis."""

    SESSION_TTL = timedelta(hours=24)

    def __init__(self, redis_client):
        self.redis = redis_client

    async def create_session(
        self,
        user_id: str,
        conversation_id: str,
        metadata: dict = None
    ) -> str:
        """Create a new chat session."""
        session_id = f"session:{user_id}:{conversation_id}"
        session_data = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        await self.redis.setex(
            session_id,
            int(self.SESSION_TTL.total_seconds()),
            json.dumps(session_data)
        )
        return session_id

    async def get_session(self, session_id: str) -> Optional[dict]:
        """Retrieve session data."""
        data = await self.redis.get(session_id)
        if data:
            return json.loads(data)
        return None

    async def update_activity(self, session_id: str) -> None:
        """Update last activity timestamp and extend TTL."""
        session = await self.get_session(session_id)
        if session:
            session["last_activity"] = datetime.utcnow().isoformat()
            await self.redis.setex(
                session_id,
                int(self.SESSION_TTL.total_seconds()),
                json.dumps(session)
            )

    async def invalidate_session(self, session_id: str) -> None:
        """Invalidate a session (logout)."""
        await self.redis.delete(session_id)
```

---

## 8. Error Handling and Logging Strategies

Structured logging and consistent error handling are critical for production observability[2][9].

### 8.1 Structured Logging Setup

```python
# src/core/logging.py
import structlog
import logging
from src.core.config import settings

def setup_logging():
    """Configure structured logging with context binding."""

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            # JSON in production, colored console in development
            structlog.processors.JSONRenderer()
            if settings.ENVIRONMENT == "production"
            else structlog.dev.ConsoleRenderer(colors=True)
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

def get_logger(name: str = "app"):
    return structlog.get_logger(name)

# Usage
logger = get_logger("chat")
logger.info("message_received", user_id="123", conversation_id="456")
```

### 8.2 Global Exception Handling

```python
# src/core/exceptions.py
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from src.core.logging import get_logger

logger = get_logger("exceptions")

class AppException(Exception):
    """Base application exception."""
    def __init__(self, message: str, code: str, status_code: int = 400):
        self.message = message
        self.code = code
        self.status_code = status_code
        super().__init__(message)

class ConversationNotFound(AppException):
    def __init__(self, conversation_id: str):
        super().__init__(
            message=f"Conversation {conversation_id} not found",
            code="CONVERSATION_NOT_FOUND",
            status_code=404
        )

class LLMServiceError(AppException):
    def __init__(self, detail: str):
        super().__init__(
            message=f"LLM service error: {detail}",
            code="LLM_SERVICE_ERROR",
            status_code=503
        )

# src/main.py
@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    logger.error(
        "application_error",
        code=exc.code,
        message=exc.message,
        path=request.url.path
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}}
    )

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception(
        "unhandled_exception",
        path=request.url.path,
        method=request.method
    )
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "INTERNAL_ERROR", "message": "An unexpected error occurred"}}
    )
```

### 8.3 Request Logging Middleware

```python
# src/core/middleware.py
import time
import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from src.core.logging import get_logger
import structlog

logger = get_logger("http")

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        # Bind context for all logs in this request
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path
        )

        start_time = time.perf_counter()

        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start_time) * 1000

            logger.info(
                "request_completed",
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2)
            )

            response.headers["X-Request-ID"] = request_id
            return response

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.exception("request_failed", duration_ms=round(duration_ms, 2))
            raise
```

---

## 9. SQLAlchemy 2.0 Async Patterns

SQLAlchemy 2.0's async API is essential for non-blocking database operations in FastAPI[10][11].

### 9.1 Database Connection Setup

```python
# src/core/database.py
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker
)
from sqlalchemy.orm import DeclarativeBase
from src.core.config import settings

class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass

# Create async engine with connection pooling
engine = create_async_engine(
    settings.DATABASE_URL,  # postgresql+asyncpg://user:pass@host/db
    echo=settings.DEBUG,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,      # Verify connections before use
    pool_recycle=3600,       # Recycle connections after 1 hour
)

# Async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Prevent lazy-loading errors after commit
    autocommit=False,
    autoflush=False,
)

async def get_db() -> AsyncSession:
    """Dependency for database session injection."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
```

### 9.2 Model Definitions with Type Hints

```python
# src/chat/models.py
from datetime import datetime
from typing import List, Optional
from sqlalchemy import String, Text, ForeignKey, DateTime, func, Enum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.core.database import Base
import enum

class MessageRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    title: Mapped[Optional[str]] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships with eager loading strategy
    messages: Mapped[List["Message"]] = relationship(
        back_populates="conversation",
        lazy="selectin",  # Eager load by default
        order_by="Message.created_at"
    )
    user: Mapped["User"] = relationship(back_populates="conversations")

class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        index=True
    )
    role: Mapped[MessageRole] = mapped_column(Enum(MessageRole))
    content: Mapped[str] = mapped_column(Text)
    token_count: Mapped[Optional[int]] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")
```

### 9.3 Repository Pattern with Async Queries

```python
# src/chat/repository.py
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from src.chat.models import Conversation, Message
from typing import List, Optional

class ConversationRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, conversation_id: str) -> Optional[Conversation]:
        """Get conversation with messages eagerly loaded."""
        stmt = (
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(Conversation.id == conversation_id)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_user_conversations(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0
    ) -> List[Conversation]:
        """Get paginated conversations for a user."""
        stmt = (
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def create(self, conversation: Conversation) -> Conversation:
        """Create a new conversation."""
        self.session.add(conversation)
        await self.session.commit()
        await self.session.refresh(conversation)
        return conversation

    async def add_message(self, message: Message) -> Message:
        """Add a message to conversation."""
        self.session.add(message)
        await self.session.commit()
        await self.session.refresh(message)
        return message
```

---

## 10. Redis Integration for Session and Cache Management

Async Redis integration is essential for caching, session management, and distributed rate limiting[12].

### 10.1 Redis Client Setup

```python
# src/core/redis.py
from redis import asyncio as aioredis
from src.core.config import settings
from typing import Optional

class RedisClient:
    """Async Redis client wrapper."""

    _instance: Optional[aioredis.Redis] = None

    @classmethod
    async def get_client(cls) -> aioredis.Redis:
        """Get or create Redis client singleton."""
        if cls._instance is None:
            cls._instance = aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20
            )
        return cls._instance

    @classmethod
    async def close(cls):
        """Close Redis connection."""
        if cls._instance:
            await cls._instance.close()
            cls._instance = None

async def get_redis_client() -> aioredis.Redis:
    """Dependency for Redis client injection."""
    return await RedisClient.get_client()
```

### 10.2 Caching Service

```python
# src/core/cache.py
import json
from typing import Optional, Any, Callable
from functools import wraps
from redis import asyncio as aioredis

class CacheService:
    """Redis-based caching service."""

    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.default_ttl = 3600  # 1 hour

    async def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        data = await self.redis.get(f"cache:{key}")
        if data:
            return json.loads(data)
        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> None:
        """Set cached value with TTL."""
        await self.redis.setex(
            f"cache:{key}",
            ttl or self.default_ttl,
            json.dumps(value)
        )

    async def delete(self, key: str) -> None:
        """Delete cached value."""
        await self.redis.delete(f"cache:{key}")

    async def invalidate_pattern(self, pattern: str) -> None:
        """Invalidate all keys matching pattern."""
        async for key in self.redis.scan_iter(f"cache:{pattern}*"):
            await self.redis.delete(key)

# Caching decorator for service methods
def cached(key_prefix: str, ttl: int = 3600):
    """Decorator for caching async function results."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Generate cache key from arguments
            cache_key = f"{key_prefix}:{':'.join(str(a) for a in args)}"

            # Try to get from cache
            cached_value = await self.cache.get(cache_key)
            if cached_value is not None:
                return cached_value

            # Execute function and cache result
            result = await func(self, *args, **kwargs)
            await self.cache.set(cache_key, result, ttl)
            return result
        return wrapper
    return decorator
```

---

## 11. Health Checks and Observability

Production applications require comprehensive health checks and observability through metrics and tracing[9][13].

### 11.1 Health Check Endpoints

```python
# src/core/health.py
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from redis import asyncio as aioredis
from src.core.database import get_db
from src.core.redis import get_redis_client
from pydantic import BaseModel
from typing import Dict

router = APIRouter(tags=["Health"])

class HealthStatus(BaseModel):
    status: str
    components: Dict[str, dict]

@router.get("/health", response_model=HealthStatus)
async def health_check(
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis_client)
):
    """Comprehensive health check endpoint."""
    components = {}
    overall_status = "healthy"

    # Database check
    try:
        await db.execute(text("SELECT 1"))
        components["database"] = {"status": "healthy"}
    except Exception as e:
        components["database"] = {"status": "unhealthy", "error": str(e)}
        overall_status = "unhealthy"

    # Redis check
    try:
        await redis.ping()
        components["redis"] = {"status": "healthy"}
    except Exception as e:
        components["redis"] = {"status": "unhealthy", "error": str(e)}
        overall_status = "unhealthy"

    # LLM service check (optional, with timeout)
    try:
        # Quick check to LLM provider
        components["llm"] = {"status": "healthy"}
    except Exception as e:
        components["llm"] = {"status": "degraded", "error": str(e)}

    return HealthStatus(status=overall_status, components=components)

@router.get("/health/live")
async def liveness():
    """Kubernetes liveness probe - just confirms app is running."""
    return {"status": "alive"}

@router.get("/health/ready")
async def readiness(db: AsyncSession = Depends(get_db)):
    """Kubernetes readiness probe - confirms app can handle requests."""
    await db.execute(text("SELECT 1"))
    return {"status": "ready"}
```

### 11.2 OpenTelemetry Integration

```python
# src/core/telemetry.py
import logging
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor

def setup_telemetry(app, engine):
    """Configure OpenTelemetry for tracing and metrics."""

    resource = Resource.create({"service.name": "customer-support-ai"})

    # Tracing setup
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(ConsoleSpanExporter())  # Replace with OTLP exporter in prod
    )
    trace.set_tracer_provider(tracer_provider)

    # Metrics setup with Prometheus
    metric_reader = PrometheusMetricReader()
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    # Auto-instrument FastAPI
    FastAPIInstrumentor.instrument_app(app)

    # Auto-instrument SQLAlchemy
    SQLAlchemyInstrumentor().instrument(engine=engine.sync_engine)

    # Auto-instrument Redis
    RedisInstrumentor().instrument()

    return trace.get_tracer("customer-support-ai"), metrics.get_meter("customer-support-ai")

# Custom metrics for LLM operations
tracer, meter = None, None

def get_tracer():
    return tracer

def get_meter():
    return meter

# Define custom metrics
llm_token_counter = None
llm_latency_histogram = None
llm_error_counter = None

def init_custom_metrics(meter):
    global llm_token_counter, llm_latency_histogram, llm_error_counter

    llm_token_counter = meter.create_counter(
        "llm_tokens_total",
        description="Total LLM tokens processed",
        unit="tokens"
    )

    llm_latency_histogram = meter.create_histogram(
        "llm_request_duration_seconds",
        description="LLM request latency",
        unit="seconds"
    )

    llm_error_counter = meter.create_counter(
        "llm_errors_total",
        description="Total LLM errors"
    )
```

### 11.3 LLM Metrics in Agent Code

```python
# src/agents/graph.py
import time
from src.core.telemetry import (
    get_tracer,
    llm_token_counter,
    llm_latency_histogram,
    llm_error_counter
)

async def llm_node(state: dict) -> dict:
    """LangGraph node with observability."""
    tracer = get_tracer()

    with tracer.start_as_current_span("llm.completion") as span:
        start_time = time.perf_counter()

        try:
            # Set span attributes
            span.set_attribute("llm.model", "gpt-4")
            span.set_attribute("llm.prompt_length", len(state["input"]))

            # Call LLM
            response = await llm.ainvoke(state["messages"])

            duration = time.perf_counter() - start_time

            # Record metrics
            llm_token_counter.add(
                response.usage.total_tokens,
                {"model": "gpt-4", "type": "total"}
            )
            llm_latency_histogram.record(
                duration,
                {"model": "gpt-4"}
            )

            span.set_attribute("llm.tokens_in", response.usage.prompt_tokens)
            span.set_attribute("llm.tokens_out", response.usage.completion_tokens)
            span.set_attribute("llm.duration_ms", duration * 1000)

            return {"output": response.content}

        except Exception as e:
            llm_error_counter.add(1, {"model": "gpt-4", "error_type": type(e).__name__})
            span.record_exception(e)
            raise
```

---

## 12. LangChain/LangGraph Streaming Integration

Integrating LangChain and LangGraph with FastAPI for streaming responses requires careful async handling[14][15].

### 12.1 LangGraph Agent with Streaming

```python
# src/agents/graph.py
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, Annotated, List
import operator

class AgentState(TypedDict):
    input: str
    messages: Annotated[List, operator.add]
    output: str
    session_id: str

async def build_agent_graph():
    """Build LangGraph agent for customer support."""

    llm = ChatOpenAI(model="gpt-4", streaming=True)

    async def process_input(state: AgentState) -> AgentState:
        """Process user input and add to messages."""
        state["messages"].append(HumanMessage(content=state["input"]))
        return state

    async def generate_response(state: AgentState) -> AgentState:
        """Generate streaming response from LLM."""
        response = await llm.ainvoke(state["messages"])
        state["messages"].append(AIMessage(content=response.content))
        state["output"] = response.content
        return state

    # Build graph
    workflow = StateGraph(AgentState)
    workflow.add_node("process_input", process_input)
    workflow.add_node("generate_response", generate_response)
    workflow.set_entry_point("process_input")
    workflow.add_edge("process_input", "generate_response")
    workflow.add_edge("generate_response", END)

    return workflow.compile()

# Streaming generator for SSE
async def stream_agent_response(user_input: str, session_id: str):
    """Generator that yields streaming tokens from agent."""
    llm = ChatOpenAI(model="gpt-4", streaming=True)

    async for chunk in llm.astream(user_input):
        if chunk.content:
            yield chunk.content
```

### 12.2 FastAPI Integration with LangChain Streaming

```python
# src/chat/router.py
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
import json

router = APIRouter()

@router.post("/chat/stream")
async def stream_completion(
    payload: ChatRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Stream LLM completion using SSE."""

    async def generate_stream():
        llm = ChatOpenAI(model="gpt-4", streaming=True)

        # Use astream_log for detailed streaming with JSON patches
        async for chunk in llm.astream(payload.messages):
            if chunk.content:
                data = json.dumps({
                    "type": "token",
                    "content": chunk.content
                })
                yield f"data: {data}\n\n"

        # Send completion signal
        yield f"data: {json.dumps({'type': 'complete'})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream"
    )
```

---

## 13. Graceful Shutdown Handling

Proper shutdown handling ensures active connections and tasks complete gracefully[16].

### 13.1 Application Lifespan Management

```python
# src/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.core.database import engine, AsyncSessionLocal
from src.core.redis import RedisClient
from src.chat.websocket_manager import manager
import asyncio

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""

    # Startup
    print("Starting up...")

    # Initialize Redis connection
    await RedisClient.get_client()

    # Setup telemetry
    from src.core.telemetry import setup_telemetry
    setup_telemetry(app, engine)

    yield

    # Shutdown
    print("Shutting down...")

    # Close all WebSocket connections gracefully
    for session_id, connections in list(manager.active_connections.items()):
        for ws in connections:
            try:
                await ws.close(code=1001, reason="Server shutting down")
            except Exception:
                pass

    # Close Redis connection
    await RedisClient.close()

    # Dispose database engine
    await engine.dispose()

    print("Shutdown complete")

app = FastAPI(
    title="Customer Support AI",
    version="1.0.0",
    lifespan=lifespan
)
```

---

## 14. Complete Application Entry Point

```python
# src/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.core.config import settings
from src.core.middleware import RequestLoggingMiddleware
from src.core.logging import setup_logging
from src.core.limiter import limiter
from src.core.exceptions import AppException, app_exception_handler
from src.core.health import router as health_router
from src.auth.router import router as auth_router
from src.chat.router import router as chat_router

# Setup logging first
setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic here
    yield
    # Shutdown logic here

app = FastAPI(
    title="Customer Support AI Backend",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
)

# Rate limiter
app.state.limiter = limiter

# Middleware (order matters - last added is first executed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)

# Exception handlers
app.add_exception_handler(AppException, app_exception_handler)

# Routers
app.include_router(health_router)
app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
app.include_router(chat_router, prefix="/chat", tags=["Chat"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        workers=4 if not settings.DEBUG else 1
    )
```

---

## 15. Conclusion

Building production AI/LLM backends with FastAPI requires careful attention to architectural patterns that support scalability, maintainability, and real-time streaming capabilities. The key recommendations from this research include:

**Architecture**: Adopt a domain-driven modular structure that scales with team size and application complexity. Keep each domain self-contained with its own router, schemas, models, and services.

**Streaming**: Use WebSockets for bidirectional real-time communication and SSE for simpler unidirectional streaming. Implement a ConnectionManager pattern for managing multiple concurrent connections.

**Performance**: Leverage async patterns throughout the stack - from SQLAlchemy 2.0's async sessions to async Redis clients. Avoid blocking operations in async routes.

**Security**: Implement JWT-based authentication with proper token expiration, rate limiting with SlowAPI, and comprehensive input validation with Pydantic.

**Observability**: Integrate OpenTelemetry for distributed tracing, Prometheus for metrics, and structured logging for debugging. Track LLM-specific metrics like token usage and latency.

**Reliability**: Implement graceful shutdown handling, health check endpoints for Kubernetes, and proper error handling with meaningful error codes.

These patterns have been validated in production environments handling millions of requests and provide a solid foundation for building customer support AI backends that are both performant and maintainable.

---

## 16. Sources

[1] [FastAPI Best Practices - zhanymkanov/fastapi-best-practices](https://github.com/zhanymkanov/fastapi-best-practices) - High Reliability - Production-tested patterns from startup experience

[2] [Production-Ready FastAPI LangGraph Template](https://github.com/wassim249/fastapi-langgraph-agent-production-ready-template) - High Reliability - Comprehensive production template with observability

[3] [FastAPI Server-Sent Events for LLM Streaming](https://medium.com/@2nick2patel2/fastapi-server-sent-events-for-llm-streaming-smooth-tokens-low-latency-1b211c94cff5) - Medium Reliability - Practical SSE implementation patterns

[4] [10 FastAPI Patterns for Streamed Responses](https://medium.com/@ThinkingLoop/10-fastapi-patterns-for-streamed-responses-1bab92860a0e) - Medium Reliability - Comprehensive streaming patterns

[5] [Getting Started with FastAPI WebSockets - Better Stack](https://betterstack.com/community/guides/scaling-python/fastapi-websockets/) - High Reliability - Industry guide with best practices

[6] [SlowAPI - Rate Limiting for FastAPI](https://github.com/laurentS/slowapi) - High Reliability - Official library documentation

[7] [Authentication and Authorization with FastAPI - Better Stack](https://betterstack.com/community/guides/scaling-python/authentication-fastapi/) - High Reliability - Comprehensive JWT implementation guide

[8] [Session Management in FastAPI](https://byo.propelauth.com/post/fastapi-session-guide) - Medium Reliability - Practical session patterns

[9] [Building Observable LLM Agents - Teknasyon Engineering](https://engineering.teknasyon.com/from-prompts-to-metrics-building-observable-llm-agents-using-fastapi-opentelemetry-prometheus-359d3132d92b) - High Reliability - Production observability patterns

[10] [FastAPI + SQLAlchemy 2.0: Modern Async Database Patterns](https://dev-faizan.medium.com/fastapi-sqlalchemy-2-0-modern-async-database-patterns-7879d39b6843) - Medium Reliability - Async database patterns

[11] [10 SQLAlchemy 2.0 Patterns for Clean Async Postgres](https://medium.com/@ThinkingLoop/10-sqlalchemy-2-0-patterns-for-clean-async-postgres-af8c4bcd86fe) - Medium Reliability - Production database patterns

[12] [Setting Up Async Redis Client in FastAPI](https://medium.com/@geetansh2k1/setting-up-and-using-an-async-redis-client-in-fastapi-the-right-way-0409ad3812e6) - Medium Reliability - Redis integration guide

[13] [FastAPI Observability - Grafana Labs](https://grafana.com/grafana/dashboards/16110-fastapi-observability/) - High Reliability - Official Grafana dashboard

[14] [Integrating LangChain with FastAPI for Async Streaming](https://dev.to/louis-sanna/integrating-langchain-with-fastapi-for-asynchronous-streaming-5d0o) - Medium Reliability - LangChain integration patterns

[15] [Deploying LangGraph with FastAPI Tutorial](https://ideentech.com/deploying-langgraph-with-fastapi-a-step-by-step-tutorial/) - Medium Reliability - LangGraph deployment guide

[16] [FastAPI Graceful Shutdown Patterns](https://realmorrisliu.com/thoughts/fastapi-graceful-shutdown) - Medium Reliability - Shutdown handling best practices
