Phase 1: Foundation Setup — Meticulous Planning & Execution
Phase 1 Execution Plan
Pre-Implementation Analysis
text

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           PHASE 1: FOUNDATION SETUP                                 │
│                              Detailed Execution Plan                                │
└─────────────────────────────────────────────────────────────────────────────────────┘

OBJECTIVE: Establish rock-solid infrastructure foundation
DURATION: 5-7 days
RISK LEVEL: Low (foundational work)

┌─────────────────────────────────────────────────────────────────────────────────────┐
│ TASK BREAKDOWN                                                                      │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│ 1.1 PROJECT INITIALIZATION                              Estimated: 0.5 days        │
│     ├── Monorepo structure with clear separation                                    │
│     ├── Frontend: Vite + React 18 + TypeScript 5.x                                  │
│     ├── Backend: Python 3.12 + Poetry + FastAPI                                     │
│     ├── Shared environment configuration                                            │
│     └── Code quality tooling (linting, formatting)                                  │
│                                                                                     │
│ 1.2 INFRASTRUCTURE (Docker)                             Estimated: 1 day           │
│     ├── docker-compose.yml with all services                                        │
│     ├── PostgreSQL 16 with initial setup                                            │
│     ├── Redis 7 with persistence config                                             │
│     ├── Qdrant latest with collection schemas                                       │
│     ├── Network configuration                                                       │
│     └── Volume management for data persistence                                      │
│                                                                                     │
│ 1.3 DATABASE SCHEMA                                     Estimated: 1.5 days         │
│     ├── SQLAlchemy 2.0 async models                                                 │
│     ├── Pydantic v2 schemas (all domain models)                                     │
│     ├── Alembic migration setup                                                     │
│     ├── Initial migrations                                                          │
│     └── Seed data scripts                                                           │
│                                                                                     │
│ 1.4 CORE API STRUCTURE                                  Estimated: 1.5 days         │
│     ├── FastAPI application factory                                                 │
│     ├── Dependency injection system                                                 │
│     ├── Configuration management (Pydantic Settings)                                │
│     ├── Database connection pooling                                                 │
│     ├── Redis connection management                                                 │
│     ├── Qdrant client setup                                                         │
│     ├── Health check endpoints                                                      │
│     ├── Structured logging                                                          │
│     └── Error handling middleware                                                   │
│                                                                                     │
│ 1.5 FRONTEND SCAFFOLD                                   Estimated: 0.5 days         │
│     ├── Vite configuration                                                          │
│     ├── Tailwind CSS 4.0 setup                                                      │
│     ├── Shadcn/UI initialization                                                    │
│     ├── TypeScript strict mode                                                      │
│     └── Basic App shell                                                             │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
Execution: Complete Implementation
1.1 Project Root Structure
text

singapore-smb-support-agent/
├── frontend/                    # React application
├── backend/                     # FastAPI application
├── infrastructure/              # Docker & deployment
├── knowledge_base/              # Source documents (placeholder)
├── docs/                        # Documentation
├── .env.example                 # Environment template
├── .gitignore
├── Makefile                     # Development commands
└── README.md
File: .env.example
env

# =============================================================================
# SINGAPORE SMB SUPPORT AGENT - ENVIRONMENT CONFIGURATION
# =============================================================================
# Copy this file to .env and fill in your values
# NEVER commit .env to version control

# -----------------------------------------------------------------------------
# APPLICATION
# -----------------------------------------------------------------------------
APP_NAME="Singapore SMB Support Agent"
APP_ENV=development
DEBUG=true
LOG_LEVEL=DEBUG

# -----------------------------------------------------------------------------
# API CONFIGURATION
# -----------------------------------------------------------------------------
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true
CORS_ORIGINS=["http://localhost:5173","http://localhost:3000"]

# -----------------------------------------------------------------------------
# DATABASE - PostgreSQL
# -----------------------------------------------------------------------------
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=smb_agent
POSTGRES_PASSWORD=secure_password_change_me
POSTGRES_DB=smb_support_agent
DATABASE_URL=postgresql+asyncpg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}

# -----------------------------------------------------------------------------
# CACHE - Redis
# -----------------------------------------------------------------------------
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0
REDIS_URL=redis://${REDIS_HOST}:${REDIS_PORT}/${REDIS_DB}

# Session TTL in seconds (24 hours)
SESSION_TTL=86400

# -----------------------------------------------------------------------------
# VECTOR DATABASE - Qdrant
# -----------------------------------------------------------------------------
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_GRPC_PORT=6334
QDRANT_API_KEY=

# Collection names
QDRANT_KNOWLEDGE_COLLECTION=knowledge_base
QDRANT_FAQ_COLLECTION=faqs
QDRANT_MEMORY_COLLECTION=conversation_memory

# -----------------------------------------------------------------------------
# LLM CONFIGURATION - OpenAI
# -----------------------------------------------------------------------------
OPENAI_API_KEY=sk-your-api-key-here

# Model selection
LLM_MODEL_PRIMARY=gpt-4o-mini
LLM_MODEL_COMPLEX=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536

# Token limits
MAX_CONTEXT_TOKENS=4000
MAX_OUTPUT_TOKENS=1000

# -----------------------------------------------------------------------------
# RERANKER
# -----------------------------------------------------------------------------
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-12-v2

# -----------------------------------------------------------------------------
# BUSINESS CONFIGURATION - Singapore SMB
# -----------------------------------------------------------------------------
BUSINESS_NAME="Your Business Name"
BUSINESS_TIMEZONE=Asia/Singapore
SUPPORT_EMAIL=support@yourbusiness.com
ESCALATION_EMAIL=escalation@yourbusiness.com

# Operating hours (24h format, SGT)
BUSINESS_HOURS_START=09:00
BUSINESS_HOURS_END=18:00
BUSINESS_DAYS=["Monday","Tuesday","Wednesday","Thursday","Friday"]

# -----------------------------------------------------------------------------
# SECURITY
# -----------------------------------------------------------------------------
SECRET_KEY=your-secret-key-min-32-chars-change-in-production
API_KEY_HEADER=X-API-Key

# Rate limiting (requests per minute)
RATE_LIMIT_PER_MINUTE=60

# -----------------------------------------------------------------------------
# PDPA COMPLIANCE
# -----------------------------------------------------------------------------
DATA_RETENTION_DAYS=730
REQUIRE_CONSENT=true
File: .gitignore
gitignore

# =============================================================================
# SINGAPORE SMB SUPPORT AGENT - GITIGNORE
# =============================================================================

# -----------------------------------------------------------------------------
# Environment & Secrets
# -----------------------------------------------------------------------------
.env
.env.local
.env.*.local
*.pem
*.key
secrets/

# -----------------------------------------------------------------------------
# Python
# -----------------------------------------------------------------------------
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
.venv/
venv/
ENV/
env/

# Poetry
poetry.lock

# PyInstaller
*.manifest
*.spec

# Pytest
.pytest_cache/
.coverage
htmlcov/
.tox/
.nox/

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Ruff
.ruff_cache/

# -----------------------------------------------------------------------------
# Node.js / Frontend
# -----------------------------------------------------------------------------
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
pnpm-debug.log*
.pnpm-store/

# Build outputs
frontend/dist/
frontend/build/
*.local

# -----------------------------------------------------------------------------
# IDE & Editors
# -----------------------------------------------------------------------------
.idea/
.vscode/
*.swp
*.swo
*~
.project
.classpath
.settings/
*.sublime-workspace
*.sublime-project

# -----------------------------------------------------------------------------
# OS Generated
# -----------------------------------------------------------------------------
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db
desktop.ini

# -----------------------------------------------------------------------------
# Docker
# -----------------------------------------------------------------------------
docker-compose.override.yml

# -----------------------------------------------------------------------------
# Data & Logs
# -----------------------------------------------------------------------------
*.log
logs/
data/
*.sqlite
*.db

# Qdrant data
qdrant_data/

# PostgreSQL data
postgres_data/

# Redis data
redis_data/

# -----------------------------------------------------------------------------
# Knowledge Base (keep structure, ignore content)
# -----------------------------------------------------------------------------
knowledge_base/**/*.pdf
knowledge_base/**/*.docx
knowledge_base/**/*.xlsx
!knowledge_base/.gitkeep
!knowledge_base/**/.gitkeep

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
.coverage
coverage/
*.cover
*.coverage
htmlcov/
test-results/
junit.xml

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
*.bak
*.tmp
*.temp
.cache/
File: Makefile
Makefile

# =============================================================================
# SINGAPORE SMB SUPPORT AGENT - DEVELOPMENT COMMANDS
# =============================================================================

.PHONY: help install dev up down logs clean test lint format migrate seed

# Default target
help:
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║          Singapore SMB Support Agent - Development Commands          ║"
	@echo "╠══════════════════════════════════════════════════════════════════════╣"
	@echo "║                                                                      ║"
	@echo "║  SETUP                                                               ║"
	@echo "║    make install      Install all dependencies                        ║"
	@echo "║    make setup        Complete project setup                          ║"
	@echo "║                                                                      ║"
	@echo "║  DEVELOPMENT                                                         ║"
	@echo "║    make dev          Start all services in dev mode                  ║"
	@echo "║    make up           Start Docker infrastructure only                ║"
	@echo "║    make down         Stop all Docker services                        ║"
	@echo "║    make logs         Tail logs from all services                     ║"
	@echo "║                                                                      ║"
	@echo "║  DATABASE                                                            ║"
	@echo "║    make migrate      Run database migrations                         ║"
	@echo "║    make migrate-new  Create new migration (MSG=description)          ║"
	@echo "║    make seed         Seed database with test data                    ║"
	@echo "║                                                                      ║"
	@echo "║  QUALITY                                                             ║"
	@echo "║    make lint         Run linters                                     ║"
	@echo "║    make format       Format code                                     ║"
	@echo "║    make test         Run all tests                                   ║"
	@echo "║    make test-cov     Run tests with coverage                         ║"
	@echo "║                                                                      ║"
	@echo "║  CLEANUP                                                             ║"
	@echo "║    make clean        Remove build artifacts                          ║"
	@echo "║    make clean-all    Remove all generated files and volumes          ║"
	@echo "║                                                                      ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"

# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------

install: install-backend install-frontend
	@echo "✅ All dependencies installed"

install-backend:
	@echo "📦 Installing backend dependencies..."
	cd backend && poetry install

install-frontend:
	@echo "📦 Installing frontend dependencies..."
	cd frontend && pnpm install

setup: install
	@echo "🔧 Setting up project..."
	cp -n .env.example .env || true
	@echo "⚠️  Please edit .env with your configuration"
	make up
	sleep 5
	make migrate
	@echo "✅ Setup complete!"

# -----------------------------------------------------------------------------
# DEVELOPMENT
# -----------------------------------------------------------------------------

dev:
	@echo "🚀 Starting development environment..."
	make up
	@echo "Starting backend and frontend..."
	(cd backend && poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000) & \
	(cd frontend && pnpm dev) & \
	wait

up:
	@echo "🐳 Starting Docker services..."
	docker-compose -f infrastructure/docker-compose.yml up -d
	@echo "✅ Services started"
	@echo "   PostgreSQL: localhost:5432"
	@echo "   Redis:      localhost:6379"
	@echo "   Qdrant:     localhost:6333 (HTTP) / 6334 (gRPC)"

down:
	@echo "🛑 Stopping Docker services..."
	docker-compose -f infrastructure/docker-compose.yml down
	@echo "✅ Services stopped"

logs:
	docker-compose -f infrastructure/docker-compose.yml logs -f

logs-api:
	docker-compose -f infrastructure/docker-compose.yml logs -f api

# -----------------------------------------------------------------------------
# DATABASE
# -----------------------------------------------------------------------------

migrate:
	@echo "🗃️  Running database migrations..."
	cd backend && poetry run alembic upgrade head
	@echo "✅ Migrations complete"

migrate-new:
	@echo "📝 Creating new migration: $(MSG)"
	cd backend && poetry run alembic revision --autogenerate -m "$(MSG)"

migrate-down:
	@echo "⏪ Rolling back last migration..."
	cd backend && poetry run alembic downgrade -1

seed:
	@echo "🌱 Seeding database..."
	cd backend && poetry run python scripts/seed_data.py
	@echo "✅ Database seeded"

# -----------------------------------------------------------------------------
# QUALITY
# -----------------------------------------------------------------------------

lint: lint-backend lint-frontend
	@echo "✅ Linting complete"

lint-backend:
	@echo "🔍 Linting backend..."
	cd backend && poetry run ruff check .
	cd backend && poetry run mypy app

lint-frontend:
	@echo "🔍 Linting frontend..."
	cd frontend && pnpm lint

format: format-backend format-frontend
	@echo "✅ Formatting complete"

format-backend:
	@echo "🎨 Formatting backend..."
	cd backend && poetry run ruff format .
	cd backend && poetry run ruff check --fix .

format-frontend:
	@echo "🎨 Formatting frontend..."
	cd frontend && pnpm format

test: test-backend test-frontend
	@echo "✅ All tests passed"

test-backend:
	@echo "🧪 Running backend tests..."
	cd backend && poetry run pytest

test-frontend:
	@echo "🧪 Running frontend tests..."
	cd frontend && pnpm test

test-cov:
	@echo "🧪 Running tests with coverage..."
	cd backend && poetry run pytest --cov=app --cov-report=html
	@echo "📊 Coverage report: backend/htmlcov/index.html"

# -----------------------------------------------------------------------------
# QDRANT
# -----------------------------------------------------------------------------

qdrant-init:
	@echo "🔧 Initializing Qdrant collections..."
	cd backend && poetry run python scripts/init_qdrant.py
	@echo "✅ Qdrant collections created"

# -----------------------------------------------------------------------------
# CLEANUP
# -----------------------------------------------------------------------------

clean:
	@echo "🧹 Cleaning build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "node_modules" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
	rm -rf frontend/dist backend/htmlcov
	@echo "✅ Clean complete"

clean-all: clean down
	@echo "🧹 Removing all data volumes..."
	docker-compose -f infrastructure/docker-compose.yml down -v
	rm -rf postgres_data redis_data qdrant_data
	@echo "✅ Full cleanup complete"
File: infrastructure/docker-compose.yml
YAML

# =============================================================================
# SINGAPORE SMB SUPPORT AGENT - DOCKER COMPOSE
# Development Infrastructure
# =============================================================================

version: "3.9"

services:
  # ---------------------------------------------------------------------------
  # PostgreSQL 16 - Primary Database
  # ---------------------------------------------------------------------------
  postgres:
    image: postgres:16-alpine
    container_name: smb_agent_postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-smb_agent}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-secure_password_change_me}
      POSTGRES_DB: ${POSTGRES_DB:-smb_support_agent}
      PGDATA: /var/lib/postgresql/data/pgdata
    ports:
      - "${POSTGRES_PORT:-5432}:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-scripts/postgres:/docker-entrypoint-initdb.d:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-smb_agent} -d ${POSTGRES_DB:-smb_support_agent}"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - smb_agent_network

  # ---------------------------------------------------------------------------
  # Redis 7 - Session Cache & Short-term Memory
  # ---------------------------------------------------------------------------
  redis:
    image: redis:7-alpine
    container_name: smb_agent_redis
    restart: unless-stopped
    command: >
      redis-server
      --appendonly yes
      --maxmemory 256mb
      --maxmemory-policy allkeys-lru
    ports:
      - "${REDIS_PORT:-6379}:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - smb_agent_network

  # ---------------------------------------------------------------------------
  # Qdrant - Vector Database for RAG & Semantic Memory
  # ---------------------------------------------------------------------------
  qdrant:
    image: qdrant/qdrant:latest
    container_name: smb_agent_qdrant
    restart: unless-stopped
    environment:
      QDRANT__SERVICE__GRPC_PORT: 6334
      QDRANT__SERVICE__HTTP_PORT: 6333
      QDRANT__LOG_LEVEL: INFO
    ports:
      - "${QDRANT_PORT:-6333}:6333"
      - "${QDRANT_GRPC_PORT:-6334}:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:6333/readyz"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - smb_agent_network

  # ---------------------------------------------------------------------------
  # Qdrant Dashboard (Optional - Development Only)
  # ---------------------------------------------------------------------------
  qdrant-dashboard:
    image: qdrant/qdrant-web-ui:latest
    container_name: smb_agent_qdrant_ui
    restart: unless-stopped
    environment:
      QDRANT_URL: http://qdrant:6333
    ports:
      - "6335:80"
    depends_on:
      qdrant:
        condition: service_healthy
    networks:
      - smb_agent_network
    profiles:
      - dashboard

# -----------------------------------------------------------------------------
# Networks
# -----------------------------------------------------------------------------
networks:
  smb_agent_network:
    driver: bridge
    name: smb_agent_network

# -----------------------------------------------------------------------------
# Volumes
# -----------------------------------------------------------------------------
volumes:
  postgres_data:
    name: smb_agent_postgres_data
  redis_data:
    name: smb_agent_redis_data
  qdrant_data:
    name: smb_agent_qdrant_data
File: infrastructure/init-scripts/postgres/01-init.sql
SQL

-- =============================================================================
-- SINGAPORE SMB SUPPORT AGENT - PostgreSQL Initialization
-- =============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For fuzzy text search

-- Create custom types
DO $$ BEGIN
    CREATE TYPE language_code AS ENUM ('en', 'zh', 'ms', 'ta', 'en-SG');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE intent_category AS ENUM (
        'product_inquiry',
        'pricing',
        'order_status',
        'booking',
        'complaint',
        'returns',
        'operating_hours',
        'general_faq',
        'escalation',
        'greeting',
        'unknown'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE conversation_status AS ENUM (
        'active',
        'resolved',
        'escalated',
        'abandoned'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Grant privileges (for application user if different from owner)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO smb_agent;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO smb_agent;

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'PostgreSQL initialization complete for SMB Support Agent';
END $$;
File: backend/pyproject.toml
toml

[tool.poetry]
name = "smb-support-agent"
version = "0.1.0"
description = "AI-powered customer support agent for Singapore SMBs"
authors = ["Your Name <your.email@example.com>"]
readme = "README.md"
packages = [{include = "app"}]

[tool.poetry.dependencies]
python = "^3.12"

# Web Framework
fastapi = "^0.115.0"
uvicorn = {extras = ["standard"], version = "^0.32.0"}
python-multipart = "^0.0.12"
websockets = "^13.0"

# Database
sqlalchemy = {extras = ["asyncio"], version = "^2.0.35"}
asyncpg = "^0.30.0"
alembic = "^1.14.0"
greenlet = "^3.1.0"

# Cache
redis = {extras = ["hiredis"], version = "^5.2.0"}

# Vector Database
qdrant-client = "^1.12.0"

# LangChain Ecosystem
langchain = "^0.3.7"
langchain-openai = "^0.2.6"
langchain-community = "^0.3.5"
langgraph = "^0.2.45"

# Embeddings & ML
sentence-transformers = "^3.2.0"
tiktoken = "^0.8.0"

# Validation & Settings
pydantic = "^2.9.0"
pydantic-settings = "^2.6.0"
email-validator = "^2.2.0"

# Utilities
python-dotenv = "^1.0.1"
httpx = "^0.27.0"
tenacity = "^9.0.0"
structlog = "^24.4.0"
python-dateutil = "^2.9.0"
pytz = "^2024.2"

[tool.poetry.group.dev.dependencies]
pytest = "^8.3.0"
pytest-asyncio = "^0.24.0"
pytest-cov = "^5.0.0"
httpx = "^0.27.0"
ruff = "^0.7.0"
mypy = "^1.13.0"
pre-commit = "^4.0.0"

# Type stubs
types-redis = "^4.6.0"
types-python-dateutil = "^2.9.0"
types-pytz = "^2024.2.0"

[tool.poetry.group.evaluation.dependencies]
ragas = "^0.2.0"
deepeval = "^1.0.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

# -----------------------------------------------------------------------------
# RUFF CONFIGURATION
# -----------------------------------------------------------------------------
[tool.ruff]
target-version = "py312"
line-length = 100
exclude = [
    ".git",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "__pycache__",
    "alembic/versions",
]

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # Pyflakes
    "I",      # isort
    "B",      # flake8-bugbear
    "C4",     # flake8-comprehensions
    "UP",     # pyupgrade
    "ARG",    # flake8-unused-arguments
    "SIM",    # flake8-simplify
]
ignore = [
    "E501",   # line too long (handled by formatter)
    "B008",   # do not perform function calls in argument defaults
    "B904",   # raise without from inside except
]

[tool.ruff.lint.isort]
known-first-party = ["app"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false
line-ending = "auto"

# -----------------------------------------------------------------------------
# MYPY CONFIGURATION
# -----------------------------------------------------------------------------
[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
follow_imports = "silent"
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = [
    "langchain.*",
    "langgraph.*",
    "qdrant_client.*",
    "sentence_transformers.*",
]
ignore_missing_imports = true

# -----------------------------------------------------------------------------
# PYTEST CONFIGURATION
# -----------------------------------------------------------------------------
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
filterwarnings = [
    "ignore::DeprecationWarning",
]
addopts = "-v --tb=short"

[tool.coverage.run]
source = ["app"]
branch = true
omit = ["*/tests/*", "*/migrations/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if TYPE_CHECKING:",
]
File: backend/app/__init__.py
Python

"""
Singapore SMB Support Agent - Backend Application

A production-ready AI-powered customer support agent built with:
- FastAPI for the web framework
- LangChain + LangGraph for agent orchestration
- Qdrant for vector storage
- PostgreSQL for persistence
- Redis for session caching
"""

__version__ = "0.1.0"
__author__ = "Your Company"
File: backend/app/core/config.py
Python

"""
Configuration Management

Centralized configuration using Pydantic Settings with:
- Environment variable loading
- Type validation
- Sensible defaults
- Singapore-specific business settings
"""

from functools import lru_cache
from typing import Any

from pydantic import Field, PostgresDsn, RedisDsn, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    All settings are validated at startup to fail fast on misconfiguration.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # -------------------------------------------------------------------------
    # Application
    # -------------------------------------------------------------------------
    app_name: str = Field(default="Singapore SMB Support Agent")
    app_env: str = Field(default="development")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    
    # -------------------------------------------------------------------------
    # API
    # -------------------------------------------------------------------------
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_reload: bool = Field(default=False)
    cors_origins: list[str] = Field(
        default=["http://localhost:5173", "http://localhost:3000"]
    )
    
    # -------------------------------------------------------------------------
    # Database - PostgreSQL
    # -------------------------------------------------------------------------
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_user: str = Field(default="smb_agent")
    postgres_password: str = Field(default="")
    postgres_db: str = Field(default="smb_support_agent")
    database_url: str | None = Field(default=None)
    
    # Connection pool settings
    db_pool_size: int = Field(default=5)
    db_max_overflow: int = Field(default=10)
    db_pool_timeout: int = Field(default=30)
    
    @model_validator(mode="after")
    def build_database_url(self) -> "Settings":
        """Construct database URL if not explicitly provided."""
        if not self.database_url:
            self.database_url = (
                f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            )
        return self
    
    # -------------------------------------------------------------------------
    # Cache - Redis
    # -------------------------------------------------------------------------
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_password: str = Field(default="")
    redis_db: int = Field(default=0)
    redis_url: str | None = Field(default=None)
    session_ttl: int = Field(default=86400)  # 24 hours
    
    @model_validator(mode="after")
    def build_redis_url(self) -> "Settings":
        """Construct Redis URL if not explicitly provided."""
        if not self.redis_url:
            if self.redis_password:
                self.redis_url = (
                    f"redis://:{self.redis_password}@{self.redis_host}:"
                    f"{self.redis_port}/{self.redis_db}"
                )
            else:
                self.redis_url = (
                    f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
                )
        return self
    
    # -------------------------------------------------------------------------
    # Vector Database - Qdrant
    # -------------------------------------------------------------------------
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_grpc_port: int = Field(default=6334)
    qdrant_api_key: str | None = Field(default=None)
    
    # Collection names
    qdrant_knowledge_collection: str = Field(default="knowledge_base")
    qdrant_faq_collection: str = Field(default="faqs")
    qdrant_memory_collection: str = Field(default="conversation_memory")
    
    # -------------------------------------------------------------------------
    # LLM Configuration
    # -------------------------------------------------------------------------
    openai_api_key: str = Field(default="")
    llm_model_primary: str = Field(default="gpt-4o-mini")
    llm_model_complex: str = Field(default="gpt-4o")
    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_dimensions: int = Field(default=1536)
    
    # Token limits
    max_context_tokens: int = Field(default=4000)
    max_output_tokens: int = Field(default=1000)
    
    # -------------------------------------------------------------------------
    # Reranker
    # -------------------------------------------------------------------------
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    
    # -------------------------------------------------------------------------
    # Business Configuration - Singapore SMB
    # -------------------------------------------------------------------------
    business_name: str = Field(default="Your Business")
    business_timezone: str = Field(default="Asia/Singapore")
    support_email: str = Field(default="support@example.com")
    escalation_email: str = Field(default="escalation@example.com")
    
    business_hours_start: str = Field(default="09:00")
    business_hours_end: str = Field(default="18:00")
    business_days: list[str] = Field(
        default=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    )
    
    # -------------------------------------------------------------------------
    # Security
    # -------------------------------------------------------------------------
    secret_key: str = Field(default="change-me-in-production-min-32-chars")
    api_key_header: str = Field(default="X-API-Key")
    rate_limit_per_minute: int = Field(default=60)
    
    # -------------------------------------------------------------------------
    # PDPA Compliance
    # -------------------------------------------------------------------------
    data_retention_days: int = Field(default=730)  # 2 years
    require_consent: bool = Field(default=True)
    
    # -------------------------------------------------------------------------
    # Computed Properties
    # -------------------------------------------------------------------------
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.app_env.lower() in ("development", "dev", "local")
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.app_env.lower() in ("production", "prod")
    
    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        """Ensure secret key is sufficiently long."""
        if len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters")
        return v


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses lru_cache to ensure settings are only loaded once.
    """
    return Settings()


# Export singleton for easy import
settings = get_settings()
File: backend/app/core/logging.py
Python

"""
Structured Logging Configuration

Implements structured JSON logging with:
- Correlation IDs for request tracing
- Context-aware log enrichment
- Human-readable development format
- Machine-readable production format
"""

import logging
import sys
from contextvars import ContextVar
from typing import Any
from uuid import uuid4

import structlog
from structlog.types import Processor

from app.core.config import settings

# Context variable for correlation ID
correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def get_correlation_id() -> str:
    """Get current correlation ID or generate new one."""
    cid = correlation_id_var.get()
    if cid is None:
        cid = str(uuid4())
        correlation_id_var.set(cid)
    return cid


def set_correlation_id(cid: str) -> None:
    """Set correlation ID for current context."""
    correlation_id_var.set(cid)


def add_correlation_id(
    logger: logging.Logger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add correlation ID to log event."""
    cid = correlation_id_var.get()
    if cid:
        event_dict["correlation_id"] = cid
    return event_dict


def add_app_context(
    logger: logging.Logger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add application context to log event."""
    event_dict["app"] = settings.app_name
    event_dict["env"] = settings.app_env
    return event_dict


def setup_logging() -> None:
    """
    Configure structured logging for the application.
    
    Development: Human-readable colored output
    Production: JSON format for log aggregation
    """
    
    # Shared processors
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        add_correlation_id,
        add_app_context,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    
    if settings.is_development:
        # Development: colored console output
        processors: list[Processor] = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
            ),
        ]
    else:
        # Production: JSON output
        processors = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging to use structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """
    Get a configured logger instance.
    
    Args:
        name: Optional logger name (typically __name__)
        
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)
File: backend/app/models/enums.py
Python

"""
Enumeration Types

Centralized enum definitions for type safety and consistency.
These mirror the PostgreSQL enum types defined in the database.
"""

from enum import Enum


class LanguageCode(str, Enum):
    """
    Supported language codes for Singapore multi-lingual context.
    
    Includes Singlish (en-SG) as a distinct variant.
    """
    
    ENGLISH = "en"
    MANDARIN = "zh"
    MALAY = "ms"
    TAMIL = "ta"
    SINGLISH = "en-SG"
    
    @classmethod
    def from_string(cls, value: str) -> "LanguageCode":
        """Convert string to LanguageCode with fallback to English."""
        try:
            return cls(value)
        except ValueError:
            return cls.ENGLISH


class IntentCategory(str, Enum):
    """
    Customer enquiry intent categories.
    
    Based on typical Singapore SMB support patterns.
    """
    
    PRODUCT_INQUIRY = "product_inquiry"
    PRICING = "pricing"
    ORDER_STATUS = "order_status"
    BOOKING = "booking"
    COMPLAINT = "complaint"
    RETURNS = "returns"
    OPERATING_HOURS = "operating_hours"
    GENERAL_FAQ = "general_faq"
    ESCALATION = "escalation"
    GREETING = "greeting"
    UNKNOWN = "unknown"
    
    @property
    def requires_rag(self) -> bool:
        """Check if this intent typically requires RAG retrieval."""
        return self in {
            IntentCategory.PRODUCT_INQUIRY,
            IntentCategory.PRICING,
            IntentCategory.GENERAL_FAQ,
            IntentCategory.RETURNS,
        }
    
    @property
    def may_escalate(self) -> bool:
        """Check if this intent may require human escalation."""
        return self in {
            IntentCategory.COMPLAINT,
            IntentCategory.ESCALATION,
            IntentCategory.ORDER_STATUS,  # If complex issues
        }


class ConversationStatus(str, Enum):
    """Status of a customer conversation."""
    
    ACTIVE = "active"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    ABANDONED = "abandoned"


class MessageRole(str, Enum):
    """Role of a message sender in conversation."""
    
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class EventType(str, Enum):
    """Types of interaction events for logging."""
    
    MESSAGE_RECEIVED = "message_received"
    MESSAGE_SENT = "message_sent"
    TOOL_CALLED = "tool_called"
    RAG_RETRIEVAL = "rag_retrieval"
    INTENT_CLASSIFIED = "intent_classified"
    ESCALATION_TRIGGERED = "escalation_triggered"
    FEEDBACK_RECEIVED = "feedback_received"
    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    ERROR_OCCURRED = "error_occurred"
File: backend/app/models/schemas.py
Python

"""
Pydantic Schemas

Comprehensive data validation schemas for:
- API request/response models
- Domain entities
- Agent state management
- Memory structures

All schemas use Pydantic v2 with strict validation.
"""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.enums import (
    ConversationStatus,
    EventType,
    IntentCategory,
    LanguageCode,
    MessageRole,
)


# =============================================================================
# BASE SCHEMAS
# =============================================================================

class BaseSchema(BaseModel):
    """Base schema with common configuration."""
    
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        use_enum_values=True,
        validate_assignment=True,
    )


class TimestampMixin(BaseModel):
    """Mixin for timestamp fields."""
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# CUSTOMER SCHEMAS
# =============================================================================

class CustomerBase(BaseSchema):
    """Base customer fields."""
    
    name: str | None = Field(default=None, max_length=255)
    email: str | None = Field(default=None, max_length=255)
    phone: str | None = Field(default=None, max_length=20)
    preferred_language: LanguageCode = Field(default=LanguageCode.ENGLISH)
    timezone: str = Field(default="Asia/Singapore")
    
    @field_validator("phone")
    @classmethod
    def validate_singapore_phone(cls, v: str | None) -> str | None:
        """Validate and normalize Singapore phone numbers."""
        if v is None:
            return None
        # Remove spaces and dashes
        cleaned = v.replace(" ", "").replace("-", "")
        # Add +65 prefix if not present
        if cleaned.startswith("65"):
            cleaned = "+" + cleaned
        elif not cleaned.startswith("+65") and len(cleaned) == 8:
            cleaned = "+65" + cleaned
        return cleaned


class CustomerCreate(CustomerBase):
    """Schema for creating a new customer."""
    
    pdpa_consent: bool = Field(default=False)
    
    @field_validator("pdpa_consent")
    @classmethod
    def require_consent_for_data(cls, v: bool, info: Any) -> bool:
        """Warn if storing customer data without consent."""
        # In production, you might want to enforce this
        return v


class CustomerProfile(CustomerBase, TimestampMixin):
    """Complete customer profile for agent context."""
    
    id: UUID = Field(default_factory=uuid4)
    first_interaction: datetime = Field(default_factory=datetime.utcnow)
    total_interactions: int = Field(default=0, ge=0)
    sentiment_trend: float = Field(default=0.0, ge=-1.0, le=1.0)
    preferences: dict[str, Any] = Field(default_factory=dict)
    past_issues_summary: str | None = Field(default=None)
    tags: list[str] = Field(default_factory=list)
    pdpa_consent: bool = Field(default=False)
    consent_timestamp: datetime | None = Field(default=None)
    
    @property
    def display_name(self) -> str:
        """Get display name or fallback."""
        return self.name or "Valued Customer"
    
    @property
    def is_returning(self) -> bool:
        """Check if this is a returning customer."""
        return self.total_interactions > 1


class CustomerUpdate(BaseSchema):
    """Schema for updating customer fields."""
    
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    preferred_language: LanguageCode | None = None
    preferences: dict[str, Any] | None = None
    tags: list[str] | None = None


# =============================================================================
# MESSAGE SCHEMAS
# =============================================================================

class MessageBase(BaseSchema):
    """Base message fields."""
    
    role: MessageRole
    content: str = Field(..., min_length=1, max_length=10000)


class ConversationMessage(MessageBase, TimestampMixin):
    """Single message in a conversation."""
    
    id: UUID = Field(default_factory=uuid4)
    language_detected: LanguageCode | None = Field(default=None)
    intent: IntentCategory | None = Field(default=None)
    entities: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    sources_used: list[str] = Field(default_factory=list)
    token_count: int | None = Field(default=None, ge=0)
    
    @field_validator("content")
    @classmethod
    def strip_content(cls, v: str) -> str:
        """Strip whitespace from content."""
        return v.strip()


class MessageCreate(MessageBase):
    """Schema for creating a new message."""
    
    session_id: str = Field(..., min_length=1)


# =============================================================================
# RETRIEVED CONTEXT SCHEMAS
# =============================================================================

class RetrievedChunk(BaseSchema):
    """Document chunk retrieved from knowledge base."""
    
    chunk_id: str
    content: str
    source_document: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    @property
    def citation(self) -> str:
        """Generate citation string for this chunk."""
        source = self.metadata.get("title", self.source_document)
        page = self.metadata.get("page")
        if page:
            return f"{source} (p. {page})"
        return source


class RetrievalResult(BaseSchema):
    """Complete retrieval result with multiple chunks."""
    
    query: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    total_found: int = Field(default=0, ge=0)
    retrieval_time_ms: float = Field(default=0.0, ge=0.0)
    
    @property
    def has_results(self) -> bool:
        """Check if any results were found."""
        return len(self.chunks) > 0
    
    @property
    def top_chunk(self) -> RetrievedChunk | None:
        """Get the highest-scoring chunk."""
        return self.chunks[0] if self.chunks else None


# =============================================================================
# CONVERSATION STATE SCHEMAS
# =============================================================================

class ConversationState(BaseSchema):
    """
    Complete state for LangGraph agent.
    
    This is the central state object passed through the agent graph.
    """
    
    session_id: str = Field(..., min_length=1)
    customer: CustomerProfile = Field(default_factory=CustomerProfile)
    messages: list[ConversationMessage] = Field(default_factory=list)
    current_intent: IntentCategory | None = Field(default=None)
    detected_language: LanguageCode = Field(default=LanguageCode.ENGLISH)
    entities: dict[str, Any] = Field(default_factory=dict)
    retrieved_contexts: list[RetrievedChunk] = Field(default_factory=list)
    tool_calls_made: list[str] = Field(default_factory=list)
    requires_escalation: bool = Field(default=False)
    escalation_reason: str | None = Field(default=None)
    conversation_summary: str | None = Field(default=None)
    turn_count: int = Field(default=0, ge=0)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    
    @field_validator("messages")
    @classmethod
    def limit_message_buffer(cls, v: list[ConversationMessage]) -> list[ConversationMessage]:
        """Keep only last 20 messages to manage context window."""
        return v[-20:] if len(v) > 20 else v
    
    @property
    def last_user_message(self) -> ConversationMessage | None:
        """Get the most recent user message."""
        for msg in reversed(self.messages):
            if msg.role == MessageRole.USER:
                return msg
        return None
    
    @property
    def last_assistant_message(self) -> ConversationMessage | None:
        """Get the most recent assistant message."""
        for msg in reversed(self.messages):
            if msg.role == MessageRole.ASSISTANT:
                return msg
        return None
    
    def add_message(self, role: MessageRole, content: str, **kwargs: Any) -> None:
        """Add a new message to the conversation."""
        message = ConversationMessage(
            role=role,
            content=content,
            **kwargs,
        )
        self.messages.append(message)
        if role == MessageRole.USER:
            self.turn_count += 1


# =============================================================================
# AGENT RESPONSE SCHEMAS
# =============================================================================

class AgentResponse(BaseSchema):
    """Validated agent output."""
    
    content: str = Field(..., min_length=1)
    language: LanguageCode = Field(default=LanguageCode.ENGLISH)
    intent_handled: IntentCategory = Field(default=IntentCategory.UNKNOWN)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    sources: list[str] = Field(default_factory=list)
    suggested_actions: list[str] = Field(default_factory=list)
    requires_followup: bool = Field(default=False)
    escalated: bool = Field(default=False)
    metadata: dict[str, Any] = Field(default_factory=dict)
    processing_time_ms: float = Field(default=0.0, ge=0.0)


# =============================================================================
# CONVERSATION SUMMARY SCHEMAS
# =============================================================================

class ConversationSummary(BaseSchema, TimestampMixin):
    """Summary of a completed conversation for long-term memory."""
    
    id: UUID = Field(default_factory=uuid4)
    session_id: str
    customer_id: UUID
    summary: str = Field(..., min_length=10)
    intent_categories: list[IntentCategory] = Field(default_factory=list)
    status: ConversationStatus = Field(default=ConversationStatus.RESOLVED)
    turn_count: int = Field(default=0, ge=0)
    sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    resolution_notes: str | None = Field(default=None)
    duration_seconds: int = Field(default=0, ge=0)
    
    # For vector storage
    embedding: list[float] | None = Field(default=None, exclude=True)


# =============================================================================
# EVENT LOGGING SCHEMAS
# =============================================================================

class InteractionEvent(BaseSchema):
    """Single interaction event for logging and analytics."""
    
    id: UUID = Field(default_factory=uuid4)
    conversation_id: UUID
    event_type: EventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    details: dict[str, Any] = Field(default_factory=dict)
    
    @classmethod
    def message_received(
        cls,
        conversation_id: UUID,
        content_length: int,
        language: LanguageCode,
    ) -> "InteractionEvent":
        """Create a message received event."""
        return cls(
            conversation_id=conversation_id,
            event_type=EventType.MESSAGE_RECEIVED,
            details={
                "content_length": content_length,
                "language": language.value,
            },
        )
    
    @classmethod
    def tool_called(
        cls,
        conversation_id: UUID,
        tool_name: str,
        duration_ms: float,
        success: bool,
    ) -> "InteractionEvent":
        """Create a tool call event."""
        return cls(
            conversation_id=conversation_id,
            event_type=EventType.TOOL_CALLED,
            details={
                "tool_name": tool_name,
                "duration_ms": duration_ms,
                "success": success,
            },
        )


# =============================================================================
# API REQUEST/RESPONSE SCHEMAS
# =============================================================================

class ChatRequest(BaseSchema):
    """Incoming chat message request."""
    
    session_id: str = Field(..., min_length=1, max_length=100)
    message: str = Field(..., min_length=1, max_length=5000)
    customer_id: UUID | None = Field(default=None)
    
    @field_validator("message")
    @classmethod
    def clean_message(cls, v: str) -> str:
        """Clean and validate message content."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Message cannot be empty")
        return cleaned


class ChatResponse(BaseSchema):
    """Chat response to client."""
    
    session_id: str
    message: str
    sources: list[str] = Field(default_factory=list)
    suggested_actions: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    intent: IntentCategory = Field(default=IntentCategory.UNKNOWN)
    requires_followup: bool = Field(default=False)
    escalated: bool = Field(default=False)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class StreamingChunk(BaseSchema):
    """Single chunk in a streaming response."""
    
    type: str = Field(..., pattern="^(token|source|complete|error)$")
    content: str = Field(default="")
    metadata: dict[str, Any] = Field(default_factory=dict)


class FeedbackRequest(BaseSchema):
    """User feedback on a response."""
    
    session_id: str
    message_id: UUID
    rating: int = Field(..., ge=1, le=5)
    comment: str | None = Field(default=None, max_length=1000)


class HealthResponse(BaseSchema):
    """Health check response."""
    
    status: str = Field(default="healthy")
    version: str
    environment: str
    services: dict[str, bool] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
File: backend/app/models/database.py
Python

"""
SQLAlchemy Database Models

Async SQLAlchemy 2.0 models for PostgreSQL with:
- UUID primary keys
- Proper indexing
- JSONB for flexible fields
- Type-safe column definitions
"""

from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all database models."""
    
    type_annotation_map = {
        dict[str, Any]: JSONB,
        list[str]: ARRAY(String),
    }


class TimestampMixin:
    """Mixin for created_at and updated_at timestamps."""
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class Customer(Base, TimestampMixin):
    """
    Customer profile for long-term memory.
    
    Stores customer preferences, interaction history, and PDPA consent.
    """
    
    __tablename__ = "customers"
    
    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    
    # Identity
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True, unique=True)
    phone: Mapped[str | None] = mapped_column(String(20), nullable=True)
    
    # Preferences
    preferred_language: Mapped[str] = mapped_column(
        String(10),
        default="en",
        nullable=False,
    )
    timezone: Mapped[str] = mapped_column(
        String(50),
        default="Asia/Singapore",
        nullable=False,
    )
    preferences: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )
    tags: Mapped[list[str]] = mapped_column(
        ARRAY(String),
        default=list,
        nullable=False,
    )
    
    # Analytics
    first_interaction: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    total_interactions: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    sentiment_trend: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False,
    )
    past_issues_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # PDPA Compliance
    pdpa_consent: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )
    consent_timestamp: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    
    # Relationships
    conversations: Mapped[list["Conversation"]] = relationship(
        "Conversation",
        back_populates="customer",
        lazy="selectin",
    )
    
    # Indexes
    __table_args__ = (
        Index("ix_customers_email", "email", unique=True, postgresql_where=email.isnot(None)),
        Index("ix_customers_phone", "phone"),
        Index("ix_customers_created_at", "created_at"),
    )
    
    def __repr__(self) -> str:
        return f"<Customer(id={self.id}, name={self.name})>"


class Conversation(Base, TimestampMixin):
    """
    Conversation record for persistence and analytics.
    
    Stores summaries and metadata, not full message history
    (which is in Redis for active sessions).
    """
    
    __tablename__ = "conversations"
    
    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    session_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        unique=True,
    )
    
    # Customer relationship
    customer_id: Mapped[UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("customers.id", ondelete="SET NULL"),
        nullable=True,
    )
    customer: Mapped["Customer | None"] = relationship(
        "Customer",
        back_populates="conversations",
    )
    
    # Timing
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    
    # Content
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    intent_categories: Mapped[list[str]] = mapped_column(
        ARRAY(String),
        default=list,
        nullable=False,
    )
    
    # Status
    status: Mapped[str] = mapped_column(
        String(20),
        default="active",
        nullable=False,
    )
    
    # Analytics
    turn_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    sentiment_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    resolution_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Relationships
    events: Mapped[list["InteractionEvent"]] = relationship(
        "InteractionEvent",
        back_populates="conversation",
        lazy="selectin",
        cascade="all, delete-orphan",
    )
    
    # Indexes
    __table_args__ = (
        Index("ix_conversations_session_id", "session_id", unique=True),
        Index("ix_conversations_customer_id", "customer_id"),
        Index("ix_conversations_started_at", "started_at"),
        Index("ix_conversations_status", "status"),
    )
    
    def __repr__(self) -> str:
        return f"<Conversation(id={self.id}, session={self.session_id})>"


class InteractionEvent(Base):
    """
    Interaction event for logging and analytics.
    
    Captures all significant events during a conversation.
    """
    
    __tablename__ = "interaction_events"
    
    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    
    # Conversation relationship
    conversation_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    conversation: Mapped["Conversation"] = relationship(
        "Conversation",
        back_populates="events",
    )
    
    # Event data
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    details: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )
    
    # Indexes
    __table_args__ = (
        Index("ix_events_conversation_id", "conversation_id"),
        Index("ix_events_event_type", "event_type"),
        Index("ix_events_timestamp", "timestamp"),
    )
    
    def __repr__(self) -> str:
        return f"<InteractionEvent(id={self.id}, type={self.event_type})>"


class KnowledgeDocument(Base, TimestampMixin):
    """
    Metadata for ingested knowledge base documents.
    
    The actual content and embeddings are in Qdrant,
    this tracks document-level metadata and processing status.
    """
    
    __tablename__ = "knowledge_documents"
    
    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    
    # Document identity
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False, unique=True)
    file_type: Mapped[str] = mapped_column(String(50), nullable=False)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    
    # Metadata
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    category: Mapped[str | None] = mapped_column(String(100), nullable=True)
    language: Mapped[str] = mapped_column(String(10), default="en", nullable=False)
    metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    
    # Processing status
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_processed: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    processing_errors: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index("ix_knowledge_docs_file_path", "file_path", unique=True),
        Index("ix_knowledge_docs_category", "category"),
        Index("ix_knowledge_docs_is_active", "is_active"),
    )
    
    def __repr__(self) -> str:
        return f"<KnowledgeDocument(id={self.id}, filename={self.filename})>"
File: backend/app/core/database.py
Python

"""
Database Connection Management

Async SQLAlchemy engine and session management with:
- Connection pooling
- Proper cleanup
- Health checks
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.core.logging import get_logger
from app.models.database import Base

logger = get_logger(__name__)

# Create async engine
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_timeout=settings.db_pool_timeout,
    pool_pre_ping=True,  # Verify connections before use
)

# Session factory
async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def init_database() -> None:
    """
    Initialize database tables.
    
    In production, use Alembic migrations instead.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialized")


async def close_database() -> None:
    """Close database connections."""
    await engine.dispose()
    logger.info("Database connections closed")


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get database session for dependency injection.
    
    Usage:
        @router.get("/items")
        async def get_items(session: AsyncSession = Depends(get_session)):
            ...
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_session_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for database sessions.
    
    Usage:
        async with get_session_context() as session:
            ...
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def check_database_health() -> bool:
    """Check if database is accessible."""
    try:
        async with async_session_factory() as session:
            await session.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error("Database health check failed", error=str(e))
        return False
File: backend/app/core/redis.py
Python

"""
Redis Connection Management

Async Redis client for session caching with:
- Connection pooling
- Automatic serialization
- Health checks
"""

from typing import Any

import redis.asyncio as redis
from redis.asyncio import ConnectionPool, Redis

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Connection pool
_pool: ConnectionPool | None = None
_client: Redis | None = None


async def init_redis() -> Redis:
    """Initialize Redis connection pool."""
    global _pool, _client
    
    _pool = ConnectionPool.from_url(
        settings.redis_url,
        max_connections=20,
        decode_responses=True,
    )
    _client = Redis(connection_pool=_pool)
    
    # Verify connection
    await _client.ping()
    logger.info("Redis connection established")
    
    return _client


async def close_redis() -> None:
    """Close Redis connections."""
    global _pool, _client
    
    if _client:
        await _client.close()
    if _pool:
        await _pool.disconnect()
    
    _client = None
    _pool = None
    logger.info("Redis connections closed")


def get_redis() -> Redis:
    """
    Get Redis client for dependency injection.
    
    Usage:
        @router.get("/session")
        async def get_session(redis: Redis = Depends(get_redis)):
            ...
    """
    if _client is None:
        raise RuntimeError("Redis not initialized. Call init_redis() first.")
    return _client


async def check_redis_health() -> bool:
    """Check if Redis is accessible."""
    try:
        if _client is None:
            return False
        await _client.ping()
        return True
    except Exception as e:
        logger.error("Redis health check failed", error=str(e))
        return False


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def set_json(key: str, value: dict[str, Any], ttl: int | None = None) -> None:
    """Set a JSON value in Redis."""
    import json
    
    client = get_redis()
    serialized = json.dumps(value, default=str)
    
    if ttl:
        await client.setex(key, ttl, serialized)
    else:
        await client.set(key, serialized)


async def get_json(key: str) -> dict[str, Any] | None:
    """Get a JSON value from Redis."""
    import json
    
    client = get_redis()
    value = await client.get(key)
    
    if value is None:
        return None
    
    return json.loads(value)


async def delete_key(key: str) -> bool:
    """Delete a key from Redis."""
    client = get_redis()
    result = await client.delete(key)
    return result > 0
File: backend/app/core/qdrant.py
Python

"""
Qdrant Vector Database Client

Manages vector collections for:
- Knowledge base (main documents)
- FAQs (optimized for Q&A)
- Conversation memory (semantic search over past interactions)
"""

from typing import Any

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Global client
_client: AsyncQdrantClient | None = None


async def init_qdrant() -> AsyncQdrantClient:
    """Initialize Qdrant client and create collections."""
    global _client
    
    _client = AsyncQdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        api_key=settings.qdrant_api_key if settings.qdrant_api_key else None,
    )
    
    # Verify connection
    await _client.get_collections()
    logger.info("Qdrant connection established")
    
    # Initialize collections
    await _create_collections()
    
    return _client


async def _create_collections() -> None:
    """Create required collections if they don't exist."""
    if _client is None:
        raise RuntimeError("Qdrant not initialized")
    
    collections_config = [
        {
            "name": settings.qdrant_knowledge_collection,
            "description": "Main knowledge base documents",
        },
        {
            "name": settings.qdrant_faq_collection,
            "description": "FAQ documents optimized for Q&A",
        },
        {
            "name": settings.qdrant_memory_collection,
            "description": "Conversation summaries for semantic memory",
        },
    ]
    
    for config in collections_config:
        try:
            await _client.get_collection(config["name"])
            logger.debug(f"Collection {config['name']} already exists")
        except UnexpectedResponse:
            # Collection doesn't exist, create it
            await _client.create_collection(
                collection_name=config["name"],
                vectors_config=models.VectorParams(
                    size=settings.embedding_dimensions,
                    distance=models.Distance.COSINE,
                    on_disk=True,
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=16,
                    ef_construct=128,
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=20000,
                ),
            )
            logger.info(f"Created collection: {config['name']}")


async def close_qdrant() -> None:
    """Close Qdrant connection."""
    global _client
    
    if _client:
        await _client.close()
        _client = None
    
    logger.info("Qdrant connection closed")


def get_qdrant() -> AsyncQdrantClient:
    """
    Get Qdrant client for dependency injection.
    
    Usage:
        @router.get("/search")
        async def search(qdrant: AsyncQdrantClient = Depends(get_qdrant)):
            ...
    """
    if _client is None:
        raise RuntimeError("Qdrant not initialized. Call init_qdrant() first.")
    return _client


async def check_qdrant_health() -> bool:
    """Check if Qdrant is accessible."""
    try:
        if _client is None:
            return False
        await _client.get_collections()
        return True
    except Exception as e:
        logger.error("Qdrant health check failed", error=str(e))
        return False


# =============================================================================
# COLLECTION OPERATIONS
# =============================================================================

async def get_collection_info(collection_name: str) -> dict[str, Any]:
    """Get collection statistics."""
    client = get_qdrant()
    info = await client.get_collection(collection_name)
    return {
        "name": collection_name,
        "vectors_count": info.vectors_count,
        "points_count": info.points_count,
        "status": info.status.value,
    }


async def upsert_vectors(
    collection_name: str,
    ids: list[str],
    vectors: list[list[float]],
    payloads: list[dict[str, Any]],
) -> None:
    """Insert or update vectors in a collection."""
    client = get_qdrant()
    
    points = [
        models.PointStruct(
            id=id_,
            vector=vector,
            payload=payload,
        )
        for id_, vector, payload in zip(ids, vectors, payloads)
    ]
    
    await client.upsert(
        collection_name=collection_name,
        points=points,
    )


async def search_vectors(
    collection_name: str,
    query_vector: list[float],
    limit: int = 10,
    score_threshold: float = 0.0,
    filter_conditions: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Search for similar vectors."""
    client = get_qdrant()
    
    # Build filter if provided
    query_filter = None
    if filter_conditions:
        must_conditions = [
            models.FieldCondition(
                key=key,
                match=models.MatchValue(value=value),
            )
            for key, value in filter_conditions.items()
        ]
        query_filter = models.Filter(must=must_conditions)
    
    results = await client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=limit,
        score_threshold=score_threshold,
        query_filter=query_filter,
    )
    
    return [
        {
            "id": str(result.id),
            "score": result.score,
            "payload": result.payload,
        }
        for result in results
    ]
File: backend/app/api/deps.py
Python

"""
FastAPI Dependencies

Centralized dependency injection for:
- Database sessions
- Redis client
- Qdrant client
- Current user/session
"""

from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_session
from app.core.logging import get_correlation_id, get_logger, set_correlation_id
from app.core.qdrant import get_qdrant
from app.core.redis import get_redis

logger = get_logger(__name__)


# =============================================================================
# DATABASE
# =============================================================================

DbSession = Annotated[AsyncSession, Depends(get_session)]


# =============================================================================
# REDIS
# =============================================================================

RedisClient = Annotated[Redis, Depends(get_redis)]


# =============================================================================
# QDRANT
# =============================================================================

QdrantClient = Annotated[AsyncQdrantClient, Depends(get_qdrant)]


# =============================================================================
# CORRELATION ID
# =============================================================================

async def inject_correlation_id(
    x_correlation_id: str | None = Header(default=None, alias="X-Correlation-ID"),
) -> str:
    """Extract or generate correlation ID for request tracing."""
    if x_correlation_id:
        set_correlation_id(x_correlation_id)
        return x_correlation_id
    return get_correlation_id()


CorrelationId = Annotated[str, Depends(inject_correlation_id)]


# =============================================================================
# API KEY VALIDATION (Optional)
# =============================================================================

async def validate_api_key(
    api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> str | None:
    """
    Validate API key if configured.
    
    For public-facing widgets, this can be optional.
    """
    # In development, skip API key validation
    if settings.is_development:
        return api_key
    
    # If API key is required in production, validate it here
    # For now, we'll make it optional
    return api_key


ApiKey = Annotated[str | None, Depends(validate_api_key)]


# =============================================================================
# RATE LIMITING
# =============================================================================

class RateLimiter:
    """
    Simple rate limiter using Redis.
    
    For production, consider using a more robust solution like slowapi.
    """
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.window_seconds = 60
    
    async def check_rate_limit(
        self,
        redis: Redis,
        key: str,
    ) -> bool:
        """Check if request is within rate limit."""
        current = await redis.incr(key)
        
        if current == 1:
            await redis.expire(key, self.window_seconds)
        
        return current <= self.requests_per_minute


async def check_rate_limit(
    redis: RedisClient,
    correlation_id: CorrelationId,
) -> None:
    """
    Dependency to check rate limit.
    
    Uses correlation ID as the rate limit key.
    In production, use client IP or session ID.
    """
    if settings.is_development:
        return
    
    limiter = RateLimiter(settings.rate_limit_per_minute)
    key = f"rate_limit:{correlation_id}"
    
    if not await limiter.check_rate_limit(redis, key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later.",
        )


RateLimitCheck = Annotated[None, Depends(check_rate_limit)]
File: backend/app/api/routes/health.py
Python

"""
Health Check Endpoints

Provides health and readiness endpoints for:
- Kubernetes liveness/readiness probes
- Load balancer health checks
- Service monitoring
"""

from fastapi import APIRouter, status

from app import __version__
from app.core.config import settings
from app.core.database import check_database_health
from app.core.logging import get_logger
from app.core.qdrant import check_qdrant_health
from app.core.redis import check_redis_health
from app.models.schemas import HealthResponse

logger = get_logger(__name__)
router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check the health of the application and its dependencies.",
)
async def health_check() -> HealthResponse:
    """
    Comprehensive health check endpoint.
    
    Checks:
    - Database connectivity
    - Redis connectivity
    - Qdrant connectivity
    
    Returns 200 if all services are healthy, 503 otherwise.
    """
    # Check all services
    db_healthy = await check_database_health()
    redis_healthy = await check_redis_health()
    qdrant_healthy = await check_qdrant_health()
    
    services = {
        "database": db_healthy,
        "redis": redis_healthy,
        "qdrant": qdrant_healthy,
    }
    
    all_healthy = all(services.values())
    
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        version=__version__,
        environment=settings.app_env,
        services=services,
    )


@router.get(
    "/health/live",
    status_code=status.HTTP_200_OK,
    summary="Liveness Probe",
    description="Simple liveness check for Kubernetes.",
)
async def liveness() -> dict[str, str]:
    """
    Kubernetes liveness probe.
    
    Returns 200 if the application is running.
    Does not check dependencies.
    """
    return {"status": "alive"}


@router.get(
    "/health/ready",
    status_code=status.HTTP_200_OK,
    summary="Readiness Probe",
    description="Readiness check for Kubernetes.",
)
async def readiness() -> dict[str, str]:
    """
    Kubernetes readiness probe.
    
    Returns 200 if the application is ready to accept traffic.
    Checks critical dependencies.
    """
    # Check critical dependencies
    db_healthy = await check_database_health()
    redis_healthy = await check_redis_health()
    
    if not db_healthy or not redis_healthy:
        # Return 503 by raising exception
        from fastapi import HTTPException
        
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready",
        )
    
    return {"status": "ready"}
File: backend/app/main.py
Python

"""
FastAPI Application Entry Point

Production-ready FastAPI application with:
- Lifespan management for startup/shutdown
- CORS configuration
- Error handling middleware
- Structured logging
- Health check endpoints
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api.routes import health
from app.core.config import settings
from app.core.database import close_database, init_database
from app.core.logging import get_correlation_id, get_logger, set_correlation_id, setup_logging
from app.core.qdrant import close_qdrant, init_qdrant
from app.core.redis import close_redis, init_redis

# Initialize logging first
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    
    Handles startup and shutdown of:
    - Database connections
    - Redis connections
    - Qdrant connections
    """
    logger.info(
        "Starting application",
        app_name=settings.app_name,
        version=__version__,
        environment=settings.app_env,
    )
    
    # Startup
    try:
        await init_database()
        await init_redis()
        await init_qdrant()
        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize services", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")
    await close_qdrant()
    await close_redis()
    await close_database()
    logger.info("Application shutdown complete")


def create_application() -> FastAPI:
    """
    Application factory.
    
    Creates and configures the FastAPI application.
    """
    app = FastAPI(
        title=settings.app_name,
        description=(
            "AI-powered customer support agent for Singapore SMBs. "
            "Features RAG-based knowledge retrieval, multi-turn conversations, "
            "and context-aware responses."
        ),
        version=__version__,
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
        openapi_url="/openapi.json" if settings.is_development else None,
        lifespan=lifespan,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Correlation-ID"],
    )
    
    # Register routes
    app.include_router(health.router)
    
    # Add middleware
    @app.middleware("http")
    async def correlation_id_middleware(request: Request, call_next):
        """Add correlation ID to all requests and responses."""
        # Extract or generate correlation ID
        correlation_id = request.headers.get("X-Correlation-ID")
        if correlation_id:
            set_correlation_id(correlation_id)
        else:
            correlation_id = get_correlation_id()
        
        # Process request
        response = await call_next(request)
        
        # Add correlation ID to response
        response.headers["X-Correlation-ID"] = correlation_id
        
        return response
    
    @app.middleware("http")
    async def logging_middleware(request: Request, call_next):
        """Log all requests."""
        logger.info(
            "Request received",
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else "unknown",
        )
        
        response = await call_next(request)
        
        logger.info(
            "Request completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
        )
        
        return response
    
    # Exception handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        """Handle validation errors with structured response."""
        logger.warning(
            "Validation error",
            path=request.url.path,
            errors=exc.errors(),
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "validation_error",
                "message": "Request validation failed",
                "details": exc.errors(),
            },
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        """Handle unexpected errors."""
        logger.exception(
            "Unexpected error",
            path=request.url.path,
            error=str(exc),
        )
        
        # Don't expose internal errors in production
        if settings.is_production:
            message = "An unexpected error occurred"
        else:
            message = str(exc)
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "internal_error",
                "message": message,
            },
        )
    
    return app


# Create application instance
app = create_application()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
    )
File: backend/alembic.ini
ini

# Alembic Configuration

[alembic]
script_location = alembic
prepend_sys_path = .
version_path_separator = os

# Use async driver
sqlalchemy.url = driver://user:pass@localhost/dbname

[post_write_hooks]
hooks = ruff
ruff.type = exec
ruff.executable = ruff
ruff.options = format REVISION_SCRIPT_FILENAME

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
File: backend/alembic/env.py
Python

"""
Alembic Migration Environment

Configured for async SQLAlchemy with PostgreSQL.
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from app.core.config import settings
from app.models.database import Base

# Alembic Config object
config = context.config

# Set database URL from settings
config.set_main_option("sqlalchemy.url", settings.database_url)

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Target metadata for autogenerate
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.
    
    Configures the context with just a URL and not an Engine.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with connection."""
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in async mode."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
File: backend/alembic/versions/001_initial_schema.py
Python

"""Initial schema

Revision ID: 001_initial_schema
Revises:
Create Date: 2024-01-01 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers
revision: str = "001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create customers table
    op.create_table(
        "customers",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=True),
        sa.Column("email", sa.String(length=255), nullable=True),
        
So
