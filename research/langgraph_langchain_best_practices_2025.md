# LangChain 1.0 and LangGraph Best Practices for Production AI Agents (2025)

## Executive Summary

This comprehensive technical report examines the latest best practices for building production AI agents using LangChain 1.0 and LangGraph in 2025. The frameworks have matured significantly, with LangGraph emerging as the preferred solution for stateful, multi-step agent workflows. Key architectural patterns covered include state machine design with StateGraph, LCEL (LangChain Expression Language) composition, tool binding, memory management through checkpointers, streaming responses, error handling, and production deployment strategies. Organizations implementing these patterns report a 30% reduction in debugging time and up to 40% improvement in resource utilization efficiency.

---

## 1. Introduction

LangGraph has become the foundation for building reliable AI agent workflows in 2025. Unlike simple chain-based approaches, LangGraph models agent workflows as directed graphs where nodes represent processing steps, edges define transitions, and state maintains context across the entire execution. This architecture enables sophisticated patterns including conditional routing, human-in-the-loop interactions, and fault-tolerant execution with persistent memory.

LangChain 1.0 provides the underlying components and LCEL (LangChain Expression Language) for composing LLM interactions, while LangGraph orchestrates these components into production-ready workflows. Together, they form a powerful ecosystem for enterprise AI applications.

---

## 2. LangGraph State Machine Patterns

### 2.1 Core Architecture: Nodes, Edges, and State

LangGraph implements state machines as directed graphs with three fundamental components:

**State** represents the shared data structure that persists across the workflow. It defines the schema that all nodes read from and write to.

**Nodes** are Python functions that perform the actual work, whether calling LLMs, executing tools, or processing data. Each node receives the current state and returns updates to that state.

**Edges** define the transitions between nodes, either as fixed routes or conditional branches based on the current state.

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from operator import add

# Define the state schema
class AgentState(TypedDict):
    messages: Annotated[list, add]  # Uses add reducer for list concatenation
    current_step: str
    result: str

# Define nodes as functions
def analyze_input(state: AgentState) -> dict:
    """First node: analyze the user input"""
    return {
        "messages": [{"role": "system", "content": "Analyzing input..."}],
        "current_step": "analysis_complete"
    }

def generate_response(state: AgentState) -> dict:
    """Second node: generate the final response"""
    return {
        "result": f"Processed: {state['messages'][-1]}",
        "current_step": "complete"
    }

# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("analyze", analyze_input)
workflow.add_node("generate", generate_response)

# Define edges
workflow.add_edge(START, "analyze")
workflow.add_edge("analyze", "generate")
workflow.add_edge("generate", END)

# Compile the graph
graph = workflow.compile()
```

### 2.2 StateGraph vs MessageGraph

**StateGraph** is the primary class for building custom agent workflows. It accepts a user-defined state schema and provides full control over the graph structure, node definitions, and edge routing.

**MessagesState** (formerly MessageGraph) is a pre-built convenience class for chat-based applications. It provides a single `messages` key with the `add_messages` reducer pre-configured, making it ideal for conversational agents.

```python
from langgraph.graph import MessagesState, StateGraph, START, END

# Using MessagesState for chat applications
class ChatState(MessagesState):
    """Extends MessagesState with additional fields"""
    documents: list[str]
    user_intent: str

# MessagesState includes: messages: Annotated[list[AnyMessage], add_messages]
# This handles message appending, updating by ID, and serialization automatically

workflow = StateGraph(ChatState)
```

**When to use each:**

StateGraph is appropriate when you need custom state fields beyond messages, want explicit control over state updates, or are building non-conversational workflows like data processing pipelines.

MessagesState is preferred for chatbots and conversational agents, applications that primarily exchange messages, and quick prototyping of chat-based systems.

### 2.3 Conditional Edge Patterns

Conditional edges enable dynamic routing based on the current state, which is essential for decision-making in agent workflows.

```python
from typing import Literal
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    query: str
    needs_retrieval: bool
    answer: str

def router(state: State) -> Literal["retrieve", "direct_answer"]:
    """Route based on whether retrieval is needed"""
    if state.get("needs_retrieval", False):
        return "retrieve"
    return "direct_answer"

def analyze_query(state: State) -> dict:
    """Analyze if the query needs retrieval"""
    # Simplified logic - in practice, use an LLM
    needs_rag = "document" in state["query"].lower()
    return {"needs_retrieval": needs_rag}

def retrieve_and_answer(state: State) -> dict:
    """Retrieve documents and generate answer"""
    return {"answer": "Answer based on retrieved documents"}

def direct_answer(state: State) -> dict:
    """Answer directly without retrieval"""
    return {"answer": "Direct answer without retrieval"}

# Build graph with conditional routing
workflow = StateGraph(State)
workflow.add_node("analyze", analyze_query)
workflow.add_node("retrieve", retrieve_and_answer)
workflow.add_node("direct_answer", direct_answer)

workflow.add_edge(START, "analyze")
workflow.add_conditional_edges("analyze", router)
workflow.add_edge("retrieve", END)
workflow.add_edge("direct_answer", END)

graph = workflow.compile()
```

### 2.4 Using Command for Combined State Updates and Routing

The `Command` object allows combining state updates and control flow in a single return value, which is particularly useful for agent handoffs and complex routing scenarios.

```python
from langgraph.types import Command
from typing import Literal

def agent_node(state: State) -> Command[Literal["tool_executor", "human_review", "finish"]]:
    """Agent that decides next action and updates state"""

    if state.get("needs_human_review"):
        return Command(
            update={"status": "awaiting_review"},
            goto="human_review"
        )
    elif state.get("tool_calls"):
        return Command(
            update={"status": "executing_tools"},
            goto="tool_executor"
        )
    else:
        return Command(
            update={"status": "complete", "final_answer": "..."},
            goto="finish"
        )
```

---

## 3. Typing State with Pydantic and TypedDict

### 3.1 TypedDict Approach (Recommended for Performance)

TypedDict is the standard approach for defining state schemas in LangGraph. It provides clear type hints while maintaining performance.

```python
from typing import TypedDict, Annotated
from typing_extensions import TypedDict
from operator import add
from langgraph.graph.message import add_messages
from langchain.messages import AnyMessage

class AgentState(TypedDict):
    # Simple field - overwrites on update
    current_task: str

    # Field with reducer - appends new items
    messages: Annotated[list[AnyMessage], add_messages]

    # Field with custom reducer
    error_count: Annotated[int, add]

    # List with add reducer
    tool_calls: Annotated[list[dict], add]
```

### 3.2 Pydantic Approach (For Validation)

Pydantic models enable runtime validation of state inputs, which is valuable for catching errors early in production systems.

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from langgraph.graph import StateGraph, START, END

class ValidatedState(BaseModel):
    """State with Pydantic validation"""
    query: str = Field(..., min_length=1, description="User query")
    max_iterations: int = Field(default=5, ge=1, le=20)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    context: Optional[List[str]] = None

    @validator('query')
    def query_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v

def process_node(state: ValidatedState) -> dict:
    """Node receives validated Pydantic object"""
    return {"context": [f"Processed: {state.query}"]}

# Build graph with Pydantic state
builder = StateGraph(ValidatedState)
builder.add_node("process", process_node)
builder.add_edge(START, "process")
builder.add_edge("process", END)
graph = builder.compile()

# Valid input
result = graph.invoke({"query": "What is AI?", "max_iterations": 3})

# Invalid input raises ValidationError
try:
    graph.invoke({"query": "", "max_iterations": 100})  # Fails validation
except Exception as e:
    print(f"Validation error: {e}")
```

**Important Pydantic Limitations:**

Runtime validation only occurs on inputs to the first node, not on outputs or intermediate states. The output of the graph is a dictionary, not a Pydantic model instance. Pydantic's recursive validation can impact performance in high-throughput scenarios.

### 3.3 Reducers for State Updates

Reducers define how node outputs are merged with existing state. Without a reducer, values are overwritten; with a reducer, values are combined according to the reducer's logic.

```python
from typing import Annotated
from operator import add

def custom_message_reducer(existing: list, new: list) -> list:
    """Custom reducer that deduplicates messages by ID"""
    existing_ids = {msg.get("id") for msg in existing if msg.get("id")}
    result = existing.copy()
    for msg in new:
        if msg.get("id") not in existing_ids:
            result.append(msg)
    return result

class StateWithCustomReducer(TypedDict):
    # Built-in add_messages handles message deduplication
    messages: Annotated[list, add_messages]

    # Simple append reducer
    logs: Annotated[list[str], add]

    # Counter reducer
    iteration: Annotated[int, add]
```

### 3.4 Using Overwrite to Bypass Reducers

When you need to completely replace a value rather than merge it, use the `Overwrite` type:

```python
from langgraph.types import Overwrite

def reset_messages(state: State) -> dict:
    """Reset messages instead of appending"""
    return {
        "messages": Overwrite([{"role": "system", "content": "Conversation reset"}])
    }
```

---

## 4. Tool Definition and Binding Best Practices

### 4.1 Tool Definition with @tool Decorator

The `@tool` decorator is the standard way to define tools that agents can use:

```python
from langchain.tools import tool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    """Input schema for search tool"""
    query: str = Field(description="The search query")
    max_results: int = Field(default=5, description="Maximum results to return")

@tool(args_schema=SearchInput)
def search_documents(query: str, max_results: int = 5) -> str:
    """Search the document database for relevant information.

    Use this tool when you need to find specific information from the knowledge base.
    """
    # Implementation
    results = perform_search(query, max_results)
    return "\n".join(results)

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A valid Python mathematical expression

    Returns:
        The result of the calculation
    """
    try:
        result = eval(expression)  # Use safer evaluation in production
        return str(result)
    except Exception as e:
        return f"Error: {e}"
```

### 4.2 Binding Tools to Models

Tools are bound to chat models using the `bind_tools` method, which formats tool schemas appropriately for the model's API:

```python
from langchain.chat_models import init_chat_model

# Initialize model
model = init_chat_model("gpt-4o", temperature=0)

# Define tools
tools = [search_documents, calculate]

# Bind tools to model
model_with_tools = model.bind_tools(tools)

# The model can now decide to call tools
response = model_with_tools.invoke([
    {"role": "user", "content": "Search for information about LangGraph"}
])

# Check for tool calls
if response.tool_calls:
    for tool_call in response.tool_calls:
        print(f"Tool: {tool_call['name']}, Args: {tool_call['args']}")
```

### 4.3 Using ToolNode for Automatic Tool Execution

LangGraph provides `ToolNode` for automatic tool execution:

```python
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, MessagesState, START, END

# Define tools
tools = [search_documents, calculate]

# Create tool node
tool_node = ToolNode(tools)

# Bind tools to model
model = init_chat_model("gpt-4o", temperature=0).bind_tools(tools)

def call_model(state: MessagesState):
    """Node that calls the model"""
    response = model.invoke(state["messages"])
    return {"messages": [response]}

# Build the ReAct agent pattern
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    tools_condition,  # Routes to "tools" if tool calls present, else END
    {"tools": "tools", END: END}
)
workflow.add_edge("tools", "agent")  # Loop back after tool execution

graph = workflow.compile()
```

### 4.4 Structured Output with Tools

For reliable structured outputs, use `with_structured_output`:

```python
from pydantic import BaseModel, Field

class AnalysisResult(BaseModel):
    """Structured output for analysis"""
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(ge=0, le=1, description="Confidence score")
    key_topics: list[str] = Field(description="Main topics identified")
    summary: str = Field(description="Brief summary")

# Create model with structured output
analysis_model = init_chat_model("gpt-4o").with_structured_output(AnalysisResult)

def analyze_text(state: State) -> dict:
    """Analyze text and return structured result"""
    result = analysis_model.invoke([
        {"role": "user", "content": f"Analyze this text: {state['text']}"}
    ])
    return {
        "sentiment": result.sentiment,
        "confidence": result.confidence,
        "topics": result.key_topics
    }
```

---

## 5. Memory Management Patterns in LangGraph

### 5.1 Checkpointers for State Persistence

Checkpointers save the graph state at every step, enabling features like conversation memory, human-in-the-loop, time travel, and fault tolerance.

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.sqlite import SqliteSaver

# In-memory checkpointer (development)
memory_checkpointer = InMemorySaver()

# SQLite checkpointer (single-node production)
sqlite_checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

# PostgreSQL checkpointer (production)
postgres_checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost/langgraph"
)

# Compile graph with checkpointer
graph = workflow.compile(checkpointer=postgres_checkpointer)

# Invoke with thread_id for persistence
config = {"configurable": {"thread_id": "user-session-123"}}
result = graph.invoke({"messages": [{"role": "user", "content": "Hello"}]}, config)

# Continue the conversation (state is restored)
result = graph.invoke(
    {"messages": [{"role": "user", "content": "What did I just say?"}]},
    config
)
```

### 5.2 Thread Management

Each conversation or workflow instance is identified by a `thread_id`. The checkpointer uses this ID to store and retrieve state:

```python
# Different threads maintain separate conversation states
config_user_1 = {"configurable": {"thread_id": "user-1-session"}}
config_user_2 = {"configurable": {"thread_id": "user-2-session"}}

# User 1's conversation
graph.invoke({"messages": [{"role": "user", "content": "I'm Alice"}]}, config_user_1)

# User 2's conversation (completely separate)
graph.invoke({"messages": [{"role": "user", "content": "I'm Bob"}]}, config_user_2)

# Later, each user continues their own conversation
graph.invoke({"messages": [{"role": "user", "content": "What's my name?"}]}, config_user_1)
# Returns: "Your name is Alice"
```

### 5.3 Accessing State History

Retrieve the full history of a thread for debugging or analysis:

```python
# Get current state
current_state = graph.get_state(config)
print(f"Current state values: {current_state.values}")
print(f"Next nodes to execute: {current_state.next}")

# Get full state history
history = list(graph.get_state_history(config))
for i, snapshot in enumerate(history):
    print(f"Step {i}: {snapshot.values.get('current_step')}")
```

### 5.4 Time Travel and State Replay

Resume from any previous checkpoint using `checkpoint_id`:

```python
# Get a specific checkpoint
history = list(graph.get_state_history(config))
old_checkpoint = history[2]  # Get the third checkpoint

# Resume from that checkpoint (creates a new branch)
resume_config = {
    "configurable": {
        "thread_id": "user-session-123",
        "checkpoint_id": old_checkpoint.config["configurable"]["checkpoint_id"]
    }
}

# This replays steps up to the checkpoint, then continues
result = graph.invoke({"messages": [{"role": "user", "content": "New direction"}]}, resume_config)
```

### 5.5 Updating State Externally

Modify the graph state without invoking the graph:

```python
# Update state manually
graph.update_state(
    config,
    {"messages": [{"role": "system", "content": "Context has been updated"}]},
    as_node="agent"  # Update appears to come from this node
)
```

---

## 6. Human-in-the-Loop Patterns

### 6.1 Using interrupt() for Dynamic Pauses

The `interrupt()` function pauses execution and returns control to the caller:

```python
from langgraph.types import interrupt, Command

def approval_node(state: State):
    """Pause for human approval before proceeding"""
    # This pauses execution and returns the payload to the caller
    user_response = interrupt({
        "question": "Do you approve this action?",
        "details": state["proposed_action"],
        "options": ["approve", "reject", "modify"]
    })

    if user_response["decision"] == "approve":
        return {"status": "approved", "action": state["proposed_action"]}
    elif user_response["decision"] == "reject":
        return {"status": "rejected", "action": None}
    else:
        return {"status": "modified", "action": user_response.get("modified_action")}

# Build graph with checkpointer (required for interrupts)
graph = workflow.compile(checkpointer=InMemorySaver())

# First invocation - pauses at interrupt
config = {"configurable": {"thread_id": "approval-flow-1"}}
result = graph.invoke({"proposed_action": "Delete user data"}, config)

# Check the interrupt payload
print(result["__interrupt__"])
# [Interrupt(value={'question': 'Do you approve...', 'details': '...'})]

# Resume with human response
from langgraph.types import Command
result = graph.invoke(
    Command(resume={"decision": "approve"}),
    config
)
```

### 6.2 Review and Edit State Pattern

Allow humans to review and modify LLM-generated content:

```python
from langgraph.types import interrupt

def review_content_node(state: State):
    """Pause for human review of generated content"""
    edited_content = interrupt({
        "instruction": "Please review and edit this content",
        "generated_text": state["draft_response"],
        "guidelines": "Ensure accuracy and appropriate tone"
    })

    # The resume value becomes the return of interrupt()
    return {"final_response": edited_content}

# Resume with edited content
graph.invoke(
    Command(resume="The corrected and improved response text"),
    config
)
```

### 6.3 Tool Approval Pattern

Place interrupts inside tools for human oversight of tool calls:

```python
from langchain.tools import tool
from langgraph.types import interrupt

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient (requires human approval)."""

    # Pause for approval before sending
    response = interrupt({
        "action": "send_email",
        "to": to,
        "subject": subject,
        "body": body,
        "message": "Approve sending this email?"
    })

    if response.get("approved"):
        # Use potentially modified values
        final_to = response.get("to", to)
        final_subject = response.get("subject", subject)
        final_body = response.get("body", body)

        # Actually send the email
        send_actual_email(final_to, final_subject, final_body)
        return f"Email sent to {final_to}"

    return "Email cancelled by user"
```

### 6.4 Important Rules for Interrupts

**Do not wrap interrupt() in try/except** as it raises a special exception:

```python
# WRONG - catches the interrupt exception
def bad_node(state):
    try:
        result = interrupt("Question?")
    except Exception:
        pass  # This catches the interrupt!

# CORRECT - separate interrupt from error handling
def good_node(state):
    result = interrupt("Question?")
    try:
        process_result(result)
    except ValueError as e:
        handle_error(e)
```

**Keep interrupt order consistent** - LangGraph matches resume values by index:

```python
# CORRECT - consistent order
def consistent_node(state):
    name = interrupt("What's your name?")
    age = interrupt("What's your age?")
    return {"name": name, "age": age}

# WRONG - conditional interrupts cause index mismatch
def inconsistent_node(state):
    name = interrupt("What's your name?")
    if state.get("needs_age"):  # Might skip this interrupt
        age = interrupt("What's your age?")  # Index changes!
```

---

## 7. Streaming Response Patterns

### 7.1 Stream Modes Overview

LangGraph supports multiple streaming modes that can be combined:

| Mode | Description |
|------|-------------|
| `values` | Full state after each step |
| `updates` | Only the delta/changes from each step |
| `messages` | LLM tokens with metadata |
| `custom` | User-defined data from nodes |
| `debug` | Comprehensive execution information |

### 7.2 Basic Streaming

```python
from langgraph.graph import StateGraph, START, END

# Stream state updates
for chunk in graph.stream(
    {"topic": "AI agents"},
    stream_mode="updates"
):
    print(chunk)
# {'analyze': {'analysis': '...'}}
# {'generate': {'response': '...'}}

# Stream full state values
for chunk in graph.stream(
    {"topic": "AI agents"},
    stream_mode="values"
):
    print(chunk)
# {'topic': 'AI agents', 'analysis': '...', 'response': ''}
# {'topic': 'AI agents', 'analysis': '...', 'response': '...'}
```

### 7.3 Streaming LLM Tokens

Stream token-by-token output from LLMs:

```python
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START

model = init_chat_model("gpt-4o-mini")

def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(MessagesState).add_node("agent", call_model)
# ... add edges and compile

# Stream tokens
for message_chunk, metadata in graph.stream(
    {"messages": [{"role": "user", "content": "Write a poem about AI"}]},
    stream_mode="messages"
):
    if message_chunk.content:
        print(message_chunk.content, end="", flush=True)
```

### 7.4 Custom Streaming from Nodes

Send custom progress updates from within nodes:

```python
from langgraph.config import get_stream_writer

@tool
def process_documents(query: str) -> str:
    """Process documents with progress updates"""
    writer = get_stream_writer()

    documents = fetch_documents(query)
    total = len(documents)

    for i, doc in enumerate(documents):
        writer({
            "type": "progress",
            "current": i + 1,
            "total": total,
            "message": f"Processing document {i + 1}/{total}"
        })
        process_document(doc)

    return f"Processed {total} documents"

# Consume custom stream
for chunk in graph.stream(inputs, stream_mode="custom"):
    if chunk.get("type") == "progress":
        print(f"Progress: {chunk['current']}/{chunk['total']}")
```

### 7.5 Multiple Stream Modes

Combine multiple stream modes:

```python
for mode, chunk in graph.stream(
    inputs,
    stream_mode=["updates", "messages", "custom"]
):
    if mode == "updates":
        print(f"State update: {chunk}")
    elif mode == "messages":
        msg, metadata = chunk
        print(f"Token: {msg.content}")
    elif mode == "custom":
        print(f"Custom: {chunk}")
```

### 7.6 Async Streaming

For async applications:

```python
async def stream_response():
    async for chunk in graph.astream(
        {"messages": [{"role": "user", "content": "Hello"}]},
        stream_mode="messages"
    ):
        message_chunk, metadata = chunk
        if message_chunk.content:
            yield message_chunk.content
```

---

## 8. Error Handling and Retry Patterns

### 8.1 Node-Level Error Handling

Implement error handling within individual nodes:

```python
from typing import TypedDict, Annotated
from operator import add

class StateWithErrors(TypedDict):
    messages: list
    error_count: Annotated[int, add]
    error_history: Annotated[list[str], add]
    last_error: str

def resilient_node(state: StateWithErrors) -> dict:
    """Node with built-in error handling"""
    try:
        result = perform_operation(state)
        return {"messages": [result]}
    except ConnectionError as e:
        return {
            "error_count": 1,
            "error_history": [f"ConnectionError: {str(e)}"],
            "last_error": str(e)
        }
    except ValueError as e:
        return {
            "error_count": 1,
            "error_history": [f"ValueError: {str(e)}"],
            "last_error": str(e)
        }
```

### 8.2 Conditional Error Routing

Route to error handling nodes based on state:

```python
from typing import Literal

def error_router(state: StateWithErrors) -> Literal["retry", "fallback", "continue"]:
    """Route based on error state"""
    if state.get("error_count", 0) == 0:
        return "continue"
    elif state.get("error_count", 0) < 3:
        return "retry"
    else:
        return "fallback"

def retry_node(state: StateWithErrors) -> dict:
    """Attempt retry with backoff"""
    import time
    backoff = 2 ** state.get("error_count", 1)
    time.sleep(min(backoff, 30))  # Max 30 second backoff
    return {"last_error": ""}  # Clear error for retry

def fallback_node(state: StateWithErrors) -> dict:
    """Graceful degradation"""
    return {
        "messages": [{"role": "system", "content": "Using fallback response"}],
        "status": "degraded"
    }

# Add conditional routing for errors
workflow.add_conditional_edges("main_node", error_router)
```

### 8.3 Recursion Limit Handling

Proactively handle recursion limits to prevent crashes:

```python
from langgraph.managed import RemainingSteps
from typing import Literal

class StateWithSteps(TypedDict):
    messages: list
    remaining_steps: RemainingSteps  # Managed value

def step_aware_router(state: StateWithSteps) -> Literal["continue", "wrap_up"]:
    """Route based on remaining steps"""
    if state["remaining_steps"] <= 2:
        return "wrap_up"
    return "continue"

def wrap_up_node(state: StateWithSteps) -> dict:
    """Provide best-effort response when approaching limit"""
    return {
        "messages": [{
            "role": "assistant",
            "content": "I've reached my processing limit. Here's my best answer so far..."
        }]
    }

# Invoke with recursion limit
result = graph.invoke(
    inputs,
    config={"recursion_limit": 25}
)
```

### 8.4 Retry Policies with Tenacity

Use the tenacity library for sophisticated retry logic:

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError))
)
def call_external_api(query: str) -> dict:
    """API call with automatic retries"""
    response = httpx.get(f"https://api.example.com/search?q={query}", timeout=10)
    response.raise_for_status()
    return response.json()

def api_node(state: State) -> dict:
    """Node that calls external API with retries"""
    try:
        result = call_external_api(state["query"])
        return {"api_result": result}
    except Exception as e:
        return {"error": str(e), "api_result": None}
```

---

## 9. LCEL (LangChain Expression Language) Patterns

### 9.1 Core LCEL Concepts

LCEL uses the pipe operator (`|`) to compose components into chains:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

# Basic LCEL chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}")
])
model = init_chat_model("gpt-4o")
parser = StrOutputParser()

# Compose with pipe operator
chain = prompt | model | parser

# Invoke the chain
result = chain.invoke({"input": "What is LangGraph?"})
```

### 9.2 Parallel Execution with RunnableParallel

Execute multiple chains in parallel:

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Define parallel branches
analyze_sentiment = prompt_sentiment | model | parser
extract_entities = prompt_entities | model | entity_parser
summarize = prompt_summary | model | parser

# Run all in parallel
parallel_chain = RunnableParallel(
    sentiment=analyze_sentiment,
    entities=extract_entities,
    summary=summarize,
    original=RunnablePassthrough()  # Pass through original input
)

result = parallel_chain.invoke({"text": "..."})
# result = {"sentiment": "...", "entities": [...], "summary": "...", "original": {...}}
```

### 9.3 Conditional Branching with RunnableBranch

Route to different chains based on input:

```python
from langchain_core.runnables import RunnableBranch

# Define condition functions
def is_technical(input_dict):
    return "code" in input_dict.get("query", "").lower()

def is_creative(input_dict):
    return any(word in input_dict.get("query", "").lower()
               for word in ["write", "create", "story"])

# Create branching chain
router = RunnableBranch(
    (is_technical, technical_chain),
    (is_creative, creative_chain),
    general_chain  # Default branch
)

result = router.invoke({"query": "Write me a story about robots"})
```

### 9.4 Fallback Chains

Implement fallbacks for reliability:

```python
from langchain_core.runnables import RunnableWithFallbacks

# Primary and fallback models
primary_model = init_chat_model("gpt-4o")
fallback_model = init_chat_model("gpt-3.5-turbo")

# Chain with fallback
robust_chain = (prompt | primary_model | parser).with_fallbacks(
    [prompt | fallback_model | parser]
)
```

### 9.5 Binding Configuration

Bind default configuration to runnables:

```python
# Bind configuration
configured_model = model.bind(
    temperature=0.7,
    max_tokens=1000
)

# Bind tools
model_with_tools = model.bind_tools([search_tool, calculate_tool])

# Bind stop sequences
model_with_stop = model.bind(stop=["\n\nHuman:", "\n\nAssistant:"])
```

---

## 10. RAG Integration with Vector Stores

### 10.1 Basic RAG Agent Pattern

```python
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = InMemoryVectorStore(embeddings)

# Add documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://example.com/docs")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = splitter.split_documents(docs)
vectorstore.add_documents(splits)

# Create retriever tool
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for relevant information."""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

# Build RAG agent
tools = [search_knowledge_base]
model = init_chat_model("gpt-4o").bind_tools(tools)

def agent(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

workflow = StateGraph(MessagesState)
workflow.add_node("agent", agent)
workflow.add_node("tools", ToolNode(tools))
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

graph = workflow.compile()
```

### 10.2 Adaptive RAG with Document Grading

```python
from pydantic import BaseModel, Field
from typing import Literal

class GradeDocuments(BaseModel):
    """Grade document relevance"""
    score: Literal["relevant", "not_relevant"] = Field(
        description="Is the document relevant to the question?"
    )

grader = init_chat_model("gpt-4o").with_structured_output(GradeDocuments)

def grade_documents(state: MessagesState) -> Literal["generate", "rewrite_query"]:
    """Grade retrieved documents for relevance"""
    question = state["messages"][0].content
    documents = state["messages"][-1].content

    result = grader.invoke([{
        "role": "user",
        "content": f"Question: {question}\n\nDocument: {documents}\n\nIs this relevant?"
    }])

    if result.score == "relevant":
        return "generate"
    return "rewrite_query"

def rewrite_query(state: MessagesState):
    """Rewrite the query for better retrieval"""
    original = state["messages"][0].content
    rewriter = init_chat_model("gpt-4o")

    response = rewriter.invoke([{
        "role": "user",
        "content": f"Rewrite this query for better search results: {original}"
    }])

    return {"messages": [{"role": "user", "content": response.content}]}

# Add to workflow
workflow.add_conditional_edges("retrieve", grade_documents)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_edge("rewrite_query", "agent")  # Loop back to agent
```

---

## 11. Production Deployment Considerations

### 11.1 Containerization with Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LANGCHAIN_TRACING_V2=true

# Run the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 11.2 API Server with FastAPI

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    thread_id: str

@app.post("/chat")
async def chat(request: ChatRequest):
    config = {"configurable": {"thread_id": request.thread_id}}

    try:
        result = await graph.ainvoke(
            {"messages": [{"role": "user", "content": request.message}]},
            config
        )
        return {"response": result["messages"][-1].content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    config = {"configurable": {"thread_id": request.thread_id}}

    async def generate():
        async for chunk, metadata in graph.astream(
            {"messages": [{"role": "user", "content": request.message}]},
            config,
            stream_mode="messages"
        ):
            if chunk.content:
                yield f"data: {chunk.content}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

### 11.3 Monitoring with LangSmith

```python
import os

# Enable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "production-agent"

# Traces are automatically captured for all LangChain/LangGraph operations
```

### 11.4 Database-Backed Checkpointers for Production

```python
from langgraph.checkpoint.postgres import PostgresSaver
import os

# Production PostgreSQL checkpointer
DATABASE_URL = os.environ["DATABASE_URL"]

checkpointer = PostgresSaver.from_conn_string(
    DATABASE_URL,
    pool_size=20,  # Connection pool size
    max_overflow=10  # Additional connections allowed
)

# Compile with production checkpointer
graph = workflow.compile(checkpointer=checkpointer)
```

### 11.5 Health Checks and Graceful Shutdown

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await checkpointer.setup()  # Initialize database tables
    yield
    # Shutdown
    await checkpointer.close()  # Close connections

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

@app.get("/ready")
async def readiness_check():
    # Check dependencies
    try:
        await checkpointer.ping()
        return {"status": "ready"}
    except Exception:
        raise HTTPException(status_code=503, detail="Not ready")
```

---

## 12. Conclusion

LangChain 1.0 and LangGraph represent a mature ecosystem for building production AI agents in 2025. The key takeaways for building reliable systems include:

**State Management:** Use TypedDict with explicit reducers for performance-critical applications, and Pydantic for scenarios requiring input validation. The `add_messages` reducer should be the default for conversation-based agents.

**Graph Architecture:** StateGraph provides the flexibility needed for complex workflows. Use conditional edges for dynamic routing and Command objects for combined state updates with routing decisions.

**Memory and Persistence:** Always use checkpointers in production to enable conversation continuity, human-in-the-loop workflows, and fault tolerance. PostgreSQL-backed checkpointers are recommended for production deployments.

**Streaming:** Implement streaming from the start using the appropriate stream modes. The `messages` mode is essential for real-time token streaming, while `custom` mode enables progress updates from tools.

**Error Handling:** Build error handling into the graph structure with dedicated error nodes and conditional routing. Use retry policies with exponential backoff for transient failures.

**Production Deployment:** Containerize applications with Docker, use LangSmith for observability, and implement proper health checks and graceful shutdown procedures.

These patterns enable organizations to build AI agents that are reliable, maintainable, and scalable for enterprise production environments.

---

## Sources

1. [LangGraph Persistence Documentation](https://docs.langchain.com/oss/python/langgraph/persistence) - High Reliability - Official LangChain documentation
2. [LangGraph Graph API Documentation](https://docs.langchain.com/oss/python/langgraph/graph-api) - High Reliability - Official LangChain documentation
3. [LangGraph Interrupts Documentation](https://docs.langchain.com/oss/python/langgraph/interrupts) - High Reliability - Official LangChain documentation
4. [LangGraph Streaming Documentation](https://docs.langchain.com/oss/python/langgraph/streaming) - High Reliability - Official LangChain documentation
5. [LangGraph State Management Guide](https://docs.langchain.com/oss/python/langgraph/use-graph-api) - High Reliability - Official LangChain documentation
6. [LangGraph Agentic RAG Tutorial](https://docs.langchain.com/oss/python/langgraph/agentic-rag) - High Reliability - Official LangChain documentation
7. [Mastering LangGraph State Management in 2025](https://sparkco.ai/blog/mastering-langgraph-state-management-in-2025) - Medium Reliability - Industry blog
8. [LangChain Ecosystem Deployment Best Practices 2025](https://sparkco.ai/blog/langchain-ecosystem-deployment-best-practices-2025) - Medium Reliability - Industry blog
9. [Advanced Error Handling in LangGraph Applications](https://sparkco.ai/blog/advanced-error-handling-strategies-in-langgraph-applications) - Medium Reliability - Industry blog
10. [LangGraph Redis Integration](https://redis.io/blog/langgraph-redis-build-smarter-ai-agents-with-memory-persistence/) - High Reliability - Redis official blog
