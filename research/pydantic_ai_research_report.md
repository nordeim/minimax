# Pydantic AI Framework: Comprehensive Research Report for Building AI Agents in 2025

## Executive Summary

Pydantic AI is a dedicated Python agent framework developed by the Pydantic team, designed specifically for building production-grade AI applications with Large Language Models. Unlike using plain Pydantic for data validation, Pydantic AI provides a complete agent orchestration layer that combines type safety, dependency injection, structured outputs, and tool calling into a cohesive framework. The library has emerged as a compelling alternative to LangChain for teams prioritizing reliability, transparency, and maintainability in production environments. This report provides an in-depth analysis of Pydantic AI's core concepts, integration patterns, and best practices for building customer support AI agents, based on official documentation and community implementations as of early 2025.

---

## 1. Introduction

The research objective was to comprehensively analyze the Pydantic AI framework, clarify its distinction from plain Pydantic usage, and document patterns for building production AI agents. Pydantic AI represents a significant evolution in the AI agent ecosystem, bringing the same validation-first philosophy that made Pydantic the de facto standard for Python data modeling to the domain of LLM-powered applications[1].

The framework addresses critical challenges in AI development including unreliable LLM outputs, complex dependency management, testing difficulties, and the need for type-safe interactions with multiple model providers. With support for major LLM providers including OpenAI, Anthropic, Google Gemini, Amazon Bedrock, and many others, Pydantic AI positions itself as a model-agnostic solution for enterprise AI development[1][2].

---

## 2. What is Pydantic AI: Core Concepts and Differentiation

### Clarification: Pydantic AI vs Plain Pydantic

Pydantic AI (installed via `pip install pydantic-ai`) is a distinct library from the core Pydantic validation library. While plain Pydantic focuses on data validation and serialization for Python applications, Pydantic AI extends these capabilities into a complete agent framework specifically designed for LLM interactions[1].

The key distinction lies in their primary purposes. Plain Pydantic validates data structures and enforces type constraints at runtime, whereas Pydantic AI orchestrates entire AI agent workflows including model communication, tool execution, dependency injection, and structured output generation. Pydantic AI leverages the core Pydantic library internally for output validation, creating a synergy where developers familiar with Pydantic models can immediately benefit from type-safe AI development[2].

### Core Architecture Components

The Pydantic AI architecture revolves around several interconnected components that work together to create robust AI agents. The **Agent** class serves as the primary interface for LLM interactions, encapsulating model configuration, system prompts, tools, and output specifications. This centralized design enables agents to be reused, tested, and composed into larger systems[3].

**Dependencies** provide a type-safe mechanism for injecting data, connections, and services into agent components. Unlike global state or configuration files, dependencies are explicitly declared and passed through the execution context, making the data flow transparent and testable[4].

**Function Tools** extend agent capabilities by allowing LLMs to call Python functions during response generation. These tools follow a schema-driven approach where Pydantic AI automatically extracts parameter information from function signatures and docstrings, generating the JSON schemas that LLMs require for function calling[5].

**Structured Outputs** ensure that LLM responses conform to predefined Pydantic models. When an agent specifies an `output_type`, Pydantic AI handles the complex process of instructing the model, parsing responses, and validating results against the schema. Validation errors are automatically fed back to the model for retry attempts[3].

---

## 3. Agent Definition Patterns

### Basic Agent Creation

Creating an agent in Pydantic AI follows a declarative pattern that emphasizes clarity and type safety. The most fundamental agent requires only a model specification:

```python
from pydantic_ai import Agent

agent = Agent(
    'anthropic:claude-sonnet-4-0',
    instructions='Be concise, reply with one sentence.',
)

result = agent.run_sync('Where does "hello world" come from?')
print(result.output)
# Output: The first known use of "hello, world" was in a 1974 textbook about the C programming language.
```

More sophisticated agents declare dependencies, output types, and system prompts. The agent class is generic over both dependency and output types (`Agent[DepsType, OutputType]`), enabling static type checkers to verify correct usage throughout the codebase[3].

### System Prompts and Instructions

Pydantic AI distinguishes between **system prompts** and **instructions**, though both guide LLM behavior. Instructions are preferred for most use cases because they are automatically excluded from message history when conversations span multiple runs, preventing prompt duplication. System prompts persist in conversation history and are appropriate when this retention is intentionally desired[3].

Both static and dynamic prompts are supported. Static prompts are strings passed to the agent constructor, while dynamic prompts are functions decorated with `@agent.instructions` or `@agent.system_prompt` that receive the `RunContext` and return prompt strings:

```python
from datetime import date
from pydantic_ai import Agent, RunContext

agent = Agent(
    'openai:gpt-4o',
    deps_type=str,
    instructions="Use the customer's name while replying to them.",
)

@agent.instructions
def add_the_users_name(ctx: RunContext[str]) -> str:
    return f"The user's name is {ctx.deps}."

@agent.instructions
def add_the_date() -> str:
    return f'The date is {date.today()}.'

result = agent.run_sync('What is the date?', deps='Frank')
print(result.output)  # Hello Frank, the date today is 2025-01-02.
```

### Configuration and Limits

Agents support comprehensive configuration through `UsageLimits` and `ModelSettings`. Usage limits prevent runaway costs and infinite loops by capping response tokens, request counts, and tool calls. Model settings control parameters like temperature, max tokens, and timeout durations, with a clear precedence system where runtime overrides take priority over agent defaults, which override model defaults[3].

---

## 4. Structured Output Generation

### Defining Output Types

Structured outputs transform Pydantic AI from a chat interface into a reliable data extraction and generation system. By specifying an `output_type`, developers ensure that every agent response matches a predetermined schema:

```python
from pydantic import BaseModel
from pydantic_ai import Agent

class ChatResult(BaseModel):
    user_id: int
    message: str
    sentiment: str

agent = Agent(
    'openai:gpt-4o',
    output_type=ChatResult,
)

result = agent.run_sync('Analyze: "I love this product! Best purchase ever."')
print(result.output)
# ChatResult(user_id=0, message='Positive review expressing satisfaction', sentiment='positive')
```

### Validation and Retry Mechanisms

When LLM outputs fail validation, Pydantic AI automatically sends the validation errors back to the model with instructions to correct its response. The `retries` parameter controls how many correction attempts are allowed before raising an exception. This self-correction mechanism significantly improves reliability without requiring manual intervention[3].

The `ModelRetry` exception enables custom retry logic within tools and output validators:

```python
from pydantic_ai import Agent, RunContext, ModelRetry

@agent.tool(retries=2)
def get_user_by_name(ctx: RunContext[DatabaseConn], name: str) -> int:
    """Get a user's ID from their full name."""
    user_id = ctx.deps.users.get(name)
    if user_id is None:
        raise ModelRetry(f'No user found with name {name!r}, remember to provide their full name')
    return user_id
```

---

## 5. Tool/Function Calling Patterns

### Tool Registration Methods

Pydantic AI provides multiple approaches for registering tools with agents. The decorator approach offers the cleanest syntax for tools defined alongside the agent:

```python
from pydantic_ai import Agent, RunContext

agent = Agent('google-gla:gemini-2.5-flash', deps_type=str)

@agent.tool_plain
def roll_dice() -> str:
    """Roll a six-sided die and return the result."""
    return str(random.randint(1, 6))

@agent.tool
def get_player_name(ctx: RunContext[str]) -> str:
    """Get the player's name."""
    return ctx.deps
```

The distinction between `@agent.tool` and `@agent.tool_plain` determines whether the function receives the `RunContext`. Tools needing access to dependencies, retry counts, or other contextual information use `@agent.tool`, while stateless tools use `@agent.tool_plain`[5].

For tools defined elsewhere or requiring explicit configuration, the `tools` parameter accepts both plain functions and `Tool` instances:

```python
from pydantic_ai import Agent, Tool

agent = Agent(
    'openai:gpt-4o',
    deps_type=str,
    tools=[
        roll_dice,  # Automatic context detection
        Tool(get_player_name, takes_ctx=True),  # Explicit specification
    ],
)
```

### Schema Generation and Documentation

Pydantic AI automatically generates JSON schemas from function signatures and docstrings. The framework uses the `griffe` library to parse Google, NumPy, and Sphinx-style docstrings, extracting parameter descriptions that become part of the tool schema sent to the LLM[5]. This automatic documentation significantly reduces the boilerplate typically required for function calling:

```python
@agent.tool_plain(docstring_format='google', require_parameter_descriptions=True)
def search_products(query: str, category: str, max_price: float) -> list[dict]:
    """Search for products matching criteria.

    Args:
        query: Search terms for product name or description
        category: Product category filter (electronics, clothing, home)
        max_price: Maximum price in USD
    """
    # Implementation
```

### Built-in Tools

Pydantic AI includes several built-in tools that leverage provider-native capabilities. These tools execute on the provider's infrastructure rather than locally, offering enhanced performance and security for specific use cases[6]:

- **WebSearchTool**: Enables web searches for up-to-date information
- **CodeExecutionTool**: Runs code in a secure sandbox environment
- **ImageGenerationTool**: Creates images based on descriptions
- **FileSearchTool**: Performs vector search over uploaded documents
- **MCPServerTool**: Connects to Model Context Protocol servers

---

## 6. Dependency Injection Patterns

### Defining and Using Dependencies

The dependency injection system in Pydantic AI promotes testable, modular code by explicitly declaring what external resources an agent requires. Dependencies are typically defined as dataclasses containing the required services and data[4]:

```python
from dataclasses import dataclass
import httpx
from pydantic_ai import Agent, RunContext

@dataclass
class MyDeps:
    api_key: str
    http_client: httpx.AsyncClient

agent = Agent('openai:gpt-4o', deps_type=MyDeps)

@agent.system_prompt
async def get_system_prompt(ctx: RunContext[MyDeps]) -> str:
    response = await ctx.deps.http_client.get(
        'https://api.example.com/config',
        headers={'Authorization': f'Bearer {ctx.deps.api_key}'},
    )
    return f'Configuration: {response.json()}'

async def main():
    async with httpx.AsyncClient() as client:
        deps = MyDeps('secret-key', client)
        result = await agent.run('Process this request', deps=deps)
```

### RunContext Deep Dive

The `RunContext` object is the thread that connects all agent components. Beyond dependencies, it provides access to retry information (`ctx.retry`), usage statistics, and other execution metadata. The context is parameterized with the dependency type, enabling IDE auto-completion and static type checking throughout tool and prompt implementations[4][5].

### Testing with Dependency Overrides

The `agent.override()` context manager enables dependency injection for testing without modifying application code:

```python
class MockDatabase:
    async def get_customer(self, customer_id: int):
        return Customer(id=customer_id, name="Test User", email="test@example.com")

async def test_customer_lookup():
    mock_deps = CustomerServiceDeps(customer_id=1, db=MockDatabase())
    with customer_service_agent.override(deps=mock_deps):
        result = await handle_customer_inquiry(1, "What is my info?", MockDatabase())
        assert result.customer_info.name == "Test User"
```

---

## 7. Integration with LangChain/LangGraph

### Complementary Usage Patterns

Pydantic AI and LangChain serve different architectural philosophies but can complement each other in production systems. LangChain provides a comprehensive ecosystem with pre-built chains, memory modules, and extensive integrations, making it suitable for rapid prototyping. Pydantic AI offers a leaner, schema-first approach emphasizing reliability and control[7][8].

A hybrid architecture might use LangChain for memory management and complex routing while employing Pydantic AI as a validation layer for all LLM outputs. This pattern leverages LangChain's ecosystem benefits while ensuring output reliability through Pydantic's strict validation[7].

### Pydantic Graph for Multi-Agent Workflows

For complex multi-agent scenarios, Pydantic AI includes `pydantic-graph`, a state machine library for defining workflows through type hints. Unlike LangGraph's explicit graph construction, Pydantic Graph infers workflow structure from node return types:

```python
from dataclasses import dataclass
from pydantic_graph import BaseNode, End, Graph

@dataclass
class AnalyzeIntent(BaseNode[State, Deps, str]):
    user_message: str

    async def run(self, ctx) -> 'RouteToAgent | End[str]':
        intent = await analyze_intent(self.user_message)
        if intent == 'simple':
            return End(f"Simple response to: {self.user_message}")
        return RouteToAgent(intent=intent)

@dataclass
class RouteToAgent(BaseNode[State, Deps, str]):
    intent: str

    async def run(self, ctx) -> 'End[str]':
        response = await specialized_agent.run(ctx.state.context)
        return End(response.output)
```

The graph system supports state persistence, enabling long-running workflows that can survive interruptions and resume from saved checkpoints[9].

---

## 8. Validation Patterns for LLM Outputs

### Output Validators

Beyond schema validation, Pydantic AI supports custom output validators that can perform semantic checks and trigger retries for logically invalid responses:

```python
@agent.output_validator
async def validate_output(ctx: RunContext[MyDeps], output: str) -> str:
    response = await ctx.deps.http_client.post(
        'https://api.example.com/validate',
        json={'content': output}
    )
    if response.status_code == 400:
        raise ModelRetry(f'Invalid response: {response.json()["error"]}')
    return output
```

### Handling Validation Failures

The `capture_run_messages` context manager aids debugging by preserving the complete message history when validation or model errors occur:

```python
from pydantic_ai import Agent, capture_run_messages, UnexpectedModelBehavior

with capture_run_messages() as messages:
    try:
        result = agent.run_sync('Complex request')
    except UnexpectedModelBehavior as e:
        print('Error:', e)
        print('Message history:', messages)
```

---

## 9. Best Practices for Production Use

### Schema Design

Production schemas should be minimal and focused, containing only fields the workflow actually requires. Smaller schemas reduce token usage and improve model accuracy. Using `Literal` types for enumerated values and explicit `Field` descriptions guides models toward correct outputs[10]:

```python
from pydantic import BaseModel, Field
from typing import Literal

class CustomerServiceResult(BaseModel):
    action: Literal['info', 'order', 'ticket', 'escalate'] = Field(
        description="The primary action category for this request"
    )
    confidence: float = Field(ge=0, le=1, description="Confidence score 0-1")
    response: str = Field(max_length=1000, description="Customer-facing response")
```

### Error Handling Strategy

Robust production agents implement layered error handling: `ModelRetry` for recoverable issues, `ValidationError` catching for schema problems, and `UnexpectedModelBehavior` for terminal failures. Usage limits prevent cost overruns during retry loops[3].

### Observability Integration

Pydantic AI integrates with Pydantic Logfire for OpenTelemetry-based observability, enabling real-time debugging, performance monitoring, and cost tracking. For teams using other observability platforms, the OpenTelemetry compatibility ensures integration with existing monitoring infrastructure[1].

### Testing Approaches

The dependency injection system facilitates unit testing through mock dependencies. Integration tests should use `TestModel` for deterministic behavior verification:

```python
from pydantic_ai.models.test import TestModel

def test_agent_tool_calls():
    test_model = TestModel()
    result = agent.run_sync('Test prompt', model=test_model)

    # Inspect what tools were called
    assert test_model.last_model_request_parameters.function_tools
```

---

## 10. Customer Support Agent: Complete Implementation

The following implementation demonstrates a production-ready customer support agent incorporating all discussed patterns:

```python
from dataclasses import dataclass
from datetime import date
from typing import List, Literal
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, ModelRetry

# Data Models
class Customer(BaseModel):
    id: int
    name: str
    email: str
    tier: Literal['standard', 'premium', 'enterprise']

class Order(BaseModel):
    order_id: int
    customer_id: int
    items: List[str]
    total: float
    status: Literal['pending', 'shipped', 'delivered', 'returned']

class SupportTicket(BaseModel):
    ticket_id: int
    customer_id: int
    subject: str
    priority: Literal['low', 'medium', 'high', 'urgent']
    status: Literal['open', 'in_progress', 'resolved']

class CustomerServiceResult(BaseModel):
    customer_info: Customer | None = None
    orders: List[Order] = Field(default_factory=list)
    tickets: List[SupportTicket] = Field(default_factory=list)
    response: str = Field(description="Customer-facing response message")
    action_taken: str = Field(description="Internal log of actions performed")

# Dependencies
@dataclass
class CustomerServiceDeps:
    customer_id: int
    db: 'CustomerDatabase'

# Agent Definition
customer_agent = Agent(
    'anthropic:claude-sonnet-4-0',
    deps_type=CustomerServiceDeps,
    output_type=CustomerServiceResult,
    retries=2,
    instructions="""You are a helpful customer service assistant.
    Always verify customer identity before providing account information.
    Be empathetic and solution-oriented in your responses.
    For urgent issues, offer to escalate to a human representative.""",
)

@customer_agent.tool
async def get_customer_info(ctx: RunContext[CustomerServiceDeps]) -> Customer:
    """Retrieve customer profile information."""
    customer = await ctx.deps.db.get_customer(ctx.deps.customer_id)
    if not customer:
        raise ModelRetry("Customer not found. Please verify the customer ID.")
    return customer

@customer_agent.tool
async def get_customer_orders(ctx: RunContext[CustomerServiceDeps]) -> List[Order]:
    """Retrieve all orders for the current customer."""
    return await ctx.deps.db.get_orders(ctx.deps.customer_id)

@customer_agent.tool
async def create_support_ticket(
    ctx: RunContext[CustomerServiceDeps],
    subject: str,
    priority: Literal['low', 'medium', 'high', 'urgent'],
) -> SupportTicket:
    """Create a new support ticket for customer issues."""
    ticket = SupportTicket(
        ticket_id=await ctx.deps.db.next_ticket_id(),
        customer_id=ctx.deps.customer_id,
        subject=subject,
        priority=priority,
        status='open',
    )
    await ctx.deps.db.save_ticket(ticket)
    return ticket

# Usage
async def handle_inquiry(customer_id: int, inquiry: str, db: 'CustomerDatabase'):
    deps = CustomerServiceDeps(customer_id=customer_id, db=db)
    result = await customer_agent.run(inquiry, deps=deps)
    return result.output
```

---

## 11. Conclusion

Pydantic AI represents a mature, production-focused approach to building AI agents that prioritizes type safety, testability, and reliability over feature breadth. The framework's tight integration with Pydantic's validation ecosystem provides a natural extension for teams already using Pydantic in their Python applications. For customer support and similar structured interaction patterns, Pydantic AI's combination of dependency injection, structured outputs, and comprehensive tool support creates a robust foundation for enterprise AI development.

The framework is particularly well-suited for teams that value explicit over implicit behavior, require strong typing guarantees, and need to maintain AI systems in production over extended periods. While LangChain may offer faster initial prototyping, Pydantic AI's emphasis on transparency and control often proves advantageous as projects mature and reliability requirements increase.

---

## Sources

[1] [Pydantic AI Official Documentation](https://ai.pydantic.dev/) - High Reliability - Official framework documentation from the Pydantic team

[2] [Pydantic AI Agents Documentation](https://ai.pydantic.dev/agents/) - High Reliability - Official agent patterns and configuration reference

[3] [Langfuse - Comparing Open-Source AI Agent Frameworks](https://langfuse.com/blog/2025-03-19-ai-agent-comparison) - High Reliability - Independent technical comparison of AI frameworks

[4] [Pydantic AI Dependencies Documentation](https://ai.pydantic.dev/dependencies/) - High Reliability - Official dependency injection documentation

[5] [Pydantic AI Function Tools Documentation](https://ai.pydantic.dev/tools/) - High Reliability - Official tool definition patterns

[6] [Pydantic AI Built-in Tools Documentation](https://ai.pydantic.dev/builtin-tools/) - High Reliability - Official built-in tools reference

[7] [LangChain vs Pydantic AI: Two Roads to Building Smarter Agents](https://medium.com/@oaistack/langchain-vs-pydantic-ai-two-roads-to-building-smarter-agents-463d2b360d54) - Medium Reliability - Community comparison analysis

[8] [Reddit: Comparing LangChain vs PydanticAI](https://www.reddit.com/r/LangChain/comments/1ics69v/comparing_langchain_vs_pydanticai_for_building_an/) - Medium Reliability - Developer community discussion

[9] [Pydantic AI Graph Documentation](https://ai.pydantic.dev/graph/) - High Reliability - Official multi-agent workflow documentation

[10] [Building Intelligent Customer Service Agents with Pydantic-AI](https://landeros-labs.com/posts/cs-agents-pydantic/) - Medium Reliability - Detailed customer service implementation tutorial
