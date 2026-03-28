# Agno Documentation - Complete Guide

## Introduction

### What is Agno?

Agno is the runtime for agentic software. Build agents, teams, and workflows. Run them as scalable services. Monitor and manage them in production.

#### Core Components

| Layer | What it does |
|---|---|
| **Framework** | Build agents, teams, and workflows with memory, knowledge, guardrails, and 100+ integrations. |
| **Runtime** | Serve your system in production with a stateless, session-scoped FastAPI backend. |
| **Control Plane** | Test, monitor, and manage your system using the AgentOS UI. |

#### What You Can Build

Agno powers real agentic systems:

- **Pal** - A personal agent that learns your preferences
- **Dash** - A self-learning data agent grounded in six layers of context
- **Scout** - A self-learning context agent that manages enterprise context knowledge
- **Gcode** - A post-IDE coding agent that improves over time
- **Investment Team** - A multi-agent investment committee that debates and allocates capital

#### Key Concepts

- **Agents** - Build intelligent programs that reason, use tools, and take action
- **Teams** - Coordinate multiple agents to collaborate and reach decisions
- **Workflows** - Orchestrate deterministic and agentic steps into structured systems
- **AgentOS** - Deploy, govern, and operate agents in production

#### Built for Production

Agno runs in your infrastructure, not ours.

- Stateless, horizontally scalable runtime
- 50+ APIs and background execution
- Per-user and per-session isolation
- Runtime approval enforcement
- Native tracing and full auditability
- Sessions, memory, knowledge, and traces stored in your database

---

## Your First Agent

### Build and Run Your First Agent in 20 Lines of Code

In this guide, you'll build an agent that:
- Connects to an MCP server
- Stores and retrieves past conversations
- Runs as a production API

All in about 20 lines of code.

### 1. Define the Agent

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.anthropic import Claude
from agno.os import AgentOS
from agno.tools.mcp import MCPTools

agno_assist = Agent(
    name="Agno Assist",
    model=Claude(id="claude-sonnet-4-5"),
    db=SqliteDb(db_file="agno.db"),  # session storage
    tools=[MCPTools(url="https://docs.agno.com/mcp")],  # Agno docs via MCP
    add_datetime_to_context=True,
    add_history_to_context=True,  # include past runs
    num_history_runs=3,  # last 3 conversations
    markdown=True,
)

# Serve via AgentOS → streaming, auth, session isolation, API endpoints
agent_os = AgentOS(agents=[agno_assist], tracing=True)
app = agent_os.get_app()
```

You now have:
- A stateful agent
- Streaming responses
- Per-user session isolation
- A production-ready API
- Tracing enabled out of the box

### 2. Run Your AgentOS

**Setup your virtual environment:**
```bash
uv venv --python 3.12
source .venv/bin/activate
```

**Install dependencies:**
```bash
uv pip install -U 'agno[os]' anthropic mcp
```

**Export your Anthropic API key:**
```bash
export ANTHROPIC_API_KEY=sk-***
```

**Run your AgentOS:**
```bash
fastapi dev agno_assist.py
```

Your AgentOS is now running at: `http://localhost:8000`
API documentation is automatically available at: `http://localhost:8000/docs`

### 3. Connect to the AgentOS UI

The AgentOS UI connects directly from your browser to your runtime.

1. Open `os.agno.com` and sign in
2. Click "Add new OS" in the top navigation
3. Select "Local" to connect to a local AgentOS
4. Enter your endpoint URL (default: `http://localhost:8000`)
5. Name it something like "Development OS"
6. Click "Connect"

---

## Agents

### What are Agents?

Agents are AI programs that use tools to accomplish tasks.

Agents are a stateful control loop around a stateless model. The model reasons and calls tools in a loop, guided by instructions. Add memory, knowledge, storage, human-in-the-loop, and guardrails as needed.

### Building Agents

Start simple: a model, tools, and instructions.

To build effective agents, start simple: a model, tools, and instructions. Once that works, layer in more functionality as needed.

**Simplest possible agent:**
```python
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools.hackernews import HackerNewsTools

agent = Agent(
    model=Claude(id="claude-sonnet-4-5"),
    tools=[HackerNewsTools()],
    instructions="Write a report on the topic. Output only the report.",
    markdown=True,
)

agent.print_response("Trending startups and products.", stream=True)
```

### Running Agents

**Development:**
```python
agent.print_response("Your message", stream=True)
```

**Production:**
```python
from typing import Iterator
from agno.agent import Agent, RunOutputEvent, RunEvent

# Stream the response
stream: Iterator[RunOutputEvent] = agent.run("Trending products", stream=True)
for chunk in stream:
    if chunk.event == RunEvent.run_content:
        print(chunk.content)
```

### Callable Factories

Pass a function instead of a static list for `tools` or `knowledge`. The function is called at the start of each run, so the toolset or knowledge base can vary per user or session.

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.run import RunContext
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools

def get_tools(run_context: RunContext):
    role = (run_context.session_state or {}).get("role", "general")
    if role == "finance":
        return [YFinanceTools()]
    return [DuckDuckGoTools()]

agent = Agent(
    model=OpenAIResponses(id="gpt-5-mini"),
    tools=get_tools,
)

agent.print_response(
    "AAPL stock price?",
    session_state={"role": "finance"},
    stream=True
)
```

---

## Teams

### What are Teams?

Groups of agents that collaborate to solve complex tasks.

A Team is a collection of agents (or sub-teams) that work together. The team leader delegates tasks to members based on their roles.

```python
from agno.team import Team
from agno.agent import Agent

team = Team(members=[
    Agent(name="English Agent", role="You answer questions in English"),
    Agent(name="Chinese Agent", role="You answer questions in Chinese"),
    Team(
        name="Germanic Team",
        role="You coordinate the team members to answer questions in German and Dutch",
        members=[
            Agent(name="German Agent", role="You answer questions in German"),
            Agent(name="Dutch Agent", role="You answer questions in Dutch"),
        ],
    ),
])
```

### Why Teams?

Single agents hit limits fast:
- **Specialization** - Each agent masters one domain instead of being mediocre at everything
- **Parallel processing** - Multiple agents work simultaneously on independent subtasks
- **Maintainability** - When something breaks, you know exactly which agent to fix
- **Scalability** - Add capabilities by adding agents, not rewriting everything

### Team Modes

Team 2.0 introduces `TeamMode` to make collaboration styles explicit:

```python
from agno.team import Team, TeamMode

team = Team(
    name="Research Team",
    members=[...],
    mode=TeamMode.broadcast,
)
```

| Mode | Configuration | Use case |
|---|---|---|
| **Coordinate** | `mode=TeamMode.coordinate` (default) | Decompose work, delegate to members, synthesize results |
| **Route** | `mode=TeamMode.route` | Route to a single specialist and return their response directly |
| **Broadcast** | `mode=TeamMode.broadcast` | Delegate the same task to all members and synthesize |
| **Tasks** | `mode=TeamMode.tasks` | Run a task list loop until the goal is complete |

---

## Workflows

### What are Workflows?

Workflows orchestrate agents, teams, and functions through defined steps for repeatable tasks.

A Workflow orchestrates agents, teams, and functions as a collection of steps. Steps can run sequentially, in parallel, in loops, or conditionally based on results.

### Your First Workflow

```python
from agno.agent import Agent
from agno.workflow import Workflow
from agno.tools.hackernews import HackerNewsTools

researcher = Agent(
    name="Researcher",
    instructions="Find relevant information about the topic",
    tools=[HackerNewsTools()]
)

writer = Agent(
    name="Writer",
    instructions="Write a clear, engaging article based on the research"
)

content_workflow = Workflow(
    name="Content Creation",
    steps=[researcher, writer]
)

content_workflow.print_response("Write an article about AI trends", stream=True)
```

### When to Use Workflows

Use a workflow when:
- You need predictable, repeatable execution
- Tasks have clear sequential steps with defined inputs and outputs
- You want audit trails and consistent results across runs

### What Can Be a Step?

| Step Type | Description |
|---|---|
| **Agent** | Individual AI executor with specific tools and instructions |
| **Team** | Coordinated group of agents for complex sub-tasks |
| **Function** | Custom Python function for specialized logic |

---

## Tools

### What are Tools?

Tools are functions Agents call to interact with external systems.

Tools are what make Agents and Teams capable of real-world action. While using LLMs directly you can only generate text responses, Agents and Teams equipped with tools can interact with external systems and perform practical actions.

### Creating Tools

```python
import random
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools import tool

def get_weather(city: str) -> str:
    """Get the weather for the given city.
    
    Args:
        city (str): The city to get the weather for.
    """
    weather_conditions = ["sunny", "cloudy", "rainy", "snowy", "windy"]
    random_weather = random.choice(weather_conditions)
    return f"The weather in {city} is {random_weather}."

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[get_weather],
    markdown=True,
)

agent.print_response("What is the weather in San Francisco?", stream=True)
```

### How Do Tools Work?

The heart of Agent execution is the LLM loop:

1. The agent sends the run context (system message, user message, chat history, etc) and tool definitions to the model
2. The model responds with a message or a tool call
3. If the model makes a tool call, the tool is executed and the result is returned to the model
4. The model processes the updated context, repeating this loop until it produces a final message without any tool calls
5. The agent returns this final response to the caller

### Tool Definitions

Agno automatically converts your tool functions into the required tool definition format for the model.

```python
def get_weather(city: str) -> str:
    """Get the weather for a given city.
    
    Args:
        city (str): The city to get the weather for.
    """
    return f"The weather in {city} is sunny."
```

This will be converted into a JSON schema that describes the parameters and return type of the tool.

### Using a Toolkit

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.hackernews import HackerNewsTools

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[HackerNewsTools()],
)

agent.print_response(
    "What are the top stories on HackerNews?",
    markdown=True
)
```

### Tool Built-in Parameters

Agno automatically provides special parameters to your tools:

**Using the Run Context:**
```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.run import RunContext

def add_item(run_context: RunContext, item: str) -> str:
    """Add an item to the shopping list."""
    if not run_context.session_state:
        run_context.session_state = {}
    run_context.session_state["shopping_list"].append(item)
    return f"The shopping list is now {run_context.session_state['shopping_list']}"

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    session_state={"shopping_list": []},
    db=SqliteDb(db_file="tmp/agents.db"),
    tools=[add_item],
    instructions="Current state (shopping list) is: {shopping_list}",
    markdown=True,
)

agent.print_response(
    "Add milk, eggs, and bread to the shopping list",
    stream=True
)
```

### Tool Results

**Simple Return Types:**
```python
@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny and 75°F"

@tool
def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers."""
    return a + b

@tool
def get_user_info(user_id: str) -> dict:
    """Get user information."""
    return {
        "user_id": user_id,
        "name": "John Doe",
        "email": "john@example.com",
        "status": "active"
    }
```

**Media Content with ToolResult:**
```python
from agno.tools.function import ToolResult
from agno.media import Image

@tool
def generate_image(prompt: str) -> ToolResult:
    """Generate an image from a prompt."""
    image_artifact = Image(
        id="img_123",
        url="https://example.com/generated-image.jpg",
        original_prompt=prompt
    )
    return ToolResult(
        content=f"Generated image for: {prompt}",
        images=[image_artifact]
    )
```

### Callable Tool Factories

```python
def get_tools(run_context: RunContext):
    """Return different tools based on the user's role."""
    role = (run_context.session_state or {}).get("role", "viewer")
    base_tools = [search_web]
    if role == "admin":
        base_tools.append(search_internal_docs)
    if role in ("admin", "finance"):
        base_tools.append(get_account_balance)
    return base_tools

agent = Agent(
    model=OpenAIResponses(id="gpt-5-mini"),
    tools=get_tools,
    instructions=[
        "You are a helpful assistant.",
        "Use the tools available to you to answer the user's question.",
    ],
)
```

---

## Memory

### What is Memory?

Give your agents the ability to remember user preferences, context, and past interactions for truly personalized experiences.