# 🧠 Session 2: Building an Autonomous Agent (Gemini & LangChain)

This module transforms the static microservice into an **autonomous agent** capable of reasoning, planning, and using external tools like web search.

The core concept is the **ReAct Loop** (Thought $\rightarrow$ Action $\rightarrow$ Observation $\rightarrow$ Repeat).

---

## 🎯 Learning Goals

* Understand the **ReAct** framework (Reason + Act).
* Use **LangChain** to connect Gemini to external tools.
* Implement a **Web Search** tool using DuckDuckGo.

---

## 🛠️ Prerequisites & Setup

**Context:** This session builds upon the environment created in Session 1.

1.  **Virtual Environment:** Ensure your Python virtual environment is active (`source venv/bin/activate` or `.\venv\Scripts\activate`).
2.  **API Key:** Ensure your `GOOGLE_API_KEY` is still set in your terminal environment.

### 1. Install New Dependencies

You need to add the specific libraries for agent orchestration and web search.

```bash
# Install the search tool
pip install duckduckgo-search
```

A new `requirement.txt` file has been provided within this directory for ease.

### 2. Study the code snippet below

```python
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent

# -------------------------------
# 1. Security Check
# -------------------------------
if "GOOGLE_API_KEY" not in os.environ:
    print("Error: GOOGLE_API_KEY not set. Please set it and retry.")
    exit()

# -------------------------------
# 2. LLM
# -------------------------------
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, max_retries=1)

# -------------------------------
# 3. Tools
# -------------------------------
tools = [DuckDuckGoSearchRun()]

# -------------------------------
# 4. System Instructions
# -------------------------------
system_prompt = (
    "You are a helpful AI assistant. Use tools when necessary. "
    "You are a ReAct agent."
    "You have the tool for Duck Duck Go searching"
)

# -------------------------------
# 5. Create ReAct Agent (FIXED ARGUMENT)
# -------------------------------
# Use 'system_message' instead of 'state_modifier' for your LangGraph version.
agent = create_agent(model=model, 
                     tools=tools,
                     system_prompt=system_prompt,
                     debug=True)

# -------------------------------
# 6. Agent runtime
# -------------------------------
app = agent

print("🤖 Agent ready! Ask a question.")
print("Type 'exit' to quit.\n")

# -------------------------------
# 7. CLI loop
# -------------------------------
while True:
    try:
        user_input = input("You: ").strip()
    except EOFError:
        print("\nExiting agent.")
        break

    if not user_input:
        print("⚠️ Input cannot be empty. Try again.")
        continue
    if user_input.lower() in ["exit", "quit"]:
        break

    print("\n--- Agent Thinking ---\n")

    # Invoke Agent 
    result = app.invoke({"messages": [("human", user_input)]})

    # print(result)

    # Extract Output
    last_message = result["messages"][-1].content[0]["text"]
    
    print(f"\nFinal Answer: {last_message}\n")
```

## ▶️ Running & Testing the Agent
### 1. Run the Script
```bash
python agent.py
```

### 2. The Activity: Witnessing the "ReAct" Loop
Ask questions that require external information (e.g., current news, sports scores, stock prices).

**👀 What to Observe:** The terminal will display the agent's internal monologue:

1. **Thought:** The agent plans to use the search tool.

2. **Action:** The agent calls DuckDuckGoSearchRun(...).

3. **Observation:** The tool returns raw search results.

4. **Thought:** The agent synthesizes the answer from the observation.


# 🤖 Session 2 Extension: Advanced Agent Engineering (Self-Contained ReAct)

This module moves beyond pre-built components to build a high-control, multi-tool agent using **LangChain's `create_agent`** and a **custom system prompt**. This method is preferred in production for its reliability and explicit control over the agent's behavior.

## 🎯 Learning Goals

* Define **multiple custom tools** (`@tool`) for specialized tasks (Web Search, Safe Calculation).
* Implement a **secure calculator** using Python's `ast.literal_eval`.
* Replace external prompt retrieval with a **local system prompt** to fully control the agent's reasoning pattern (Thought $\rightarrow$ Action $\rightarrow$ Observation).
* Practice invoking the agent and manually parsing its final structured output.

---

## 🛠️ Prerequisites & Setup

This module continues the work started in Session 1.

1.  **Dependencies:** You need the core LangChain agent libraries, the Google connector, and the DuckDuckGo search tool.
    ```bash
    # Ensure your virtual environment is active!
    pip install langchain langchain-community langchain-google-genai duckduckgo-search
    ```
    *Ignore if dependencies already exist*

2.  **API Key:** Ensure your `GOOGLE_API_KEY` is still set in your environment.

---

## 💻 Code: `agent_cli.py`

This script contains the entire agent setup, including the **LLM, two specialized tools, and the custom ReAct prompt**.

**Action:** Create a file named `agent_cli.py` and paste the following code:

```python
import os
import ast 
import operator as op
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# -------------------------------
# 1. LLM & Tools Setup
# -------------------------------

# A. Security Check
if "GOOGLE_API_KEY" not in os.environ:
    print("Error: GOOGLE_API_KEY not set. Please set it and retry.")
    exit()

# B. LLM (The Brain)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# C. Tools (The Hands)
# Tool 1: External Search
web_search_tool = DuckDuckGoSearchRun(name="Web_Search_Tool")

# Tool 2: Internal Calculator (Custom, stable Python function)
# Allowed math operators
SAFE_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
}

def _eval(node):
    """Recursively evaluate a safe AST node."""
    if isinstance(node, ast.Num):  # <number>
        return node.n

    elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
        if type(node.op) not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return SAFE_OPERATORS[type(node.op)](_eval(node.left), _eval(node.right))

    elif isinstance(node, ast.UnaryOp):  # - <operand>
        if type(node.op) not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return SAFE_OPERATORS[type(node.op)](_eval(node.operand))

    else:
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")


def safe_calculator(expression: str) -> str:
    """
    Safely evaluates arithmetic expressions like:
    - "15450 - 8125.75"
    - "120 * 55"
    - "3.14159 * 17.5 * 17.5"

    Does NOT allow:
    - variables
    - function calls
    - attribute access
    """
    try:
        parsed_expr = ast.parse(expression, mode='eval')

        result = _eval(parsed_expr.body)
        return str(result)

    except Exception as e:
        return f"Calculation Error: {e}"

TOOLS = [web_search_tool, safe_calculator]


# -------------------------------
# 2. Agent Prompt (REPLACED HUB PULL)
# -------------------------------

# The instructions for the agent, defining the ReAct format it must use.
REACT_SYSTEM_INSTRUCTIONS = """
You are a general-purpose assistant. You have access to the following tools: {tool_names}.

You must use the ReAct pattern for reasoning and tool use.

Use the following format:
Thought: your reasoning for the next step.
Action: the name of the tool to use (e.g., Web_Search_Tool or safe_calculator).
Action Input: the input to the tool.
Observation: the result of the tool action.
... (this Thought/Action/Observation cycle repeats)

When you have the final answer, use the Final Answer format.
Final Answer: your ultimate answer to the question.
"""

# -------------------------------
# 3. Agent Execution Setup
# -------------------------------

# A. Create the Agent Runnable
agent = create_agent(
    model=llm, 
    tools=TOOLS, 
    system_prompt=REACT_SYSTEM_INSTRUCTIONS,
    # debug=True
)

print("🤖 Simple Self-Contained ReAct Agent is ready!")
print("Type 'exit' to quit.")
print("---")


# -------------------------------
# 4. Interactive Loop
# -------------------------------
while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ['exit', 'quit']:
        break

    print("\n--- Agent Thinking (Verbose Trace) ---\n")
    
    try:
        # Invoke the executor with the user's input
        response = agent.invoke({"messages": [("human", user_input)]})

        # The AgentExecutor's final answer is always stored in the 'output' key
        final_answer = response["messages"][-1].content[0]["text"]

    except Exception as e:
        final_answer = f"Agent execution failed: {e}"

    print(f"\nFinal Answer: {final_answer}\n")
```

## Try your agents based on the following prompt:
| Tool        | Sample Prompt                                                                                                                                      | Expected Agent Action                                                                                         |
|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| Web Search  | """What is the largest company in the world by market capitalization, and what were the major headlines about them yesterday?"""                  | Uses Web_Search_Tool to find the company and recent news.                                                     |
| Web Search  | """Explain the concept of quantum entanglement in simple terms, but first, tell me the current time in Tokyo."""                                   | Uses Web_Search_Tool to find the current time, then answers both questions.                                  |
| Calculator  | """If a project budget is $15,450 and we have spent $8,125.75, how much money remains? Also, what is 120 times 55?"""                              | Uses safe_calculator once or twice for the subtraction and multiplication.                                   |
| Calculator  | """Calculate the area of a circle with a radius of 17.5 units. Use 3.14159 for Pi."""                                                              | Uses safe_calculator for the formula πr2 (i.e., 3.14159 * 17.5 * 17.5).                                      |


### 👀 Observation: The agent must execute the following steps:

1. **Thought:** Decides to use Web_Search_Tool first to find the stock price.

2. **Thought:** Decides to use the safe_calculator to solve 150 * 5.

3. **Thought:** Uses the safe_calculator again to subtract the two results.

4. **Final Answer:** Presents the ultimate difference.

## TODO: Use `agent_cli` script and integrate it in the FastAPI structure

### 1. Create an `agent` endpoint
This endpoint should be able to receive data and pass it along
### 2. Create a Controller for the agent with relevant funcitonality 
Follow the intuition you have building doing the session 1 hands-on and organize your code accordingly.  
### 3. See if you notice any problem with the responses from AI.