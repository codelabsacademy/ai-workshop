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


🌟 Advanced Hands-On: Multi-Tool Autonomous Audit Agent
Use Case: GreenVault Compliance Audit
The Agent's Goal is to perform a full compliance audit:

Look up internal policy (RAG Tool).

Find external financial data (Web Search Tool).

Calculate a composite score (Calculator Tool).

Provide a structured, final recommendation.

1. New Dependencies & RAG Setup
Since this combines previous extensions, ensure you have all the following libraries installed and your policy.txt file is in the root directory.