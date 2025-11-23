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
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun
from langchain import hub 

# 1. Security Check
if "GOOGLE_API_KEY" not in os.environ:
    print("Error: GOOGLE_API_KEY not set. Please set it and retry.")
    exit()

# 2. Set up the LLM ("The Brain")
# Use Gemini 1.5 Flash with temperature=0 for consistent planning.
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# 3. Set up the Tools ("The Hands")
tools = [
    DuckDuckGoSearchRun()
]

# 4. Get the Agent Instructions (The Prompt)
# We pull the standard "ReAct" prompt from the LangChain Hub.
prompt = hub.pull("hwchase17/react")

# 5. Create the Agent
agent = create_react_agent(llm, tools, prompt)

# 6. Create the Runtime Executor
# verbose=True is CRITICAL to see the Thought -> Action -> Observation loop.
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print("🤖 Agent is ready! Ask a question about current events.")
print("Type 'exit' to quit.")

# 7. Interactive Loop
while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ['exit', 'quit']:
        break

    print("\n--- Agent Thinking ---\n")
    
    # The invoke command triggers the ReAct loop
    response = agent_executor.invoke({
        "input": user_input
    })

    print(f"\nFinal Answer: {response['output']}\n")
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