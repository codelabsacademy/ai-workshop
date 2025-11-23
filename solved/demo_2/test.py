import os
import ast 
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
@tool
def safe_calculator(expression: str) -> str:
    """
    Use this tool to calculate simple arithmetic expressions. 
    The input must be a valid Python arithmetic expression string (e.g., '2 * 5 + 10').
    Do not include variables or complex functions.
    """
    safe_chars = '0123456789.+-*/() '
    if not all(c in safe_chars for c in expression):
        return "Error: Expression contains unsafe characters."
    
    try:
        # ast.literal_eval is a safe way to evaluate arithmetic expressions
        result = ast.literal_eval(expression)
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
    system_prompt=REACT_SYSTEM_INSTRUCTIONS
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