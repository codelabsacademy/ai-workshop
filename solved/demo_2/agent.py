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