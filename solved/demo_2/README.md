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


## 🌟 Advanced Hands-On: Multi-Tool Autonomous Audit Agent
### Use Case: GreenVault Compliance Audit

1. The Agent's **Goal** is to perform a full compliance audit:

2. Look up **internal policy** (RAG Tool).

3. Find external **financial data** (Web Search Tool).

4. **Calculate** a composite score (Calculator Tool).

5. Provide a structured, final recommendation.

### 1. New Dependencies & RAG Setup
Since this combines previous extensions, ensure you have all the following libraries installed and your policy.txt file is in the root directory.

```bash
# Install everything needed for Agents, RAG, Web Search, and Math
pip install langchain langchain-google-genai langchain-community duckduckgo-search numpy numexpr faiss-cpu
```
*No need to install if libraries already exist*

### 2. Pydantic Schema for Final Output
We will resuse the structured output technique to define the final audit report format.
```python
# Defined within the complex_agent_service.py file
from pydantic import BaseModel, Field
from typing import List

class AuditDecision(BaseModel):
    """The structured output format for the final compliance decision."""
    compliance_status: str = Field(description="FINAL compliance decision: 'PASS' or 'FAIL'.")
    score_metric: float = Field(description="The calculated composite ESG score metric (0.0 to 1.0).")
    rationale: str = Field(description="The key policy and external data points used to justify the decision.")
    tools_used: List[str] = Field(description="List of tools the agent explicitly used (e.g., Web_Search, Policy_Lookup, Calculator).")
```

### 3. Code:
The following script is for your assistance, the script defines three specialized tools and creates that central agent that orchestrates them.

```python
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun, tool
from langchain_community.tools.llm_math.tool import LLMMathTool
from langchain import hub
from langchain_core.pydantic_v1 import BaseModel as PydanticBaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

# --- 1. TOOL DEFINITIONS ---

# A. Retrieval Tool (RAG - Internal Data)
POLICY_FILE = "policy.txt"
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize RAG retriever (assumes policy.txt exists and is loaded)
try:
    loader = TextLoader(POLICY_FILE)
    docs = CharacterTextSplitter(chunk_size=500, chunk_overlap=0).split_documents(loader.load())
    vectorstore = FAISS.from_documents(docs, embeddings)
    rag_retriever = vectorstore.as_retriever()
except Exception:
    rag_retriever = None

@tool
def policy_lookup(query: str) -> str:
    """
    Use this tool to lookup internal GreenVault policies from the ESG Policy Handbook. 
    Always prioritize this tool for compliance and policy questions.
    """
    if not rag_retriever: return "Error: Policy system offline."
    
    # Create a transient QA Chain for the lookup
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=rag_retriever, chain_type="stuff")
    
    result = qa_chain.invoke({"query": query})
    return result.get("result", "Policy information not found.")

# B. Web Search Tool (External Data)
web_search_tool = DuckDuckGoSearchRun(name="Web_Search_Tool")

# C. Calculation Tool (Analysis)
# This tool uses a safe expression evaluator (numexpr) for complex math
math_tool = LLMMathTool(name="Calculator_Tool") 

# --- 2. AGENT ORCHESTRATION ---

# List of all tools available to the agent
TOOLS = [policy_lookup, web_search_tool, math_tool]

# Initialize LLM for the Agent's reasoning loop
AGENT_LLM = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.2) # Use Pro for better reasoning

# Create the Agent
agent_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(AGENT_LLM, TOOLS, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=True)

# --- 3. FASTAPI API SETUP ---

app = FastAPI(title="Multi-Tool Audit Agent Service")

class AuditRequest(BaseModel):
    company_data: str = Field(description="The unstructured text containing company's ESG data and metrics.")

class AuditDecision(PydanticBaseModel):
    compliance_status: str = Field(description="FINAL compliance decision: 'PASS' or 'FAIL'.")
    score_metric: float = Field(description="The calculated composite ESG score metric (0.0 to 1.0).")
    rationale: str = Field(description="The key policy and external data points used to justify the decision.")
    tools_used: List[str] = Field(description="List of tools the agent explicitly used (e.g., Web_Search, Policy_Lookup, Calculator).")

@app.post("/audit-company")
async def audit_company(request: AuditRequest):
    """
    Invokes the multi-tool agent to perform a complete compliance audit and returns a structured decision.
    """
    
    # Define the final structured output chain
    parser = JsonOutputParser(pydantic_object=AuditDecision)
    format_instructions = parser.get_format_instructions()

    # The prompt instructs the agent what to do and how to structure its final thought process.
    agent_task = f"""
    You are the GreenVault Senior Compliance Analyst. Your task is to audit the following company data.
    
    1. Policy Check: Use the 'policy_lookup' tool to find the required board diversity score and executive compensation requirements.
    2. External Check: Use the 'Web_Search_Tool' to find the company's current stock price or recent financial news.
    3. Calculate: Use the 'Calculator_Tool' if any arithmetic is needed (e.g., calculating a final score).
    4. FINAL OUTPUT: Based on ALL gathered information, provide the final structured decision.
    
    Audit Data: {request.company_data}
    
    After reasoning, output the final structured JSON adhering to the following format instructions:
    {format_instructions}
    
    """
    
    try:
        # 1. Run the Multi-Tool Agent
        agent_result = await agent_executor.ainvoke({"input": agent_task})
        final_text = agent_result["output"]
        
        # 2. Extract structured JSON from the final text output
        # We assume the agent's final answer contains the structured data due to the prompt instructions.
        
        # A simple method to extract and parse the JSON (robustness would require more complex parsing)
        # For the workshop, we rely on the LLM to adhere to the final instruction.
        
        # Since the ReAct loop returns a string, we feed the final result into the parser chain
        
        # We'll use a simple final prompt for parsing the structured output
        final_parsing_prompt = PromptTemplate(
            template="Given the agent's final audit summary below, extract and format the data exactly according to the JSON schema.\n{format_instructions}\n\nAgent Summary: {summary}",
            input_variables=["summary"],
            partial_variables={"format_instructions": format_instructions}
        )
        
        final_parser_chain = final_parsing_prompt | AGENT_LLM | parser

        # Use the final_text from the agent run for parsing
        final_decision = await final_parser_chain.ainvoke({"summary": final_text})
        
        return final_decision
    
    except Exception as e:
        return {"error": f"Audit Agent failed: {e}. Check if policy.txt exists and API key is valid."}
```

### 4. Sample Prompt for Testing
To test the complex multi-tool agent, you would use a detailed prompt that requires all three tools for a full audit:
```json
{
  "company_data": "Company X just released its Q3 report showing a Board Diversity Score of 0.85 and a carbon intensity of 0.25. The CEO's bonus was 15% of annual compensation. Based on this, and today's stock price, assess compliance and calculate the final score using the formula (Board Score + (1 - Carbon Intensity)) / 2."
}
```

The agent will then:

1. **Look up policy:** Policy lookup (Policy Tool).

2. **Look up stock price:** Web Search Tool.

3. **Evaluate compliance:** Based on policy lookup.

4. **Calculate final score:** Calculator Tool.

5. Return final structured JSON.