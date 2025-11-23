import os
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
import ast 
import traceback

# LangChain Core/Community Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

# Agent and LCEL Imports
from langchain.agents import create_agent
# Note: create_retrieval_chain and create_stuff_documents_chain are removed
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough # ADDED: For LCEL pipeline
from operator import itemgetter # ADDED: For LCEL pipeline


# --------------------------
# 1. RAG TOOL (policy_lookup)
# --------------------------

POLICY_FILE = "policy.txt"
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
retriever = None

# RAG setup
try:
    if os.path.exists(POLICY_FILE):
        loader = UnstructuredFileLoader(POLICY_FILE)
        raw_docs = loader.load()
        docs = CharacterTextSplitter(chunk_size=500).split_documents(raw_docs)

        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever()
        print(f"✅ Loaded {len(docs)} chunks from policy file.")
    else:
        print("⚠️ Policy file not found. RAG tool disabled.")

except Exception as e:
    print(f"⚠️ Error loading RAG: {e}")


@tool
def policy_lookup(query: str) -> str:
    """
    Use this tool to lookup internal GreenVault policies. 
    Useful for checking board diversity or compensation rules.
    """
    if retriever is None:
        return "Error: Policy system offline or file missing."

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    # 1. Define the Answer Generation Prompt
    system_prompt = (
        "You are a policy assistant. Answer the user's question based strictly on the context below:\n\n"
        "{context}"
    )
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 2. Define the RAG Chain using LCEL (Alternative to create_retrieval_chain)
    rag_chain = (
        # a. Assign context: Retrieve documents and join them into a single string
        RunnablePassthrough.assign(
            context=(lambda x: x["input"]) | retriever | (lambda docs: "\n\n---\n\n".join(doc.page_content for doc in docs))
        )
        # b. Ensure only 'input' and 'context' are passed to the final prompt template
        | itemgetter("input", "context")
        # c. Pass to the prompt and then the LLM
        | answer_prompt
        | llm
    )

    # 3. Invoke the chain
    result = rag_chain.invoke({"input": query})
    
    # Return the content of the Message object
    return result.content


# --------------------------
# 2. EXTERNAL SEARCH TOOL
# --------------------------

web_search_tool = DuckDuckGoSearchRun(name="Web_Search_Tool")


# --------------------------
# 3. INTERNAL PYTHON CALCULATOR TOOL
# --------------------------

@tool
def safe_calculator(expression: str) -> str:
    """
    Use this tool to calculate simple arithmetic expressions. 
    The input must be a valid Python arithmetic expression string (e.g., '2 * 5 + 10').
    Do not include variables or complex functions.
    """
    safe_chars = '0123456789.+-*/() '
    if not all(c in safe_chars for c in expression):
        return "Error: Expression contains unsafe characters. Only numbers and basic operators are allowed."
    
    try:
        result = ast.literal_eval(expression)
        return str(result)
    except Exception as e:
        return f"Calculation Error: {e}. Please check the expression syntax."


# --------------------------
# 4. AGENT SETUP 
# --------------------------

TOOLS = [policy_lookup, web_search_tool, safe_calculator]

AGENT_LLM = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.2
)

# Inline ReAct Prompt Template
REACT_SYSTEM_INSTRUCTIONS = """
You are a helpful and meticulous Senior Compliance Analyst. You have access to the following tools: {tool_names}.

You must follow the ReAct pattern for reasoning and tool use. Your response must be in JSON format.

Use the following format:
Thought: your reasoning for the next step (e.g., check policy, search web, calculate).
Action: the name of the tool to use (e.g., policy_lookup, Web_Search_Tool, safe_calculator).
Action Input: the input to the tool.

The user input will contain the Audit Data. Analyze the data and decide the compliance status.
"""

react_prompt = ChatPromptTemplate.from_messages([
    ("system", REACT_SYSTEM_INSTRUCTIONS),
    MessagesPlaceholder("agent_scratchpad"),
    ("human", "{input}"),
])

# Using create_agent (assuming your environment uses this simple wrapper for the ReAct Runnable)
agent = create_agent(
    model=AGENT_LLM,
    tools=TOOLS,
    system_prompt=react_prompt
)

# --------------------------
# 5. FASTAPI & PYDANTIC SETUP
# --------------------------

app = FastAPI(title="Multi-Tool Audit Agent Service")


class AuditRequest(BaseModel):
    company_data: str = Field(description="Company ESG data text.")


class AuditDecision(BaseModel):
    compliance_status: str
    score_metric: float
    rationale: str
    tools_used: List[str]


@app.post("/audit-company")
async def audit_company(request: AuditRequest):

    parser = JsonOutputParser(pydantic_object=AuditDecision)
    format_instructions = parser.get_format_instructions()

    agent_task = f"""
    You are a Senior Compliance Analyst.

    1. Use `policy_lookup` to check board diversity & executive compensation rules.
    2. Use `Web_Search_Tool` to get stock price or recent news.
    3. Use `safe_calculator` for math (e.g., '2 * 5 + 10').
    4. After full reasoning, output the final structured JSON.

    Audit Data:
    {request.company_data}

    IMPORTANT: Do not return the JSON yet. Just return the analysis summary in the 'Final Answer'.
    """

    try:
        # Run ReAct agent (the runnable)
        # Note: If you encounter an error here, you might need to wrap the agent 
        # in AgentExecutor and pass verbose=True for full ReAct functionality.
        # For simplicity and sticking to the "chain-free" request, we use the runnable directly.
        result = await agent.ainvoke({"input": agent_task})
        
        # The output is the final message/text from the agent
        # The key name for the output may vary based on the specific 'create_agent' implementation.
        # We rely on the implicit final output being in the message content or "output" key.
        
        # Safely extract the final raw output string
        if isinstance(result, dict) and "output" in result:
             raw_output = result["output"]
        elif hasattr(result, "content"):
             raw_output = result.content
        else:
             raw_output = str(result)
        
        # Final chain to force the output into the required JSON structure
        formatting_prompt = PromptTemplate(
            template=(
                "You are a data extraction engine. Extract the required JSON from the agent's textual output.\n"
                "Follow this output format:\n{format_instructions}\n\n"
                "AGENT OUTPUT:\n{summary}"
            ),
            input_variables=["summary"],
            partial_variables={"format_instructions": format_instructions}
        )

        final_chain = formatting_prompt | AGENT_LLM | parser
        parsed_json = await final_chain.ainvoke({"summary": raw_output})

        return parsed_json

    except Exception as e:
        # Fallback error handling
        traceback.print_exc()
        return {
            "compliance_status": "Error", 
            "score_metric": 0.0, 
            "rationale": f"Audit Agent failed: {str(e)}", 
            "tools_used": []
        }