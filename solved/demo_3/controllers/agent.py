import os
import operator as op
import ast 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent
from langchain.tools import tool

from app.langfuse import lf_handler, langfuse

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

class AgentController:
    
    def __init__(self, model="gemini-2.5-flash", temperature=0.7):
        self.__model = model
        self.__temperature = temperature

        # -----------------------------
        # Gemini LLM with Langfuse hooks
        # -----------------------------
        self.__gemini = ChatGoogleGenerativeAI(
            model=self.__model,
            temperature=self.__temperature,
            callbacks=[lf_handler]
        )

        self.__web_search_tool = DuckDuckGoSearchRun(name="Web_Search_Tool")
        self.__calculator_tool = safe_calculator
        
        # prompt management
        self.__REACT_SYSTEM_INSTRUCTIONS = langfuse.get_prompt("demo", type="text").prompt
        print("Loaded prompt from Langfuse:")
        print(self.__REACT_SYSTEM_INSTRUCTIONS)

    async def ask_agent(self, prompt):

        TOOLS = [self.__web_search_tool, self.__calculator_tool]

        agent = create_agent(
            model=self.__gemini, 
            tools=TOOLS,
            system_prompt=self.__REACT_SYSTEM_INSTRUCTIONS
        )

        response = await agent.ainvoke({"messages": [("human", prompt)]})

        return response["messages"][-1].content[0]["text"]

