import os
import operator as op
import ast 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent
from langchain.tools import tool

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

        self.__gemini = ChatGoogleGenerativeAI(model=self.__model, temperature=self.__temperature)

        self.__web_search_tool = DuckDuckGoSearchRun(name="Web_Search_Tool")
        self.__calculator_tool = safe_calculator

        self.__REACT_SYSTEM_INSTRUCTIONS = """
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

    async def ask_agent(self, prompt):

        TOOLS = [self.__web_search_tool, self.__calculator_tool]

        agent = create_agent(
            model=self.__gemini, 
            tools=TOOLS,
            system_prompt=self.__REACT_SYSTEM_INSTRUCTIONS
        )

        response = agent.ainvoke({"messages": [("human", prompt)]})

        return response["messages"][-1].content[0]["text"]

