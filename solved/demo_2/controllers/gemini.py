import os
from fastapi import FastAPI
from pydantic import BaseModel

from app.schemas import InvestmentProfile

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage # Used in structured output

class Gemini:

    def __init__(self, model="gemini-2.5-flash", temperature=0.7):
        self.__model = model
        self.__temperature = temperature

        self.__llm = ChatGoogleGenerativeAI(model=self.__model, temperature=self.__temperature)

    def ask_gemini(self, prompt):

        try:
            # LangChain makes the call: prompt -> model -> response
            response = self.__llm.invoke(prompt)
            
            # We only want the content from the response object
            return {"ai_response": response.content}
        
        except Exception as e:
            # Return a simple error message if the API key is invalid or another issue occurs
            return {"error": str(e)}
        
    async def extract_profile(self, prompt):

        # Define the Parser and get formatting instructions
        parser = JsonOutputParser(pydantic_object=InvestmentProfile)
        format_instructions = parser.get_format_instructions()

        # create the Chain Prompt
        prompt = PromptTemplate(
                    template="Analyze the following company sustainability data and extract the required profile fields. STRICTLY follow the format instructions.\n{format_instructions}\n\nCompany Data:\n{data}",
                    input_variables=["data"],
                    partial_variables={"format_instructions": format_instructions}
                )

        # LangChain Expression Language (LCEL) chain
        analysis_chain = prompt | self.__llm | parser
        
        try:
            # 3. Invoke the asynchronous chain
            result = await analysis_chain.ainvoke({"data": prompt})
            
            # Pydantic model is automatically converted to JSON by FastAPI
            return result 
            
        except Exception as e:
            return {"error": f"Failed to parse model output: {e}. Check LLM response quality."}