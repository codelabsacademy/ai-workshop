from langchain_google_genai import ChatGoogleGenerativeAI

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
