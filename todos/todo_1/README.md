# 🚀 Gen AI Workshop: Your First AI Microservice

This repository contains the code for the hands-on exercise from Session 1 of the Generative AI Workshop. We will build a simple, scalable microservice to serve an AI endpoint powered by **LangChain** and the **Gemini** model.

---

## 🌟 Learning Goals

* Understand how to wrap an LLM in a dedicated web service (a **Microservice**).
* Use the **LangChain** framework to easily integrate with the **Gemini API**.
* Practice fundamental **Git** version control workflow (`add`, `commit`, `push`).

---

## 🛠️ Prerequisites

Please ensure you have the following installed on your system:

1.  **Python 3.9+**
2.  **Git** (Installed and configured)
3.  **A Code Editor** (e.g., VS Code)

### 🔑 API Key Setup (Crucial!)

You need a **Gemini API Key** to run the model.

1.  Visit **[Google AI Studio](https://aistudio.google.com/)** and generate a new API key.
2.  Set the key as an **environment variable** in your terminal. This keeps your secret key out of your code.

| OS/Shell | Command (Replace `YOUR_API_KEY`) |
| :--- | :--- |
| **Mac/Linux (Bash/Zsh)** | `export GOOGLE_API_KEY='YOUR_API_KEY'` |
| **Windows (CMD)** | `set GOOGLE_API_KEY="YOUR_API_KEY"` |

***Note:** The LangChain Google integration automatically looks for the `GOOGLE_API_KEY` variable.*

---

## ⚙️ Project Setup and Installation

Follow these steps:

### 1. Use the TODO directory and find the "todo-1"
This directory will have the code boiler plate setup, following the conventions of a production grade development project.

### 2. Locate the "requirement.txt" and install the dependencies
This "requirements.txt" file contains all the libraries that we will be using for this

Run the following command to install the requirements.
```python
pip install requirements.txt
```

You may use a virutal env to isolate these dependencies (Optional)

## Code:
Create a file named `main.py` and paste the following code. This service defines one **POST** endpoint *(/ask-ai)* that handles the logic.

```python
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Initialize FastAPI
app = FastAPI(title="Gemini Microservice")

# 2. Initialize the Gemini Model using LangChain
# It automatically reads the GOOGLE_API_KEY from your environment
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7 # Add a little creativity
)

# 3. Define the request structure (Pydantic validation)
class QueryRequest(BaseModel):
    prompt: str

# 4. Define the endpoint
@app.post("/ask-ai")
def ask_ai(request: QueryRequest):
    """
    Sends the user's prompt to Gemini via LangChain and returns the text response.
    """
    try:
        # LangChain makes the call: prompt -> model -> response
        response = llm.invoke(request.prompt)
        
        # We only want the content from the response object
        return {"ai_response": response.content}
        
    except Exception as e:
        # Return a simple error message if the API key is invalid or another issue occurs
        return {"error": str(e)}
```

## Running and Testing the Service

### 1. Run the Server

```bash
# This starts the Uvicorn server and watches main.py for changes
uvicorn main:app --reload
```

Your service is now live! The server will show an address like `http://127.0.0.1:8000

### 2. Test the API

FastAPFastAPI automatically generates interactive documentation (Swagger UI).

1. Open your web browser to: https://www.google.com/search?q=http://127.0.0.1:8000/docs

2. Expand the POST /ask-ai endpoint.

3. Click "Try it out".

4. In the Request body box, enter a JSON prompt:

```json
{
  "prompt": "What is the key difference between a Microservice and a Monolith?"
}
```
5. Click **Execute** and view the **Response body** from the Gemini model.

## Saving Your Work (Git Workflow)

This step ensures your code is version-controlled and ready for team collaboration.

### 1. Stop the server (Press Ctrl + C in the terminal).

### 2. Initialize Git:
```bash
git init
```

### 3. Create .gitignore file: (Prevents large, unnecessary files from being saved) - only if you used a virutal environment
```bash
echo "venv/" > .gitignore
echo "__pycache__/" >> .gitignore
echo ".env" >> .gitignore # If you store your key in a .env file later
```

### 4. Add and Commit: (Create your first permanent snapshot)
```bash
git add .
git commit -m "Initial commit: FastAPI microservice with LangChain and Gemini"
```

### 5. Push to GitHub: (If you have a remote repository set up)
```bash
# Replace the URL with your GitHub repo URL
git remote add origin [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
git branch -M main
git push -u origin main
```

## After successfully running the demo lets organize it
Once you have your implementation running check the solved directory and restructure your code as per the solution.


## Adding Structured Response Capabilities

### The Company
GreenVault is an asset management firm that invests only in companies meeting strict Environmental, Social, and Governance (ESG) criteria.

### The Problem
GreenVault analysts spend hours manually reviewing lengthy corporate sustainability reports, news articles, and compliance filings to answer two critical, recurring questions:

**Compliance Check:** Does a potential investment (a corporate sustainability report) contain the required data points (e.g., carbon emission figures, executive compensation) in a structured, comparable format?

Instead of an analyst manually reading a 100-page report and filling out a form, the FastAPI Microservice will take the document's summary text and automatically generate a structured JSON profile.

### Tasks:
#### 1. Create a pydantic schema named "InvestmentProfile" with the following fields and data type:
        carbon_intensity: float
        board_diversity_score: float
        key_risks: List[str]
        compliance_status: str

#### 2. Define the Parser and get formatting instructions
```python
    parser = JsonOutputParser(pydantic_object=InvestmentProfile)
    format_instructions = parser.get_format_instructions()
```

#### 3. Use the "PrompTemplate" function to creata a template:
```python
prompt = PromptTemplate(
        template="Analyze the following company sustainability data and extract the required profile fields. STRICTLY follow the format instructions.\n{format_instructions}\n\nCompany Data:\n{data}",
        input_variables=["data"],
        partial_variables={"format_instructions": format_instructions}
    )
```

#### 4. Create chain
```python
# LangChain Expression Language (LCEL) chain
analysis_chain = prompt | llm | parser
```

#### 5. Place it all the above snippets at appropriate places as per the flow of the data

#### 6. Try the following prompt to test
```json
{
  "prompt": "Q3 Environmental Summary: Our company emitted 1,250,000 tons of CO2 equivalent (CO2e) this quarter, while revenue was $1.5 billion. This yields a carbon intensity ratio of 0.833 tons per $1,000 of revenue. Furthermore, we face growing uncertainty around new regional PFAS regulations which may impact our manufacturing costs significantly. Social Governance: Currently, three of our nine board members are women, and one is from an underrepresented ethnic group, giving us a diversity ratio of 4 out of 9, or approximately 0.44. Based on these initial figures, and the looming regulatory risk, we must exercise caution."
}
```
