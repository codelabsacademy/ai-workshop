import os
from langfuse import get_client
from langfuse.langchain import CallbackHandler

os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-b36c8bc7-7cdc-41cd-a5f3-3705ddd5f772"
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-8aa23815-86a5-4033-853d-3d24c3cb8258"
os.environ["LANGFUSE_BASE_URL"] = "https://cloud.langfuse.com"

langfuse = get_client()
lf_handler = CallbackHandler()