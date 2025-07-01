import os
from dotenv import load_dotenv
from langchain_ibm import WatsonxLLM

load_dotenv()

def get_llm():
    return WatsonxLLM(
        model_id=os.getenv("WATSONX_MODEL_ID"),
        url=os.getenv("WATSONX_URL"),
        apikey=os.getenv("WATSONX_APIKEY"),
        project_id=os.getenv("WATSONX_PROJECT_ID"),
        params={
            "decoding_method": "sample",
            "max_new_tokens": 312
        }
    )
