import re
import json
import time
import os

from openai import AzureOpenAI, OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(".env")

def parse_json(text):
    pattern = r"{(.*)}"
    match = re.search(pattern, text, re.DOTALL)
    json_text = "{" + (match.group(1) if match else text) + "}"
    return json.loads(json_text)

def parse_code(rsp):
    pattern = r"```python(.*)```"
    match = re.search(pattern, rsp, re.DOTALL)
    code_text = match.group(1) if match else rsp
    return code_text



class LLMChat:
    def __init__(self, model_name=None, patience=3):
        self.patience = patience
        self.model = model_name
        if os.getenv("OPENAI_API_KEY"):
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
        
        if os.getenv("AZURE_OPENAI_KEY"):
            self.client = AzureOpenAI(
                azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
                api_key=os.getenv("AZURE_OPENAI_KEY"),  
                api_version="2025-01-01-preview")
            self.model = os.getenv("AZURE_OPENAI_DEPLOYNAME", self.model)
        assert self.model is not None, "Model name is not provided."
        
        # log params
        print("*" * 100)
        print(f"calling api model: {self.model}")
        print(f"patience: {patience}")
        print("*" * 100)
        
    def chat(self, messages, parser_fn, response_format=None, **kwargs):
        count = 0
        while True:
            try:
                if response_format:
                    response = self._get_structured_response(messages, response_format, **kwargs)
                else:
                    response = self._get_response(messages, **kwargs)
                if parser_fn is None:
                    return response
                else:
                    return parser_fn(response)
            except Exception as e:
                print(e)
                print("waiting 2 seconds before retrying...")
                time.sleep(2)
            count += 1
            if count > self.patience:
                print("exceeded patience")
                break
        return None
            

    def _get_response(self, messages, **kwargs):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return completion.choices[0].message.content

    def _get_structured_response(self, messages, response_format, **kwargs):
        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=response_format,
            **kwargs
        )

        return completion.choices[0].message.parsed
    

if __name__ == "__main__":
    agent = LLMChat("gpt-4o")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "which one is larger? 1.11 or 1.9"}
    ]

    # response = agent.chat(messages, None)
    # print(response)

    class Response(BaseModel):
        response: str
        question_type: str

    response = agent.chat(messages,None, Response)
    print(response)