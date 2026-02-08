from langchain_ollama import ChatOllama

LLM_NAME = "llama3.1:8b"
BASE_URL = "http://localhost:11434"

llm = ChatOllama(model=LLM_NAME, temperature=0, base_url=BASE_URL)

def baseline_llm(fault):
    prompt = f"""
You are a maintenance technician.
Fault: {fault}
Give the correct maintenance action.
Return only the action.
"""
    return llm.invoke(prompt).content.strip()