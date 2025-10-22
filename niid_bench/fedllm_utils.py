
import requests
import json
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from typing import List, Dict, Union

def invoke_llm(
    messages: List[Dict[str, str]],  # list of dicts with keys: role (system/user/assistant), content (str)
    model_name: str = "llama3.2:3b",
    temperature: float = 0.1
) -> Union[str, None]:
    """
    Invoke the local Ollama LLM via LangChain interface with given messages.

    Args:
        messages: List of dicts representing chat messages. Example:
            [
                {"role": "system", "content": "You are an assistant."},
                {"role": "user", "content": "Explain FL in one sentence."}
            ]
        model_name: Ollama model to use, default "llama3.1".
        temperature: Generation temperature, default 0.3.

    Returns:
        The content (text) output from the LLM, or None if error.
    """
    try:
        # Convert dict messages to LangChain message classes
        langchain_msgs = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                langchain_msgs.append(SystemMessage(content=content))
            elif role == "user":
                langchain_msgs.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_msgs.append(AIMessage(content=content))
            else:
                # Unknown role, treat as user message (optional)
                langchain_msgs.append(HumanMessage(content=content))

        llm = ChatOllama(model=model_name, temperature=temperature)
        response = llm.invoke(langchain_msgs)
        return response.content

    except Exception as e:
        print(f"Error invoking LLM: {e}")
        return None

def invoke_jamba(messages, api_key):
    url = "https://tfrq6se0fg.execute-api.us-east-1.amazonaws.com/invoke_model"

    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }

    payload = {
        "messages": messages,
        "max_tokens": 200,
        "temperature": 0.7,
        "top_p": 0.8
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    print(response.json())
    if response.status_code == 200:
        return response.json()
    else:
        return {
            "error": "Request failed",
            "status_code": response.status_code,
            "response": response.text
        }
    

## PROMPTS 

initial_message = [
    {
        "role": "system",
        "content": (
            "You are an expert federated learning assistant. Consider all the knowledge about client selection for federated learning existent on the scientific literature"
            "Your task is to help with client selection at each training round. "
            "At each round, you wi ll receive performance data from all clients. "
            "Your goal is to minimize the number of updates (client participation) "
            "while improving the global model’s centralized accuracy. "
            "Based on the data, return the number of clients to select, and the client IDs of selected clients."
            
        )
    },
    
]

output_format_instructions = (
    "Select the appropriate number of clients to participate in the next round. "
    "Return only a valid JSON object in the following format containing the number of clients selected, the client ids of the selected clients, "
    "and a brief explanation of why those clients were selected:\n"
    "{\n"
    "  \"num_clients\": X,\n"
    "  \"selected_clients\": [\"id\", \"id2\"],\n"
    "  \"explanation\": \"Your explanation here.\"\n"
    "}\n"
    "You are free to select any clients in any round, regardless of their participation history. "
    "Avoid selecting the same clients every round unless strictly necessary. Consider exploring other clients when possible to help improve the global model more efficiently over time. "
    "Only output the JSON object in this exact format, and do not include any extra text outside the JSON."
)