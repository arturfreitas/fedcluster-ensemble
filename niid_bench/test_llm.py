#from strategy_fedllm import invoke_jamba

import requests
import json 

def invoke_jamba(question, api_key):
    url = "https://tfrq6se0fg.execute-api.us-east-1.amazonaws.com/invoke_model"

    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }

    # Wrap question in a single-message format
    messages = [
    {
        "role": "system",
        "content": (
            "You are an expert federated learning assistant. "
            "Your task is to help with client selection at each training round. "
            "At each round, you will receive performance data from all clients. "
            "Your goal is to minimize the number of updates (client participation) "
            "while improving the global model’s centralized accuracy. "
            "Based on the data, return the number of clients to select, and the client IDs."
            
        )
    },
    {
        "role": "user",
        "content": (
            "Round 3 data:\n"
            "- 1: similarity=0.91, loss=0.22\n"
            "- 2: similarity=0.84, loss=0.30\n"
            "- 3: similarity=0.78, loss=0.25\n"
            "- 4: similarity=0.94, loss=0.18\n"
            "- 5: similarity=0.65, loss=0.40\n\n"
            "Select the appropriate number of clients to participate in the next round. "
            "Return only a valid JSON object in the format:\n"
            "Return a JSON response like:\n"
            "{\"num_clients\": X, \"selected_clients\": [\"client_id_1\", \"client_id_2\"]}\n"
            "Do not include any explanation or commentary."
        )
    }
    ]


    payload = {
        "messages": messages,
        "max_tokens": 200,
        "temperature": 0.7,
        "top_p": 0.8
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))

        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": "Request failed",
                "status_code": response.status_code,
                "response": response.text
            }

    except requests.exceptions.RequestException as e:
        return {
            "error": "Request failed",
            "exception": str(e)
        }

# 👇 Fixed usage
api_key = "13qd759fs4"
question = "help me to do client selection in federated learning, I will provide client data and you will return the ids."

result = invoke_jamba(question, api_key)
print(json.dumps(result, indent=2))
