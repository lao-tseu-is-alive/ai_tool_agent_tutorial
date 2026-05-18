import requests
import json
import os

from dotenv import load_dotenv


load_dotenv()
openrouter_api_key = os.getenv("openrouter_api_key") or os.getenv("OPENROUTER_API_KEY")

if not openrouter_api_key:
    raise RuntimeError(
        "❌ Missing openrouter_api_key in .env. Add openrouter_api_key=<your key>."
    )

headers = {
    "Authorization": f"Bearer {openrouter_api_key}",
    "Content-Type": "application/json",
}


def call_openrouter(payload, call_name):
    print(f"\n✅ {call_name}: contacting OpenRouter...")

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=60,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        print(f"❌  {call_name}: OpenRouter connection failed.")
        print(f"    Error: {exc}")
        raise

    print(f"✅ {call_name}: connection is working. HTTP {response.status_code}")

    try:
        return response.json()
    except requests.exceptions.JSONDecodeError as exc:
        print(f"❌  {call_name}: OpenRouter returned invalid JSON.")
        print(f"    Error: {exc}")
        raise


def extract_message(response_data, call_name):
    try:
        return response_data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        print(f"❌  {call_name}: could not read the assistant message.")
        print(f"    Raw response: {response_data}")
        raise RuntimeError("Unexpected OpenRouter response format.") from exc


def print_call_result(call_name, message):
    content = message.get("content") or "<no content returned>"
    reasoning_details = message.get("reasoning_details")

    print(f"✅ {call_name}: assistant response:")
    print(content)

    if reasoning_details:
        print(f"✅ {call_name}: reasoning details were returned.")
    else:
        print(f"⁉️ {call_name}: no reasoning details were returned.")


print("✅ OpenRouter API key loaded from .env.")

first_payload = {
    "model": "deepseek/deepseek-v4-flash",
    "messages": [
        {
            "role": "user",
            "content": "How many r's are in the word 'strawberry'?"
        }
    ],
    "reasoning": {"enabled": True}
}

# First API call with reasoning
first_response_data = call_openrouter(first_payload, "First call")
first_message = extract_message(first_response_data, "First call")
print_call_result("First call", first_message)

# Preserve the assistant message with reasoning_details
messages = [
    {"role": "user", "content": "How many r's are in the word 'strawberry'?"},
    {
        "role": "assistant",
        "content": first_message.get("content"),
        "reasoning_details": first_message.get("reasoning_details")  # Pass back unmodified
    },
    {"role": "user", "content": "Are you sure? Think carefully."}
]

second_payload = {
    "model": "deepseek/deepseek-v4-flash",
    "messages": messages,  # Includes preserved reasoning_details
    "reasoning": {"enabled": True}
}

# Second API call - model continues reasoning from where it left off
second_response_data = call_openrouter(second_payload, "Second call")
second_message = extract_message(second_response_data, "Second call")
print_call_result("Second call", second_message)
