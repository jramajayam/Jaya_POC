import os
import requests
from dotenv import load_dotenv
from config import MODEL, REASONING_EFFORT

load_dotenv()

# ── API key from environment variable (NEVER hardcode) ──────────────────────
_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
if not _api_key:
    raise EnvironmentError(
        "AZURE_OPENAI_API_KEY not set. "
        "Copy .env.example → .env and fill in your key."
    )

_endpoint = os.environ.get(
    "AZURE_OPENAI_ENDPOINT", "https://openai-us-east2.openai.azure.com/"
)
_api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")


class OpenAI:
    def __init__(self):
        pass

    def _make_request(self, url, headers, data):
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            raise ValueError(f"Request failed: {response.status_code}, {response.text}")

    def get_openai_chat(self, 
                        messages: list, 
                        timeout: int=180, 
                        max_retries: int=5,
                        dep_id: str = MODEL,
                        effort: str = REASONING_EFFORT
                        ) -> tuple[str, str, str]:
        endpoint = _endpoint
        deployment_id = dep_id
        api_version = _api_version
        headers = {
            "Content-Type": "application/json",
            "api-key": _api_key
        }
        data = {"messages": messages, "reasoning_effort": effort}
        url = f"{endpoint}/openai/deployments/{deployment_id}/chat/completions?api-version={api_version}"

        for attempt in range(max_retries):
                print(f"\nAttempt {attempt + 1}/{max_retries}...")
                try:
                    resp = requests.post(url, headers=headers, json=data, timeout=timeout)
                    if resp.status_code != 200:
                        print(f"Request failed: {resp.status_code} {resp.text}")
                        return None, None, None
                    result = resp.json()
                    content = result['choices'][0]['message']['content']
                    usage = result.get('usage', {})
                    return content, result, usage
                except requests.Timeout:
                    print("Request timed out. Retrying...")
                    if attempt == max_retries - 1:
                        print("Max retries reached. Request failed.")
                        return None, None, None
                except requests.RequestException as e:
                    print(f"Request error: {e}")
                    return None, None, None


if __name__ == "__main__":
    # Example usage
    openai = OpenAI()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]

    content, _, _ = openai.get_openai_chat(messages)
    if content:
        print("Response:", content)

    else:
        print("Failed to get response.")