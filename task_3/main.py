import os
import time
from dotenv import load_dotenv
from openai import AzureOpenAI

def load_env_variables():
    """
    Load environment variables from the .env file and return API credentials.

    Returns:
        tuple: (api_key, endpoint, api_version)
    """
    load_dotenv()
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    if not api_key or not endpoint or not api_version:
        print("Required environment variables are missing!")
        exit(1)

    print(f"Endpoint: {endpoint}")
    print(f"API Version: {api_version}")

    return api_key, endpoint, api_version

def create_openai_client(api_key: str, endpoint: str, api_version: str) -> AzureOpenAI:
    """
    Create an AzureOpenAI client.

    Args:
        api_key (str): Azure OpenAI API key.
        endpoint (str): Azure OpenAI endpoint.
        api_version (str): Azure OpenAI API version.

    Returns:
        AzureOpenAI: Configured client instance.
    """
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )
    return client

PROMPTS = [
    "What are the pros and cons of nuclear energy?",
    "Explain entropy like I'm five years old.",
    "Explain entropy to me as if I were a five-year-old.",
]

def run_prompt(client: AzureOpenAI, prompt: str, deployment: str):
    """
    Send a prompt to the Azure OpenAI model and log the response.

    Args:
        client (AzureOpenAI): The Azure OpenAI client.
        prompt (str): The prompt to send.
        deployment (str): The model deployment name.
    """
    start = time.time()
    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False
        )

        response_dict = response.model_dump()
        usage = response_dict["usage"]
        total_tokens = usage["total_tokens"]
        prompt_tokens = usage["prompt_tokens"]
        completion_tokens = usage["completion_tokens"]
        elapsed = time.time() - start

        cost_per_1k_tokens = 0.005
        cost = total_tokens / 1000 * cost_per_1k_tokens

        log_entry = (
            f"## Prompt\n{prompt}\n\n"
            f"**Prompt Tokens**: {prompt_tokens}\n"
            f"**Completion Tokens**: {completion_tokens}\n"
            f"**Total Tokens**: {total_tokens}\n"
            f"**Estimated Cost**: ${cost:.4f}\n"
            f"**Time**: {elapsed:.2f}s\n\n"
            f"### Response\n{response_dict['choices'][0]['message']['content']}\n\n"
            "---\n\n"
        )

        os.makedirs("logs", exist_ok=True)
        with open("logs/usage.md", "a", encoding="utf-8") as f:
            f.write(log_entry)

        print(f"Prompt done (Tokens: {total_tokens}, Cost: ${cost:.4f})")

    except Exception as exc:
        print(f"Error: {exc}")

def initialize_log_file():
    """
    Initialize the log file for prompt usage.
    """
    os.makedirs("logs", exist_ok=True)
    with open("logs/usage.md", "w", encoding="utf-8") as f:
        f.write("# GPT-4o Azure Prompt Usage Log\n\n")

def main():
    """
    Main function to process prompts using Azure OpenAI and log the results.
    """
    deployment = "gpt-4o"
    api_key, endpoint, api_version = load_env_variables()
    client = create_openai_client(api_key, endpoint, api_version)
    initialize_log_file()

    for prompt in PROMPTS:
        run_prompt(client, prompt, deployment)

    print("All prompts processed.")

if __name__ == "__main__":
    main()
