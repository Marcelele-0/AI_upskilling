import os
from openai import AzureOpenAI
from dotenv import load_dotenv
import logging
from pathlib import Path

# Load env variables and set logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

def generate_invest_user_stories(custom_prompt: str = "") -> str:
    """
    Generate INVEST-style user stories using GPT-4o.
    
    Args:
        custom_prompt (str): Optional custom instruction for GPT.

    Returns:
        str: The user stories as plain text.
    """
    default_prompt = (
        "Write 3 user stories using the INVEST model. "
        "Each should follow: 'As a [user], I want to [action], so that [value]'. "
        "Add 4-5 acceptance criteria per story as Markdown bullet points. "
        "Do not include anything other than the stories."
    )

    prompt = custom_prompt or default_prompt

    logging.info("Sending prompt to GPT-4o...")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1500
    )

    content = response.choices[0].message.content.strip()
    logging.info("User stories generated successfully.")
    return content


def save_stories_to_file(stories: str, filename: str):
    """
    Save generated user stories to a file.

    Args:
        stories (str): The user stories to save.
        filename (str): The name of the file to save to.
    """
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        f.write(stories)
    logging.info(f"User stories saved to {file_path}")


def main():
    print("Generating INVEST user stories with GPT-4o...\n")
    stories = generate_invest_user_stories()
    print(stories)

    save_stories_to_file(stories, str(Path("backlog/sprint1.md")))


if __name__ == "__main__":
    main()
