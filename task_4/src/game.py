import os
import json
from openai import AzureOpenAI
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

def get_quiz_questions() -> list:
    """
    Fetch quiz questions from the GPT model.

    Returns:
        list: A list of quiz questions.
    """

    with open("prompts/best.txt", encoding="utf-8") as f:
        prompt = f.read()
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1000
    )

    json_text = response.choices[0].message.content.strip()

    try:
        logging.debug("✅ Successfully parsed GPT response as JSON.")
        return json.loads(json_text)
    except json.JSONDecodeError:
        logging.debug("❌ Failed to parse GPT response as JSON.")
        logging.debug(json_text)
        return []

def run_quiz(questions: list[dict]) -> None:
    """
    Run the quiz game with the provided questions.
    """

    score = 0
    
    for i, q in enumerate(questions, 1):
        logging.debug(f"Question {i}: {q['question']}")
        for key, val in q["options"].items():
            print(f"  {key}: {val}")
        answer = input("Your answer (A/B/C/D): ").strip().upper()
        if answer == q["answer"].upper():
            print("✅ Correct!")
            score += 1
        else:
            correct = q["options"][q["answer"]]
            print(f"❌ Wrong. Correct answer: {q['answer']} - {correct}")

    print(f"\n🎯 Final score: {score}/{len(questions)}")


def main():
    print("🎮 Welcome to the Quiz Game!")
    questions = get_quiz_questions()
    if questions:
        run_quiz(questions)
    else:
        print("⚠️ No questions available.")

if __name__ == "__main__":
    main()
