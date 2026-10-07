import os

from openai import OpenAI
from dotenv import load_dotenv

from models import Answer

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def ask(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

# Structured Outputs
def ask_structured(prompt: str) -> Answer:
    completion = client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "Extract the topic, a one-sentence summary, "
                        "and your confidence from 0 to 1."},
            {"role": "user", "content": prompt},
        ],
        response_format=Answer,
    )
    message = completion.choices[0].message
    if message.refusal:
        raise ValueError(f"Model refused: {message.refusal}")
    return message.parsed
