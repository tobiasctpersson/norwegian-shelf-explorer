import anthropic
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()


def _get_secret(name):
    if name in st.secrets:
        return st.secrets[name]
    return os.getenv(name)


def _get_client():
    api_key = _get_secret("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)

def ask_claude(question, df):
    client = _get_client()
    if client is None:
        return (
            "Anthropic is not configured for this deployment yet. "
            "Add `ANTHROPIC_API_KEY` to your local `.env` file or Streamlit secrets."
        )

    summary = df.groupby('Drilling operator')['Status'].value_counts().head(30).to_string()
    content_counts = df['Content'].value_counts().to_string()
    status_counts = df['Status'].value_counts().to_string()
    area_counts = df['Main area'].value_counts().to_string()

    context = f"""
You are an oil and gas data analyst. You have access to data on {len(df)} exploration wells 
on the Norwegian Continental Shelf (NCS).

Here is a summary of the data:

STATUS BREAKDOWN:
{status_counts}

CONTENT BREAKDOWN (what was found):
{content_counts}

TOP OPERATORS BY ACTIVITY:
{summary}

MAIN AREAS:
{area_counts}

Answer the following question based on this data. Be concise and insightful.
If relevant, mention what the numbers mean for companies trying to access historical well data.
"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[
            {"role": "user", "content": f"{context}\n\nQuestion: {question}"}
        ]
    )

    return message.content[0].text
