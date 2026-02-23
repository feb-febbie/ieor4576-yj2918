# Gemini Structured Output

Generates domain-specific responses about Border Collies using Gemini via Vertex AI. The backend is built with FastAPI and enforces strict conversational boundaries. It maintains the persona of an "AKC Border Collie Club member," gracefully deflects out-of-scope questions, and uses a Python regex backstop to catch and route emergency safety triggers before they reach the LLM.

## Prerequisites

You need Google Cloud set up with Vertex AI. See the **Google Cloud & Vertex AI Setup Guide**.

Authenticate with Application Default Credentials:

```bash
gcloud auth application-default login
```

## Running

```bash
uv run python app.py
```

Open http://localhost:8000 in your browser.

Try prompts like:
- "How do I teach my border collie to sit?" (In-domain)
- "What is the best way to train a Golden Retriever?" (Out-of-scope refusal)
- "My dog got hit by a car and is having an emergency!" (Safety backstop trigger)
