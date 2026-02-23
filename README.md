# Gemini Structured Output

Generates D&D weapons with guaranteed-valid JSON using Gemini's constrained structured output. A Pydantic model defines the weapon schema, and the API enforces it during decoding — no parsing or validation needed on our side.

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
- "a frost-enchanted greatsword wielded by a frost giant"
- "a lightweight elven dagger that glows in moonlight"
- "a heavy crossbow designed for siege warfare"
