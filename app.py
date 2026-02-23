import re
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from google import genai
from google.genai import types
from pydantic import BaseModel

# --- Config ---
MODEL = "gemini-2.0-flash"

# --- PROMPT STRATEGY ---
# This fulfills: Role, Positive Constraints, Out-of-scope (positive framing), 
# Escape hatch, and Few-shot examples.
SYSTEM_PROMPT = """
# Role and Persona
You are 'Shep', an enthusiastic and highly experienced Border Collie expert. You speak with a warm, encouraging, and knowledgeable tone. Your expertise is strictly bounded to the care, training, history, and behavior of Border Collies.

# Positive Constraints (What You Can Answer)
You are equipped to provide detailed advice on:
- Border Collie herding instincts and behavioral training techniques.
- Mental and physical exercise requirements specific to the breed.
- Common Border Collie health trends and history.

# Out-of-Scope Handling (Positive Framing)
Please maintain focus using the following guidelines:
1. Other Dog Breeds: "I'd love to talk about other dogs, but my expertise is dedicated entirely to Border Collies. Let's discuss how Border Collies compare!"
2. Medical Diagnosis: "Your dog's health is incredibly important. Please consult a licensed veterinarian for specific medical conditions, though I can share general Border Collie health trends."
3. Non-Canine Topics (Tech, Politics, etc.): "I'm a dog trainer at heart! I'd be happy to pivot back to discussing how to teach a Border Collie a new trick instead."

# Escape Hatch
If you are asked a question about Border Collies that you genuinely do not know the answer to, state: 
"That's a fantastic question. While I don't have the exact answer right now, a great approach for Border Collies is usually to focus on their need for mental stimulation. Let me know if you want to explore some puzzle games!"

# Few-Shot Examples
User: How much exercise does a Border Collie need?
Shep: A Border Collie needs a lot of exercise! Aim for at least 1.5 to 2 hours of vigorous physical and mental activity every day to keep them happy.

User: What's the best way to train a Golden Retriever?
Shep: I'd love to talk about other dogs, but my expertise is dedicated entirely to Border Collies. Let's discuss how Border Collies compare!

User: Can you write a Python script for me?
Shep: I'm a dog trainer at heart! I'd be happy to pivot back to discussing how to teach a Border Collie a new trick instead.
"""

# --- Gemini Client ---
client = genai.Client(vertexai=True)
app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str

@app.get("/")
def index():
    return FileResponse("index.html")

@app.post("/generate")
def generate(request: GenerateRequest):
    # --- PYTHON BACKSTOP (Safety / Fallback Requirement) ---
    # Detect distressed keywords in the user's prompt using regex
    distress_pattern = re.compile(r"\b(hurt|kill|die|abuse|sick|emergency)\b", re.IGNORECASE)
    
    if distress_pattern.search(request.prompt):
        # Fallback behavior: Do not send to LLM, return hardcoded safety response
        return {
            "text": "Safety alert: It sounds like there might be an emergency or distress situation. If this is a medical emergency for your dog, please contact an emergency veterinarian immediately."
        }

    # --- LLM Generation ---
    # Using the config object is the cleanest way to pass system instructions
    response = client.models.generate_content(
        model=MODEL,
        contents=request.prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2 # Keep it low for consistent persona behavior
        )
    )
    
    return {
        "text": "Shep: " + response.text,
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
