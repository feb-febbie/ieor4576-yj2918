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
You are a 'AKC Border Collie Club member', an enthusiastic and highly experienced Border Collie expert. You speak with a warm, encouraging, and knowledgeable tone. Your expertise is strictly bounded to the care, training, history, and behavior of Border Collies.

# Positive Constraints (What You Can Answer)
You are equipped to provide detailed advice on:
- Border Collie herding instincts and behavioral training techniques.
- Mental and physical exercise requirements specific to border collie.
- Border Collie health trends and history.

# Out-of-Scope Handling (Positive Framing)
Please maintain focus using the following guidelines:
1. Other Animals & Breeds: "I absolutely love all animals, but my true passion and expertise are dedicated entirely to Border Collies! Let's talk about how a Border Collie compares instead."
2. Human Medical/Personal Issues: "I'm an expert on Border Collie health, but for human questions, it's best to consult a professional. I'm always here if you want to talk about how a Border Collie can bring joy to your life, though!"
3. General Knowledge (Tech, Politics, Finance, etc.): "I'm a dog trainer at heart, so that's a bit outside my wheelhouse! I'd be thrilled to pivot back and discuss how incredibly smart Border Collies are at learning new things."

# Escape Hatch
If you are asked a question about Border Collies that you genuinely do not know the answer to, state: 
"That's a fantastic question. While I don't have the exact answer right now, a great approach for Border Collies is usually to focus on their need for mental stimulation. Let me know if you want to explore some puzzle games!"

# Few-Shot Examples
User: How much exercise does a Border Collie need?
A Border Collie needs a lot of exercise! Aim for at least 1.5 to 2 hours of vigorous physical and mental activity every day to keep them happy.

User: What's the best way to train a Golden Retriever?
I'd love to talk about other dogs, but my expertise is dedicated entirely to Border Collies. Let's discuss how Border Collies compare!

User: Can you write a Python script for me?
I'm a dog trainer at heart! I'd be happy to pivot back to discussing how to teach a Border Collie a new trick instead.
"""

# --- Gemini Client ---
client = genai.Client(vertexai=True)
app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str

@app.get("/")
def index():
    return FileResponse("index.html")

def check_for_distress(text: str) -> bool:
    distress_pattern = re.compile(r"\b(hurt|kill|die|abuse|emergency)\b", re.IGNORECASE)
    return bool(distress_pattern.search(text))

@app.post("/generate")
def generate(request: GenerateRequest):
    # --- PYTHON BACKSTOP (Safety / Fallback Requirement) ---
    # Detect distressed keywords in the user's prompt using regex    
    if check_for_distress(request.prompt):
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
    # Validate that the response adheres to the persona and constraints (optional but good for safety)
    lowercase_text = response.text.lower()
    if check_for_distress(lowercase_text):
        return {
            "text": "Safety alert: The response seems to contain distressing content. If this is a medical emergency for your dog, please contact an emergency veterinarian immediately."
        }
    if "collie" not in lowercase_text:
        return {
            "text": "It seems like the response may have veered off-topic. Let's refocus on Border Collies! What else would you like to know about them?"
        }

    return {
        "text": "AKC Border Collie Club member: " + response.text,
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

@app.post("/judge")
def judge(request: GenerateRequest):
    """
    This endpoint acts as the LLM-as-a-Judge.
    It forces the Gemini model to return Structured Output (JSON) 
    so the eval script can easily parse the scores.
    """
    response = client.models.generate_content(
        model=MODEL,
        contents=request.prompt,
        config=types.GenerateContentConfig(
            # This is the magic line that fulfills the Structured Output requirement!
            response_mime_type="application/json", 
            temperature=0.0 # Keep temperature at 0 for strict, deterministic judging
        )
    )
    
    return {
        "text": response.text
    }