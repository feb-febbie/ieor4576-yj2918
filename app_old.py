from typing import Literal

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from google import genai
from pydantic import BaseModel, Field

# --- Config ---

MODEL = "gemini-2.0-flash"

SYSTEM_PROMPT = """\
You are a D&D 2024 weapon designer. Given a description, generate a weapon
that fits the D&D 2024 rules. Be creative with the name and description
while keeping the stats balanced and consistent with official weapons."""

# --- Weapon Schema ---

# todo: change this to be facts about border collie
class DnDWeapon(BaseModel):
    name: str = Field(description="The weapon's name")
    category: Literal["Simple", "Martial"]
    weapon_type: Literal["Melee", "Ranged"]
    damage_dice: str = Field(description="Damage dice, e.g. '1d8' or '2d6'")
    damage_type: Literal["Bludgeoning", "Piercing", "Slashing"]
    weight_lb: float = Field(description="Weight in pounds")
    cost_gp: float = Field(description="Cost in gold pieces")
    properties: list[
        Literal[
            "Finesse",
            "Heavy",
            "Light",
            "Thrown",
            "Two-Handed",
            "Versatile",
            "Ammunition",
            "Loading",
            "Reach",
        ]
    ]
    mastery: Literal["Cleave", "Graze", "Nick", "Push", "Sap", "Slow", "Topple", "Vex"]
    description: str = Field(description="A brief flavor description of the weapon")


# --- Gemini Client ---

client = genai.Client(vertexai=True)


# --- FastAPI App ---

app = FastAPI()


class GenerateRequest(BaseModel):
    prompt: str


@app.get("/")
def index():
    return FileResponse("index.html")


@app.post("/generate")
def generate(request: GenerateRequest):
    response = client.models.generate_content(
        model=MODEL,
        contents=f"{SYSTEM_PROMPT}\n\nDesign a weapon based on: {request.prompt}",
        config={
            "response_mime_type": "application/json",
            "response_json_schema": DnDWeapon.model_json_schema(),
        },
    )
    weapon = DnDWeapon.model_validate_json(response.text)
    return weapon.model_dump()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
