from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import os
from openai import OpenAI

# Initialize FastAPI app
app = FastAPI(title="KeaBuilder AI Lead Processor", version="1.0")

# Note: In a real environment, this is loaded from .env
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy_key"))

# --- Schemas ---
class LeadInput(BaseModel):
    name: str
    email: str
    company_size: str
    budget: str
    timeline: str
    goal: str

class LeadResponse(BaseModel):
    lead_id: str
    classification: str
    confidence_score: float
    missing_info: list
    recommended_action: str
    generated_response: str

# --- Core Logic ---
@app.post("/api/v1/process-lead", response_model=LeadResponse)
async def process_lead(lead: LeadInput):
    """
    Takes incoming lead data from a KeaBuilder funnel, 
    classifies it, and generates a personalized response.
    """
    prompt = f"""
    You are an expert lead qualifier for KeaBuilder. Analyze this lead:
    Name: {lead.name}
    Budget: {lead.budget}
    Timeline: {lead.timeline}
    Goal: {lead.goal}

    Classify the lead as "Hot" (High budget/immediate need), "Warm" (Has need, loose timeline), or "Cold" (Vague).
    Write a personalized email response to them.
    
    Output strictly in this JSON format:
    {{
        "classification": "Hot/Warm/Cold",
        "confidence_score": 0.95,
        "missing_info": ["list any missing crucial info"],
        "recommended_action": "trigger_demo / nurture_email / manual_review",
        "generated_response": "Hi [Name]..."
    }}
    """

    try:
        # If no real API key is present, we simulate the LLM response to prove the architecture works.
        if client.api_key == "dummy_key":
            return generate_mock_response(lead)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }
        )
        
        result = json.loads(response.choices[0].message.content)
        result["lead_id"] = f"lead_{hash(lead.email) % 10000}"
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def generate_mock_response(lead: LeadInput):
    """Fallback mock response for demonstration without an API key."""
    classification = "Hot" if "immediate" in lead.timeline.lower() else "Warm"
    return {
        "lead_id": "lead_9921",
        "classification": classification,
        "confidence_score": 0.88,
        "missing_info": [] if lead.budget else ["budget"],
        "recommended_action": "trigger_demo_scheduler",
        "generated_response": f"Hi {lead.name}, I saw you're looking to achieve: {lead.goal}. Let's get you set up on KeaBuilder right away. Are you free for a call tomorrow?"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
