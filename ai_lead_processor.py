from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
import os
from openai import OpenAI

app = FastAPI(title="VARYNT AI Lead Processor")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy_key"))

# --- Strict Schemas ---
class LeadInput(BaseModel):
    lead_id: str
    name: str
    email: str
    timeline: str
    budget: str
    inquiry: str

class ClassificationResult(BaseModel):
    category: str = Field(description="Must be exactly: Hot, Warm, Cold, or Invalid")
    confidence_score: float = Field(description="Confidence from 0.0 to 1.0")

# --- Core Logic ---
def process_lead_pipeline(lead: LeadInput):
    """Background task to process the lead, ensuring UI is never blocked."""
    try:
        class_completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a lead qualification AI. Output strict JSON."},
                {"role": "user", "content": f"Budget: {lead.budget}, Timeline: {lead.timeline}, Inquiry: {lead.inquiry}"}
            ],
            response_format=ClassificationResult,
            temperature=0.1
        )
        classification = class_completion.choices[0].message.parsed
        # Save to DB simulated here
    except Exception as e:
        print(f"Error processing lead: {e}")

# --- API Endpoint ---
@app.post("/api/v1/leads", status_code=202)
async def ingest_lead(lead: LeadInput, background_tasks: BackgroundTasks):
    """Accepts lead and hands off to background worker immediately."""
    background_tasks.add_task(process_lead_pipeline, lead)
    return {"status": "accepted", "message": "Lead queued for AI processing", "lead_id": lead.lead_id}
