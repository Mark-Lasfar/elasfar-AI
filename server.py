from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

app = FastAPI()

# Read token from Environment Variable
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN", "")

# Download the form and tokenizer
model_name = "ibrahimlasfar/elasfar-AI"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGING_FACE_TOKEN)
model = AutoModelForCausalLM.from_pretrained(model_name, token=HUGGING_FACE_TOKEN)

class Query(BaseModel):
    question: str

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/ask")
async def ask_question(query: Query):
    try:
#Create context
        context = f"""
        Website: Ibrahim Al-Asfar's personal portfolio.
        Description: A full-stack web developer portfolio showcasing projects, skills, and contact information.
        Question: {query.question}
        """
        inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_length=150, num_return_sequences=1, temperature=0.7)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
