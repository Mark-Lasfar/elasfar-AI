from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

app = FastAPI()

# قراءة التوكن من Environment Variable
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN", "")

# تحميل النموذج والتوكنيزر
model_name = "ibrahimlasfar/elasfar-AI"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGING_FACE_TOKEN)
model = AutoModelForCausalLM.from_pretrained(model_name, token=HUGGING_FACE_TOKEN)

class Query(BaseModel):
    question: str

class Conversation(BaseModel):
    messages: list[dict]  # قائمة من الرسائل، كل رسالة فيها {'role': 'user'|'assistant', 'content': str}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/ask")
async def ask_question(query: Query):
    try:
        if not query.question:
            raise HTTPException(status_code=400, detail="Question is required")
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

@app.post("/api/converse")
async def converse(conversation: Conversation):
    try:
        if not conversation.messages:
            raise HTTPException(status_code=400, detail="Messages are required")
        conversation_text = ""
        for msg in conversation.messages:
            if "role" not in msg or "content" not in msg:
                raise HTTPException(status_code=400, detail="Each message must have role and content")
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation_text += f"{role}: {msg['content']}\n"
        context = f"""
        Website: Ibrahim Al-Asfar's personal portfolio.
        Description: A full-stack web developer portfolio showcasing projects, skills, and contact information.
        Conversation:
        {conversation_text}
        Assistant: """
        inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_length=150, num_return_sequences=1, temperature=0.7)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(context, "").strip()
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing conversation: {str(e)}")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
