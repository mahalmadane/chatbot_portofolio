from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from agent_ai import chat_function
import os

app = FastAPI(title="AI Agent API", version="1.0.0")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production si nécessaire
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ThemeInput(BaseModel):
    input: str
    context:list

@app.get("/")
async def root():
    return {
        "message": "AI Agent API is running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/chat")
async def chat_endpoint(theme_input: ThemeInput):
    try:
        raw_response = chat_function(theme_input.input,theme_input.context)
        return {"raw": raw_response}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8009))
    uvicorn.run(app, host="0.0.0.0", port=port)