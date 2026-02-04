from fastapi import FastAPI
from pydantic import BaseModel
from agent_ai import chat_function

app = FastAPI()

class ThemeInput(BaseModel):
    input: str


@app.post("/chat")
async def chat_endpoint(theme_input: ThemeInput):
    result = chat_function(theme_input.input)
    return {"result": result}