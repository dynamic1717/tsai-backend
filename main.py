from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, BaseSettings
import spacy
from spacy.matcher import PhraseMatcher
import uvicorn
import openai
import tiktoken
from datetime import datetime


class Settings(BaseSettings):
    OPENAI_API_KEY: str = 'OPENAI_API_KEY'

    class Config:
        env_file = '.env'


settings = Settings()
openai.api_key = settings.OPENAI_API_KEY


class Message(BaseModel):
    message: str


app = FastAPI()
# app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/')
async def hello():
    return {'message': 'Hello World'}


@app.post('/api/chat')
async def index(request: Message):

    number_of_tokens = calculate_tokens(request.message)

    if number_of_tokens > 50:
        raise HTTPException(
            status_code=400, detail="As an AI language model, I have limited number of tokens. So sorry, your message is too long.")

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "system", "content": "You are not a very useful assistant who floods with names, numbers, dates. Start your response with: As an AI language model"},
            {"role": "user", "content": request.message}
        ],
        temperature=0.5,
        max_tokens=100
    )

    result = response['choices'][0]['message']['content']  # type: ignore

    anonymized_text = anonymize_text(result)
    ts = datetime.now().timestamp()

    return {"data": {
        "response": anonymized_text,
        "timestamp": ts
    }, "status": 200, 'error': None}


def calculate_tokens(text: str):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(text))
    return num_tokens


def anonymize_text(text: str):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)

    tokens = [
        {
            'i': token.i,
            'pos': token.pos_,
            'text': token.text,
            'whitespace': token.whitespace_,
        }
        for token in doc
    ]

    replaced_text = []
    replace_mode = False

    matcher = PhraseMatcher(nlp.vocab, None)
    matcher.add('ai_start', [nlp("As an AI language model")])
    matches = matcher(doc)

    for token in tokens:
        if len(matches) > 0 and token['i'] >= matches[0][1] and token['i'] <= matches[0][2]:
            replace_mode = False
        else:
            replace_mode = True

        if replace_mode and (token['pos'] == 'PROPN' or token['pos'] == 'NOUN'):
            replaced_text.append('XXX')
        else:
            replaced_text.append(token['text'])

        replaced_text.append(token['whitespace'])

    return ''.join(replaced_text)


if __name__ == "__main__":
    uvicorn.run('app:app', host="localhost", port=5001, reload=True)
