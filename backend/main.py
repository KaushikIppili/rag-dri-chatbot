from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import openai
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

# ENV Vars
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
ICM_INDEX = "icm-index"
TSG_INDEX = "tsgindex"

openai.api_type = "azure"
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_key = AZURE_OPENAI_KEY
openai.api_version = "2024-02-01"

def search_index(query: str, index: str):
    url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{index}/docs/search?api-version=2021-04-30-Preview"
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_SEARCH_API_KEY
    }
    body = {"search": query, "top": 5}
    response = requests.post(url, headers=headers, json=body)
    return response.json().get("value", [])

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    icm_results = search_index(request.query, ICM_INDEX)
    tsg_results = search_index(request.query, TSG_INDEX)

    context = ""
    for doc in icm_results:
        context += f"\n- ICM ID: {doc.get('IncidentId')} | Title: {doc.get('Title')} | Severity: {doc.get('Severity')}\n{doc.get('DiscussionText')}\n"
    for doc in tsg_results:
        context += f"\n- TSG Title: {doc.get('Title')} | Content: {doc.get('content')}\n"

    prompt = f"""
Resolve the ICM issue below using history and TSGs.

Query: {request.query}
Data: {context}
"""

    completion = openai.ChatCompletion.create(
        deployment_id=AZURE_DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": "You are a technical assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1000
    )

    return {"answer": completion["choices"][0]["message"]["content"]}

@app.get("/")
async def serve_index():
    return FileResponse("frontend/index.html")
