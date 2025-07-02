from fastapi import FastAPI
from pydantic import BaseModel
import requests
import openai
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files from frontend folder
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

class QueryRequest(BaseModel):
    query: str

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
print("AZURE_SEARCH_ENDPOINT =", AZURE_SEARCH_ENDPOINT)
ICM_INDEX = "icm-index"
TSG_INDEX = "tsgindex"

openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_key = os.getenv("AZURE_OPENAI_KEY")
openai.api_version = "2024-02-01"
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")

def search_index(query: str, index: str):
    url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{index}/docs/search?api-version=2021-04-30-Preview"
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_SEARCH_API_KEY
    }
    body = {
        "search": query,
        "top": 5
    }
    response = requests.post(url, headers=headers, json=body)
    return response.json().get("value", [])

def build_prompt(user_query, icm_results, tsg_results):
    context = ""
    for doc in icm_results:
        context += f"\n- Incident ID: {doc.get('IncidentId')}\n  Title: {doc.get('Title')}\n  Severity: {doc.get('Severity')}\n  DiscussionText: {doc.get('DiscussionText')}\n"
    for doc in tsg_results:
        context += f"\n- TSG Title: {doc.get('Title')}\n  Content: {doc.get('content')}\n"
    return f"""... \n## Current User Query:\n\"{user_query}\"\n\n## Retrieved Data:\n{context}"""

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    try:
        icm_results = search_index(request.query, ICM_INDEX)
        tsg_results = search_index(request.query, TSG_INDEX)
        prompt = build_prompt(request.query, icm_results, tsg_results)

        completion = openai.ChatCompletion.create(
            deployment_id=AZURE_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a technical assistant for resolving incidents using TSG + ICM history."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        print(completion)

        return {"answer": completion["choices"][0]["message"]["content"]}

    except Exception as e:
        # log and return error
        print(completion)
        return {"answer": f"❌ Error: {str(e)}"}
