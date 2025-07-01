from fastapi import FastAPI
from pydantic import BaseModel
import requests
import openai
import os
from fastapi.middleware.cors import CORSMiddleware

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

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
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

    return f"""
Assist the DRI (Oncall) in resolving an ICM issue by leveraging the TSG (Troubleshooting Guide) documentation and historical data from the ICM dump. Ensure that Incident IDs from historical data include similar, not just exact, matches to the current ICM.

## Instructions:

1. Understand the Current ICM:
   Begin by extracting key details about the current ICM, including error codes, severity, impact, and relevant context.

2. Retrieve Relevant Historical Data:
   - Search the "Title" column in the past ICM dump for incidents with similar error codes, symptoms, or descriptions. Exact matches are not required.
   - Analyze the "DiscussionText" of matches to understand how those ICMs were mitigated or resolved.
   - Note down the related Incident IDs for both exact and similar matching ICMs.

3. Refer to the TSG Documentation:
   - Cross-reference TSG documentation sections related to the current ICM or past matches.
   - Extract specific troubleshooting steps or guidance.

4. Synthesize the Findings:
   - Combine insights from TSG + past ICMs to give clear, actionable recommendations.
   - If no past matches, rely solely on TSG.

5. Provide Recommendations:
   - Reasoning first (why a step is useful).
   - Follow with a list of actionable steps.

---

## Current User Query:
"{user_query}"

## Retrieved Data:
{context}

## Task:
Based on the retrieved information, generate the following:

- **Current ICM Summary:** key info about the query.
- **Insights from Past ICMs:** matched incidents, why they’re relevant, and what was done.
- **TSG Guidance:** any relevant steps or sections.
- **Recommended Resolution Steps:** a prioritized, reasoned action list for the DRI to execute.
"""

@app.post("/query")
async def query_endpoint(request: QueryRequest):
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

    return {"answer": completion["choices"][0]["message"]["content"]}
