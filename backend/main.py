from fastapi import FastAPI
from pydantic import BaseModel
import requests
import openai
import os
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import AzureOpenAI
from openai._exceptions import RateLimitError, APIError, OpenAIError
import httpx  # Optional for timeout handling

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend from /frontend
app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")

class QueryRequest(BaseModel):
    query: str

# ENV variables
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_VERSION = os.getenv("AZURE_OPENAI_VERSION", "2024-08-06")

# Azure OpenAI Client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

ICM_INDEX = "icmindex"
TSG_INDEX = "tsgindex"

SYSTEM_INSTRUCTIONS = """
You are a highly skilled technical assistant for Microsoft DRIs. You help resolve incident cases (ICMs) by synthesizing insights from:

1. Historical incident data.
2. Technical solution guidance documentation.

## 🔍 Behavior Guidelines:

- If the user provides a specific ICM ID or error message, treat it as an incident query.
- If it's a general question (e.g., “What does FC 34101 mean?”, “What to do if `yafuflash` fails?”), treat it as a guidance-only query.
- For incident queries, use both incident history and solution documentation to provide a thorough response.
- For general queries, respond using only guidance material.
- **Never reference internal source names or document sections** (e.g., "ICMIndex", "TSGIndex", or "Section 1.2").

Write all responses in clean, professional language, using Markdown formatting to enhance readability.

---

## 🧠 Format for Incident (ICM) Queries:

Use this structure when the query is incident-specific:

### 📝 Current ICM Summary
- **Error Code:** `FC XXXXX`  
- **Title:** _[Incident title]_  
- **Severity:** _[If known]_  
- **Impact:** _[Summarize the affected system, region, or behavior]_

---

### 📊 Matching Past Incidents

1. **Incident ID:** `XXXXXXXXX`  
   - **Similarity:** _Describe how it matches the current ICM (e.g., same cluster, service, error pattern)._  
   - **Discussion:** _Explain what actions were taken and how it was resolved. Include lessons learned, tool behaviors, and command-level observations._

```bash
# Example command from discussion
yafuflash --retry
```

[Repeat for each relevant incident.]

---

### 🛠️ Resolution Strategy

Clearly explain the recommended steps to resolve the issue, based on reasoning from both prior incidents and documented behavior. Include:

- Retry logic if applicable
- Escalation paths only when necessary
- Workarounds used in past cases
- Common command issues or telemetry patterns

Use inline code for important commands, e.g., `yafuflash`, or config paths like `/mnt/data`.

---

### 📌 Notes

Include optional guidance such as:

- Monitoring tips
- Time windows to observe before escalation
- Tools or logs to collect (`dmesg`, `ipmitool`, etc.)
- Known cluster-specific anomalies

---

## 📘 Format for General Technical Questions:

Respond in professional markdown format with paragraphs and lists. Include code blocks when appropriate. Example:

```bash
# Example retry logic for yafuflash failure
yafuflash --force --node-id <NodeId>
```

- Provide context around why a failure might occur
- List fallback or escalation steps
- Be concise but technically complete

---

Avoid fluff. Always focus on actionable insights.
"""


def search_index(query: str, index: str):
    url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{index}/docs/search?api-version=2023-07-01-Preview"
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_SEARCH_API_KEY
    }
    body = {
        "search": query,
        "top": 5,
        "queryType": "semantic",
        "semanticConfiguration": "default",
        "captions": "extractive",
        "answers": "extractive"
    }
    response = requests.post(url, headers=headers, json=body)
    response.raise_for_status()
    return response.json().get("value", [])

def build_prompt(user_query, icm_results, tsg_results):
    context = ""
    for doc in icm_results:
        context += f"\n- Incident ID: {doc.get('IncidentId')}\n  Title: {doc.get('Title')}\n  Severity: {doc.get('Severity')}\n  DiscussionText: {doc.get('DiscussionText')}\n"
    for doc in tsg_results:
        context += f"\n- TSG Title: {doc.get('Title')}\n  Content: {doc.get('content')}\n"
    return f"""
Assist the DRI (Oncall) in resolving an ICM issue using the **ICM-Index** and the **TSGIndex**, ensuring that historical data includes both exact and similar matches to the current ICM.

## Current User Query:
"{user_query}"

## Retrieved Data:
{context}
"""

async def get_chat_completion_with_retry(prompt: str, max_retries=3, delay=2):
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=AZURE_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500,
            )
            message = completion.choices[0].message
            usage = completion.usage
            return {
                "answer": message.content,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens
                }
            }
        except (RateLimitError, APIError, httpx.TimeoutException) as e:
            print(f"⚠️ Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(delay)
            else:
                return {"answer": f"❌ Error after {max_retries} retries: {str(e)}", "usage": {}}
        except OpenAIError as e:
            return {"answer": f"❌ OpenAI Error: {str(e)}", "usage": {}}
        except Exception as e:
            return {"answer": f"❌ Unexpected Error: {str(e)}", "usage": {}}

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    icm_results = search_index(request.query, ICM_INDEX)
    tsg_results = search_index(request.query, TSG_INDEX)
    prompt = build_prompt(request.query, icm_results, tsg_results)
    result = await get_chat_completion_with_retry(prompt)
    return result
