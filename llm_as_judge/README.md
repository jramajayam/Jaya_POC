# LLM-as-Judge: Asset Category Classifier

An **independent LLM evaluator** for the MLP asset-category classifier used in CTEM (Continuous Threat Exposure Management).

The LLM never sees the MLP predictions — it classifies each asset on its own using only 5 raw fields. A deterministic Python comparator then scores the MLP against the LLM to produce a structured `JudgmentResult`.

---

## Architecture

```
                   ┌──────────────────┐
                   │   AssetInput     │
                   │ (5 fields + MLP) │
                   └────────┬─────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
   ┌─────────────────┐            ┌──────────────────┐
   │  LLM (Azure AI) │            │  MLP predictions │
   │ receives ONLY:  │            │  (already done)  │
   │  • domain       │            │  stored in       │
   │  • ports        │            │  asset.mlp_cats   │
   │  • technologies │            └────────┬─────────┘
   │  • banner       │                     │
   │  • dns_count    │                     │
   └────────┬────────┘                     │
            │                              │
            ▼                              │
   ┌─────────────────┐                     │
   │ LLM categories  │                     │
   └────────┬────────┘                     │
            │         ┌───────────┐        │
            └────────►│  _compare │◄───────┘
                      │  (Python) │
                      └─────┬─────┘
                            ▼
                   ┌─────────────────┐
                   │ JudgmentResult  │
                   │  • judgment     │
                   │  • correct      │
                   │  • missing      │
                   │  • wrong        │
                   │  • confidence   │
                   └─────────────────┘
```

---

## Project Structure

```
.
├── llm_judge/                  # Standalone package
│   ├── __init__.py             # Public exports
│   ├── models.py               # AssetInput, JudgmentResult
│   ├── prompts.py              # Self-contained prompts & category definitions
│   ├── judge.py                # LLMJudge class (call LLM → compare → result)
│   └── json_parser.py          # Robust JSON extraction from LLM output
│
├── tests/
│   └── test_llm_judge.py       # 24 unit tests (all mocked, no LLM needed)
│
├── openai_model_manager.py     # Azure OpenAI client wrapper
├── config.py                   # MODEL / REASONING_EFFORT defaults
├── requirements.txt            # Python dependencies
├── .env.example                # Template — copy to .env, add your key
└── .gitignore                  # Excludes .env, __pycache__, model artifacts
```

---

## Quick Start

### 1. Clone the repo and navigate into the `llm_as_judge` folder

```bash
git clone https://github.com/jramajayam/Jaya_POC.git
cd Jaya_POC/llm_as_judge
```

### 2. Create a virtual environment and activate it

```bash
python3 -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs: `requests`, `python-dotenv`, `pytest`.

### 4. Run the tests (no API key needed — everything is mocked)

```bash
python -m pytest tests/test_llm_judge.py -v
```

You should see **24 passed** in under 1 second. No Azure credentials required for this step.

### 5. Set up your API key (needed only for real LLM calls)

Copy the example env file to create your own `.env`:

```bash
cp .env.example .env
```

Now open the `.env` file in any text editor and replace the placeholder:

```dotenv
# ─── BEFORE (placeholder) ───
AZURE_OPENAI_API_KEY=your-api-key-here

# ─── AFTER (your real key) ───
AZURE_OPENAI_API_KEY=abc123def456...
```

The other two values already have working defaults — only change them if your
Azure setup is different:

```dotenv
AZURE_OPENAI_ENDPOINT=https://openai-us-east2.openai.azure.com/
AZURE_OPENAI_API_VERSION=2025-04-01-preview
```

> **⚠️  Never commit `.env` to Git.** It is already in `.gitignore`.
>
> If you forget to create `.env`, you will see this error on startup:
> ```
> EnvironmentError: AZURE_OPENAI_API_KEY not set.
>                   Copy .env.example → .env and fill in your key.
> ```

---

## Usage

```python
from llm_judge import AssetInput, LLMJudge

judge = LLMJudge()

asset = AssetInput(
    asset_id="asset-001",
    domain="api.example.com",
    ports=["80", "443"],
    technologies=["nginx", "React"],
    banner="HTTP/1.1 200 OK\r\nServer: nginx/1.25",
    dns_count=150,
    mlp_categories=["web server"],           # from your MLP model
    valid_categories=["web server", "api endpoint", "mail server"],
)

result = judge.judge(asset)

print(result.judgment)            # CORRECT / PARTIALLY_CORRECT / INCORRECT
print(result.llm_categories)     # what the LLM classified independently
print(result.missing_categories) # categories the MLP missed
print(result.confidence)         # LLM's self-reported confidence (0-1)
```

---

## How It Works

| Step | What happens | Where |
|------|-------------|-------|
| 1 | Build prompt with **only** domain, ports, techs, banner, dns_count | `judge.py → _build_prompt()` |
| 2 | Send to Azure OpenAI (o1 model) | `openai_model_manager.py` |
| 3 | Parse JSON from LLM response | `json_parser.py → extract_json()` |
| 4 | Compare LLM categories vs MLP categories | `judge.py → _compare()` |
| 5 | Apply **subsumption rules** (e.g. "web server" subsumes "CDN") | `prompts.py → WEB_SERVER_SUBSUMES` |
| 6 | Return `JudgmentResult` | `models.py` |

### Subsumption Rule

If MLP predicted `"web server"` and the LLM says `"content delivery network"`, the judgment is **CORRECT** — because CDN is a specialization of web server. These categories are subsumed:

- content delivery network
- api endpoint
- certificate validation server
- software update server
- content management system
- application server
- monitoring server
- continuous integration server

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AZURE_OPENAI_API_KEY` | **Yes** | — | Your Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | No | `https://openai-us-east2.openai.azure.com/` | Azure endpoint URL |
| `AZURE_OPENAI_API_VERSION` | No | `2025-04-01-preview` | API version |

---

## Tests

All 24 tests run without an API key (the LLM client is mocked):

```
tests/test_llm_judge.py
  TestExtractJson          6 tests   — JSON parsing edge cases
  TestModels               4 tests   — AssetInput / JudgmentResult
  TestCompare              7 tests   — Deterministic comparison logic
  TestJudgeEndToEnd        4 tests   — Full flow with mocked LLM
  TestPromptBuilding       3 tests   — Verifies only 5 fields go to LLM
```

```bash
python -m pytest tests/test_llm_judge.py -v
```
