# OpsPilot AI

OpsPilot AI is a compact proof-of-concept for an AI-powered operational analytics assistant for logistics branch management.

The app combines SQLite-based structured retrieval, KPI trend analysis, lightweight anomaly detection, and an LLM-powered chat interface using LangChain prompts, LCEL chains, and RAG.

## What It Does

- Generates realistic synthetic KPI data for 8 logistics branches across 12 months
- Stores operational metrics and detail records in SQLite
- Detects meaningful KPI developments such as service deterioration, cost spikes, staffing pressure, and complaint increases
- Produces an executive management summary
- Lets users ask grounded follow-up questions in a Streamlit chatbot
- Retrieves relevant SQLite records before answering, so factual claims come from data rather than model memory

## Project Structure

```text
app/                  Streamlit entry point
analysis/             KPI trend analysis, anomaly detection, retrieval orchestration
data/                 Synthetic demo data generation
database/             SQLite connection and setup scripts
llm/                  LLM client and prompt wrappers
ui/                   Reserved for reusable UI components
requirements.txt      Python dependencies
call_llm.py           Minimal model smoke test provided during setup
```

## Architecture

```text
Synthetic data generator
        |
        v
SQLite database
        |
        v
KPI analysis engine ---- structured findings
        |                       |
        v                       v
Streamlit dashboard       LLM management summary
        |
        v
Manual RAG orchestrator -> SQL retrieval -> grounded chatbot prompt -> LLM answer
```

The project intentionally avoids heavy agent frameworks. Intent detection is implemented with simple Python rules, then mapped to SQL retrieval functions for KPI rows, delay reasons, staffing events, customer feedback, and shipment cost samples. Retrieval queries are structured using a Pydantic `RetrievalQuery` model for validated, typed input handling.

## Database Tables

Main KPI table:

- `monthly_kpis`

Detail tables:

- `shipment_details`
- `delay_reasons`
- `staffing_events`
- `customer_feedback`

The demo data includes branch-specific patterns:

- Hamburg gradually loses service quality
- Munich develops rising transportation cost per shipment
- Berlin experiences staffing shortages and overtime pressure
- Frankfurt has a one-month operational disruption

## Setup

Create or activate your virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Add your OpenAI API credentials to `.env`. The app uses the same inexpensive model as `call_llm.py`:

```text
OPENAI_API_KEY=your_key_here
```

Create the demo database:

```bash
python -m database.setup_database
```

Run the Streamlit app:

```bash
streamlit run app/main.py
```

## Example Prompts

- Why did service quality decrease in Hamburg?
- Why was customer satisfaction bad in Hamburg?
- Which branches are currently critical?
- What drives transportation costs in Munich?
- Show similar developments in other branches.
- Is Berlin's staffing situation affecting service levels?

## RAG Workflow

1. The user asks a question in the chatbot.
2. The app detects a lightweight intent such as `root_cause`, `cost_drivers`, or `critical_branches`.
3. The retrieval layer identifies relevant branches and pulls structured records from SQLite.
4. KPI findings and SQL results are serialized into compact context. Customer satisfaction questions additionally retrieve feedback categories, sentiment counts, low-rating comments, and satisfaction KPI trends.
5. The LLM receives strict instructions to answer only from retrieved context.

This keeps the assistant grounded in data while still allowing natural language follow-up analysis.

## Screenshots
Overview with branch selection on the left:
![overview](/screenshots/OpsPilotOverview.png)

Tables and Management summary:
![management_summary](/screenshots/ManagementSummary.jpg)

Conversation with Chatbot:
![conversation](/screenshots/ChatBot.png)