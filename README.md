# AI-Powered SOC Security Agent

An AI-powered Security Operations Center (SOC) agent that goes beyond answering questions. It analyzes security logs, detects suspicious activity, identifies potential attack families, maps them to official **MITRE ATT&CK techniques**, and automatically generates structured incident reports.

The goal is to build an intelligent security assistant capable of supporting the investigation process, from anomaly detection to incident documentation.

## Project Overview

The project combines three complementary components, each addressing a different question during a security investigation:

| Component | Main Purpose |
|---|---|
| **Isolation Forest** | What activity looks unusual or abnormal? |
| **XGBoost** | What attack family does the activity belong to? |
| **MITRE ATT&CK RAG** | What is the official attack technique, and how can it be mitigated? |

By combining anomaly detection, supervised classification, and retrieval-augmented generation (RAG), the system provides more context than a traditional detection pipeline.

## Key Features

- **Hybrid Threat Detection:** Combines Isolation Forest and XGBoost to identify suspicious network activity.
- **Attack Classification:** Predicts the potential attack family using a trained XGBoost model.
- **MITRE ATT&CK Mapping:** Retrieves relevant attack techniques and official references from a dedicated knowledge base.
- **Automated Incident Reporting:** Generates structured Markdown reports containing investigation findings and security recommendations.
- **AI-Powered Assistant:** Supports general-purpose tasks through a tool-based agent architecture.
- **Local LLM Integration:** Uses Ollama to run language models locally, without relying on external LLM APIs.
- **Web Dashboard:** Provides a web interface to interact with the agent.

## Available Tools

The agent can interact with different tools to perform tasks and support investigations.

| Tool | Description |
|---|---|
| `web_search` | Searches the web using DuckDuckGo. |
| `read_file` | Reads files from the `workspace/` directory. |
| `write_file` | Creates or updates files in the workspace. |
| `list_files` | Lists files available in the workspace. |
| `calculator` | Evaluates mathematical expressions. |
| `run_python` | Executes Python code for data processing, filtering, and aggregation. |
| `detect_anomalies` | Runs the hybrid Isolation Forest + XGBoost detection pipeline and returns JSON results. |
| `search_attack_knowledge` | Retrieves relevant MITRE ATT&CK information using natural-language queries. |

## Two Operating Modes

### 1. General Assistant Mode

A general-purpose AI assistant that can combine multiple tools to complete tasks.

For example, it can search for information, process data, perform calculations, read or write files, and provide a final response.

### 2. SOC Investigation Mode

A specialized security analyst mode designed to investigate security logs.

The agent follows an investigation workflow:

1. Reads the available security logs.
2. Processes and analyzes the collected events.
3. Detects anomalous activity using Isolation Forest.
4. Classifies potential attacks using XGBoost.
5. Retrieves relevant MITRE ATT&CK techniques through the RAG system.
6. Generates an incident report with findings and mitigation recommendations.

## Datasets and Knowledge Sources

The project relies on three distinct sources of data, each serving a specific purpose.

### 1. Synthetic Security Logs

Generated locally using `generate_logs.py`, these logs provide a reproducible environment for testing the SOC investigation workflow.

They simulate 30 days of activity within a small organization, including:

- Legitimate SSH authentication attempts.
- Normal firewall traffic.
- Application activity.
- Simulated malicious events and attack scenarios.

These logs serve as the primary input for the security investigation process.

### 2. UNSW-NB15 Dataset

A publicly available network intrusion detection dataset developed by UNSW Canberra.

It is used to train and evaluate the supervised XGBoost classification model.

- **82,332 network flows**
- **9 attack categories**, including DoS, Exploits, and Reconnaissance.

Dataset source: [UNSW-NB15 – Cyber Range Lab](https://www.unsw.canberra.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-UNSW-NB15-Dataset/)

### 3. MITRE ATT&CK Framework

The MITRE ATT&CK knowledge base provides a standardized reference for describing adversary tactics and techniques.

The project uses a curated subset of **63 ATT&CK techniques** relevant to the simulated attack scenarios.

This knowledge base allows the agent to retrieve official technique identifiers, descriptions, and mitigation information rather than relying solely on the language model's internal knowledge.

## RAG: Grounding Attack Identification in MITRE ATT&CK

### The Problem

Detecting an anomaly or identifying a suspicious IP address is not enough for a meaningful SOC investigation.

A security report should also explain:

- What attack technique might be involved?
- What is its official MITRE ATT&CK identifier?
- How does the technique work?
- What security measures can help prevent or mitigate it?

Asking an LLM to provide these references from memory can lead to inaccurate or fabricated technique identifiers.

### The Solution

The project implements a Retrieval-Augmented Generation (RAG) pipeline based on the MITRE ATT&CK knowledge base.

Instead of relying exclusively on the LLM's internal knowledge, the agent retrieves relevant information from a dedicated collection of ATT&CK techniques.

The process includes:

1. Extracting ATT&CK information from the official STIX dataset.
2. Selecting and preparing the 63 relevant techniques.
3. Generating vector embeddings using `nomic-embed-text`.
4. Storing and indexing the knowledge base with ChromaDB.
5. Retrieving relevant techniques through natural-language queries.
6. Using the retrieved information to support the agent's explanations and incident reports.

This approach helps ground attack identification in documented security knowledge and reduces the risk of hallucinated MITRE references.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn, XGBoost |
| Anomaly Detection | Isolation Forest |
| Vector Database | ChromaDB |
| RAG & Embeddings | RAG, `nomic-embed-text` |
| Local LLM | Ollama, Qwen3 8B |
| Security Knowledge Base | MITRE ATT&CK, STIX |
| Web Search | DuckDuckGo |
| Interface | Streamlit |
| Data Processing | Pandas, NumPy |

## Installation and Setup

### Prerequisites

Make sure you have the following installed:

- Python 3.10 or later.
- Ollama.
- Git (recommended).

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd projet_agent
```

### 2. Create a Virtual Environment

On Windows:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create your local environment configuration file:

```powershell
copy .env.example .env
```

Update the `.env` file according to your local configuration.

### 4. Download the Local Models

Install the required Ollama models:

```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

These models are downloaded once and can then be used locally.

### 5. Build the MITRE ATT&CK Knowledge Base

Run the following commands to extract the ATT&CK techniques and build the vector database:

```powershell
.\.venv\Scripts\python.exe ingest_attack.py

.\.venv\Scripts\python.exe -m agent.rag
```

This process prepares the 63 selected techniques and creates the ChromaDB index.

### 6. Train the XGBoost Model

Download the `UNSW_NB15_testing-set.csv` dataset from the official UNSW source.

Place the file in:

```text
D:\datasets\unsw_nb15\UNSW_NB15_testing-set.csv
```

Then run:

```powershell
.\.venv\Scripts\python.exe train_xgboost
```

The trained model artifacts will be saved in the `models/` directory.

## Usage

### 1. Generate Synthetic Security Logs

Generate the simulated security events:

```powershell
.\.venv\Scripts\python.exe generate_logs.py
```

This step only needs to be performed once unless you want to regenerate the logs.

### 2. Run a Complete SOC Investigation

Start the SOC agent:

```powershell
.\.venv\Scripts\python.exe main.py --soc "Analyze the logs and write an incident report"
```

The agent will investigate the available logs and generate:

```text
workspace/incident_report.md
```

### 3. Run the General Assistant

Start the interactive assistant:

```powershell
.\.venv\Scripts\python.exe main.py
```

You can also execute a one-time request:

```powershell
.\.venv\Scripts\python.exe main.py "Find the three latest AI news articles and summarize them"
```

### 4. Launch the Web Dashboard

Start the web interface:

```powershell
.\.venv\Scripts\python.exe app.py
```

Open the local URL displayed in your terminal to access the dashboard.

## Project Structure

```text
projet_agent/
│
├── agent/
│   └── rag/                  # RAG pipeline and ATT&CK retrieval
│
├── models/                   # Trained ML model artifacts
│
├── workspace/                # Input logs and generated reports
│
├── generate_logs.py           # Synthetic log generation
├── ingest_attack.py           # MITRE ATT&CK data ingestion
├── train_xgboost.py           # XGBoost model training
├── main.py                    # Main agent entry point
├── app.py                     # Web dashboard
│
├── requirements.txt           # Python dependencies
├── .env.example               # Environment configuration template
└── README.md
```

*Note: The structure above presents the main project components. Some internal files and modules may not be shown.*

## Project Goals

This project explores how machine learning and generative AI can work together to support security operations.

Its main objectives are:

- Automating the analysis of security logs.
- Combining unsupervised anomaly detection with supervised attack classification.
- Connecting detected activity to a standardized cybersecurity knowledge base.
- Reducing hallucinations through retrieval-grounded MITRE ATT&CK references.
- Generating structured and actionable incident reports.
- Running the AI components locally using open-source models.

## Future Improvements

Potential extensions include:

- Integrating real-time log ingestion.
- Supporting additional log formats and security data sources.
- Expanding the MITRE ATT&CK technique coverage.
- Adding incident severity scoring and prioritization.
- Improving detection performance through additional ML experiments.
- Integrating the agent with SIEM platforms.
- Adding interactive investigation history and visual analytics.

## Disclaimer

This project is developed for educational and research purposes.

The security logs are synthetic, and the detection models are intended for experimentation. The system should not be considered a replacement for professional SOC analysts or production-grade security monitoring solutions.

---

**Built with Python, Machine Learning, RAG, Ollama, and MITRE ATT&CK.**
