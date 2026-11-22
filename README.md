# AI Data Analysis Agent

An AI-powered data analysis web application that allows users to upload CSV or Excel datasets, clean data, ask questions in natural language, execute SQL queries, and visualize results.

## Features

- **CSV and Excel Upload:** Upload `.csv` and `.xlsx` datasets.
- **Data Cleaning:** Trim text whitespace, handle missing values, remove duplicate rows, and standardize column names.
- **Natural Language Analysis:** Ask questions about your dataset using plain English.
- **AI-Powered SQL Generation:** Use the local Ollama model to generate DuckDB SQL queries.
- **SQL Auto-Retry:** If a generated query fails, send the error to the model and attempt to repair the query up to two times.
- **Verified Data Results:** Use Python, Pandas, and DuckDB to calculate and execute data operations rather than relying on AI to guess numerical results.
- **Interactive Visualizations:** Generate bar charts, pie charts, and line charts based on query results.
- **Local AI Processing:** Use Ollama with `llama3.2:3b` without requiring a paid LLM API key.

## Technology Stack

- Python
- Streamlit
- Pandas
- DuckDB
- Ollama (`llama3.2:3b`)
- Plotly
- OpenPyXL

## How It Works

1. Upload a CSV or Excel dataset.
2. Review the data and apply available cleaning operations.
3. Ask a question in natural language.
4. Ollama generates a read-only SQL query.
5. DuckDB executes the query against the uploaded dataset.
6. If SQL execution fails, the application attempts to repair the query up to two times.
7. The application displays the query result, a suitable visualization, and an explanation grounded in the executed result.

## Installation and Setup

### Prerequisites

- Python installed on your computer
- Git
- Ollama

### 1. Install the local AI model

Install Ollama, then run:

```bash
ollama pull llama3.2:3b
```

Ensure Ollama is running before using the application.

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows Command Prompt:

```cmd
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install streamlit pandas duckdb ollama plotly openpyxl
```

### 4. Run the application

```bash
streamlit run ai_data_analyst.py
```

The application should open in your browser.

## Example Questions

- Which category has the highest total sales?
- Show total sales by category using a bar chart.
- Show each category's percentage contribution to total sales using a pie chart.
- How many missing values are there in each column?
- How many duplicate rows are present in the dataset?
- Show the monthly sales trend.

Results depend on the columns and data available in the uploaded dataset.

## Project Learning Outcomes

- Data cleaning and preprocessing with Pandas
- Natural-language-to-SQL workflows
- SQL execution and error handling with DuckDB
- Local LLM integration using Ollama
- Interactive dashboards and visualizations with Streamlit and Plotly
- Validation of AI-generated analytical results

## Acknowledgements

This project builds on a public starter project from the `awesome-llm-apps` repository. The application has been adapted and extended with local Ollama integration, data cleaning, SQL validation and retry handling, verified analysis, and visualization features.

Original repository: https://github.com/Shubhamsaboo/awesome-llm-apps

## Future Improvements

- Automated data-quality reports
- Export results and charts
- More advanced visualization controls
- Expanded automated testing
