import json
import re
import csv
import io

import pandas as pd
import streamlit as st
import ollama
import duckdb
import plotly.express as px


MODEL_NAME = "llama3.2:3b"

st.set_page_config(page_title="AI Data Analyst Agent", page_icon="📊", layout="wide")
st.title("📊 AI Data Analyst Agent")
st.caption(
    "Ask questions in plain English. Ollama writes SQL, DuckDB executes it, "
    "and Python/Pandas verifies dataset statistics."
)


def read_csv_with_fallback(uploaded_file):
    """Try common encodings so CSVs exported from Excel also load."""
    raw = uploaded_file.getvalue()
    errors = []
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(
                io.BytesIO(raw),
                encoding=encoding,
                na_values=["NA", "N/A", "missing"],
                low_memory=False,
            ), encoding
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError("Could not decode this CSV. Try saving it as UTF-8 CSV and upload again.")


def preprocess_and_save(uploaded_file):
    """Load CSV/XLSX and perform conservative, transparent type cleanup."""
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df, used_encoding = read_csv_with_fallback(uploaded_file)
            st.caption(f"CSV encoding detected: `{used_encoding}`")
        elif uploaded_file.name.lower().endswith(".xlsx"):
            try:
                df = pd.read_excel(uploaded_file, na_values=["NA", "N/A", "missing"], engine="openpyxl")
            except ImportError as exc:
                raise ImportError(
                    "Reading .xlsx files requires openpyxl. In your activated .venv, run: "
                    "python -m pip install openpyxl"
                ) from exc
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None

        # Clean whitespace in column names and ensure unique names for SQL.
        cleaned = [str(col).strip() or f"column_{i+1}" for i, col in enumerate(df.columns)]
        seen = {}
        unique = []
        for name in cleaned:
            count = seen.get(name, 0)
            seen[name] = count + 1
            unique.append(name if count == 0 else f"{name}_{count+1}")
        df.columns = unique

        # Convert date-like columns; convert text columns only when almost all
        # non-empty values are numeric, avoiding aggressive conversion.
        for col in df.columns:
            if "date" in str(col).lower():
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif pd.api.types.is_object_dtype(df[col]):
                converted = pd.to_numeric(df[col], errors="coerce")
                nonblank = int(df[col].notna().sum())
                if nonblank and converted.notna().sum() / nonblank >= 0.95:
                    df[col] = converted

        return df
    except Exception as exc:
        st.error(f"Could not process the uploaded file: {exc}")
        return None



def clean_dataset(df, trim_text=True, remove_duplicates=False, missing_strategy="Keep missing values", standardize_headers=False):
    """Return a cleaned copy; never modify the uploaded source dataframe in place."""
    cleaned = df.copy(deep=True)

    if standardize_headers:
        names = []
        used = set()
        for i, col in enumerate(cleaned.columns, start=1):
            name = re.sub(r"[^0-9a-zA-Z]+", "_", str(col).strip()).strip("_").lower()
            name = name or f"column_{i}"
            candidate, suffix = name, 2
            while candidate in used:
                candidate = f"{name}_{suffix}"
                suffix += 1
            used.add(candidate)
            names.append(candidate)
        cleaned.columns = names

    if trim_text:
        for col in cleaned.select_dtypes(include=["object", "string"]).columns:
            cleaned[col] = cleaned[col].map(
                lambda value: value.strip() if isinstance(value, str) else value
            )
            cleaned[col] = cleaned[col].replace("", pd.NA)

    if remove_duplicates:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    if missing_strategy == "Drop rows with missing values":
        cleaned = cleaned.dropna().reset_index(drop=True)
    elif missing_strategy == "Fill numeric with median; text with mode":
        for col in cleaned.columns:
            if pd.api.types.is_numeric_dtype(cleaned[col]):
                if cleaned[col].notna().any():
                    cleaned[col] = cleaned[col].fillna(cleaned[col].median())
            elif pd.api.types.is_datetime64_any_dtype(cleaned[col]):
                # Dates are deliberately left missing rather than guessed.
                continue
            else:
                modes = cleaned[col].mode(dropna=True)
                if not modes.empty:
                    cleaned[col] = cleaned[col].fillna(modes.iloc[0])

    return cleaned


def show_result_visualization(question, result_df):
    """Show a chart when the executed result has useful categorical/numeric fields."""
    if result_df is None or result_df.empty or len(result_df.columns) < 2:
        return

    numeric_cols = result_df.select_dtypes(include="number").columns.tolist()
    category_cols = [
        col for col in result_df.columns
        if col not in numeric_cols and not pd.api.types.is_datetime64_any_dtype(result_df[col])
    ]
    date_cols = [col for col in result_df.columns if pd.api.types.is_datetime64_any_dtype(result_df[col])]
    q = question.lower()

    # Prefer columns that semantically match the question.
    metric_col = next(
        (c for c in numeric_cols if any(word in str(c).lower() for word in
         ("sales", "revenue", "profit", "amount", "quantity", "count", "total", "sum", "average", "avg"))),
        numeric_cols[-1] if numeric_cols else None
    )
    category_col = next(
        (c for c in category_cols if any(word in str(c).lower() for word in
         ("category", "city", "product", "region", "segment", "month", "year", "status"))),
        category_cols[0] if category_cols else None
    )

    if metric_col is None:
        st.caption("Chart skipped: the query result has no numeric metric to plot.")
        return

    chart_df = result_df.copy()
    if len(chart_df) > 100:
        chart_df = chart_df.head(100)
        st.caption("Chart uses the first 100 rows of the query result for readability.")

    st.subheader("Data Visualization")
    chart_kind = "bar"
    if any(word in q for word in ("trend", "over time", "by date", "monthly", "daily", "yearly")) and (date_cols or category_col):
        chart_kind = "line"
    elif any(word in q for word in ("percentage", "proportion", "share", "distribution", "pie")) and category_col:
        chart_kind = "pie"

    if chart_kind == "pie" and category_col and len(chart_df) <= 30:
        pie_df = chart_df[[category_col, metric_col]].dropna()
        if not pie_df.empty:
            fig = px.pie(pie_df, names=category_col, values=metric_col, title=f"{metric_col} by {category_col}")
            st.plotly_chart(fig, use_container_width=True)
            return

    if category_col:
        plot_df = chart_df[[category_col, metric_col]].dropna()
        if plot_df.empty:
            st.caption("Chart skipped: no usable category/metric pairs in this result.")
        elif chart_kind == "line":
            x_col = date_cols[0] if date_cols else category_col
            plot_df = chart_df[[x_col, metric_col]].dropna()
            fig = px.line(plot_df, x=x_col, y=metric_col, markers=True, title=f"{metric_col} by {x_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.bar(plot_df, x=category_col, y=metric_col, title=f"{metric_col} by {category_col}")
            st.plotly_chart(fig, use_container_width=True)
    elif len(chart_df) > 1:
        st.bar_chart(chart_df[metric_col])

def json_safe(value):
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def build_verified_facts(df):
    """Compute overview/statistics with Pandas, never with the language model."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [
        col for col in df.columns
        if col not in numeric_cols and not pd.api.types.is_datetime64_any_dtype(df[col])
    ]
    facts = {
        "dataset": {
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "column_names": [str(c) for c in df.columns],
            "duplicate_rows": int(df.duplicated().sum()),
            "missing_values_total": int(df.isna().sum().sum()),
            "missing_values_by_column": {
                str(col): int(count) for col, count in df.isna().sum().items()
            },
        },
        "numeric_summary": {},
        "categorical_summary": {},
    }

    for col in numeric_cols:
        s = df[col].dropna()
        facts["numeric_summary"][str(col)] = {
            "non_missing_count": int(df[col].notna().sum()),
            "missing_count": int(df[col].isna().sum()),
            "sum": None if s.empty else float(s.sum()),
            "mean": None if s.empty else float(s.mean()),
            "median": None if s.empty else float(s.median()),
            "min": None if s.empty else float(s.min()),
            "max": None if s.empty else float(s.max()),
            "sample_standard_deviation": (
                None if len(s) < 2 else float(s.std())
            ),
        }

    for col in categorical_cols:
        counts = df[col].dropna().astype(str).value_counts().head(10)
        facts["categorical_summary"][str(col)] = {
            "unique_non_missing_values": int(df[col].nunique(dropna=True)),
            "top_values": {str(k): int(v) for k, v in counts.items()},
        }
    return facts


def quote_identifier(name):
    return '"' + str(name).replace('"', '""') + '"'


def generate_sql(question, df):
    """Use deterministic SQL for data-quality questions; Ollama handles other questions."""
    q = question.lower()
    columns = list(df.columns)

    # Data-quality SQL is generated deterministically to avoid invalid LLM patterns.
    if any(term in q for term in ("missing value", "missing values", "null value", "null values", "missing data", "nulls")):
        parts = []
        for col in columns:
            ident = quote_identifier(col)
            label = str(col).replace("'", "''")
            parts.append(
                f"SELECT '{label}' AS column_name, "
                f"COUNT(*) FILTER (WHERE {ident} IS NULL) AS missing_values "
                f"FROM uploaded_data"
            )
        return " UNION ALL ".join(parts) if parts else "SELECT 'No columns' AS column_name, 0 AS missing_values"

    if any(term in q for term in ("duplicate row", "duplicate rows", "duplicated row", "duplicated rows", "duplicate records", "duplicate entries")):
        if not columns:
            return "SELECT 0 AS duplicate_rows"
        partition_cols = ", ".join(quote_identifier(col) for col in columns)
        return (
            "SELECT COUNT(*) AS duplicate_rows FROM ("
            f"SELECT ROW_NUMBER() OVER (PARTITION BY {partition_cols} ORDER BY {partition_cols}) AS rn "
            "FROM uploaded_data) AS duplicate_check WHERE rn > 1"
        )

    schema = []
    for col, dtype in df.dtypes.items():
        schema.append({"column": str(col), "pandas_dtype": str(dtype)})

    prompt = f"""Write exactly ONE DuckDB SQL query to answer the user's question about the uploaded dataset.
The only available table is uploaded_data.
Schema: {json.dumps(schema, ensure_ascii=False)}
Rules:
- Return SQL only, no Markdown fences or explanation.
- Use only SELECT (or a WITH clause followed by SELECT).
- Do not modify data or use INSERT, UPDATE, DELETE, DROP, CREATE, COPY, ATTACH, INSTALL, or LOAD.
- Use double quotes around column names when needed. Use only exact schema column names.
- Answer the user's actual question, not a related question.
- For "which category/city/product has the highest total sales", use SUM(sales_column), GROUP BY the requested entity, ORDER BY the SUM descending, LIMIT 1. Do NOT use MAX(sales_column) unless the user asks for the single largest transaction/record.
- For "highest sales value/transaction", use ORDER BY sales_column DESC LIMIT 1.
- For lowest/minimum questions, use the equivalent ascending order and LIMIT 1.
- For average, total, count, or grouped comparisons, use the corresponding SQL aggregate.
- For a percentage contribution/share by category, return category and SUM(metric) grouped by category; do NOT divide each group by its own group total. The chart layer can calculate percentage shares from the returned grouped totals.
- For a bar chart request, return grouped category + numeric aggregate suitable for a bar chart. For a pie chart request, return category + numeric aggregate suitable for a pie chart.
- Interpret normal conversational English questions, including polite phrasing, as questions about the uploaded data.
- Do not return an unsorted list when the user asks for highest/lowest; return only the winning row for a superlative question.
- Do not invent column names. If the question cannot be answered from this schema, return:
SELECT 'The uploaded dataset does not contain enough information to answer this question.' AS message
User question: {question}
"""
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You generate safe, read-only DuckDB SQL. Return only one SQL statement."},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0, "num_predict": 180},
        keep_alive="10m",
    )
    sql = response["message"]["content"].strip()
    sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s*```$", "", sql).strip()
    sql = sql.rstrip(";").strip()

    # Guard against multiple statements, non-read-only SQL, and other tables.
    if ";" in sql:
        raise ValueError("The model returned multiple SQL statements. Please try a simpler question.")
    if not re.match(r"^(SELECT|WITH)\b", sql, flags=re.IGNORECASE):
        raise ValueError("The generated query was not a read-only SELECT statement.")
    forbidden = r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|COPY|ATTACH|DETACH|INSTALL|LOAD|EXPORT|IMPORT|PRAGMA|CALL|EXECUTE)\b"
    if re.search(forbidden, sql, flags=re.IGNORECASE):
        raise ValueError("The generated SQL included a blocked operation.")
    # Queries must operate on the uploaded data only. Literal SELECT queries without a FROM are allowed.
    table_refs = re.findall(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)", sql, flags=re.IGNORECASE)
    if any(t.lower() != "uploaded_data" for t in table_refs):
        raise ValueError("The query referenced a table other than uploaded_data.")
    return sql


def execute_sql(df, sql):
    """Execute the read-only query against the current uploaded dataframe."""
    con = duckdb.connect(database=":memory:")
    try:
        con.register("uploaded_data", df)
        # DuckDB's query() runs the one validated SELECT statement.
        result = con.execute(sql).df()
        return result
    finally:
        con.close()


def repair_sql_with_ollama(question, df, failed_sql, error_message, attempt):
    """Ask the local model to repair SQL using the exact DuckDB error message."""
    schema = [{"column": str(col), "pandas_dtype": str(dtype)} for col, dtype in df.dtypes.items()]
    prompt = f"""The SQL query you generated failed in DuckDB. Fix it and return exactly ONE corrected, read-only DuckDB SQL query.
This is repair attempt {attempt} of 2.
User's original natural-language question: {question}
Available table: uploaded_data
Exact schema: {json.dumps(schema, ensure_ascii=False)}
Failed SQL:
{failed_sql}
DuckDB error:
{error_message}
Rules:
- Return SQL only, no Markdown fences or explanations.
- Use only SELECT or WITH followed by SELECT.
- Use only the exact column names in the schema and only uploaded_data as the table.
- Do not use INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, COPY, ATTACH, INSTALL, LOAD, PRAGMA, or multiple statements.
- Answer the original question, not a different question.
- For category percentage/share questions, return grouped category totals; do not divide a group by itself.
"""
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Repair DuckDB SQL. Return one safe SELECT query only."},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0, "num_predict": 180},
        keep_alive="10m",
    )
    sql = response["message"]["content"].strip()
    sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s*```$", "", sql).strip().rstrip(";").strip()
    if ";" in sql or not re.match(r"^(SELECT|WITH)\b", sql, flags=re.IGNORECASE):
        raise ValueError("The model returned an invalid or multi-statement repair query.")
    forbidden = r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|COPY|ATTACH|DETACH|INSTALL|LOAD|EXPORT|IMPORT|PRAGMA|CALL|EXECUTE)\b"
    if re.search(forbidden, sql, flags=re.IGNORECASE):
        raise ValueError("The repaired SQL included a blocked operation.")
    table_refs = re.findall(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)", sql, flags=re.IGNORECASE)
    if any(t.lower() != "uploaded_data" for t in table_refs):
        raise ValueError("The repaired query referenced a table other than uploaded_data.")
    return sql


def execute_with_sql_retries(question, df, initial_sql, max_retries=2):
    """Execute SQL and ask Ollama to repair it after DuckDB errors, up to 2 retries."""
    sql = initial_sql
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return sql, execute_sql(df, sql), attempt
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            sql = repair_sql_with_ollama(question, df, sql, str(exc), attempt + 1)
    raise RuntimeError(
        f"SQL execution failed after {max_retries} repair attempt(s). Last DuckDB error: {last_error}"
    )


def deterministic_result_answer(question, result_df):
    """Create a grounded answer for common superlative questions directly from SQL output."""
    if result_df.empty:
        return "The executed SQL query returned no rows, so there is no result to report."

    q = question.lower()
    wants_high = any(term in q for term in ("highest", "maximum", "max ", "largest", "most"))
    wants_low = any(term in q for term in ("lowest", "minimum", "min ", "smallest", "least"))
    if not (wants_high or wants_low):
        return None

    numeric_cols = result_df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return None

    # Prefer a numeric column with a metric-like name, otherwise use the last numeric column.
    preferred = [c for c in numeric_cols if any(k in str(c).lower() for k in
                 ("sales", "sum", "total", "amount", "quantity", "count", "max", "min", "avg", "mean"))]
    metric_col = preferred[-1] if preferred else numeric_cols[-1]
    values = pd.to_numeric(result_df[metric_col], errors="coerce")
    if values.notna().sum() == 0:
        return None
    idx = values.idxmax() if wants_high else values.idxmin()
    row = result_df.loc[idx]
    entity_cols = [c for c in result_df.columns if c != metric_col and not pd.api.types.is_numeric_dtype(result_df[c])]
    entity = ", ".join(f"{c}: {row[c]}" for c in entity_cols if pd.notna(row[c]))
    value = row[metric_col]
    try:
        value_text = f"{float(value):,.4f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        value_text = str(value)
    direction = "highest" if wants_high else "lowest"
    if entity:
        return f"Verified from the executed SQL result: the {direction} value in **{metric_col}** is **{value_text}** ({entity})."
    return f"Verified from the executed SQL result: the {direction} value in **{metric_col}** is **{value_text}**."


def explain_result(question, sql, result_df, facts):
    """Use deterministic answers for data-quality and superlative questions."""
    q = question.lower()
    if any(term in q for term in ("missing value", "missing values", "null value", "null values", "missing data", "nulls")):
        if {"column_name", "missing_values"}.issubset(result_df.columns):
            total = int(pd.to_numeric(result_df["missing_values"], errors="coerce").fillna(0).sum())
            return f"Verified from DuckDB: there are **{total} missing values** across the dataset. The table above shows the count for each column."
    if any(term in q for term in ("duplicate row", "duplicate rows", "duplicated row", "duplicated rows", "duplicate records", "duplicate entries")):
        if "duplicate_rows" in result_df.columns and not result_df.empty:
            count = int(result_df.iloc[0]["duplicate_rows"])
            return f"Verified from DuckDB: the dataset contains **{count} duplicate row(s)** (identical rows beyond their first occurrence)."

    result_preview = result_df.head(100).to_dict(orient="records")
    payload = {
        "question": question,
        "executed_sql": sql,
        "sql_result_columns": [str(c) for c in result_df.columns],
        "sql_result_rows": [
            {str(k): json_safe(v) for k, v in row.items()}
            for row in result_preview
        ],
        "sql_result_row_count": int(len(result_df)),
        "verified_dataset_overview": facts["dataset"],
    }
    verified_answer = deterministic_result_answer(question, result_df)
    if verified_answer:
        # For highest/lowest questions, prefer the deterministic result over a potentially
        # contradictory LLM explanation. This also saves a second model call.
        return verified_answer

    prompt = f"""Explain the user's question using ONLY the executed SQL result and verified overview below.
Do not invent, estimate, or recalculate any number. Do not claim a category/entity is highest or lowest unless that exact conclusion is supported by the result.
If the result does not clearly answer the question, say so. Keep it concise and in plain English.
Verified data:
{json.dumps(payload, ensure_ascii=False, default=str)}
"""
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Explain executed SQL results faithfully. Never guess, invent values, or contradict the verified result."},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0, "num_predict": 100},
        keep_alive="10m",
    )
    ai_text = response["message"]["content"].strip()

    return ai_text


with st.sidebar:
    st.header("Local AI Model")
    st.success(f"Ollama: {MODEL_NAME}")
    st.caption("Python/Pandas calculates verified statistics. Ollama generates SQL; DuckDB executes it. Failed SQL is automatically repaired up to 2 times.")

uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file",
    type=["csv", "xlsx"],
    help="Use a CSV/XLSX file with column headers in the first row.",
)

if uploaded_file is not None:
    df = preprocess_and_save(uploaded_file)
    if df is not None:
        if df.empty:
            st.warning("The uploaded file has column headers but no data rows.")
        else:
            original_df = df.copy(deep=True)
            st.subheader("Uploaded Data")
            st.dataframe(original_df, use_container_width=True)

            st.subheader("Data Cleaning")
            st.caption("Choose cleaning operations below. The original upload stays unchanged; analysis runs on the cleaned preview.")
            clean_col1, clean_col2 = st.columns(2)
            with clean_col1:
                trim_text = st.checkbox("Trim extra spaces in text", value=True)
                remove_duplicates = st.checkbox("Remove exact duplicate rows", value=False)
            with clean_col2:
                standardize_headers = st.checkbox("Standardize column names", value=False)
                missing_strategy = st.selectbox(
                    "Handle missing values",
                    [
                        "Keep missing values",
                        "Fill numeric with median; text with mode",
                        "Drop rows with missing values",
                    ],
                    index=0,
                )

            df = clean_dataset(
                original_df,
                trim_text=trim_text,
                remove_duplicates=remove_duplicates,
                missing_strategy=missing_strategy,
                standardize_headers=standardize_headers,
            )
            before_missing = int(original_df.isna().sum().sum())
            after_missing = int(df.isna().sum().sum())
            before_duplicates = int(original_df.duplicated().sum())
            after_duplicates = int(df.duplicated().sum())
            clean_m1, clean_m2, clean_m3, clean_m4 = st.columns(4)
            clean_m1.metric("Rows before", len(original_df))
            clean_m2.metric("Rows after", len(df), delta=len(df) - len(original_df))
            clean_m3.metric("Missing before", before_missing)
            clean_m4.metric("Missing after", after_missing, delta=after_missing - before_missing)
            with st.expander("Cleaned data preview", expanded=False):
                st.dataframe(df, use_container_width=True)
                st.caption(f"Exact duplicate rows: {before_duplicates} before → {after_duplicates} after.")
            if df.empty:
                st.warning("Cleaning left no rows to analyze. Change the cleaning options.")
                st.stop()

            facts = build_verified_facts(df)
            overview = facts["dataset"]
            st.subheader("Verified Dataset Overview")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", overview["row_count"])
            c2.metric("Columns", overview["column_count"])
            c3.metric("Missing values", overview["missing_values_total"])
            c4.metric("Duplicate rows", overview["duplicate_rows"])

            with st.expander("Missing values by column", expanded=True):
                missing_df = pd.DataFrame(
                    list(overview["missing_values_by_column"].items()),
                    columns=["Column", "Missing values"],
                )
                st.dataframe(missing_df, hide_index=True, use_container_width=True)

            if facts["numeric_summary"]:
                with st.expander("Pandas-calculated numeric summary", expanded=True):
                    numeric_df = pd.DataFrame.from_dict(facts["numeric_summary"], orient="index")
                    numeric_df.index.name = "Column"
                    st.dataframe(numeric_df.reset_index(), hide_index=True, use_container_width=True)
            else:
                st.info("No numeric columns were detected. Numeric summaries are unavailable.")

            if facts["categorical_summary"]:
                with st.expander("Categorical summary", expanded=True):
                    # Show a clean tabular overview instead of raw nested dictionaries.
                    category_overview_rows = []
                    top_value_rows = []
                    for col, summary in facts["categorical_summary"].items():
                        top_values = summary["top_values"]
                        first_item = next(iter(top_values.items()), (None, None))
                        category_overview_rows.append({
                            "Column": col,
                            "Unique non-missing values": summary["unique_non_missing_values"],
                            "Most frequent value": first_item[0] if first_item[0] is not None else "—",
                            "Frequency": first_item[1] if first_item[1] is not None else 0,
                        })
                        for value, count in top_values.items():
                            top_value_rows.append({
                                "Column": col,
                                "Category / Value": value,
                                "Count": count,
                            })

                    st.markdown("**Categorical column overview**")
                    st.dataframe(
                        pd.DataFrame(category_overview_rows),
                        hide_index=True,
                        use_container_width=True,
                    )
                    st.markdown("**Top values by categorical column (up to 10 each)**")
                    if top_value_rows:
                        st.dataframe(
                            pd.DataFrame(top_value_rows),
                            hide_index=True,
                            use_container_width=True,
                        )
                    else:
                        st.info("No non-missing categorical values were found.")

            st.subheader("Ask a question about your data")
            st.caption("Example: What is total sales? Which city has the highest total sales? Show average quantity by category.")
            st.info("For Excel .xlsx uploads, install openpyxl once if prompted: python -m pip install openpyxl")
            user_query = st.text_area(
                "Your question",
                placeholder="Type your question here...",
            )

            if st.button("Analyze with AI", type="primary"):
                if not user_query.strip():
                    st.warning("Please enter a question.")
                else:
                    try:
                        with st.spinner("Ollama is writing SQL..."):
                            sql_query = generate_sql(user_query.strip(), df)
                        with st.spinner("DuckDB is executing SQL; Ollama will repair a failed query up to 2 times..."):
                            sql_query, sql_result, repair_attempts = execute_with_sql_retries(
                                user_query.strip(), df, sql_query, max_retries=2
                            )
                        st.subheader("Generated SQL Query")
                        st.code(sql_query, language="sql")
                        if repair_attempts:
                            st.success(f"SQL was automatically repaired and executed successfully (repair attempts: {repair_attempts}/2).")
                        st.subheader("SQL Query Result")
                        st.dataframe(sql_result, use_container_width=True)
                        show_result_visualization(user_query.strip(), sql_result)

                        with st.spinner("Ollama is explaining the verified result..."):
                            answer = explain_result(user_query.strip(), sql_query, sql_result, facts)
                        st.subheader("AI Explanation")
                        st.markdown(answer)
                        st.caption(
                            "Trust note: the displayed SQL result is produced by DuckDB from your uploaded file. "
                            "The AI explanation is grounded in that executed result; verify the query for complex questions."
                        )
                    except Exception as exc:
                        st.error(f"Could not complete the analysis: {exc}")
                        st.info(
                            "Make sure Ollama is running and the model is installed. "
                            "In Command Prompt, run: ollama list"
                        )
else:
    st.info("Upload a CSV or Excel file to begin.")
