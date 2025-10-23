from openai import AzureOpenAI
from typing import Any
import json
import re
import concurrent.futures
from aiconfig import AIConfiguration


class LLMPIIChecker:
    def __init__(self, config: AIConfiguration):
        self.config: AIConfiguration = config
        self.client: AzureOpenAI = AzureOpenAI(
            azure_endpoint=self.config.api_base,
            api_key=self.config.api_key,
            api_version=self.config.api_version,
        )

    def check_pii(
        self, table: str, rows: list[dict[str, Any]], columns: list[str]
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Parallelized: submit one request per row using ThreadPoolExecutor.
        Tune max_workers to control concurrency (default min(8, len(rows))).
        """
        pii_rows: list[dict[str, Any]] = []

        def call_for_row(row: dict[str, Any]) -> dict | None:
            prompt: str = self._build_prompt(row, table, columns)
            try:
                response = self.client.chat.completions.create(
                    model=self.config.api_engine,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a data privacy expert. For each row, return a single JSON object listing the columns "
                                "containing PII, the table name, and the actual PII value. Only include columns where you detect PII. "
                                "PII does not include where column names may be a name or email, but the value is clearly not a PII value. "
                                "For example, a column named 'name' with value '3fb841a7' is not PII. "
                                "Additionally, where a date of birth (or similar) column contains all the same value for every row, and especially if it "
                                "is fairly recent (e.g. 2025), that is likely to have been obfuscated and is not PII. "
                                "Do not count references as PII. "
                                "Respond with only valid JSON using this schema: "
                                '{"table":"<table_name>","columns_with_pii":[{"column":"<column_name>","value":"<pii_value>"}]}'
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    stream=self.config.stream,
                    max_completion_tokens=16384,
                    reasoning_effort="low",
                )
                try:
                    response_message = response.choices[0].message  # type: ignore
                    completion_text: str = response_message.content  # type: ignore
                except Exception:
                    completion_text = str(response)
            except Exception as e:
                # request-level failure -> treat as no-detect for this row
                print(f"Error calling LLM for row {row}: {e}")
                return None

            # parse JSON (best-effort)
            parsed = None
            try:
                parsed = json.loads(completion_text)
            except Exception:
                m = re.search(r"(\{.*\})", completion_text, re.S)
                if m:
                    try:
                        parsed = json.loads(m.group(1))
                    except Exception:
                        parsed = None

            if not parsed or not isinstance(parsed, dict):
                return None

            # Validate parsed output: table must match and columns must be known
            parsed_table = parsed.get("table")
            raw_cols_with_pii = parsed.get("columns_with_pii", [])

            if parsed_table != table:
                return None

            valid_cols: list[dict[str, Any]] = []
            if isinstance(raw_cols_with_pii, list):
                for item in raw_cols_with_pii:
                    if not isinstance(item, dict):
                        continue
                    col_name = item.get("column")
                    val = item.get("value")
                    if isinstance(col_name, str) and col_name in columns:
                        valid_cols.append({"column": col_name, "value": val})

            if valid_cols:
                return {"table": table, "row": row, "columns_with_pii": valid_cols}
            return None

        # Tune concurrency: cap workers to avoid blasting the API
        max_workers = min(8, max(1, len(rows)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as exc:
            futures = [exc.submit(call_for_row, r) for r in rows]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    res = fut.result()
                    if res:
                        pii_rows.append(res)
                except Exception:
                    # keep going on per-task failure
                    continue

        return {"pii_rows": pii_rows}

    def _build_prompt(self, row: dict[str, Any], table: str, columns: list[str]) -> str:
        # include explicit column list and values to reduce ambiguity; use provided columns list when available
        cols_str = ", ".join(columns) if columns else ", ".join(list(row.keys()))
        field_str = "\n".join([f"- {key}: {value}" for key, value in row.items()])
        return (
            f"Table: {table}\n"
            f"Columns: {cols_str}\n"
            f"Row\n{field_str}\n\n"
            "Task: Identify which columns, if any, contain PII. "
            "Respond in JSON format exactly like this example:\n"
            '{"table":"<table_name>","columns_with_pii":[{"column":"<column_name>","value":"<pii_value>"}]}'
        )
