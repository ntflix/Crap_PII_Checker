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

    # Accept either a single object or an array of objects from the LLM.
    def _validate_and_extract(self, item: dict, columns, table, row) -> dict | None:
        if not isinstance(item, dict):
            return None
        parsed_table = item.get("table")
        raw_cols_with_pii = item.get("columns_with_pii", [])
        if parsed_table != table:
            return None
        valid_cols: list[dict[str, Any]] = []
        if isinstance(raw_cols_with_pii, list):
            for it in raw_cols_with_pii:
                if not isinstance(it, dict):
                    continue
                col_name = it.get("column")
                val = it.get("value")
                if isinstance(col_name, str) and col_name in columns:
                    valid_cols.append({"column": col_name, "value": val})
        if valid_cols:
            return {"table": table, "row": row, "columns_with_pii": valid_cols}
        return None

    def check_pii(
        self, table: str, rows: list[dict[str, Any]], columns: list[str]
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Send all sampled rows in a single LLM request. The LLM should return a JSON array
        where each object references the originating row by `row_index` and lists detected PII.
        Schema example:
        [{"table":"<table_name>","row_index":0,"columns_with_pii":[{"column":"<column_name>","value":"<pii_value>"}]}]
        """
        pii_rows: list[dict[str, Any]] = []

        prompt: str = self._build_prompt(rows, table, columns)
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
                            "For example, a column named 'username' with value '3fb841a7' is not PII, so use common sense to avoid false positives. "
                            "Similarly, IDs (e.g. learner_id, user_id, x_identity_value or similar) that are just numeric or alphanumeric codes should **not** be counted as PII. "
                            "Additionally, where a date of birth (or similar) column contains all the same value for every row, and especially if it "
                            "is fairly recent (e.g. 2025), that is likely to have been obfuscated and is not PII. "
                            "Do not count references, IDs, random alpha&/numeric strings, or similar as PII. "
                            "Respond with only valid JSON using this schema: "
                            '[{"table":"<table_name>","row_index":0,"columns_with_pii":[{"column":"<column_name>","value":"<pii_value>"}]}]'
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
            print(f"Error calling LLM for table {table}: {e}")
            return {"pii_rows": pii_rows}

        # parse JSON (best-effort)
        parsed = None
        try:
            parsed = json.loads(completion_text)
        except Exception:
            m = re.search(r"(\[.*\]|\{.*\})", completion_text, re.S)
            if m:
                try:
                    parsed = json.loads(m.group(1))
                except Exception:
                    parsed = None

        if not parsed:
            return {"pii_rows": pii_rows}

        # Accept either a single object or an array of objects from the LLM.
        items = parsed if isinstance(parsed, list) else [parsed]

        for item in items:
            if not isinstance(item, dict):
                continue
            # prefer explicit row_index
            row_index = item.get("row_index")
            if isinstance(row_index, int) and 0 <= row_index < len(rows):
                row_context = rows[row_index]
                validated = self._validate_and_extract(
                    item, columns, table, row_context
                )
                if validated:
                    pii_rows.append(validated)
            else:
                # fallback: try to validate against each provided row (match by column values)
                for candidate_row in rows:
                    validated = self._validate_and_extract(
                        item, columns, table, candidate_row
                    )
                    if validated:
                        pii_rows.append(validated)
                        break

        return {"pii_rows": pii_rows}

    def _build_prompt(
        self, rows: list[dict[str, Any]], table: str, columns: list[str]
    ) -> str:
        # include explicit column list and values to reduce ambiguity; use provided columns list when available
        cols_str = ", ".join(columns) if columns else ", ".join(list(row.keys()))
        rows_string: str = ""
        for row in rows:
            field_str = "\n".join([f"- {key}: {value}" for key, value in row.items()])
            rows_string += field_str + "\n"
        return (
            f"Table: {table}\n"
            f"Columns: {cols_str}\n"
            f"Rows: <ROWS>\n{rows_string}\n</ROWS>\n"
            "Task: Identify which columns, if any, contain PII. "
            "Respond in JSON format with an array of objects - exactly like this example:\n"
            '[{"table":"<table_name>","columns_with_pii":[{"column":"<column_name>","value":"<pii_value>"}]}]'
        )
