from openai import AzureOpenAI
from typing import Any
import json
import re

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
        For each row, call the LLM with an explicit prompt containing:
          - Table: <table_name>
          - Row: - column: value
        and ask the model to return JSON:
          {"table": "<table_name>", "columns_with_pii": [{"column": "<column_name>", "value": "<pii_value>"}]}
        """
        pii_rows: list[dict[str, Any]] = []

        for row in rows:
            prompt: str = self._build_prompt(row, table, columns)

            response = self.client.chat.completions.create(
                model=self.config.api_engine,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a data privacy expert. For each row, return a single JSON object listing the columns "
                            "containing PII, the table name, and the actual PII value. Only include columns where you detect PII. "
                            "Respond with only valid JSON using this schema: "
                            '{"table":"<table_name>","columns_with_pii":[{"column":"<column_name>","value":"<pii_value>"}]}'
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                stream=self.config.stream,
                temperature=0.0,
                max_tokens=256,
            )

            # Extract text from response (best-effort)
            try:
                response_message = response.choices[0].message  # type: ignore
                completion_text: str = response_message.content  # type: ignore
            except Exception:
                completion_text = str(response)

            # Try to parse JSON directly; if fails, extract JSON substring
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

            if parsed and isinstance(parsed, dict):
                cols_with_pii = parsed.get("columns_with_pii", [])
                if cols_with_pii:
                    pii_rows.append(
                        {"table": table, "row": row, "columns_with_pii": cols_with_pii}
                    )

        return {"pii_rows": pii_rows}

    def _build_prompt(self, row: dict[str, Any], table: str, columns: list[str]) -> str:
        # include explicit column list and values to reduce ambiguity
        cols_str = ", ".join(columns) if (columns := list(row.keys())) else ""
        field_str = "\n".join([f"- {key}: {value}" for key, value in row.items()])
        return (
            f"Table: {table}\n"
            f"Columns: {cols_str}\n"
            f"Row\n{field_str}\n\n"
            "Task: Identify which columns, if any, contain PII. "
            "Respond in JSON format exactly like this example:\n"
            '{"table":"<table_name>","columns_with_pii":[{"column":"<column_name>","value":"<pii_value>"}]}'
        )
