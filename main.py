import argparse
from dataclasses import dataclass
from typing import Any
from aiconfig import PIICheckerConfig
import mysql.connector
import random
from llm_pii_checker import LLMPIIChecker


@dataclass
class MySQLConfig:
    host: str
    user: str
    password: str
    port: int = 3306


@dataclass
class AppConfig:
    mysql_config: MySQLConfig
    sample_size: int = 100


class MySQLPIIInspector:
    def __init__(self, app_config: AppConfig):
        self.mysql_config = app_config.mysql_config
        self.llm_checker = LLMPIIChecker(config=PIICheckerConfig())
        self.sample_size = app_config.sample_size

    def _get_connection(self):
        return mysql.connector.connect(
            host=self.mysql_config.host,
            user=self.mysql_config.user,
            password=self.mysql_config.password,
            port=self.mysql_config.port,
        )

    def inspect_all(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SHOW DATABASES")
        databases = [row[0] for row in cursor.fetchall()]

        for db_name in databases:
            if db_name in ["information_schema", "mysql", "performance_schema", "sys"]:
                continue
            cursor.execute(f"USE `{db_name}`")
            cursor.execute("SHOW TABLES")
            tables = [row[0] for row in cursor.fetchall()]
            for table in tables:
                sampled_rows, columns = self._sample_table(cursor, table)
                # pass table name, sampled rows, and explicit column names so the LLM can reference table + columns
                pii_result = self.llm_checker.check_pii(table, sampled_rows, columns)
                print(
                    f"DB: {db_name}, Table: {table} -- PII Found: {len(pii_result['pii_rows'])}"
                )
                for pii_row in pii_result["pii_rows"]:
                    print(
                        f"\tRow: {pii_row['row']}, Columns with PII: {pii_row['columns_with_pii']}"
                    )

        cursor.close()
        conn.close()

    def _sample_table(
        self, cursor, table_name: str
    ) -> list[dict[str, Any]]:  # pyright: ignore[reportExplicitAny]
        cursor.execute(f"SELECT * FROM `{table_name}` LIMIT {self.sample_size}")
        columns = list(cursor.column_names)
        rows = cursor.fetchall()
        random.shuffle(rows)  # Shuffle and take up to sample_size if more are present
        rows = rows[: self.sample_size]
        return [dict(zip(columns, row)) for row in rows], columns


# ----- Command-Line Interface -----
def main():
    parser = argparse.ArgumentParser(description="MySQL PII Inspector")
    parser.add_argument("--host", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--port", type=int, default=3306)
    parser.add_argument("--sample-size", type=int, default=100)
    args = parser.parse_args()

    mysql_cfg = MySQLConfig(
        host=args.host, user=args.user, password=args.password, port=args.port
    )
    app_cfg = AppConfig(mysql_config=mysql_cfg, sample_size=args.sample_size)
    inspector = MySQLPIIInspector(app_cfg)
    inspector.inspect_all()


if __name__ == "__main__":
    main()
