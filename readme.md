quick and dirty script to sample MySQL database and check for PII using an LLM.

connects to a mysql database, samples 5 rows of all databases' tables, and checks for PII using an LLM connector. class-based, entirely configurable with structs/classes for configuration values and takes args for mysql details (host, user, pass, optional port).

obvs only use this w trusted LLM providers/local.

[blog post about it](https://felixweber.co.uk/scripting/database/cybersecurity/personal_data/obfuscation/2025/10/23/automated-personal-data-checking.html)
