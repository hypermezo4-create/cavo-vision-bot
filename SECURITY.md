# Security

Do not publish Telegram tokens, Gemini keys, Google credentials, the private catalog,
or the generated recognition index.

Report a vulnerability privately to the repository owner. Include reproduction steps,
affected files, and the expected impact. Do not open a public issue containing secrets,
customer photos, private inventory data, or an exploitable proof of concept.

Production defaults are fail-closed: `ALLOW_PUBLIC=false`, explicit numeric user IDs are
required, disabled inventory rows are hidden, and uncertain image matches require human
selection instead of forced classification.
