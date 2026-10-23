# Loan Review Assistant — rules for agents

## Project
Multi-step LangGraph assistant that helps a human lender review consumer loan
applications. Built from scratch on the public Home Credit Default Risk dataset.
No real client code or data.

## Non-negotiables
- The LLM NEVER makes or recommends the final credit decision. A human reviewer does.
- The LLM never chooses reasons. Reasons come from SHAP + a fixed reason-code
  dictionary + failed policy rules. The LLM only verbalizes them.
- Protected / prohibited attributes (CODE_GENDER, DAYS_BIRTH/age, NAME_FAMILY_STATUS)
  are NEVER model features. They are used ONLY in the fairness audit.
- Every decision writes exactly one append-only, hash-chained audit record.
- All tool and LLM I/O is validated with Pydantic v2. On validation failure: retry
  once, then route to human review with an error flag. Never silently coerce.

## Stack
Python 3.11, uv, LightGBM, shap, pandas/polars, Pydantic v2, LangGraph,
langchain-aws (Bedrock), FastAPI, SQLite (dev) / Postgres (prod), pytest, ruff, mypy.

## Layout
src/loan_review/{data,model,policy,llm,graph,audit,fairness,evals,api}/
configs/ (YAML: policy rules, reason codes, thresholds)
data/raw/ (gitignored), artifacts/ (gitignored), tests/, notebooks/

## Conventions
- Every config and artifact is versioned; versions are recorded in audit records.
- Deterministic seeds everywhere. No network calls in tests; use a FakeLLM.
- Run `ruff check`, `mypy src`, and `pytest -q` before declaring a task done.
- Don't add dependencies beyond the stack without saying why.
- If something in a prompt conflicts with this file, stop and ask.
