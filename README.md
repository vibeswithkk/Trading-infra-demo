# Quant Trading Infra Demo (Clean-Room)

Enterprise-style demo repository showing backend architecture patterns for an algorithmic trading system **without any proprietary logic**.

> ✅ Clean-room implementation. No private code or data included.

## Highlights
- Python 3.11+, Async-first
- Clean architecture: domain, infra, repository, services (router)
- Repository Pattern + Unit of Work (async SQLAlchemy)
- Config via Pydantic Settings (12-factor friendly)
- Structured JSON logging
- Simple Smart Order Router demo (mock brokers)
- FastAPI demo endpoints
- Pytest tests using in-memory SQLite

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn qtinfa.api.main:app --reload
```

### Run tests
```bash
pytest -q
```

> Optional: You can wire Postgres/Redis later via `docker-compose.yml` (stub provided).

## Architecture

```mermaid
flowchart LR
    UI[FastAPI] --> SOR[SmartOrderRouter]
    SOR --> Repo[Repository (SQLAlchemy Async)]
    SOR --> Brokers[Mock Brokers]
    Repo --> DB[(DB)]
```

## DISCLAIMER
This project is for **portfolio demonstration** only. It does **not** include your private strategies, datasets, broker credentials, or any business-sensitive code.