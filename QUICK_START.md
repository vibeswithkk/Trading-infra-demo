# Quick Deployment Guide

## 🚀 One-Command Demo

**Get the system running in under 2 minutes:**

```bash
# Clone and run (copy-paste friendly)
git clone https://github.com/yourusername/trading-infra-demo.git && \
cd trading-infra-demo && \
python -m venv .venv && \
source .venv/bin/activate && \
pip install -r requirements.txt && \
python examples/elite_trading_demo.py
```

**Windows PowerShell:**
```powershell
git clone https://github.com/yourusername/trading-infra-demo.git; cd trading-infra-demo; python -m venv .venv; .venv\Scripts\activate; pip install -r requirements.txt; python examples/elite_trading_demo.py
```

## 📊 Expected Results

You should see:
```
🚀 ELITE TRADING INFRASTRUCTURE DEMO
====================================
✅ Custom exception hierarchy
✅ Domain-driven design
✅ Transaction safety
✅ Validation layers
✅ Execution algorithms
✅ Security & compliance
🎉 DEMO COMPLETED SUCCESSFULLY!
```

## 🔧 Development Server

**Start the API server:**
```bash
uvicorn qtinfra.api.main:app --reload
```

**Access points:**
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/healthz

## 🧪 Run Tests

```bash
# Complete test suite
pytest -v

# Performance benchmarks  
python tests/benchmark_logging.py

# System verification
python verify_observability.py
```

## 🐳 Docker Deployment

```bash
# Build and run with Docker
docker-compose up -d

# View logs
docker-compose logs -f
```

## ☁️ Cloud Deployment

**AWS/GCP/Azure ready:**
- Kubernetes manifests included
- Environment-based configuration
- Health checks and metrics endpoints
- Auto-scaling configuration

## 📈 Monitoring Setup

**Prometheus + Grafana:**
```bash
# Metrics endpoint
curl http://localhost:8000/metrics

# Health check
curl http://localhost:8000/healthz
```

## 🔍 Troubleshooting

**Common issues:**

1. **Python version**: Requires Python 3.11+
2. **Dependencies**: Run `pip install -r requirements.txt`  
3. **Port conflicts**: Change port with `--port 8001`
4. **Database**: Uses SQLite by default (no setup needed)

## 💡 Pro Tips

- Use `--reload` for development
- Check `logs/` for detailed logs
- Monitor `/metrics` for performance
- Use `/healthz` for load balancer checks

---

**Ready to explore enterprise-grade trading infrastructure!** 🏆