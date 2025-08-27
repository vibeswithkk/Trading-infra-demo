import asyncio
import os
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager

from qtinfra.infra.logging import (
    EnterpriseLogger, setup_enterprise_logging, 
    set_trace_context, set_user_context, health_check
)
from qtinfra.infra.middleware import LoggingMiddleware, CorrelationIdMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    config = {
        'service_name': 'trading-api',
        'version': '2.0.0',
        'environment': os.getenv('ENVIRONMENT', 'development'),
        'destinations': [
            {'type': 'console', 'level': 'INFO'},
            {'type': 'file', 'filename': 'logs/app.log', 'level': 'DEBUG'},
            {'type': 'kafka', 'topic': 'trading-logs', 'bootstrap_servers': ['localhost:9092']},
            {'type': 'elasticsearch', 'hosts': ['localhost:9200'], 'index_name': 'trading-logs'}
        ],
        'async_logging': True,
        'security': {
            'enable_encryption': True,
            'enable_signing': True,
            'enable_scrubbing': True
        },
        'integrations': {
            'uvicorn': True,
            'fastapi': True,
            'sqlalchemy': True
        }
    }
    
    setup_enterprise_logging('trading-api', config)
    
    EnterpriseLogger.configure_sampling(0.8)
    EnterpriseLogger.configure_rate_limiting(1000, 1.0)
    
    logger = EnterpriseLogger(__name__, "trading-api", config)
    logger.info("Trading API starting up", version="2.0.0")
    
    yield
    
    logger.info("Trading API shutting down")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    LoggingMiddleware,
    jwt_secret_key=os.getenv('JWT_SECRET_KEY', 'your-secret-key')
)
app.add_middleware(CorrelationIdMiddleware)

logger = EnterpriseLogger(__name__, 'trading-api')

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "Trading API v2.0"}

@app.get("/orders/{order_id}")
async def get_order(order_id: str, request: Request):
    with logger.span("get_order_operation", order_id=order_id) as span:
        logger.info("Retrieving order", order_id=order_id)
        
        try:
            order_data = {"id": order_id, "status": "filled", "amount": 1000.50}
            
            logger.info("Order retrieved successfully", 
                       order_id=order_id, 
                       status=order_data["status"])
            
            return order_data
            
        except Exception as e:
            logger.error("Failed to retrieve order", 
                        order_id=order_id, 
                        error=str(e))
            raise

@app.post("/orders")
async def create_order(order_data: dict):
    set_user_context(user_id="trader123", session_id="sess_456")
    
    logger.audit("CREATE", "order", "trader123", 
                amount=order_data.get("amount"),
                symbol=order_data.get("symbol"))
    
    with logger.span("create_order_operation") as span:
        order_id = f"ord_{int(asyncio.get_event_loop().time())}"
        
        logger.info("Creating new order", 
                   order_id=order_id,
                   symbol=order_data.get("symbol"),
                   amount=order_data.get("amount"))
        
        # Simulate order processing
        await asyncio.sleep(0.1)
        
        logger.info("Order created successfully", order_id=order_id)
        
        return {"order_id": order_id, "status": "created"}

@app.get("/health")
async def health_endpoint():
    health_status = health_check()
    
    logger.info("Health check requested", 
               status=health_status["status"],
               metrics=health_status["metrics"])
    
    return health_status

@app.get("/metrics")
async def metrics_endpoint():
    metrics = EnterpriseLogger.get_metrics()
    
    return {
        "logging_metrics": metrics,
        "system_info": {
            "service": "trading-api",
            "version": "2.0.0"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    print("Starting Trading API with Enterprise Logging")
    print("Features enabled:")
    print("✓ Distributed tracing")
    print("✓ PII scrubbing") 
    print("✓ AES-GCM encryption")
    print("✓ Prometheus metrics")
    print("✓ Multiple destinations (Console, File, Kafka, Elasticsearch)")
    print("✓ Circuit breaker protection")
    print("✓ Rate limiting")
    print("✓ Async logging with queue management")
    print("✓ FastAPI middleware integration")
    print("✓ Schema validation")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None)