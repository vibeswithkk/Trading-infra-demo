from __future__ import annotations
import json
import jwt
import time
import uuid
from typing import Callable, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from .logging import set_trace_context, set_user_context, set_request_context, clear_all_context

class LoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, jwt_secret_key: Optional[str] = None, jwt_algorithm: str = "HS256"):
        super().__init__(app)
        self.jwt_secret_key = jwt_secret_key
        self.jwt_algorithm = jwt_algorithm
    
    async def dispatch(self, request: Request, call_next: Callable):
        start_time = time.time()
        
        trace_id = request.headers.get("x-request-id") or request.headers.get("x-trace-id") or str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        
        set_trace_context(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=None,
            operation_name=f"{request.method} {request.url.path}"
        )
        
        set_request_context(
            request_id=trace_id,
            method=request.method,
            path=str(request.url.path),
            query_params=dict(request.query_params),
            user_agent=request.headers.get("user-agent"),
            remote_addr=request.client.host if request.client else None
        )
        
        user_context = self._extract_user_context(request)
        if user_context:
            set_user_context(**user_context)
        
        try:
            response = await call_next(request)
            
            response.headers["x-trace-id"] = trace_id
            response.headers["x-span-id"] = span_id
            
            duration = time.time() - start_time
            
            set_request_context(
                duration=duration,
                status_code=response.status_code,
                response_size=len(response.body) if hasattr(response, 'body') else 0
            )
            
            return response
            
        except Exception as e:
            set_request_context(
                duration=time.time() - start_time,
                status_code=500,
                error=str(e)
            )
            raise
        finally:
            clear_all_context()
    
    def _extract_user_context(self, request: Request) -> Optional[dict]:
        user_context = {}
        
        authorization = request.headers.get("authorization")
        if authorization and authorization.startswith("Bearer ") and self.jwt_secret_key:
            token = authorization[7:]
            try:
                payload = jwt.decode(token, self.jwt_secret_key, algorithms=[self.jwt_algorithm])
                user_context.update({
                    "user_id": payload.get("sub") or payload.get("user_id"),
                    "username": payload.get("username"),
                    "email": payload.get("email"),
                    "roles": payload.get("roles", []),
                    "session_id": payload.get("session_id")
                })
            except jwt.InvalidTokenError:
                pass
        
        session_id = request.headers.get("x-session-id") or request.cookies.get("session_id")
        if session_id:
            user_context["session_id"] = session_id
        
        user_id = request.headers.get("x-user-id")
        if user_id:
            user_context["user_id"] = user_id
        
        return user_context if user_context else None

class CorrelationIdMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, header_name: str = "x-correlation-id"):
        super().__init__(app)
        self.header_name = header_name
    
    async def dispatch(self, request: Request, call_next: Callable):
        correlation_id = request.headers.get(self.header_name) or str(uuid.uuid4())
        
        set_trace_context(correlation_id=correlation_id)
        
        response = await call_next(request)
        response.headers[self.header_name] = correlation_id
        
        return response