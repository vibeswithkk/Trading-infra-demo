from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    load_dotenv = lambda *args, **kwargs: None

try:
    import boto3
    from botocore.exceptions import ClientError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    boto3 = None
    ClientError = Exception

try:
    from google.cloud import secretmanager
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False
    secretmanager = None

try:
    import hvac
    VAULT_AVAILABLE = True
except ImportError:
    VAULT_AVAILABLE = False
    hvac = None

EnvironmentType = Literal["dev", "staging", "prod"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
DatabaseType = Literal["sqlite", "postgresql", "mysql", "oracle"]
SecretsProvider = Literal["env", "file", "aws", "gcp", "vault"]
CacheBackend = Literal["memory", "redis", "memcached"]
AuthProvider = Literal["jwt", "oauth2", "saml", "ldap"]
EncryptionAlgorithm = Literal["AES256", "RSA2048", "ECDSA"]
ComplianceFramework = Literal["SOX", "GDPR", "PCI_DSS", "HIPAA", "ISO27001"]

class DatabaseConfig(BaseModel):
    type: DatabaseType = "sqlite"
    host: str = "localhost"
    port: int = 5432
    name: str = "qtinfra"
    username: str = "qtinfra"
    password: SecretStr = Field(default=SecretStr(""))
    ssl_mode: str = "require"
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600
    query_timeout: int = 30
    connection_retries: int = 3
    backup_enabled: bool = True
    backup_schedule: str = "0 2 * * *"
    read_replicas: List[str] = Field(default_factory=list)
    charset: str = "utf8mb4"
    timezone: str = "UTC"
    
    @field_validator('port')
    @classmethod
    def validate_port(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v
    
    def get_url(self) -> str:
        if self.type == "sqlite":
            return f"sqlite+aiosqlite:///{self.name}.db"
        elif self.type == "postgresql":
            return f"postgresql+asyncpg://{self.username}:{self.password.get_secret_value()}@{self.host}:{self.port}/{self.name}"
        elif self.type == "mysql":
            return f"mysql+aiomysql://{self.username}:{self.password.get_secret_value()}@{self.host}:{self.port}/{self.name}"
        elif self.type == "oracle":
            return f"oracle+cx_oracle://{self.username}:{self.password.get_secret_value()}@{self.host}:{self.port}/{self.name}"
        else:
            raise ValueError(f"Unsupported database type: {self.type}")

class RedisConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: SecretStr = Field(default=SecretStr(""))
    ssl_enabled: bool = False
    connection_pool_size: int = 10
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True
    max_connections: int = 50
    cluster_mode: bool = False
    sentinel_hosts: List[str] = Field(default_factory=list)
    master_name: str = "mymaster"
    
    def get_url(self) -> str:
        protocol = "rediss" if self.ssl_enabled else "redis"
        auth = f":{self.password.get_secret_value()}@" if self.password.get_secret_value() else ""
        return f"{protocol}://{auth}{self.host}:{self.port}/{self.db}"

class SecurityConfig(BaseModel):
    secret_key: SecretStr = Field(default=SecretStr(""))
    encryption_algorithm: EncryptionAlgorithm = "AES256"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 30
    password_min_length: int = 12
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_numbers: bool = True
    password_require_symbols: bool = True
    session_timeout_minutes: int = 60
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    enable_2fa: bool = True
    allowed_origins: List[str] = Field(default_factory=lambda: ["*"])
    ssl_required: bool = True
    hsts_max_age: int = 31536000
    content_security_policy: str = "default-src 'self'"
    rate_limit_per_minute: int = 100
    
    @field_validator('secret_key')
    @classmethod
    def validate_secret_key(cls, v: SecretStr) -> SecretStr:
        if len(v.get_secret_value()) < 32:
            raise ValueError("Secret key must be at least 32 characters long")
        return v

class ObservabilityConfig(BaseModel):
    enabled: bool = True
    metrics_enabled: bool = True
    tracing_enabled: bool = True
    logging_enabled: bool = True
    health_checks_enabled: bool = True
    prometheus_port: int = 9090
    jaeger_endpoint: str = "http://localhost:14268/api/traces"
    grafana_dashboard_enabled: bool = True
    alert_manager_enabled: bool = True
    log_retention_days: int = 30
    metrics_retention_days: int = 90
    trace_sampling_rate: float = 0.1
    custom_metrics: Dict[str, Any] = Field(default_factory=dict)
    alert_rules: List[Dict[str, Any]] = Field(default_factory=list)
    dashboard_configs: Dict[str, Any] = Field(default_factory=dict)

class PerformanceConfig(BaseModel):
    api_rate_limit: int = 1000
    max_connections: int = 1000
    connection_timeout: int = 30
    read_timeout: int = 30
    cache_enabled: bool = True
    cache_backend: CacheBackend = "redis"
    cache_ttl_seconds: int = 3600
    cache_max_size: int = 10000
    enable_compression: bool = True
    compression_level: int = 6
    worker_processes: int = 4
    worker_threads: int = 100
    async_pool_size: int = 20
    batch_size: int = 1000
    bulk_insert_size: int = 10000
    query_optimization: bool = True
    connection_pooling: bool = True
    prepared_statements: bool = True
    
    @field_validator('api_rate_limit')
    @classmethod
    def validate_rate_limit(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("API rate limit must be positive")
        return v

class ComplianceConfig(BaseModel):
    frameworks: List[ComplianceFramework] = Field(default_factory=list)
    audit_logging_enabled: bool = True
    data_encryption_enabled: bool = True
    pii_detection_enabled: bool = True
    gdpr_compliance: bool = False
    hipaa_compliance: bool = False
    sox_compliance: bool = False
    pci_dss_compliance: bool = False
    data_retention_days: int = 2555
    audit_trail_retention_days: int = 7300
    anonymization_enabled: bool = True
    right_to_erasure_enabled: bool = True
    consent_management_enabled: bool = True
    privacy_by_design: bool = True
    data_minimization: bool = True
    purpose_limitation: bool = True

class FeatureFlags(BaseModel):
    enable_authentication: bool = True
    enable_authorization: bool = True
    enable_rate_limiting: bool = True
    enable_caching: bool = True
    enable_monitoring: bool = True
    enable_audit_logging: bool = True
    enable_data_validation: bool = True
    enable_encryption: bool = True
    enable_backup: bool = True
    enable_real_time_sync: bool = False
    enable_advanced_analytics: bool = False
    enable_machine_learning: bool = False
    enable_auto_scaling: bool = False
    enable_disaster_recovery: bool = True
    enable_multi_tenant: bool = False
    enable_federation: bool = False
    enable_blockchain: bool = False
    enable_quantum_security: bool = False
    enable_edge_computing: bool = False
    experimental_features: Dict[str, bool] = Field(default_factory=dict)

class SecretsManager:
    def __init__(self, provider: SecretsProvider, config: Dict[str, Any]):
        self.provider = provider
        self.config = config
        self._client = None
    
    def get_secret(self, key: str) -> str:
        if self.provider == "env":
            return os.getenv(key, "")
        elif self.provider == "file":
            return self._get_from_file(key)
        elif self.provider == "aws" and AWS_AVAILABLE:
            return self._get_from_aws(key)
        elif self.provider == "gcp" and GCP_AVAILABLE:
            return self._get_from_gcp(key)
        elif self.provider == "vault" and VAULT_AVAILABLE:
            return self._get_from_vault(key)
        else:
            return ""
    
    def _get_from_file(self, key: str) -> str:
        secrets_file = self.config.get("secrets_file", "secrets.json")
        if Path(secrets_file).exists():
            with open(secrets_file) as f:
                secrets = json.load(f)
                return secrets.get(key, "")
        return ""
    
    def _get_from_aws(self, key: str) -> str:
        if not self._client:
            self._client = boto3.client("secretsmanager", region_name=self.config.get("region", "us-east-1"))
        try:
            response = self._client.get_secret_value(SecretId=key)
            return response["SecretString"]
        except ClientError:
            return ""
    
    def _get_from_gcp(self, key: str) -> str:
        if not self._client:
            self._client = secretmanager.SecretManagerServiceClient()
        try:
            name = f"projects/{self.config['project_id']}/secrets/{key}/versions/latest"
            response = self._client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
        except Exception:
            return ""
    
    def _get_from_vault(self, key: str) -> str:
        if not self._client:
            self._client = hvac.Client(
                url=self.config.get("url", "http://localhost:8200"),
                token=self.config.get("token")
            )
        try:
            response = self._client.secrets.kv.v2.read_secret_version(path=key)
            return response["data"]["data"]["value"]
        except Exception:
            return ""

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="QTINFRA_",
        extra="ignore",
        case_sensitive=False,
        env_nested_delimiter="__",
        secrets_dir="/run/secrets" if os.path.exists("/run/secrets") else None
    )
    
    app_name: str = "qtinfra-enterprise"
    app_version: str = "1.0.0"
    app_description: str = "Enterprise-grade Qt Infrastructure Platform"
    environment: EnvironmentType = "dev"
    debug: bool = False
    testing: bool = False
    
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    
    log_level: LogLevel = "INFO"
    log_format: str = "json"
    log_file: Optional[str] = None
    log_max_size: int = 100
    log_backup_count: int = 5
    
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    compliance: ComplianceConfig = Field(default_factory=ComplianceConfig)
    feature_flags: FeatureFlags = Field(default_factory=FeatureFlags)
    
    secrets_provider: SecretsProvider = "env"
    secrets_config: Dict[str, Any] = Field(default_factory=dict)
    
    timezone: str = "UTC"
    locale: str = "en_US.UTF-8"
    currency: str = "USD"
    
    external_apis: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    integrations: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    startup_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    config_version: str = "2.0.0"
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v: str) -> str:
        if v not in ["dev", "staging", "prod"]:
            raise ValueError("Environment must be dev, staging, or prod")
        return v
    
    @field_validator('port')
    @classmethod
    def validate_port(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v
    
    @model_validator(mode='after')
    def configure_environment_defaults(self):
        if self.environment == "dev":
            self.database.type = "sqlite"
            self.database.name = "qtinfra_dev"
            self.debug = True
            self.reload = True
            self.log_level = "DEBUG"
            self.security.ssl_required = False
            self.observability.trace_sampling_rate = 1.0
        elif self.environment == "staging":
            self.database.type = "postgresql"
            self.database.host = "staging-db.internal"
            self.database.name = "qtinfra_staging"
            self.debug = False
            self.reload = False
            self.log_level = "INFO"
            self.security.ssl_required = True
            self.observability.trace_sampling_rate = 0.5
        elif self.environment == "prod":
            self.database.type = "postgresql"
            self.database.host = "prod-db.aws-rds.com"
            self.database.name = "qtinfra_production"
            self.debug = False
            self.reload = False
            self.log_level = "WARNING"
            self.security.ssl_required = True
            self.observability.trace_sampling_rate = 0.1
            self.compliance.gdpr_compliance = True
            self.compliance.sox_compliance = True
        return self
    
    @model_validator(mode='after')
    def load_secrets(self):
        secrets_manager = SecretsManager(self.secrets_provider, self.secrets_config)
        
        if self.database.password.get_secret_value() == "":
            db_password = secrets_manager.get_secret("database_password")
            if db_password:
                self.database.password = SecretStr(db_password)
        
        if self.redis.password.get_secret_value() == "":
            redis_password = secrets_manager.get_secret("redis_password")
            if redis_password:
                self.redis.password = SecretStr(redis_password)
        
        if self.security.secret_key.get_secret_value() == "":
            secret_key = secrets_manager.get_secret("secret_key")
            if secret_key:
                self.security.secret_key = SecretStr(secret_key)
        
        return self
    
    @property
    def db_url(self) -> str:
        return self.database.get_url()
    
    @property
    def redis_url(self) -> str:
        return self.redis.get_url()
    
    @property
    def is_production(self) -> bool:
        return self.environment == "prod"
    
    @property
    def is_development(self) -> bool:
        return self.environment == "dev"
    
    @property
    def is_staging(self) -> bool:
        return self.environment == "staging"
    
    def to_dict(self, include_secrets: bool = False) -> Dict[str, Any]:
        data = self.model_dump(mode="json")
        if not include_secrets:
            self._mask_secrets(data)
        return data
    
    def _mask_secrets(self, data: Any) -> None:
        if isinstance(data, dict):
            for key, value in data.items():
                if "password" in key.lower() or "secret" in key.lower() or "token" in key.lower():
                    data[key] = "***MASKED***"
                else:
                    self._mask_secrets(value)
        elif isinstance(data, list):
            for item in data:
                self._mask_secrets(item)
    
    def export_config(self, format_type: Literal["json", "yaml"] = "json", include_secrets: bool = False) -> str:
        data = self.to_dict(include_secrets=include_secrets)
        if format_type == "json":
            return json.dumps(data, indent=2, default=str)
        elif format_type == "yaml":
            return yaml.dump(data, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError("Format must be 'json' or 'yaml'")
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Settings':
        data = json.loads(json_str)
        return cls(**data)
    
    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'Settings':
        data = yaml.safe_load(yaml_str)
        return cls(**data)
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'Settings':
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        content = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            return cls.from_json(content)
        elif path.suffix.lower() in [".yaml", ".yml"]:
            return cls.from_yaml(content)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def validate_database_url(self) -> bool:
        try:
            parsed = urlparse(self.db_url)
            return all([parsed.scheme, parsed.netloc or parsed.path])
        except Exception:
            return False
    
    def health_check(self) -> Dict[str, Any]:
        return {
            "app_name": self.app_name,
            "version": self.app_version,
            "environment": self.environment,
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": (datetime.now(timezone.utc) - self.startup_time).total_seconds(),
            "database_configured": self.validate_database_url(),
            "debug_mode": self.debug,
            "feature_flags_count": len([k for k, v in self.feature_flags.model_dump().items() if v]),
            "compliance_frameworks": self.compliance.frameworks
        }

def load_environment_file() -> None:
    if not DOTENV_AVAILABLE:
        return
    
    env = os.getenv("QTINFRA_ENVIRONMENT", "dev")
    env_files = [
        f".env.{env}",
        ".env.local",
        ".env"
    ]
    
    for env_file in env_files:
        if Path(env_file).exists():
            load_dotenv(env_file, override=False)

@lru_cache()
def get_settings() -> Settings:
    load_environment_file()
    return Settings()

settings = get_settings()