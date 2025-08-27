#!/usr/bin/env python3

import json
import os
from pathlib import Path

import sys
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from qtinfra.infra.config import Settings, get_settings, load_environment_file

def demo_configuration():
    print("=== Enterprise Configuration System Demo ===\n")
    
    print("1. Loading configuration with auto-environment detection...")
    settings = get_settings()
    
    print(f"Environment: {settings.environment}")
    print(f"App Name: {settings.app_name}")
    print(f"Database Type: {settings.database.type}")
    print(f"Database URL: {settings.db_url}")
    print(f"Debug Mode: {settings.debug}")
    print(f"Log Level: {settings.log_level}")
    
    print("\n2. Configuration Export (JSON):")
    config_json = settings.export_config("json", include_secrets=False)
    print(config_json[:500] + "..." if len(config_json) > 500 else config_json)
    
    print("\n3. Health Check:")
    health = settings.health_check()
    print(json.dumps(health, indent=2))
    
    print("\n4. Feature Flags Status:")
    flags = settings.feature_flags.model_dump()
    enabled_flags = [k for k, v in flags.items() if v]
    print(f"Enabled Features: {enabled_flags}")
    
    print("\n5. Security Configuration:")
    print(f"SSL Required: {settings.security.ssl_required}")
    print(f"2FA Enabled: {settings.security.enable_2fa}")
    print(f"Rate Limit: {settings.security.rate_limit_per_minute}/min")
    
    print("\n6. Performance Settings:")
    print(f"API Rate Limit: {settings.performance.api_rate_limit}")
    print(f"Max Connections: {settings.performance.max_connections}")
    print(f"Cache Enabled: {settings.performance.cache_enabled}")
    
    print("\n7. Compliance Frameworks:")
    print(f"GDPR: {settings.compliance.gdpr_compliance}")
    print(f"SOX: {settings.compliance.sox_compliance}")
    print(f"Audit Logging: {settings.compliance.audit_logging_enabled}")

if __name__ == "__main__":
    demo_configuration()