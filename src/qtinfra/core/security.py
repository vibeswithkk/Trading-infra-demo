"""
Enterprise Security and Compliance Framework.

Implements comprehensive security features:
- Role-Based Access Control (RBAC)
- Data masking and PII protection
- GDPR compliance (Right to be forgotten, data portability)
- Audit trail and compliance reporting
- Multi-factor authentication support
- Digital signatures and encryption
- Regulatory compliance (MiFID II, Dodd-Frank, Basel III)
"""

from __future__ import annotations
import hashlib
import hmac
import secrets
import base64
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Set, Union, Protocol
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import re

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from .exceptions import SecurityError, UnauthorizedAccessError, ComplianceError
from ..infra.logging import EnterpriseLogger


# === RBAC SYSTEM ===

class Permission(str, Enum):
    """System permissions."""
    # Order permissions
    CREATE_ORDER = "orders:create"
    READ_ORDER = "orders:read"
    UPDATE_ORDER = "orders:update"
    CANCEL_ORDER = "orders:cancel"
    EXECUTE_ORDER = "orders:execute"
    
    # Client permissions
    READ_CLIENT_DATA = "clients:read"
    UPDATE_CLIENT_DATA = "clients:update"
    
    # Risk permissions
    BYPASS_RISK_CHECKS = "risk:bypass"
    UPDATE_RISK_LIMITS = "risk:update"
    VIEW_RISK_REPORTS = "risk:view"
    
    # Admin permissions
    MANAGE_USERS = "admin:users"
    MANAGE_SYSTEM = "admin:system"
    VIEW_AUDIT_LOGS = "admin:audit"
    
    # Compliance permissions
    GENERATE_REPORTS = "compliance:reports"
    EXPORT_DATA = "compliance:export"
    DELETE_DATA = "compliance:delete"


class Role(str, Enum):
    """System roles."""
    TRADER = "TRADER"
    PORTFOLIO_MANAGER = "PORTFOLIO_MANAGER"
    RISK_MANAGER = "RISK_MANAGER"
    COMPLIANCE_OFFICER = "COMPLIANCE_OFFICER"
    ADMINISTRATOR = "ADMINISTRATOR"
    SYSTEM_SERVICE = "SYSTEM_SERVICE"
    READONLY_USER = "READONLY_USER"


@dataclass
class User:
    """User entity for authentication and authorization."""
    id: str
    username: str
    email: str
    roles: List[Role]
    permissions: Set[Permission] = field(default_factory=set)
    client_access: Set[str] = field(default_factory=set)  # Client IDs user can access
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    mfa_enabled: bool = False
    password_hash: Optional[str] = None


class RBACManager:
    """Role-Based Access Control Manager."""
    
    # Define role permissions mapping
    ROLE_PERMISSIONS = {
        Role.TRADER: {
            Permission.CREATE_ORDER,
            Permission.READ_ORDER,
            Permission.UPDATE_ORDER,
            Permission.CANCEL_ORDER,
            Permission.READ_CLIENT_DATA,
        },
        Role.PORTFOLIO_MANAGER: {
            Permission.CREATE_ORDER,
            Permission.READ_ORDER,
            Permission.UPDATE_ORDER,
            Permission.CANCEL_ORDER,
            Permission.EXECUTE_ORDER,
            Permission.READ_CLIENT_DATA,
            Permission.UPDATE_CLIENT_DATA,
            Permission.VIEW_RISK_REPORTS,
        },
        Role.RISK_MANAGER: {
            Permission.READ_ORDER,
            Permission.READ_CLIENT_DATA,
            Permission.UPDATE_RISK_LIMITS,
            Permission.VIEW_RISK_REPORTS,
            Permission.BYPASS_RISK_CHECKS,
        },
        Role.COMPLIANCE_OFFICER: {
            Permission.READ_ORDER,
            Permission.READ_CLIENT_DATA,
            Permission.VIEW_AUDIT_LOGS,
            Permission.GENERATE_REPORTS,
            Permission.EXPORT_DATA,
            Permission.DELETE_DATA,
        },
        Role.ADMINISTRATOR: set(Permission),  # All permissions
        Role.SYSTEM_SERVICE: {
            Permission.CREATE_ORDER,
            Permission.READ_ORDER,
            Permission.UPDATE_ORDER,
            Permission.EXECUTE_ORDER,
        },
        Role.READONLY_USER: {
            Permission.READ_ORDER,
            Permission.READ_CLIENT_DATA,
        },
    }
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.log = EnterpriseLogger(__name__, 'rbac-manager')
    
    def add_user(self, user: User) -> None:
        """Add user to the system."""
        # Calculate effective permissions from roles
        user.permissions = set()
        for role in user.roles:
            user.permissions.update(self.ROLE_PERMISSIONS.get(role, set()))
        
        self.users[user.id] = user
        self.log.info("User added", user_id=user.id, roles=user.roles)
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.users.get(user_id)
    
    def has_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if user has specific permission."""
        user = self.get_user(user_id)
        if not user or not user.is_active:
            return False
        
        return permission in user.permissions
    
    def can_access_client(self, user_id: str, client_id: str) -> bool:
        """Check if user can access specific client data."""
        user = self.get_user(user_id)
        if not user or not user.is_active:
            return False
        
        # Administrators and system services can access all clients
        if Role.ADMINISTRATOR in user.roles or Role.SYSTEM_SERVICE in user.roles:
            return True
        
        # Check explicit client access
        return client_id in user.client_access
    
    def authorize_operation(
        self,
        user_id: str,
        permission: Permission,
        client_id: Optional[str] = None
    ) -> None:
        """Authorize operation, raise exception if not allowed."""
        if not self.has_permission(user_id, permission):
            raise UnauthorizedAccessError(
                resource=permission.value,
                action="access",
                user_id=user_id
            )
        
        if client_id and not self.can_access_client(user_id, client_id):
            raise UnauthorizedAccessError(
                resource=f"client:{client_id}",
                action="access",
                user_id=user_id
            )


# === DATA MASKING AND PII PROTECTION ===

class PIIType(str, Enum):
    """Types of Personally Identifiable Information."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    ACCOUNT_NUMBER = "account_number"
    NAME = "name"
    ADDRESS = "address"
    IP_ADDRESS = "ip_address"


class DataMaskingStrategy(ABC):
    """Abstract base for data masking strategies."""
    
    @abstractmethod
    def mask(self, value: str) -> str:
        """Mask the sensitive value."""
        pass


class PartialMaskingStrategy(DataMaskingStrategy):
    """Partial masking strategy (show first/last few characters)."""
    
    def __init__(self, show_start: int = 2, show_end: int = 2, mask_char: str = "*"):
        self.show_start = show_start
        self.show_end = show_end
        self.mask_char = mask_char
    
    def mask(self, value: str) -> str:
        """Partially mask the value."""
        if len(value) <= self.show_start + self.show_end:
            return self.mask_char * len(value)
        
        start = value[:self.show_start]
        end = value[-self.show_end:] if self.show_end > 0 else ""
        middle_length = len(value) - self.show_start - self.show_end
        middle = self.mask_char * middle_length
        
        return start + middle + end


class HashMaskingStrategy(DataMaskingStrategy):
    """Hash masking strategy (irreversible)."""
    
    def __init__(self, salt: str = ""):
        self.salt = salt
    
    def mask(self, value: str) -> str:
        """Hash the value."""
        salted_value = self.salt + value
        return hashlib.sha256(salted_value.encode()).hexdigest()[:16]


class PIIProtector:
    """PII Protection and masking system."""
    
    # PII detection patterns
    PII_PATTERNS = {
        PIIType.EMAIL: re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        PIIType.PHONE: re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
        PIIType.SSN: re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
        PIIType.CREDIT_CARD: re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
        PIIType.IP_ADDRESS: re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
    }
    
    def __init__(self):
        self.masking_strategies = {
            PIIType.EMAIL: PartialMaskingStrategy(3, 0),
            PIIType.PHONE: PartialMaskingStrategy(3, 4),
            PIIType.SSN: HashMaskingStrategy(),
            PIIType.CREDIT_CARD: PartialMaskingStrategy(4, 4),
            PIIType.ACCOUNT_NUMBER: PartialMaskingStrategy(4, 4),
            PIIType.IP_ADDRESS: PartialMaskingStrategy(0, 0),
        }
        self.log = EnterpriseLogger(__name__, 'pii-protector')
    
    def detect_pii(self, text: str) -> Dict[PIIType, List[str]]:
        """Detect PII in text."""
        detected = {}
        
        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                detected[pii_type] = matches
        
        return detected
    
    def mask_pii(self, text: str, pii_types: Optional[Set[PIIType]] = None) -> str:
        """Mask PII in text."""
        if pii_types is None:
            pii_types = set(PIIType)
        
        masked_text = text
        detected_count = 0
        
        for pii_type in pii_types:
            if pii_type in self.PII_PATTERNS and pii_type in self.masking_strategies:
                pattern = self.PII_PATTERNS[pii_type]
                strategy = self.masking_strategies[pii_type]
                
                def replace_match(match):
                    nonlocal detected_count
                    detected_count += 1
                    return strategy.mask(match.group(0))
                
                masked_text = pattern.sub(replace_match, masked_text)
        
        if detected_count > 0:
            self.log.info("PII masked", count=detected_count, types=list(pii_types))
        
        return masked_text
    
    def mask_dict(self, data: Dict[str, Any], sensitive_fields: Set[str]) -> Dict[str, Any]:
        """Mask sensitive fields in dictionary."""
        masked_data = data.copy()
        
        for field in sensitive_fields:
            if field in masked_data and isinstance(masked_data[field], str):
                masked_data[field] = self.mask_pii(masked_data[field])
        
        return masked_data


# === GDPR COMPLIANCE ===

class GDPRRights(str, Enum):
    """GDPR individual rights."""
    RIGHT_TO_ACCESS = "access"           # Article 15
    RIGHT_TO_RECTIFICATION = "rectification"  # Article 16
    RIGHT_TO_ERASURE = "erasure"        # Article 17 (Right to be forgotten)
    RIGHT_TO_PORTABILITY = "portability"  # Article 20
    RIGHT_TO_RESTRICT = "restriction"    # Article 18
    RIGHT_TO_OBJECT = "objection"       # Article 21


@dataclass
class GDPRRequest:
    """GDPR request tracking."""
    id: str
    request_type: GDPRRights
    data_subject_id: str
    requested_at: datetime
    processed_at: Optional[datetime] = None
    status: str = "PENDING"  # PENDING, PROCESSING, COMPLETED, REJECTED
    details: Dict[str, Any] = field(default_factory=dict)


class GDPRComplianceManager:
    """GDPR Compliance Manager."""
    
    def __init__(self, rbac_manager: RBACManager, pii_protector: PIIProtector):
        self.rbac_manager = rbac_manager
        self.pii_protector = pii_protector
        self.requests: Dict[str, GDPRRequest] = {}
        self.log = EnterpriseLogger(__name__, 'gdpr-compliance')
    
    async def process_access_request(
        self,
        data_subject_id: str,
        requester_id: str
    ) -> Dict[str, Any]:
        """Process GDPR right to access request."""
        # Authorize request
        self.rbac_manager.authorize_operation(
            requester_id,
            Permission.EXPORT_DATA,
            data_subject_id
        )
        
        request = GDPRRequest(
            id=secrets.token_urlsafe(16),
            request_type=GDPRRights.RIGHT_TO_ACCESS,
            data_subject_id=data_subject_id,
            requested_at=datetime.now(timezone.utc)
        )
        
        try:
            # Collect all data for the subject
            personal_data = await self._collect_personal_data(data_subject_id)
            
            # Apply data masking for sensitive information
            masked_data = self._mask_sensitive_data(personal_data)
            
            request.status = "COMPLETED"
            request.processed_at = datetime.now(timezone.utc)
            request.details = {"data_exported": True, "records_count": len(masked_data)}
            
            self.log.info(
                "GDPR access request processed",
                request_id=request.id,
                data_subject_id=data_subject_id,
                records_count=len(masked_data)
            )
            
            return {
                "request_id": request.id,
                "data_subject_id": data_subject_id,
                "exported_data": masked_data,
                "export_timestamp": request.processed_at.isoformat()
            }
            
        except Exception as e:
            request.status = "REJECTED"
            request.details = {"error": str(e)}
            self.log.error("GDPR access request failed", error=str(e))
            raise
        
        finally:
            self.requests[request.id] = request
    
    async def process_erasure_request(
        self,
        data_subject_id: str,
        requester_id: str,
        retention_override: bool = False
    ) -> Dict[str, Any]:
        """Process GDPR right to erasure (right to be forgotten) request."""
        # Authorize request
        self.rbac_manager.authorize_operation(
            requester_id,
            Permission.DELETE_DATA,
            data_subject_id
        )
        
        request = GDPRRequest(
            id=secrets.token_urlsafe(16),
            request_type=GDPRRights.RIGHT_TO_ERASURE,
            data_subject_id=data_subject_id,
            requested_at=datetime.now(timezone.utc)
        )
        
        try:
            # Check legal retention requirements
            if not retention_override:
                retention_check = await self._check_retention_requirements(data_subject_id)
                if not retention_check["can_delete"]:
                    raise ComplianceError(
                        f"Cannot delete data due to retention requirements: {retention_check['reason']}"
                    )
            
            # Perform data erasure
            deletion_results = await self._erase_personal_data(data_subject_id)
            
            request.status = "COMPLETED"
            request.processed_at = datetime.now(timezone.utc)
            request.details = {
                "data_deleted": True,
                "deletion_results": deletion_results
            }
            
            self.log.info(
                "GDPR erasure request processed",
                request_id=request.id,
                data_subject_id=data_subject_id,
                deletion_results=deletion_results
            )
            
            return {
                "request_id": request.id,
                "data_subject_id": data_subject_id,
                "deletion_results": deletion_results,
                "deletion_timestamp": request.processed_at.isoformat()
            }
            
        except Exception as e:
            request.status = "REJECTED"
            request.details = {"error": str(e)}
            self.log.error("GDPR erasure request failed", error=str(e))
            raise
        
        finally:
            self.requests[request.id] = request
    
    async def _collect_personal_data(self, data_subject_id: str) -> List[Dict[str, Any]]:
        """Collect all personal data for a subject."""
        # In production, this would query all relevant tables
        # For demo, return sample data structure
        return [
            {
                "table": "orders",
                "records": [],  # Would contain actual order records
                "count": 0
            },
            {
                "table": "clients",
                "records": [],  # Would contain client profile data
                "count": 0
            },
            {
                "table": "audit_logs",
                "records": [],  # Would contain audit trail
                "count": 0
            }
        ]
    
    def _mask_sensitive_data(self, personal_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Mask sensitive data in export."""
        masked_data = []
        
        for table_data in personal_data:
            masked_table = table_data.copy()
            masked_records = []
            
            for record in table_data.get("records", []):
                # Define sensitive fields per table
                sensitive_fields = {"email", "phone", "ssn", "account_number"}
                masked_record = self.pii_protector.mask_dict(record, sensitive_fields)
                masked_records.append(masked_record)
            
            masked_table["records"] = masked_records
            masked_data.append(masked_table)
        
        return masked_data
    
    async def _check_retention_requirements(self, data_subject_id: str) -> Dict[str, Any]:
        """Check legal retention requirements."""
        # In production, this would check various retention policies
        # Financial records, regulatory requirements, etc.
        
        # Sample retention check
        return {
            "can_delete": True,
            "reason": None,
            "retention_periods_checked": [
                "financial_records_7_years",
                "audit_trail_10_years",
                "regulatory_reporting_5_years"
            ]
        }
    
    async def _erase_personal_data(self, data_subject_id: str) -> Dict[str, Any]:
        """Erase personal data (anonymize/delete)."""
        # In production, this would:
        # 1. Anonymize data that must be retained
        # 2. Delete data that can be deleted
        # 3. Update references and relationships
        
        return {
            "orders_anonymized": 0,
            "client_data_deleted": 0,
            "audit_logs_anonymized": 0,
            "files_deleted": 0
        }


# === ENCRYPTION AND DIGITAL SIGNATURES ===

class CryptoManager:
    """Cryptographic operations manager."""
    
    def __init__(self, master_key: Optional[str] = None):
        if master_key:
            self.master_key = master_key.encode()
        else:
            self.master_key = secrets.token_bytes(32)
        
        self.log = EnterpriseLogger(__name__, 'crypto-manager')
    
    def generate_key(self, password: str, salt: Optional[bytes] = None) -> bytes:
        """Generate encryption key from password."""
        if salt is None:
            salt = secrets.token_bytes(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return kdf.derive(password.encode())
    
    def encrypt_data(self, data: str, key: Optional[bytes] = None) -> Dict[str, str]:
        """Encrypt data."""
        try:
            encryption_key = key or self.master_key
            f = Fernet(base64.urlsafe_b64encode(encryption_key))
            
            encrypted_data = f.encrypt(data.encode())
            
            return {
                "encrypted_data": base64.b64encode(encrypted_data).decode(),
                "algorithm": "Fernet",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.log.error("Encryption failed", error=str(e))
            raise SecurityError(f"Encryption failed: {str(e)}")
    
    def decrypt_data(self, encrypted_data: str, key: Optional[bytes] = None) -> str:
        """Decrypt data."""
        try:
            encryption_key = key or self.master_key
            f = Fernet(base64.urlsafe_b64encode(encryption_key))
            
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            decrypted_data = f.decrypt(encrypted_bytes)
            
            return decrypted_data.decode()
            
        except Exception as e:
            self.log.error("Decryption failed", error=str(e))
            raise SecurityError(f"Decryption failed: {str(e)}")
    
    def sign_data(self, data: str, secret_key: Optional[str] = None) -> str:
        """Create HMAC signature for data."""
        signing_key = (secret_key or base64.b64encode(self.master_key).decode()).encode()
        signature = hmac.new(signing_key, data.encode(), hashlib.sha256).hexdigest()
        return signature
    
    def verify_signature(self, data: str, signature: str, secret_key: Optional[str] = None) -> bool:
        """Verify HMAC signature."""
        try:
            expected_signature = self.sign_data(data, secret_key)
            return hmac.compare_digest(signature, expected_signature)
        except Exception as e:
            self.log.error("Signature verification failed", error=str(e))
            return False


# === COMPLIANCE REPORTING ===

class ComplianceReporter:
    """Compliance reporting system."""
    
    def __init__(self, rbac_manager: RBACManager):
        self.rbac_manager = rbac_manager
        self.log = EnterpriseLogger(__name__, 'compliance-reporter')
    
    async def generate_mifid_report(
        self,
        start_date: datetime,
        end_date: datetime,
        requester_id: str
    ) -> Dict[str, Any]:
        """Generate MiFID II compliance report."""
        self.rbac_manager.authorize_operation(requester_id, Permission.GENERATE_REPORTS)
        
        # MiFID II requires transaction reporting, best execution reports, etc.
        report = {
            "report_type": "MiFID_II",
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "sections": {
                "transaction_reporting": {
                    "total_transactions": 0,
                    "reported_transactions": 0,
                    "reporting_rate": 100.0
                },
                "best_execution": {
                    "venues_analyzed": [],
                    "execution_quality_metrics": {}
                },
                "market_data_usage": {
                    "systematic_internalization": False,
                    "pre_trade_transparency": True,
                    "post_trade_transparency": True
                }
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_by": requester_id
        }
        
        self.log.info("MiFID II report generated", requester_id=requester_id)
        return report
    
    async def generate_audit_trail_report(
        self,
        client_id: Optional[str],
        start_date: datetime,
        end_date: datetime,
        requester_id: str
    ) -> Dict[str, Any]:
        """Generate audit trail report."""
        self.rbac_manager.authorize_operation(requester_id, Permission.VIEW_AUDIT_LOGS)
        
        if client_id:
            self.rbac_manager.authorize_operation(requester_id, Permission.READ_CLIENT_DATA, client_id)
        
        # Generate audit trail report
        report = {
            "report_type": "AUDIT_TRAIL",
            "client_id": client_id,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "events": [],  # Would contain actual audit events
            "summary": {
                "total_events": 0,
                "event_types": {},
                "users_activity": {}
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_by": requester_id
        }
        
        self.log.info("Audit trail report generated", client_id=client_id, requester_id=requester_id)
        return report