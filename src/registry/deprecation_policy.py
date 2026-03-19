"""
Deprecation Policy Loader and Validator
Phase 6.8: Model Deprecation & Retirement Policy
Reads: config/deprecation_policy.yaml
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError


class DeprecationPolicyConfig(BaseModel):
    """Pydantic model for deprecation policy configuration"""

    version: str
    environment: str = "development"

    class DeprecationConfig(BaseModel):
        warning_period_days: int = Field(ge=0, default=30)
        require_migration_guide: bool = True
        allowed_reasons: list[str] = Field(default_factory=list)

    class RetirementConfig(BaseModel):
        soft_delete_default: bool = True
        audit_retention_days: int = Field(ge=1, default=365)
        require_approval: bool = False
        allowed_approvers: list[str] = Field(default_factory=list)

    class NotificationConfig(BaseModel):
        notify_on_deprecate: bool = True
        channels: list[str] = Field(default_factory=list)

    deprecation: DeprecationConfig = Field(default_factory=DeprecationConfig)
    retirement: RetirementConfig = Field(default_factory=RetirementConfig)
    notifications: NotificationConfig = Field(default_factory=NotificationConfig)


class PolicyViolationError(Exception):
    """Raised when a deprecation/retirement request violates policy"""

    def __init__(self, message: str, field: str | None = None, value: Any = None):
        self.field = field
        self.value = value
        super().__init__(message)


class DeprecationPolicy:
    """
    Policy enforcement engine for model deprecation and retirement

    Usage:
        policy = DeprecationPolicy.load()
        policy.validate_deprecation_request(model_name, version, reason, migration_guide)
        policy.validate_retirement_request(model_name, version, actor)
    """

    _instance: Optional["DeprecationPolicy"] = None
    _config_path = Path("config/deprecation_policy.yaml")

    def __init__(self, config: DeprecationPolicyConfig):
        self.config = config

    @classmethod
    def load(cls, config_path: Path | None = None) -> "DeprecationPolicy":
        """Load policy from YAML configuration file"""
        path = config_path or cls._config_path

        if not path.exists():
            # Return default policy if config file missing (iR&D fallback)
            return cls(DeprecationPolicyConfig(version="0.1.0", environment="development"))

        with open(path, encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        try:
            config = DeprecationPolicyConfig(**raw_config)
            return cls(config)
        except ValidationError as e:
            # Log warning and fall back to defaults for iR&D
            print(f"Warning: Policy config validation failed: {e}")
            print("Using default policy configuration")
            return cls(DeprecationPolicyConfig(version="0.1.0", environment="development"))

    @classmethod
    def get_instance(cls) -> "DeprecationPolicy":
        """Singleton accessor for policy instance"""
        if cls._instance is None:
            cls._instance = cls.load()
        return cls._instance

    @classmethod
    def reload(cls) -> None:
        """Force reload of policy configuration (useful for testing)"""
        cls._instance = cls.load()

    def validate_deprecation_request(
        self,
        model_name: str,
        version: str,
        reason: str,
        migration_guide: str | None = None,
        effective_date: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Validate a deprecation request against policy rules

        Returns:
            Dict with 'valid': bool and 'warnings': List[str]

        Raises:
            PolicyViolationError: If request violates hard policy constraints
        """
        warnings = []

        # Rule 1: Reason must be in allowed list (if configured)
        if self.config.deprecation.allowed_reasons:
            if reason not in self.config.deprecation.allowed_reasons:
                raise PolicyViolationError(
                    f"Reason '{reason}' not in allowed reasons: {self.config.deprecation.allowed_reasons}",
                    field="reason",
                    value=reason,
                )

        # Rule 2: Migration guide required if policy enforces
        if self.config.deprecation.require_migration_guide and not migration_guide:
            raise PolicyViolationError(
                "Migration guide is required by policy but not provided",
                field="migration_guide",
                value=None,
            )

        # Rule 3: Effective date must be in future (if provided)
        if effective_date and effective_date < datetime.now():
            warnings.append(f"Effective date {effective_date} is in the past")

        # Rule 4: Warning period calculation
        if effective_date:
            warning_end = effective_date + timedelta(
                days=self.config.deprecation.warning_period_days
            )
            warnings.append(
                f"Deprecation warning period: {datetime.now().date()} to {warning_end.date()} "
                f"({self.config.deprecation.warning_period_days} days)"
            )

        return {
            "valid": True,
            "warnings": warnings,
            "warning_period_days": self.config.deprecation.warning_period_days,
            "requires_migration_guide": self.config.deprecation.require_migration_guide,
        }

    def validate_retirement_request(
        self, model_name: str, version: str, actor: str, soft_delete: bool | None = None
    ) -> dict[str, Any]:
        """
        Validate a retirement request against policy rules

        Returns:
            Dict with 'valid': bool and 'requirements': Dict[str, Any]

        Raises:
            PolicyViolationError: If request violates hard policy constraints
        """
        requirements = {}

        # Rule 1: Soft delete default
        if soft_delete is None:
            requirements["soft_delete"] = self.config.retirement.soft_delete_default
        else:
            requirements["soft_delete"] = soft_delete

        # Rule 2: Approval required (if configured)
        if self.config.retirement.require_approval:
            if actor not in self.config.retirement.allowed_approvers:
                raise PolicyViolationError(
                    f"Actor '{actor}' not authorized for retirement. "
                    f"Allowed: {self.config.retirement.allowed_approvers}",
                    field="actor",
                    value=actor,
                )
            requirements["approval_status"] = "pending"

        # Rule 3: Audit retention period
        requirements["audit_retention_until"] = (
            (datetime.now() + timedelta(days=self.config.retirement.audit_retention_days))
            .date()
            .isoformat()
        )

        return {
            "valid": True,
            "requirements": requirements,
            "audit_retention_days": self.config.retirement.audit_retention_days,
        }

    def is_model_deprecated(self, deprecation_date: datetime) -> bool:
        """Check if a model's deprecation warning period has elapsed"""
        warning_end = deprecation_date + timedelta(days=self.config.deprecation.warning_period_days)
        return datetime.now() > warning_end

    def get_retirement_eligibility(
        self, deprecation_date: datetime, model_name: str
    ) -> dict[str, Any]:
        """
        Determine if a deprecated model is eligible for retirement

        Returns:
            Dict with eligibility status and timeline info
        """
        warning_end = deprecation_date + timedelta(days=self.config.deprecation.warning_period_days)
        now = datetime.now()

        if now < warning_end:
            days_remaining = (warning_end - now).days
            return {
                "eligible": False,
                "reason": f"Warning period not elapsed ({days_remaining} days remaining)",
                "eligible_after": warning_end.isoformat(),
            }

        return {
            "eligible": True,
            "reason": "Warning period elapsed",
            "deprecated_since": deprecation_date.isoformat(),
            "warning_period_days": self.config.deprecation.warning_period_days,
        }
