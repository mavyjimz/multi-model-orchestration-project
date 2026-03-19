"""
Pydantic schemas for Model Registry API
Extended for Phase 6.8: Deprecation & Retirement Policy
"""
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelRegisterRequest(BaseModel):
    """Request schema for registering a new model version"""
    name: str = Field(..., min_length=1, description="Model name")
    version: str = Field(..., description="Version as string (e.g., '2' or '1.2.0')")
    source: str = Field(..., description="MLflow source URI (runs:/<run-id>/path)")
    run_id: str | None = Field(None, description="MLflow run ID")
    description: str | None = Field(None, description="Model description")
    metadata: dict[str, Any] | None = Field(
        default_factory=dict,
        description="Custom metadata tags"
    )

    @field_validator('version')
    @classmethod
    def validate_version_format(cls, v: str) -> str:
        """Accept integer strings or semver; normalize to string for MLflow"""
        normalized = v.lstrip('v')
        if not normalized.replace('.', '').isdigit():
            raise ValueError('Version must be numeric or semver format')
        return normalized

    model_config = ConfigDict(populate_by_name=True)


class ModelPromoteRequest(BaseModel):
    """Request schema for promoting model to stage"""
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Target version")
    stage: str = Field(..., pattern="^(Staging|Production|Archived)$")
    comment: str | None = Field(None, description="Audit comment for transition")

    model_config = ConfigDict(populate_by_name=True)


class ModelInfo(BaseModel):
    """Response schema for model details - EXTENDED FOR DEPRECATION"""
    name: str
    version: str
    stage: str | None = None
    source: str | None = None
    run_id: str | None = None
    description: str | None = None
    created_timestamp: int | None = None
    last_updated_timestamp: int | None = None
    metadata: dict[str, Any] | None = Field(default_factory=dict)

    # Phase 6.8: Deprecation & Retirement fields
    deprecation_status: str | None = Field(
        None,
        pattern="^(active|deprecated|retired|archived)$",
        description="Current deprecation lifecycle state"
    )
    deprecation_date: datetime | None = Field(
        None,
        description="Timestamp when model was marked deprecated"
    )
    deprecation_reason: str | None = Field(
        None,
        description="Reason for deprecation (required if deprecated)"
    )
    migration_guide: str | None = Field(
        None,
        description="Path to migration documentation"
    )
    retirement_scheduled: bool | None = Field(
        False,
        description="Whether retirement is scheduled"
    )
    retirement_date: datetime | None = Field(
        None,
        description="Planned retirement date"
    )

    model_config = ConfigDict(from_attributes=True, populate_by_name=True, protected_namespaces=())


class ModelListResponse(BaseModel):
    """Response schema for listing models"""
    models: list[ModelInfo]
    total: int
    page: int = 1
    page_size: int = 50

    model_config = ConfigDict(populate_by_name=True)


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = "healthy"
    mlflow_connected: bool
    registry_version: str = "0.6.7"
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(populate_by_name=True)


class DeprecationRequest(BaseModel):
    """Request schema for deprecating a model - Phase 6.8"""
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Version to deprecate")
    reason: str = Field(..., min_length=10, description="Reason for deprecation")
    migration_guide: str | None = Field(
        None,
        description="Path to migration documentation (required if policy enforces)"
    )
    effective_date: datetime | None = Field(
        None,
        description="When deprecation takes effect (default: now)"
    )
    notify_stakeholders: bool = Field(
        True,
        description="Whether to trigger notification workflow"
    )

    model_config = ConfigDict(populate_by_name=True)


class RetirementRequest(BaseModel):
    """Request schema for retiring a model - Phase 6.8"""
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Version to retire")
    soft_delete: bool = Field(
        True,
        description="Whether to soft-delete (preserve audit trail)"
    )
    confirmation: str = Field(
        ...,
        pattern="^I confirm retirement$",
        description="Must be exactly: 'I confirm retirement'"
    )
    archive_location: str | None = Field(
        None,
        description="Target archive path for retired artifacts"
    )

    model_config = ConfigDict(populate_by_name=True)


class AuditLogEntry(BaseModel):
    """Schema for audit log entries - Phase 6.8"""
    timestamp: datetime
    action: str
    model_name: str
    version: str
    actor: str = "system"
    metadata: dict[str, Any] | None = Field(default_factory=dict)
    ip_address: str | None = None
    status: str = "success"
    error_message: str | None = None

    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())

class ModelQueryRequest(BaseModel):
    """Request schema for querying models - Phase 6.7"""
    name: str | None = Field(None, description="Model name filter")
    stage: str | None = Field(None, description="Stage filter (Staging/Production/Archived)")
    version: str | None = Field(None, description="Specific version")
    limit: int = Field(50, ge=1, le=500, description="Max results")

    model_config = ConfigDict(populate_by_name=True)


# Alias for api.py compatibility (HealthResponse is the canonical name)
RegistryHealthResponse = HealthResponse



# ============================================================================
# Backup & Recovery API Schemas (Phase 6.9)
# ============================================================================

class BackupRequest(BaseModel):
    """Request schema for POST /backup endpoint"""
    components: list[str] | None = Field(
        default=None,
        description="List of components to backup. If None, uses policy defaults."
    )
    compression: str | None = Field(
        default="gzip",
        description="Compression algorithm: gzip, bzip2, or none"
    )
    include_encryption: bool = Field(
        default=False,
        description="Whether to encrypt backup archive (requires encryption_key config)"
    )
    dry_run: bool = Field(
        default=False,
        description="If true, simulate backup without writing files"
    )
    policy_override: dict[str, Any] | None = Field(
        default=None,
        description="Optional policy parameters to override config/backup_policy.yaml"
    )

    @field_validator('compression')
    @classmethod
    def validate_compression(cls, v):
        if v not in ["gzip", "bzip2", "none", None]:
            raise ValueError('compression must be gzip, bzip2, or none')
        return v


class BackupResponse(BaseModel):
    """Response schema for POST /backup endpoint"""
    job_id: str
    status: str  # "initiated", "completed", "failed", "dry_run"
    manifest: dict[str, Any] | None = None
    message: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ListBackupsQuery(BaseModel):
    """Query parameters for GET /backups endpoint"""
    component: str | None = Field(default=None, description="Filter by component name")
    status: str | None = Field(default=None, description="Filter by backup status")
    limit: int = Field(default=50, ge=1, le=200, description="Max results to return")
    offset: int = Field(default=0, ge=0, description="Pagination offset")


class ListBackupsResponse(BaseModel):
    """Response schema for GET /backups endpoint"""
    backups: list[dict[str, Any]]
    total_count: int
    page: int = 1
    page_size: int = 50
