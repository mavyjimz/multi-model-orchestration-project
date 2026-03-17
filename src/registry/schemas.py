"""
Pydantic schemas for Model Registry API
Extended for Phase 6.8: Deprecation & Retirement Policy
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict


class ModelRegisterRequest(BaseModel):
    """Request schema for registering a new model version"""
    name: str = Field(..., min_length=1, description="Model name")
    version: str = Field(..., description="Version as string (e.g., '2' or '1.2.0')")
    source: str = Field(..., description="MLflow source URI (runs:/<run-id>/path)")
    run_id: Optional[str] = Field(None, description="MLflow run ID")
    description: Optional[str] = Field(None, description="Model description")
    metadata: Optional[Dict[str, Any]] = Field(
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
    comment: Optional[str] = Field(None, description="Audit comment for transition")
    
    model_config = ConfigDict(populate_by_name=True)


class ModelInfo(BaseModel):
    """Response schema for model details - EXTENDED FOR DEPRECATION"""
    name: str
    version: str
    stage: Optional[str] = None
    source: Optional[str] = None
    run_id: Optional[str] = None
    description: Optional[str] = None
    created_timestamp: Optional[int] = None
    last_updated_timestamp: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    # Phase 6.8: Deprecation & Retirement fields
    deprecation_status: Optional[str] = Field(
        None, 
        pattern="^(active|deprecated|retired|archived)$",
        description="Current deprecation lifecycle state"
    )
    deprecation_date: Optional[datetime] = Field(
        None, 
        description="Timestamp when model was marked deprecated"
    )
    deprecation_reason: Optional[str] = Field(
        None, 
        description="Reason for deprecation (required if deprecated)"
    )
    migration_guide: Optional[str] = Field(
        None, 
        description="Path to migration documentation"
    )
    retirement_scheduled: Optional[bool] = Field(
        False, 
        description="Whether retirement is scheduled"
    )
    retirement_date: Optional[datetime] = Field(
        None, 
        description="Planned retirement date"
    )
    
    model_config = ConfigDict(from_attributes=True, populate_by_name=True, protected_namespaces=())


class ModelListResponse(BaseModel):
    """Response schema for listing models"""
    models: List[ModelInfo]
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
    migration_guide: Optional[str] = Field(
        None, 
        description="Path to migration documentation (required if policy enforces)"
    )
    effective_date: Optional[datetime] = Field(
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
    archive_location: Optional[str] = Field(
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
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    ip_address: Optional[str] = None
    status: str = "success"
    error_message: Optional[str] = None
    
    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())

class ModelQueryRequest(BaseModel):
    """Request schema for querying models - Phase 6.7"""
    name: Optional[str] = Field(None, description="Model name filter")
    stage: Optional[str] = Field(None, description="Stage filter (Staging/Production/Archived)")
    version: Optional[str] = Field(None, description="Specific version")
    limit: int = Field(50, ge=1, le=500, description="Max results")
    
    model_config = ConfigDict(populate_by_name=True)


# Alias for api.py compatibility (HealthResponse is the canonical name)
RegistryHealthResponse = HealthResponse



# ============================================================================
# Backup & Recovery API Schemas (Phase 6.9)
# ============================================================================

class BackupRequest(BaseModel):
    """Request schema for POST /backup endpoint"""
    components: Optional[List[str]] = Field(
        default=None,
        description="List of components to backup. If None, uses policy defaults."
    )
    compression: Optional[str] = Field(
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
    policy_override: Optional[Dict[str, Any]] = Field(
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
    manifest: Optional[Dict[str, Any]] = None
    message: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ListBackupsQuery(BaseModel):
    """Query parameters for GET /backups endpoint"""
    component: Optional[str] = Field(default=None, description="Filter by component name")
    status: Optional[str] = Field(default=None, description="Filter by backup status")
    limit: int = Field(default=50, ge=1, le=200, description="Max results to return")
    offset: int = Field(default=0, ge=0, description="Pagination offset")


class ListBackupsResponse(BaseModel):
    """Response schema for GET /backups endpoint"""
    backups: List[Dict[str, Any]]
    total_count: int
    page: int = 1
    page_size: int = 50
