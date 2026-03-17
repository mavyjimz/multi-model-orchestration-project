"""
Pydantic schemas for Model Registry API
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Literal
from datetime import datetime

# Valid stage transitions
VALID_STAGES = Literal["None", "Staging", "Production", "Archived"]

class ModelRegisterRequest(BaseModel):
    """Request schema for registering a new model version"""
    name: str = Field(..., min_length=1, description="Model name")
    version: str = Field(..., pattern=r'^\d+\.\d+\.\d+.*$', description="Semantic version")
    description: Optional[str] = Field(default="", max_length=500)
    source_path: str = Field(..., description="MLflow run URI or artifact path")
    run_id: Optional[str] = Field(default=None, description="Associated MLflow run ID")
    metadata: Optional[dict] = Field(default_factory=dict, description="Custom metadata")

    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if not v.replace('-', '_').replace('.', '').isalnum():
            raise ValueError('Name must contain only alphanumeric, hyphens, underscores, or dots')
        return v

class ModelPromotionRequest(BaseModel):
    """Request schema for promoting a model to a new stage"""
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    stage: VALID_STAGES = Field(..., description="Target stage")
    comment: Optional[str] = Field(default=None, description="Promotion rationale")

class ModelQueryRequest(BaseModel):
    """Request schema for querying models"""
    name: Optional[str] = Field(default=None, description="Filter by model name")
    stage: Optional[VALID_STAGES] = Field(default=None, description="Filter by stage")
    min_version: Optional[str] = Field(default=None, description="Minimum version filter")
    limit: int = Field(default=50, ge=1, le=200, description="Max results to return")

class ModelInfo(BaseModel):
    """Response schema for model information"""
    name: str
    version: str
    stage: VALID_STAGES
    status: str  # READY, FAILED, PENDING
    creation_time: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    description: Optional[str] = None
    source_path: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Optional[dict] = None
    metrics_summary: Optional[dict] = None

class ModelListResponse(BaseModel):
    """Response schema for listing models"""
    models: List[ModelInfo]
    total: int
    page: int = 1
    limit: int

class RegistryHealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    mlflow_connected: bool
    timestamp: datetime
