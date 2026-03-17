"""
Pydantic schemas for Model Registry API - MLflow 2.11.0 Compatible
Minimal, working version with relaxed validation
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Literal
from datetime import datetime
import re

VALID_STAGES = Literal["None", "Staging", "Production", "Archived"]

class ModelRegisterRequest(BaseModel):
    """Request schema for registering a new model version"""
    name: str = Field(..., min_length=1, max_length=255)
    version: str = Field(..., description="Version: integer or semver")
    description: Optional[str] = Field(default="", max_length=500)
    source_path: str = Field(...)
    run_id: Optional[str] = Field(default=None)
    metadata: Optional[dict] = Field(default_factory=dict)
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if not re.match(r'^[a-zA-Z0-9._-]+$', v):
            raise ValueError('Name: alphanumeric, hyphens, underscores, or dots only')
        return v
    
    @field_validator('version')
    @classmethod
    def validate_version(cls, v):
        # Accept integer OR semver
        if v.isdigit() or re.match(r'^\d+(\.\d+)*.*$', v):
            return v
        raise ValueError('Version must be integer (2) or semver (1.2.0)')

class ModelPromotionRequest(BaseModel):
    name: str = Field(...)
    version: str = Field(...)
    stage: VALID_STAGES = Field(...)
    comment: Optional[str] = Field(default=None)
    
    @field_validator('version')
    @classmethod
    def validate_version(cls, v):
        if v.isdigit() or re.match(r'^\d+(\.\d+)*.*$', v):
            return v
        raise ValueError('Version must be integer or semver')

class ModelQueryRequest(BaseModel):
    name: Optional[str] = Field(default=None)
    stage: Optional[VALID_STAGES] = Field(default=None)
    min_version: Optional[str] = Field(default=None)
    limit: int = Field(default=50, ge=1, le=200)

class ModelInfo(BaseModel):
    name: str
    version: str
    stage: VALID_STAGES
    status: str
    creation_time: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    description: Optional[str] = None
    source_path: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Optional[dict] = None
    metrics_summary: Optional[dict] = None

class ModelListResponse(BaseModel):
    models: List[ModelInfo]
    total: int
    page: int = 1
    limit: int

class RegistryHealthResponse(BaseModel):
    status: str
    service: str
    mlflow_connected: bool
    timestamp: datetime
