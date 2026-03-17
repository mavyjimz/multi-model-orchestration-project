"""
FastAPI application for Model Registry
Provides REST endpoints for model lifecycle management
"""
from fastapi import FastAPI, HTTPException, status, Query, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
import logging

from .schemas import (
    ModelRegisterRequest, ModelPromotionRequest, ModelQueryRequest,
    ModelInfo, ModelListResponse, RegistryHealthResponse
)

# Configure logging
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multi-Model Orchestration Registry API",
    description="REST API for managing model registry operations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global MLflow client (initialized on startup)
mlflow_client: Optional[MlflowClient] = None

def get_mlflow_client() -> MlflowClient:
    """Dependency to get MLflow client"""
    if mlflow_client is None:
        raise HTTPException(status_code=503, detail="MLflow client not initialized")
    return mlflow_client

@app.on_event("startup")
async def startup_event():
    """Initialize MLflow connection on startup"""
    global mlflow_client
    # Use environment variable or default from Phase 1 config
    tracking_uri = "http://localhost:5000"
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow_client = MlflowClient(tracking_uri=tracking_uri)
        logger.info(f"Connected to MLflow at {tracking_uri}")
    except Exception as e:
        logger.error(f"Failed to connect to MLflow: {e}")
        # Don't crash the app, but mark as unhealthy
        mlflow_client = None

@app.get("/health", response_model=RegistryHealthResponse)
async def health_check():
    """Health check endpoint"""
    mlflow_ok = False
    if mlflow_client:
        try:
            mlflow_client.list_registered_models(max_results=1)
            mlflow_ok = True
        except:
            pass
    
    return RegistryHealthResponse(
        status="healthy" if mlflow_ok else "degraded",
        service="model-registry",
        mlflow_connected=mlflow_ok,
        timestamp=datetime.utcnow()
    )

@app.post("/register", response_model=ModelInfo, status_code=status.HTTP_201_CREATED)
async def register_model(
    request: ModelRegisterRequest,
    client: MlflowClient = Depends(get_mlflow_client)
):
    """Register a new model version in the registry"""
    try:
        # Check if model exists, create if not
        try:
            client.get_registered_model(request.name)
        except MlflowException:
            client.create_registered_model(
                name=request.name,
                tags={"created_by": "registry-api", "project": "multi-model-orchestration"}
            )
            logger.info(f"Created new registered model: {request.name}")
        
        # Create model version
        mv = client.create_model_version(
            name=request.name,
            source=request.source_path,
            run_id=request.run_id,
            tags={"version": request.version, "description": request.description}
        )
        
        # Attach custom metadata as tags
        if request.metadata:
            for key, value in request.metadata.items():
                try:
                    client.set_model_version_tag(
                        name=request.name,
                        version=mv.version,
                        key=key,
                        value=str(value)
                    )
                except Exception as e:
                    logger.warning(f"Failed to set metadata tag {key}: {e}")
        
        return ModelInfo(
            name=mv.name,
            version=mv.version,
            stage="None",
            status="READY",
            description=request.description,
            source_path=request.source_path,
            run_id=request.run_id,
            metadata=request.metadata,
            creation_time=datetime.fromtimestamp(mv.creation_timestamp / 1000) if mv.creation_timestamp else None
        )
        
    except MlflowException as e:
        logger.error(f"MLflow error during registration: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during registration: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/models", response_model=ModelListResponse)
async def list_models(
    query: ModelQueryRequest = Depends(),
    client: MlflowClient = Depends(get_mlflow_client)
):
    """List registered models with optional filtering"""
    try:
        # Build filter string for MLflow
        filter_string = None
        if query.name:
            filter_string = f"name = '{query.name}'"
        
        models = client.search_registered_models(
            filter_string=filter_string,
            max_results=query.limit
        )
        
        model_infos = []
        for m in models:
            # Get latest version info
            latest = m.latest_versions[0] if m.latest_versions else None
            
            # Fetch metrics from associated run if available
            metrics_summary = None
            if latest and latest.run_id:
                try:
                    run = client.get_run(latest.run_id)
                    metrics_summary = dict(run.data.metrics)
                except:
                    pass
            
            model_infos.append(ModelInfo(
                name=m.name,
                version=latest.version if latest else "0",
                stage=latest.current_stage if latest else "None",
                status="READY",
                description=m.description,
                creation_time=datetime.fromtimestamp(m.creation_timestamp / 1000) if m.creation_timestamp else None,
                metadata=dict(m.tags) if m.tags else None,
                metrics_summary=metrics_summary
            ))
        
        # Apply stage filtering post-fetch if needed
        if query.stage and query.stage != "None":
            model_infos = [m for m in model_infos if m.stage == query.stage]
        
        return ModelListResponse(
            models=model_infos,
            total=len(model_infos),
            page=1,
            limit=query.limit
        )
        
    except MlflowException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/promote", response_model=ModelInfo)
async def promote_model(
    request: ModelPromotionRequest,
    client: MlflowClient = Depends(get_mlflow_client)
):
    """Promote a model version to a new stage"""
    try:
        # Validate transition (basic business logic)
        current_mv = client.get_model_version(request.name, request.version)
        current_stage = current_mv.current_stage
        
        # Log promotion attempt
        logger.info(f"Promoting {request.name}:{request.version} from {current_stage} to {request.stage}")
        
        # Execute transition
        client.transition_model_version_stage(
            name=request.name,
            version=request.version,
            stage=request.stage,
            archive_existing_versions=(request.stage == "Production")
        )
        
        # Add comment as tag if provided
        if request.comment:
            client.set_model_version_tag(
                name=request.name,
                version=request.version,
                key="promotion_comment",
                value=request.comment
            )
        
        return ModelInfo(
            name=request.name,
            version=request.version,
            stage=request.stage,
            status="TRANSITIONED",
            last_updated=datetime.utcnow()
        )
        
    except MlflowException as e:
        logger.error(f"MLflow error during promotion: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during promotion: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/models/{name}/versions/{version}", response_model=ModelInfo)
async def get_model_version(
    name: str,
    version: str,
    client: MlflowClient = Depends(get_mlflow_client)
):
    """Get detailed information about a specific model version"""
    try:
        mv = client.get_model_version(name, version)
        
        # Fetch associated run metrics
        metrics_summary = None
        if mv.run_id:
            try:
                run = client.get_run(mv.run_id)
                metrics_summary = dict(run.data.metrics)
            except:
                pass
        
        return ModelInfo(
            name=mv.name,
            version=mv.version,
            stage=mv.current_stage,
            status=mv.status,
            description=mv.description,
            source_path=mv.source,
            run_id=mv.run_id,
            metadata=dict(mv.tags) if mv.tags else None,
            metrics_summary=metrics_summary,
            creation_time=datetime.fromtimestamp(mv.creation_timestamp / 1000) if mv.creation_timestamp else None
        )
        
    except MlflowException as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.delete("/models/{name}/versions/{version}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model_version(
    name: str,
    version: str,
    client: MlflowClient = Depends(get_mlflow_client)
):
    """Delete a model version (soft delete: archive first)"""
    try:
        mv = client.get_model_version(name, version)
        
        # Cannot delete Production models directly
        if mv.current_stage == "Production":
            raise HTTPException(
                status_code=400, 
                detail="Cannot delete Production model. Demote to Archived first."
            )
        
        # Archive then delete
        if mv.current_stage != "Archived":
            client.transition_model_version_stage(
                name=name, version=version, stage="Archived"
            )
        
        client.delete_model_version(name=name, version=version)
        return JSONResponse(status_code=204, content=None)
        
    except MlflowException as e:
        raise HTTPException(status_code=400, detail=str(e))
