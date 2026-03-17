"""
FastAPI application for Model Registry - MLflow 2.11.0 Compatible
"""
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
import logging

from .schemas import (
    ModelRegisterRequest, ModelPromoteRequest, ModelQueryRequest,
    ModelInfo, ModelListResponse, RegistryHealthResponse,
    DeprecationRequest, RetirementRequest, AuditLogEntry
)

logger = logging.getLogger(__name__)
app = FastAPI(title="Multi-Model Orchestration Registry API", version="1.0.0")

mlflow_client: Optional[MlflowClient] = None

def get_mlflow_client() -> MlflowClient:
    if mlflow_client is None:
        raise HTTPException(status_code=503, detail="MLflow client not initialized")
    return mlflow_client

@app.on_event("startup")
async def startup_event():
    global mlflow_client
    from .config import get_mlflow_tracking_uri
    tracking_uri = get_mlflow_tracking_uri()
    logger.info(f"Connecting to MLflow at: {tracking_uri}")
    try:
        mlflow.set_tracking_uri(tracking_uri)
        test_client = MlflowClient(tracking_uri=tracking_uri)
        list(test_client.search_registered_models(max_results=1))
        mlflow_client = test_client
        logger.info("✓ Connected to MLflow")
    except Exception as e:
        logger.error(f"✗ MLflow connection failed: {type(e).__name__}: {e}")
        mlflow_client = None

@app.get("/health", response_model=RegistryHealthResponse)
async def health_check():
    mlflow_ok = False
    if mlflow_client:
        try:
            list(mlflow_client.search_registered_models(max_results=1))
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
async def register_model(request: ModelRegisterRequest, client: MlflowClient = Depends(get_mlflow_client)):
    try:
        # Create registered model if not exists
        try:
            client.get_registered_model(request.name)
        except MlflowException:
            client.create_registered_model(name=request.name, tags={"created_by": "registry-api"})
        
        # Create model version
        mv = client.create_model_version(
            name=request.name, 
            source=request.source_path, 
            run_id=request.run_id,
            tags={"version": request.version, "description": request.description}
        )
        
        # Attach custom metadata as tags
        if request.metadata:
            for k, v in request.metadata.items():
                try:
                    client.set_model_version_tag(name=request.name, version=mv.version, key=k, value=str(v))
                except Exception as e:
                    logger.warning(f"Failed to set tag {k}: {e}")
        
        return ModelInfo(
            name=mv.name, version=mv.version, stage="None", status="READY",
            description=request.description, source_path=request.source_path,
            run_id=request.run_id, metadata=request.metadata,
            creation_time=datetime.fromtimestamp(mv.creation_timestamp/1000) if mv.creation_timestamp else None
        )
    except MlflowException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Register error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/models", response_model=ModelListResponse)
async def list_models(query: ModelQueryRequest = Depends(), client: MlflowClient = Depends(get_mlflow_client)):
    try:
        filter_str = f"name = '{query.name}'" if query.name else None
        models = list(client.search_registered_models(filter_string=filter_str, max_results=query.limit))
        
        infos = []
        for m in models:
            latest = m.latest_versions[0] if m.latest_versions else None
            metrics = None
            if latest and latest.run_id:
                try:
                    run = client.get_run(latest.run_id)
                    metrics = dict(run.data.metrics)
                except: 
                    pass
            infos.append(ModelInfo(
                name=m.name, 
                version=latest.version if latest else "0",
                stage=latest.current_stage if latest else "None", 
                status="READY",
                description=m.description,
                creation_time=datetime.fromtimestamp(m.creation_timestamp/1000) if m.creation_timestamp else None,
                metadata=dict(m.tags) if m.tags else None, 
                metrics_summary=metrics
            ))
        if query.stage and query.stage != "None":
            infos = [i for i in infos if i.stage == query.stage]
        return ModelListResponse(models=infos, total=len(infos), page=1, limit=query.limit)
    except MlflowException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"List error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/promote", response_model=ModelInfo)
async def promote_model(request: ModelPromoteRequest, client: MlflowClient = Depends(get_mlflow_client)):
    try:
        client.transition_model_version_stage(
            name=request.name, 
            version=request.version, 
            stage=request.stage,
            archive_existing_versions=(request.stage == "Production")
        )
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
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Promote error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/models/{name}/versions/{version}", response_model=ModelInfo)
async def get_model_version(name: str, version: str, client: MlflowClient = Depends(get_mlflow_client)):
    try:
        mv = client.get_model_version(name, version)
        metrics = None
        if mv.run_id:
            try:
                run = client.get_run(mv.run_id)
                metrics = dict(run.data.metrics)
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
            metrics_summary=metrics,
            creation_time=datetime.fromtimestamp(mv.creation_timestamp/1000) if mv.creation_timestamp else None
        )
    except MlflowException as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.delete("/models/{name}/versions/{version}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model_version(name: str, version: str, client: MlflowClient = Depends(get_mlflow_client)):
    try:
        mv = client.get_model_version(name, version)
        if mv.current_stage == "Production":
            raise HTTPException(status_code=400, detail="Cannot delete Production model. Demote to Archived first.")
        if mv.current_stage != "Archived":
            client.transition_model_version_stage(name=name, version=version, stage="Archived")
        client.delete_model_version(name=name, version=version)
        return JSONResponse(status_code=204, content=None)
    except MlflowException as e:
        raise HTTPException(status_code=400, detail=str(e))


# =============================================================================
# Phase 6.8: Deprecation & Retirement Endpoints
# =============================================================================

@app.post("/deprecate", status_code=200, response_model=ModelInfo)
async def deprecate_model(
    request: DeprecationRequest,
    client: MlflowClient = Depends(get_mlflow_client)
):
    """
    Deprecate a model version according to policy
    
    Request body:
    {
        "name": "intent-classifier-sgd",
        "version": "1.0.0",
        "reason": "superseded by v2.0.0",
        "migration_guide": "docs/migration/v1-to-v2.md",
        "effective_date": "2026-04-16T00:00:00Z",  # optional
        "notify_stakeholders": true  # optional
    }
    """
    from .deprecation_policy import DeprecationPolicy, PolicyViolationError
    from .audit import log_deprecation, log_lifecycle_event
    
    try:
        # Validate against policy
        policy = DeprecationPolicy.get_instance()
        validation = policy.validate_deprecation_request(
            model_name=request.name,
            version=request.version,
            reason=request.reason,
            migration_guide=request.migration_guide,
            effective_date=request.effective_date
        )
        
        # Log to audit trail
        log_deprecation(
            model_name=request.name,
            version=request.version,
            reason=request.reason,
            migration_guide=request.migration_guide,
            actor="api",
            ip_address=None,  # Could extract from request.headers if needed
            policy_validated=True
        )
        
        # Update MLflow model version metadata
        try:
            # Normalize version for MLflow (strip 'v' prefix if present)
            mlflow_version = request.version.lstrip('v')
            
            client.set_model_version_tag(
                name=request.name,
                version=mlflow_version,
                key="deprecation_status",
                value="deprecated"
            )
            client.set_model_version_tag(
                name=request.name,
                version=mlflow_version,
                key="deprecation_reason",
                value=request.reason
            )
            if request.migration_guide:
                client.set_model_version_tag(
                    name=request.name,
                    version=mlflow_version,
                    key="migration_guide",
                    value=request.migration_guide
                )
            if request.effective_date:
                client.set_model_version_tag(
                    name=request.name,
                    version=mlflow_version,
                    key="deprecation_date",
                    value=request.effective_date.isoformat()
                )
        except Exception as mlflow_err:
            logger.warning(f"MLflow metadata update failed: {mlflow_err}")
            # Non-fatal: audit log already recorded
        
        # Return updated model info
        try:
            mlflow_version = request.version.lstrip('v')
            mv = client.get_model_version(request.name, mlflow_version)
            return ModelInfo(
                name=mv.name,
                version=mv.version,
                stage=mv.current_stage,
                source=mv.source,
                run_id=mv.run_id,
                description=mv.description,
                created_timestamp=mv.creation_timestamp,
                last_updated_timestamp=mv.last_updated_timestamp,
                metadata=dict(mv.tags) if mv.tags else {},
                deprecation_status="deprecated",
                deprecation_reason=request.reason,
                migration_guide=request.migration_guide
            )
        except Exception:
            # Fallback response if MLflow fetch fails
            return ModelInfo(
                name=request.name,
                version=request.version,
                deprecation_status="deprecated",
                deprecation_reason=request.reason,
                migration_guide=request.migration_guide
            )
        
    except PolicyViolationError as e:
        log_lifecycle_event(
            action="deprecate",
            model_name=request.name,
            version=request.version,
            status="failed",
            actor="api",
            error_message=str(e)
        )
        raise HTTPException(status_code=400, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        log_lifecycle_event(
            action="deprecate",
            model_name=request.name,
            version=request.version,
            status="failed",
            actor="api",
            error_message=str(e)
        )
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/retire", status_code=200, response_model=ModelInfo)
async def retire_model(
    request: RetirementRequest,
    client: MlflowClient = Depends(get_mlflow_client)
):
    """
    Retire a deprecated model version according to policy
    
    Request body:
    {
        "name": "intent-classifier-sgd",
        "version": "1.0.0",
        "soft_delete": true,
        "confirmation": "I confirm retirement",
        "archive_location": "s3://bucket/retired/"
    }
    """
    from .deprecation_policy import DeprecationPolicy, PolicyViolationError
    from .audit import log_retirement, log_lifecycle_event
    
    try:
        # Validate confirmation
        if request.confirmation != "I confirm retirement":
            raise HTTPException(
                status_code=400,
                detail="Confirmation string must be exactly: 'I confirm retirement'"
            )
        
        # Load policy and validate
        policy = DeprecationPolicy.get_instance()
        validation = policy.validate_retirement_request(
            model_name=request.name,
            version=request.version,
            actor="api",
            soft_delete=request.soft_delete
        )
        
        # Log to audit trail
        log_retirement(
            model_name=request.name,
            version=request.version,
            soft_delete=request.soft_delete,
            archive_location=request.archive_location,
            actor="api",
            policy_validated=True
        )
        
        # Perform retirement
        mlflow_version = request.version.lstrip('v')
        
        if request.soft_delete:
            # Soft delete: update tags to mark as retired
            client.set_model_version_tag(
                name=request.name,
                version=mlflow_version,
                key="deprecation_status",
                value="retired"
            )
            client.set_model_version_tag(
                name=request.name,
                version=mlflow_version,
                key="retired_date",
                value=datetime.now(timezone.utc).isoformat()
            )
            if request.archive_location:
                client.set_model_version_tag(
                    name=request.name,
                    version=mlflow_version,
                    key="archive_location",
                    value=request.archive_location
                )
        else:
            # Hard delete: remove model version (use with extreme caution)
            # In production, add approval workflow here
            logger.warning(f"Hard delete requested for {request.name} v{mlflow_version}")
            # client.delete_model_version(request.name, mlflow_version)  # Uncomment for production
        
        # Return updated model info
        try:
            mv = client.get_model_version(request.name, mlflow_version)
            return ModelInfo(
                name=mv.name,
                version=mv.version,
                stage=mv.current_stage,
                source=mv.source,
                run_id=mv.run_id,
                description=mv.description,
                created_timestamp=mv.creation_timestamp,
                last_updated_timestamp=mv.last_updated_timestamp,
                metadata=dict(mv.tags) if mv.tags else {},
                deprecation_status="retired",
                retirement_scheduled=not request.soft_delete
            )
        except Exception:
            return ModelInfo(
                name=request.name,
                version=request.version,
                deprecation_status="retired",
                retirement_scheduled=not request.soft_delete
            )
        
    except PolicyViolationError as e:
        log_lifecycle_event(
            action="retire",
            model_name=request.name,
            version=request.version,
            status="failed",
            actor="api",
            error_message=str(e)
        )
        raise HTTPException(status_code=400, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        log_lifecycle_event(
            action="retire",
            model_name=request.name,
            version=request.version,
            status="failed",
            actor="api",
            error_message=str(e)
        )
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/audit", response_model=List[AuditLogEntry])
async def query_audit(
    model_name: Optional[str] = None,
    action: Optional[str] = None,
    version: Optional[str] = None,
    limit: int = 100
):
    """
    Query audit log entries
    
    Query params:
    - model_name: Filter by model name
    - action: Filter by action type (deprecate, retire, promote, register)
    - version: Filter by version
    - limit: Max entries to return (default: 100)
    """
    from .audit import query_audit_log
    
    entries = query_audit_log(
        model_name=model_name,
        action=action,
        version=version,
        limit=limit
    )
    
    # Parse JSON strings to dicts for Pydantic validation
    results = []
    for entry in entries:
        try:
            if isinstance(entry, str):
                import json
                entry = json.loads(entry)
            results.append(AuditLogEntry(**entry))
        except Exception:
            continue  # Skip malformed entries
    
    return results

