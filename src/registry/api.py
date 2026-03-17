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
    ModelRegisterRequest, ModelPromotionRequest, ModelQueryRequest,
    ModelInfo, ModelListResponse, RegistryHealthResponse
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
async def promote_model(request: ModelPromotionRequest, client: MlflowClient = Depends(get_mlflow_client)):
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
