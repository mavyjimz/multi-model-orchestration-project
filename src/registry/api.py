"""
FastAPI application for Model Registry - MLflow 2.11.0 Compatible
"""

import logging
from datetime import UTC, datetime

import mlflow
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from pydantic import ValidationError

from .audit import log_lifecycle_event
from .deprecation_policy import DeprecationPolicy
from .schemas import (
    AuditLogEntry,
    BackupRequest,
    BackupResponse,
    DeprecationRequest,
    ModelInfo,
    ModelListResponse,
    ModelPromoteRequest,
    ModelQueryRequest,
    ModelRegisterRequest,
    RegistryHealthResponse,
    RetirementRequest,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-Model Orchestration Registry API", version="1.0.0")


def get_mlflow_client() -> MlflowClient:
    """Get MLflow client instance."""
    tracking_uri = mlflow.get_tracking_uri()
    return MlflowClient(tracking_uri)


@app.get("/health", response_model=RegistryHealthResponse)
async def health_check() -> RegistryHealthResponse:
    """Check registry health status."""
    mlflow_ok = False
    try:
        list(mlflow_client.search_registered_models(max_results=1))
        mlflow_ok = True
    except Exception:
        pass

    return RegistryHealthResponse(
        status="healthy" if mlflow_ok else "degraded",
        mlflow_connected=mlflow_ok,
        timestamp=datetime.now(UTC).isoformat(),
        service="model-registry",
    )


mlflow_client = get_mlflow_client()


@app.post("/register", response_model=ModelInfo, status_code=status.HTTP_201_CREATED)
async def register_model(
    request: ModelRegisterRequest, client: MlflowClient = Depends(get_mlflow_client)
):
    """Register a new model version."""
    try:
        client.create_registered_model(request.name)

        mv = client.create_model_version(
            name=request.name, source=request.source, run_id=request.run_id
        )

        client.set_model_version_tag(
            name=request.name,
            version=mv.version,
            key="registered_date",
            value=datetime.now(UTC).isoformat(),
        )

        log_lifecycle_event(
            model_name=request.name, version=mv.version, event="registered", user="system"
        )

        return ModelInfo(
            name=request.name,
            version=mv.version,
            stage=mv.current_stage,
            run_id=mv.run_id,
            source=mv.source,
            status=mv.status,
        )
    except MlflowException as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Register error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.get("/models", response_model=ModelListResponse)
async def list_models(
    query: ModelQueryRequest = Depends(), client: MlflowClient = Depends(get_mlflow_client)
):
    """List registered models - handles empty state and validation errors gracefully."""
    try:
        filter_str = f"name = '{query.name}'" if query.name else None
        models = client.search_registered_models(filter_string=filter_str, max_results=query.limit)

        infos = []
        for model in models:
            try:
                latest = client.get_latest_versions(model.name, stages=["Production", "Staging"])
                metrics = {}
                if latest:
                    try:
                        run = client.get_run(latest[0].run_id)
                        metrics = dict(run.data.metrics)
                    except Exception:
                        pass

                # Build ModelInfo with safe fallbacks
                model_version = latest[0].version if latest else None
                model_stage = latest[0].current_stage if latest else None

                # Only create ModelInfo if we have valid data
                if model_version is not None:
                    infos.append(
                        ModelInfo(
                            name=model.name,
                            version=model_version,
                            stage=model_stage,
                            run_id=latest[0].run_id if latest else None,
                            source=latest[0].source if latest else None,
                            status="READY",
                            metrics=metrics,
                        )
                    )
            except ValidationError as ve:
                # Skip this model if validation fails, log warning
                logger.warning(f"Skipping model {model.name} due to validation error: {ve}")
                continue
            except Exception as e:
                logger.warning(f"Error processing model {model.name}: {e}")
                continue

        return ModelListResponse(models=infos, total=len(infos), page=1, limit=query.limit)

    except ValidationError as ve:
        # Pydantic validation error on response - return empty list
        logger.warning(f"ModelListResponse validation error: {ve}")
        return ModelListResponse(models=[], total=0, page=1, limit=query.limit)
    except MlflowException as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"List error: {e}")
        # Return empty list instead of 500 for empty state
        return ModelListResponse(models=[], total=0, page=1, limit=query.limit)


@app.post("/promote", response_model=ModelInfo)
async def promote_model(
    request: ModelPromoteRequest, client: MlflowClient = Depends(get_mlflow_client)
):
    """Promote model to new stage."""
    try:
        client.transition_model_version_stage(
            name=request.name, version=request.version, stage=request.stage
        )

        mv = client.get_model_version(request.name, request.version)

        log_lifecycle_event(
            model_name=request.name,
            version=request.version,
            event="promoted",
            user="system",
            new_stage=request.stage,
        )

        return ModelInfo(
            name=mv.name,
            version=mv.version,
            stage=mv.current_stage,
            run_id=mv.run_id,
            source=mv.source,
            status=mv.status,
        )
    except MlflowException as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Promote error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.get("/models/{name}/versions/{version}", response_model=ModelInfo)
async def get_model_version(
    name: str, version: str, client: MlflowClient = Depends(get_mlflow_client)
):
    """Get specific model version."""
    try:
        mv = client.get_model_version(name, version)

        metrics = {}
        if mv.run_id:
            try:
                run = client.get_run(mv.run_id)
                metrics = dict(run.data.metrics)
            except Exception:
                pass

        return ModelInfo(
            name=mv.name,
            version=mv.version,
            stage=mv.current_stage,
            run_id=mv.run_id,
            source=mv.source,
            status=mv.status,
            metrics=metrics,
        )
    except MlflowException as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.delete("/models/{name}/versions/{version}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model_version(
    name: str, version: str, client: MlflowClient = Depends(get_mlflow_client)
):
    """Delete model version."""
    try:
        mv = client.get_model_version(name, version)
        client.delete_model_version(name, version)

        log_lifecycle_event(model_name=name, version=version, event="deleted", user="system")

        return JSONResponse(status_code=204, content=None)
    except MlflowException as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/deprecate")
async def deprecate_model(
    request: DeprecationRequest, client: MlflowClient = Depends(get_mlflow_client)
):
    """
    Deprecate a model version according to policy

    Request body:
    {
        "name": "model-name",
        "version": "1",
        "reason": "outdated"
    }
    """
    try:
        policy = DeprecationPolicy.get_instance()
        policy.validate_deprecation_request(
            model_name=request.name, version=request.version, reason=request.reason
        )

        client.set_model_version_tag(
            name=request.name, version=request.version, key="deprecated", value="true"
        )

        client.set_model_version_tag(
            name=request.name,
            version=request.version,
            key="deprecation_date",
            value=datetime.now(UTC).isoformat(),
        )

        if request.reason:
            client.set_model_version_tag(
                name=request.name,
                version=request.version,
                key="deprecation_reason",
                value=request.reason,
            )

        log_lifecycle_event(
            model_name=request.name,
            version=request.version,
            event="deprecated",
            user="system",
            reason=request.reason,
        )

        return {"status": "deprecated", "name": request.name, "version": request.version}
    except MlflowException as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors()) from e
    except Exception as e:
        log_lifecycle_event(
            model_name=request.name,
            version=request.version,
            event="deprecation_failed",
            user="system",
            error_message=str(e),
        )
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}") from e


@app.post("/retire")
async def retire_model(
    request: RetirementRequest, client: MlflowClient = Depends(get_mlflow_client)
):
    """
    Retire a deprecated model version according to policy

    Request body:
    {
        "name": "model-name",
        "version": "1",
        "archive_location": "s3://bucket/path"
    }
    """
    try:
        policy = DeprecationPolicy.get_instance()
        policy.validate_retirement_request(
            model_name=request.name,
            version=request.version,
            archive_location=request.archive_location,
        )

        mv = client.get_model_version(request.name, request.version)

        if request.archive_location:
            client.set_model_version_tag(
                name=request.name,
                version=request.version,
                key="archive_location",
                value=request.archive_location,
            )

        client.set_model_version_tag(
            name=request.name,
            version=request.version,
            key="retired_date",
            value=datetime.now(UTC).isoformat(),
        )

        client.set_model_version_tag(
            name=request.name, version=request.version, key="retired", value="true"
        )

        log_lifecycle_event(
            model_name=request.name,
            version=request.version,
            event="retired",
            user="system",
            archive_location=request.archive_location,
        )

        return {"status": "retired", "name": request.name, "version": request.version}
    except MlflowException as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors()) from e
    except Exception as e:
        log_lifecycle_event(
            model_name=request.name,
            version=request.version,
            event="retirement_failed",
            user="system",
            error_message=str(e),
        )
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}") from e


@app.get("/audit", response_model=list[AuditLogEntry])
async def get_audit_log(model_name: str = None):
    """
    Query audit log entries

    Query params:
    - model_name: Filter by model name
    """
    entries = []
    try:
        log_file = "logs/audit/registry_audit.log"
        with open(log_file) as f:
            for line in f:
                if model_name and model_name not in line:
                    continue
                entries.append(
                    AuditLogEntry(timestamp=datetime.now().isoformat(), event=line.strip())
                )
    except Exception:
        pass

    return entries


@app.post("/backup", response_model=BackupResponse)
async def trigger_backup(request: BackupRequest, background_tasks: BackgroundTasks):
    """
    Trigger backup operation with policy parameters.

    Requires REGISTRY_DEV_MODE=true for iR&D or proper authentication in production.
    Runs backup asynchronously via background task to avoid HTTP timeout.
    """
    try:
        from .backup import BackupPolicy

        policy = BackupPolicy.get_instance()

        background_tasks.add_task(
            policy.execute_backup, model_name=request.model_name, destination=request.destination
        )

        return BackupResponse(
            status="initiated", model_name=request.model_name, destination=request.destination
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
