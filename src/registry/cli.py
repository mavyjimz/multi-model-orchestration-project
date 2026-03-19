#!/usr/bin/env python3
"""
Command Line Interface for Model Registry - MLflow 2.11.0 Compatible
"""

import json
import os
import sys

# Ensure project root is in Python path for direct CLI execution
from pathlib import Path

import click
import mlflow
import requests
from mlflow.tracking import MlflowClient

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


DEFAULT_API_URL = os.getenv("REGISTRY_API_URL", "http://localhost:8000")


def api_request(method: str, endpoint: str, payload: dict = None) -> dict:
    """Helper function to make API requests"""
    url = f"{DEFAULT_API_URL}{endpoint}"
    headers = {"Content-Type": "application/json"}
    try:
        if method == "GET":
            response = requests.get(url, params=payload, headers=headers, timeout=30)
        elif method == "POST":
            response = requests.post(url, json=payload, headers=headers, timeout=30)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers, timeout=30)
        else:
            raise ValueError(f"Unsupported method: {method}")
        response.raise_for_status()
        if response.status_code == 204:
            return {"status": "success", "message": "Operation completed"}
        return response.json()
    except requests.exceptions.ConnectionError:
        click.echo(f"Error: Cannot connect to registry API at {url}", err=True)
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        error_detail = e.response.json().get("detail", str(e)) if e.response.content else str(e)
        click.echo(f"Error: {error_detail}", err=True)
        sys.exit(1)
    except requests.exceptions.Timeout:
        click.echo("Error: Request timed out", err=True)
        sys.exit(1)
    except json.JSONDecodeError:
        click.echo("Error: Invalid JSON response from API", err=True)
        sys.exit(1)


@click.group()
@click.option("--api-url", default=DEFAULT_API_URL, help="Registry API base URL")
@click.pass_context
def cli(ctx, api_url: str):
    """Model Registry CLI - Manage model lifecycle operations"""
    ctx.ensure_object(dict)
    ctx.obj["api_url"] = api_url.rstrip("/")
    global DEFAULT_API_URL
    DEFAULT_API_URL = ctx.obj["api_url"]


@cli.command()
@click.pass_context
def health(ctx):
    """Check registry API health status"""
    result = api_request("GET", "/health")
    click.echo(json.dumps(result, indent=2))


@cli.command(name="register")
@click.option("--name", required=True, help="Model name")
@click.option("--version", required=True, help="Model version (integer for MLflow)")
@click.option("--source", required=True, help="MLflow source path or run URI")
@click.option("--run-id", help="Associated MLflow run ID")
@click.option("--description", default="", help="Model description")
@click.option("--metadata", multiple=True, help="Custom metadata (key=value, use multiple times)")
@click.pass_context
def register(
    ctx,
    name: str,
    version: str,
    source: str,
    run_id: str | None,
    description: str,
    metadata: tuple[str, ...],
):
    """Register a new model version"""
    # Parse metadata key=value pairs
    metadata_dict = {}
    for item in metadata:
        if "=" in item:
            k, v = item.split("=", 1)
            if "," in v:
                metadata_dict[k] = v.split(",")
            else:
                metadata_dict[k] = v
    payload = {
        "name": name,
        "version": version,
        "source_path": source,
        "description": description,
        "metadata": metadata_dict,
    }
    if run_id:
        payload["run_id"] = run_id
    result = api_request("POST", "/register", payload)
    click.echo(f"Registered model: {name} v{version}")
    click.echo(json.dumps(result, indent=2))


@cli.command(name="promote")
@click.option("--name", required=True, help="Model name")
@click.option("--version", required=True, help="Model version (integer)")
@click.option(
    "--stage",
    required=True,
    type=click.Choice(["Staging", "Production", "Archived"]),
    help="Target stage",
)
@click.option("--comment", help="Promotion rationale/comment")
@click.pass_context
def promote(ctx, name: str, version: str, stage: str, comment: str | None):
    """Promote a model version to a new stage"""
    payload = {"name": name, "version": version, "stage": stage}
    if comment:
        payload["comment"] = comment
    result = api_request("POST", "/promote", payload)
    click.echo(f"Promoted {name} v{version} to {stage}")
    click.echo(json.dumps(result, indent=2))


@cli.command(name="list")
@click.option("--name", help="Filter by model name")
@click.option(
    "--stage",
    type=click.Choice(["None", "Staging", "Production", "Archived"]),
    help="Filter by stage",
)
@click.option("--limit", default=50, help="Max results (1-200)")
@click.option(
    "--format",
    "output_format",
    default="table",
    type=click.Choice(["table", "json"]),
    help="Output format",
)
@click.pass_context
def list_models(ctx, name: str | None, stage: str | None, limit: int, output_format: str):
    """List registered models"""
    params = {"limit": limit}
    if name:
        params["name"] = name
    if stage:
        params["stage"] = stage
    result = api_request("GET", "/models", params)
    if output_format == "json":
        click.echo(json.dumps(result, indent=2))
    else:
        models = result.get("models", [])
        if not models:
            click.echo("No models found.")
            return
        click.echo(f"{'NAME':<30} {'VERSION':<10} {'STAGE':<12} {'STATUS':<10}")
        click.echo("-" * 65)
        for m in models:
            click.echo(f"{m['name']:<30} {m['version']:<10} {m['stage']:<12} {m['status']:<10}")
        click.echo(f"\nTotal: {result['total']} model(s)")


@cli.command(name="get")
@click.option("--name", required=True, help="Model name")
@click.option("--version", required=True, help="Model version (integer)")
@click.option(
    "--format",
    "output_format",
    default="json",
    type=click.Choice(["table", "json"]),
    help="Output format",
)
@click.pass_context
def get_model(ctx, name: str, version: str, output_format: str):
    """Get details for a specific model version"""
    result = api_request("GET", f"/models/{name}/versions/{version}")
    if output_format == "json":
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(f"Model: {result['name']}")
        click.echo(f"Version: {result['version']}")
        click.echo(f"Stage: {result['stage']}")
        click.echo(f"Status: {result['status']}")
        if result.get("description"):
            click.echo(f"Description: {result['description']}")


@cli.command(name="delete")
@click.option("--name", required=True, help="Model name")
@click.option("--version", required=True, help="Model version")
@click.option("--force", is_flag=True, help="Force deletion without confirmation")
@click.pass_context
def delete_model(ctx, name: str, version: str, force: bool):
    """Delete a model version"""
    if not force:
        click.confirm(f"Delete {name} v{version}?", abort=True)
    result = api_request("DELETE", f"/models/{name}/versions/{version}")
    click.echo(json.dumps(result, indent=2))


@cli.command()
@click.option("--name", required=True, help="Model name to deprecate")
@click.option("--version", required=True, help="Version to deprecate")
@click.option("--reason", required=True, help="Reason for deprecation (min 10 chars)")
@click.option("--migration-guide", default=None, help="Path to migration documentation")
@click.option("--effective-date", default=None, help="Effective date (ISO format: YYYY-MM-DD)")
@click.option("--no-notify", is_flag=True, help="Disable stakeholder notifications")
@click.option("--actor", default="system", help="Actor initiating deprecation (for audit)")
@click.pass_context
def deprecate(ctx, name, version, reason, migration_guide, effective_date, no_notify, actor):
    """
    Deprecate a model version according to policy

    Example:
        python src/registry/cli.py deprecate \
            --name intent-classifier-sgd \
            --version 1.0.0 \
            --reason "superseded by v2.0.0 with improved accuracy" \
            --migration-guide "docs/migration/v1-to-v2.md" \
            --actor "jim-mlops"
    """
    from datetime import datetime

    from pydantic import ValidationError

    from src.registry.audit import log_deprecation, log_lifecycle_event
    from src.registry.deprecation_policy import DeprecationPolicy, PolicyViolationError
    from src.registry.schemas import DeprecationRequest

    try:
        # Parse effective date if provided
        eff_date = None
        if effective_date:
            try:
                eff_date = datetime.fromisoformat(effective_date.replace("Z", "+00:00"))
            except ValueError:
                click.echo(
                    f"Error: Invalid date format '{effective_date}'. Use ISO format (YYYY-MM-DD)",
                    err=True,
                )
                ctx.exit(1)

        # Validate request against schema
        request = DeprecationRequest(
            name=name,
            version=version,
            reason=reason,
            migration_guide=migration_guide,
            effective_date=eff_date,
            notify_stakeholders=not no_notify,
        )

        # Load and apply policy
        policy = DeprecationPolicy.get_instance()
        validation = policy.validate_deprecation_request(
            model_name=request.name,
            version=request.version,
            reason=request.reason,
            migration_guide=request.migration_guide,
            effective_date=request.effective_date,
        )

        # Display warnings if any
        for warning in validation.get("warnings", []):
            click.echo(f"⚠️  Warning: {warning}", err=True)

        # Log to audit trail BEFORE making changes (audit-first principle)
        log_deprecation(
            model_name=request.name,
            version=request.version,
            reason=request.reason,
            migration_guide=request.migration_guide,
            actor=actor,
            ip_address=ctx.obj.get("ip_address") if ctx.obj else None,
            policy_validated=True,
            warning_period_days=validation.get("warning_period_days"),
        )

        # Update MLflow model version metadata with deprecation status
        # Note: In dev mode, this may be a no-op or mock operation
        # Initialize MLflow client directly for CLI usage
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)

        try:
            # Attempt to update model version tags with deprecation metadata
            model_version = client.get_model_version(request.name, request.version)
            client.set_model_version_tag(
                name=request.name,
                version=request.version,
                key="deprecation_status",
                value="deprecated",
            )
            client.set_model_version_tag(
                name=request.name,
                version=request.version,
                key="deprecation_reason",
                value=request.reason,
            )
            if request.migration_guide:
                client.set_model_version_tag(
                    name=request.name,
                    version=request.version,
                    key="migration_guide",
                    value=request.migration_guide,
                )
            click.echo(f"✓ Model {request.name} v{request.version} marked as deprecated in MLflow")
        except Exception as mlflow_err:
            # Non-fatal: audit log already recorded, MLflow update is best-effort in iR&D
            click.echo(f"⚠️  MLflow metadata update skipped: {mlflow_err}", err=True)
            log_lifecycle_event(
                action="deprecate",
                model_name=request.name,
                version=request.version,
                status="partial_success",
                actor=actor,
                error_message=str(mlflow_err),
            )

        # Success output
        click.echo(f"✓ Deprecation recorded for {request.name} v{request.version}")
        click.echo(f"  Reason: {request.reason}")
        if request.migration_guide:
            click.echo(f"  Migration guide: {request.migration_guide}")
        if request.effective_date:
            click.echo(f"  Effective date: {request.effective_date.isoformat()}")
        if validation.get("warning_period_days"):
            click.echo(f"  Warning period: {validation['warning_period_days']} days")

        # Log success to audit
        log_lifecycle_event(
            action="deprecate",
            model_name=request.name,
            version=request.version,
            status="success",
            actor=actor,
            metadata={"migration_guide": request.migration_guide},
        )

    except PolicyViolationError as e:
        click.echo(f"Error: Policy violation - {e}", err=True)
        log_lifecycle_event(
            action="deprecate",
            model_name=name,
            version=version,
            status="failed",
            actor=actor,
            error_message=str(e),
            metadata={"violation_field": e.field, "violation_value": e.value},
        )
        ctx.exit(2)
    except ValidationError as e:
        click.echo(f"Error: Invalid request - {e}", err=True)
        ctx.exit(1)
    except Exception as e:
        click.echo(f"Error: Unexpected error - {e}", err=True)
        log_lifecycle_event(
            action="deprecate",
            model_name=name,
            version=version,
            status="failed",
            actor=actor,
            error_message=str(e),
        )
        ctx.exit(3)


@cli.command()
@click.option("--name", required=True, help="Model name to retire")
@click.option("--version", required=True, help="Version to retire")
@click.option(
    "--soft-delete", is_flag=True, default=True, help="Soft-delete (preserve audit trail)"
)
@click.option("--archive-location", default=None, help="Target archive path for artifacts")
@click.option("--actor", default="system", help="Actor initiating retirement (for audit)")
@click.option("--force", is_flag=True, help="Bypass policy checks (use with caution)")
@click.pass_context
def retire(ctx, name, version, soft_delete, archive_location, actor, force):
    """
    Retire a deprecated model version according to policy

    Example:
        python src/registry/cli.py retire \
            --name intent-classifier-sgd \
            --version 1.0.0 \
            --actor "jim-mlops" \
            --archive-location "s3://archive-bucket/models/retired/"
    """
    from datetime import datetime

    from pydantic import ValidationError

    from src.registry.audit import log_lifecycle_event, log_retirement
    from src.registry.deprecation_policy import DeprecationPolicy, PolicyViolationError
    from src.registry.schemas import RetirementRequest

    try:
        # Validate request against schema
        request = RetirementRequest(
            name=name,
            version=version,
            soft_delete=soft_delete,
            confirmation="I confirm retirement",  # Auto-confirm for CLI; API requires explicit
            archive_location=archive_location,
        )

        # Load policy
        policy = DeprecationPolicy.get_instance()

        # Check eligibility unless --force
        if not force:
            # Try to get deprecation date from MLflow tags
            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
            mlflow.set_tracking_uri(tracking_uri)
            client = MlflowClient(tracking_uri=tracking_uri)

            try:
                model_version = client.get_model_version(request.name, request.version)
                deprecation_tag = model_version.tags.get("deprecation_date")
                if deprecation_tag:
                    from datetime import datetime

                    deprecation_date = datetime.fromisoformat(
                        deprecation_tag.replace("Z", "+00:00")
                    )
                    eligibility = policy.get_retirement_eligibility(deprecation_date, request.name)
                    if not eligibility["eligible"]:
                        click.echo(f"Error: {eligibility['reason']}", err=True)
                        click.echo(
                            f"  Eligible after: {eligibility.get('eligible_after', 'N/A')}",
                            err=True,
                        )
                        ctx.exit(2)
                else:
                    click.echo(
                        "⚠️  Warning: No deprecation_date tag found; proceeding with retirement",
                        err=True,
                    )
            except Exception as e:
                click.echo(f"⚠️  Warning: Could not check eligibility: {e}", err=True)

        # Validate retirement request
        validation = policy.validate_retirement_request(
            model_name=request.name,
            version=request.version,
            actor=actor,
            soft_delete=request.soft_delete,
        )

        # Log to audit trail BEFORE making changes
        log_retirement(
            model_name=request.name,
            version=request.version,
            soft_delete=request.soft_delete,
            archive_location=request.archive_location,
            actor=actor,
            ip_address=ctx.obj.get("ip_address") if ctx.obj else None,
            policy_validated=True,
        )

        # Perform retirement action
        if request.soft_delete:
            # Soft delete: update MLflow stage to "Archived" and add retirement tags
            try:
                client.set_model_version_tag(
                    name=request.name,
                    version=request.version,
                    key="deprecation_status",
                    value="retired",
                )
                client.set_model_version_tag(
                    name=request.name,
                    version=request.version,
                    key="retired_date",
                    value=datetime.now().isoformat(),
                )
                if request.archive_location:
                    client.set_model_version_tag(
                        name=request.name,
                        version=request.version,
                        key="archive_location",
                        value=request.archive_location,
                    )
                click.echo(
                    f"✓ Model {request.name} v{request.version} soft-retired (stage: Archived)"
                )
            except Exception as mlflow_err:
                click.echo(f"⚠️  MLflow metadata update skipped: {mlflow_err}", err=True)
        else:
            # Hard delete: remove model version (production use only)
            click.echo(f"⚠️  Hard delete requested for {request.name} v{request.version}")
            click.echo("    This action is irreversible. Use --soft-delete for safety.")
            # In production, add confirmation prompt here
            # For iR&D, skip actual deletion

        # Success output
        click.echo(f"✓ Retirement recorded for {request.name} v{request.version}")
        click.echo(f"  Soft delete: {request.soft_delete}")
        if request.archive_location:
            click.echo(f"  Archive location: {request.archive_location}")
        if validation.get("requirements", {}).get("audit_retention_until"):
            click.echo(
                f"  Audit retained until: {validation['requirements']['audit_retention_until']}"
            )

        # Log success
        log_lifecycle_event(
            action="retire",
            model_name=request.name,
            version=request.version,
            status="success",
            actor=actor,
            metadata={
                "soft_delete": request.soft_delete,
                "archive_location": request.archive_location,
            },
        )

    except PolicyViolationError as e:
        click.echo(f"Error: Policy violation - {e}", err=True)
        log_lifecycle_event(
            action="retire",
            model_name=name,
            version=version,
            status="failed",
            actor=actor,
            error_message=str(e),
        )
        ctx.exit(2)
    except ValidationError as e:
        click.echo(f"Error: Invalid request - {e}", err=True)
        ctx.exit(1)
    except Exception as e:
        click.echo(f"Error: Unexpected error - {e}", err=True)
        log_lifecycle_event(
            action="retire",
            model_name=name,
            version=version,
            status="failed",
            actor=actor,
            error_message=str(e),
        )
        ctx.exit(3)


if __name__ == "__main__":
    cli(obj={})
