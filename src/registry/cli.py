#!/usr/bin/env python3
"""
Command Line Interface for Model Registry
Provides developer-friendly commands for model lifecycle management
"""
import click
import requests
import json
import sys
import os
from typing import Optional

# Default API endpoint (can be overridden via env var)
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
        
        # Handle 204 No Content
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
        click.echo(f"Error: Invalid JSON response from API", err=True)
        sys.exit(1)

@click.group()
@click.option('--api-url', default=DEFAULT_API_URL, help='Registry API base URL')
@click.pass_context
def cli(ctx, api_url: str):
    """Model Registry CLI - Manage model lifecycle operations"""
    ctx.ensure_object(dict)
    ctx.obj['api_url'] = api_url.rstrip('/')
    global DEFAULT_API_URL
    DEFAULT_API_URL = ctx.obj['api_url']

@cli.command()
@click.pass_context
def health(ctx):
    """Check registry API health status"""
    result = api_request("GET", "/health")
    click.echo(json.dumps(result, indent=2))

@cli.command(name='register')
@click.option('--name', required=True, help='Model name (e.g., intent-classifier)')
@click.option('--version', required=True, help='Semantic version (e.g., 1.2.0)')
@click.option('--source', required=True, help='MLflow source path or run URI')
@click.option('--run-id', help='Associated MLflow run ID')
@click.option('--description', default='', help='Model description')
@click.option('--metadata', multiple=True, help='Custom metadata (key=value format)')
@click.pass_context
def register(ctx, name: str, version: str, source: str, run_id: Optional[str], 
             description: str, metadata: tuple):
    """Register a new model version"""
    # Parse metadata key=value pairs
    metadata_dict = {}
    for item in metadata:
        if '=' in item:
            k, v = item.split('=', 1)
            metadata_dict[k] = v
    
    payload = {
        "name": name,
        "version": version,
        "source_path": source,
        "description": description,
        "metadata": metadata_dict
    }
    if run_id:
        payload["run_id"] = run_id
    
    result = api_request("POST", "/register", payload)
    click.echo(f"Registered model: {name} v{version}")
    click.echo(json.dumps(result, indent=2))

@cli.command(name='promote')
@click.option('--name', required=True, help='Model name')
@click.option('--version', required=True, help='Model version')
@click.option('--stage', required=True, 
              type=click.Choice(['Staging', 'Production', 'Archived']),
              help='Target stage')
@click.option('--comment', help='Promotion rationale/comment')
@click.pass_context
def promote(ctx, name: str, version: str, stage: str, comment: Optional[str]):
    """Promote a model version to a new stage"""
    payload = {
        "name": name,
        "version": version,
        "stage": stage
    }
    if comment:
        payload["comment"] = comment
    
    result = api_request("POST", "/promote", payload)
    click.echo(f"Promoted {name} v{version} to {stage}")
    click.echo(json.dumps(result, indent=2))

@cli.command(name='list')
@click.option('--name', help='Filter by model name')
@click.option('--stage', type=click.Choice(['None', 'Staging', 'Production', 'Archived']),
              help='Filter by stage')
@click.option('--limit', default=50, help='Max results (1-200)')
@click.option('--format', 'output_format', default='table', 
              type=click.Choice(['table', 'json']), help='Output format')
@click.pass_context
def list_models(ctx, name: Optional[str], stage: Optional[str], 
                limit: int, output_format: str):
    """List registered models"""
    params = {"limit": limit}
    if name:
        params["name"] = name
    if stage:
        params["stage"] = stage
    
    result = api_request("GET", "/models", params)
    
    if output_format == 'json':
        click.echo(json.dumps(result, indent=2))
    else:
        # Table format
        models = result.get('models', [])
        if not models:
            click.echo("No models found.")
            return
        
        # Header
        click.echo(f"{'NAME':<25} {'VERSION':<12} {'STAGE':<12} {'STATUS':<10}")
        click.echo("-" * 60)
        
        for m in models:
            click.echo(f"{m['name']:<25} {m['version']:<12} {m['stage']:<12} {m['status']:<10}")
        
        click.echo(f"\nTotal: {result['total']} model(s)")

@cli.command(name='get')
@click.option('--name', required=True, help='Model name')
@click.option('--version', required=True, help='Model version')
@click.option('--format', 'output_format', default='json',
              type=click.Choice(['table', 'json']), help='Output format')
@click.pass_context
def get_model(ctx, name: str, version: str, output_format: str):
    """Get details for a specific model version"""
    result = api_request("GET", f"/models/{name}/versions/{version}")
    
    if output_format == 'json':
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(f"Model: {result['name']}")
        click.echo(f"Version: {result['version']}")
        click.echo(f"Stage: {result['stage']}")
        click.echo(f"Status: {result['status']}")
        if result.get('description'):
            click.echo(f"Description: {result['description']}")
        if result.get('source_path'):
            click.echo(f"Source: {result['source_path']}")
        if result.get('metrics_summary'):
            click.echo("\nMetrics:")
            for k, v in result['metrics_summary'].items():
                click.echo(f"  {k}: {v}")

@cli.command(name='delete')
@click.option('--name', required=True, help='Model name')
@click.option('--version', required=True, help='Model version')
@click.option('--force', is_flag=True, help='Force deletion without confirmation')
@click.pass_context
def delete_model(ctx, name: str, version: str, force: bool):
    """Delete a model version (archives first if not already)"""
    if not force:
        click.confirm(f"Delete {name} v{version}?", abort=True)
    
    result = api_request("DELETE", f"/models/{name}/versions/{version}")
    click.echo(json.dumps(result, indent=2))

@cli.command(name='config')
@click.pass_context
def show_config(ctx):
    """Show current CLI configuration"""
    click.echo(f"API URL: {DEFAULT_API_URL}")
    click.echo(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")

if __name__ == '__main__':
    cli(obj={})
