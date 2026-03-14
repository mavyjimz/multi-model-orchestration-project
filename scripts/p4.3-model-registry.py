#!/usr/bin/env python3
"""
Phase 4.3: Model Registry & Versioning
=======================================
Implements centralized model registry with:
- Model lifecycle management (development → staging → production)
- Model lineage tracking (data + code + hyperparameters)
- Registry indexing and metadata management
- Model promotion/demotion workflow
- Integration with MLflow Model Registry

Author: MLOps Team
Date: March 2026
Version: 1.0.0
"""

import os
import sys
import json
import shutil
import pickle
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from enum import Enum
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Third-party imports
import yaml
import mlflow
import mlflow.sklearn
from mlflow.entities import LifecycleStage
import pandas as pd

# Suppress non-critical warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/p4.3-model-registry.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ModelStage(Enum):
    """Model lifecycle stages."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ModelRegistry:
    """Production-grade model registry with lifecycle management."""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialize model registry."""
        self.config = self._load_config(config_path)
        self.project_root = Path(__file__).parent.parent
        self.registry_path = self.project_root / 'models' / 'registry'
        self.mlflow_tracking_uri = str(self.project_root / 'mlruns')
        
        # Initialize MLflow
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.config['phase4']['mlflow_experiment'])
        
        logger.info(f"Initialized Model Registry at: {self.registry_path}")
        logger.info(f"MLflow tracking URI: {self.mlflow_tracking_uri}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def _setup_registry_structure(self) -> None:
        """Create registry directory structure."""
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        for stage in ModelStage:
            (self.registry_path / stage.value).mkdir(exist_ok=True)
        
        logger.info(f"Registry structure created at {self.registry_path}")
    
    def register_model(self, model_name: str, model_version: str, 
                       stage: ModelStage = ModelStage.DEVELOPMENT,
                       metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Register a model in the registry.
        
        Args:
            model_name: Name of the model (e.g., 'sgd', 'random_forest')
            model_version: Version string (e.g., 'v1.0.1')
            stage: Initial lifecycle stage
            metadata: Additional metadata
        
        Returns:
            Registration details
        """
        logger.info(f"Registering model: {model_name} {model_version} → {stage.value}")
        
        # Setup registry structure
        self._setup_registry_structure()
        
        # Source model files
        source_model_path = self.project_root / 'models' / 'phase4' / f"{model_name}_{model_version}.pkl"
        source_manifest_path = self.project_root / 'models' / 'phase4' / f"model_manifest_{model_version}.json"
        
        if not source_model_path.exists():
            raise FileNotFoundError(f"Model not found: {source_model_path}")
        
        # Load the actual model object for MLflow
        with open(source_model_path, 'rb') as f:
            model_object = pickle.load(f)
        
        # Create registry entry directory
        registry_entry_dir = self.registry_path / stage.value / f"{model_name}_{model_version}"
        registry_entry_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model file
        dest_model_path = registry_entry_dir / f"{model_name}.pkl"
        shutil.copy2(source_model_path, dest_model_path)
        
        # Load and enhance metadata
        model_metadata = self._create_model_metadata(
            model_name=model_name,
            model_version=model_version,
            stage=stage,
            source_manifest_path=source_manifest_path,
            custom_metadata=metadata
        )
        
        # Save metadata
        metadata_path = registry_entry_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        # Register with MLflow Model Registry (pass actual model object)
        mlflow_model_uri = self._register_with_mlflow(
            model_name, model_version, model_object, str(dest_model_path)
        )
        
        logger.info(f"Model registered successfully: {registry_entry_dir}")
        logger.info(f"MLflow URI: {mlflow_model_uri}")
        
        return {
            'model_name': model_name,
            'version': model_version,
            'stage': stage.value,
            'path': str(registry_entry_dir),
            'metadata_path': str(metadata_path),
            'mlflow_uri': mlflow_model_uri,
            'registered_at': datetime.now().isoformat()
        }
    
    def _create_model_metadata(self, model_name: str, model_version: str,
                               stage: ModelStage, source_manifest_path: Path,
                               custom_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create comprehensive model metadata with lineage tracking."""
        
        # Load source manifest if exists
        source_metadata = {}
        if source_manifest_path.exists():
            with open(source_manifest_path, 'r') as f:
                source_metadata = json.load(f)
        
        # Get git commit hash
        try:
            import subprocess
            git_commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                cwd=self.project_root,
                stderr=subprocess.DEVNULL
            ).decode('ascii').strip()
        except:
            git_commit = "unknown"
        
        # Build lineage information
        lineage = {
            'code_version': {
                'git_commit': git_commit,
                'git_branch': self._get_git_branch(),
                'script': 'p4.2-training-pipeline.py'
            },
            'data_version': {
                'embeddings_version': 'v2.0',
                'train_samples': 3341,
                'val_samples': 716,
                'test_samples': 717,
                'num_classes': 41
            },
            'hyperparameters': self._get_model_hyperparams(model_name),
            'training_metrics': source_metadata.get('models', {}).get(model_name, {}).get('metrics', {})
        }
        
        # Create comprehensive metadata
        metadata = {
            'model_name': model_name,
            'version': model_version,
            'stage': stage.value,
            'registered_at': datetime.now().isoformat(),
            'registered_by': 'mavyjimz',
            'lineage': lineage,
            'performance': {
                'accuracy': source_metadata.get('models', {}).get(model_name, {}).get('metrics', {}).get('test_accuracy', 0),
                'f1_score': source_metadata.get('models', {}).get(model_name, {}).get('metrics', {}).get('test_f1_macro', 0),
                'meets_target': source_metadata.get('models', {}).get(model_name, {}).get('metrics', {}).get('test_accuracy', 0) >= 0.90
            },
            'deployment': {
                'ready_for_staging': False,
                'ready_for_production': False,
                'deployment_notes': ''
            },
            'custom_metadata': custom_metadata or {}
        }
        
        return metadata
    
    def _get_git_branch(self) -> str:
        """Get current git branch."""
        try:
            import subprocess
            branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=self.project_root,
                stderr=subprocess.DEVNULL
            ).decode('ascii').strip()
            return branch
        except:
            return "unknown"
    
    def _get_model_hyperparams(self, model_name: str) -> Dict[str, Any]:
        """Get hyperparameters for a model."""
        hyperparams = {
            'sgd': {
                'loss': 'hinge',
                'penalty': 'l2',
                'alpha': 0.0001,
                'max_iter': 1000,
                'tol': 0.001,
                'random_state': 42
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 20,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'class_weight': 'balanced'
            }
        }
        return hyperparams.get(model_name, {})
    
    def _register_with_mlflow(self, model_name: str, model_version: str, 
                              model_object: Any, model_path: str) -> str:
        """Register model with MLflow Model Registry using actual model object."""
        try:
            # Create MLflow model name
            mlflow_model_name = f"multi_model_orchestration_{model_name}"
            
            # Log model to MLflow with proper sklearn flavor
            with mlflow.start_run():
                mlflow.sklearn.log_model(
                    sk_model=model_object,  # Pass actual model object, not path
                    artifact_path="model",
                    registered_model_name=mlflow_model_name,
                    input_example=None  # Could add sample input for schema
                )
                
                # Log additional metadata
                mlflow.log_param("model_version", model_version)
                mlflow.log_param("model_path", model_path)
            
            model_uri = f"models:/{mlflow_model_name}/{model_version}"
            logger.info(f"Registered with MLflow: {model_uri}")
            return model_uri
            
        except Exception as e:
            logger.warning(f"MLflow registration failed: {e}")
            return f"models:/phase4/{model_name}_{model_version}"
    
    def promote_model(self, model_name: str, model_version: str,
                      from_stage: ModelStage, to_stage: ModelStage,
                      notes: str = "") -> Dict[str, Any]:
        """
        Promote a model to a higher stage.
        
        Args:
            model_name: Name of the model
            model_version: Version to promote
            from_stage: Current stage
            to_stage: Target stage
            notes: Promotion notes
        
        Returns:
            Promotion details
        """
        logger.info(f"Promoting {model_name} {model_version}: {from_stage.value} → {to_stage.value}")
        
        # Validate promotion path
        stage_order = [ModelStage.DEVELOPMENT, ModelStage.STAGING, 
                      ModelStage.PRODUCTION, ModelStage.ARCHIVED]
        
        if stage_order.index(to_stage) <= stage_order.index(from_stage):
            if to_stage != ModelStage.ARCHIVED:
                raise ValueError(f"Invalid promotion: {from_stage.value} → {to_stage.value}")
        
        # Source and destination paths
        source_path = self.registry_path / from_stage.value / f"{model_name}_{model_version}"
        dest_path = self.registry_path / to_stage.value / f"{model_name}_{model_version}"
        
        if not source_path.exists():
            raise FileNotFoundError(f"Model not found in {from_stage.value}: {source_path}")
        
        # Copy to new stage
        shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
        
        # Update metadata
        metadata_path = dest_path / 'metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        metadata['stage'] = to_stage.value
        metadata['promoted_at'] = datetime.now().isoformat()
        metadata['promotion_history'] = metadata.get('promotion_history', [])
        metadata['promotion_history'].append({
            'from_stage': from_stage.value,
            'to_stage': to_stage.value,
            'timestamp': metadata['promoted_at'],
            'notes': notes
        })
        
        # Update deployment readiness
        if to_stage == ModelStage.STAGING:
            metadata['deployment']['ready_for_staging'] = True
        elif to_stage == ModelStage.PRODUCTION:
            metadata['deployment']['ready_for_production'] = True
            metadata['deployment']['deployment_notes'] = notes
        
        # Save updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model promoted successfully to {to_stage.value}")
        
        return {
            'model_name': model_name,
            'version': model_version,
            'from_stage': from_stage.value,
            'to_stage': to_stage.value,
            'promoted_at': metadata['promoted_at'],
            'notes': notes
        }
    
    def list_models(self, stage: Optional[ModelStage] = None) -> List[Dict[str, Any]]:
        """List all registered models, optionally filtered by stage."""
        models = []
        
        stages_to_check = [stage] if stage else [s for s in ModelStage]
        
        for stage_to_check in stages_to_check:
            stage_path = self.registry_path / stage_to_check.value
            
            if not stage_path.exists():
                continue
            
            for model_dir in stage_path.iterdir():
                if model_dir.is_dir():
                    metadata_path = model_dir / 'metadata.json'
                    if metadata_path.exists():
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        models.append(metadata)
        
        return models
    
    def get_model_info(self, model_name: str, model_version: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific model."""
        for stage in ModelStage:
            model_path = self.registry_path / stage.value / f"{model_name}_{model_version}"
            metadata_path = model_path / 'metadata.json'
            
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    return json.load(f)
        
        return None
    
    def generate_registry_index(self) -> Dict[str, Any]:
        """Generate a comprehensive registry index."""
        logger.info("Generating registry index...")
        
        registry_index = {
            'generated_at': datetime.now().isoformat(),
            'total_models': 0,
            'by_stage': {},
            'models': []
        }
        
        for stage in ModelStage:
            models = self.list_models(stage=stage)
            registry_index['by_stage'][stage.value] = len(models)
            registry_index['total_models'] += len(models)
            registry_index['models'].extend(models)
        
        # Save index
        index_path = self.registry_path / 'registry_index.json'
        with open(index_path, 'w') as f:
            json.dump(registry_index, f, indent=2)
        
        logger.info(f"Registry index saved to: {index_path}")
        logger.info(f"Total registered models: {registry_index['total_models']}")
        
        return registry_index


def main():
    """Main registry setup and model registration."""
    logger.info("="*60)
    logger.info("PHASE 4.3: MODEL REGISTRY & VERSIONING")
    logger.info("="*60)
    
    try:
        # Initialize registry
        registry = ModelRegistry(config_path='config/config.yaml')
        
        # Register models from Phase 4.2
        models_to_register = [
            {
                'name': 'sgd',
                'version': 'v1.0.1',
                'initial_stage': ModelStage.DEVELOPMENT
            },
            {
                'name': 'random_forest',
                'version': 'v1.0.1',
                'initial_stage': ModelStage.DEVELOPMENT
            }
        ]
        
        registered_models = []
        for model_info in models_to_register:
            registration = registry.register_model(
                model_name=model_info['name'],
                model_version=model_info['version'],
                stage=model_info['initial_stage']
            )
            registered_models.append(registration)
            logger.info(f"✓ Registered: {model_info['name']} {model_info['version']}")
        
        # Promote best model (SGD) to staging
        logger.info("\nPromoting best model (SGD) to staging...")
        promotion = registry.promote_model(
            model_name='sgd',
            model_version='v1.0.1',
            from_stage=ModelStage.DEVELOPMENT,
            to_stage=ModelStage.STAGING,
            notes="Best performing model (96.93% accuracy). Ready for validation."
        )
        logger.info(f"✓ Promoted SGD v1.0.1 to staging")
        
        # Generate registry index
        registry_index = registry.generate_registry_index()
        
        # List all models
        logger.info("\n" + "="*60)
        logger.info("REGISTERED MODELS SUMMARY")
        logger.info("="*60)
        
        for stage in ModelStage:
            models = registry.list_models(stage=stage)
            if models:
                logger.info(f"\n{stage.value.upper()} ({len(models)} models):")
                for model in models:
                    logger.info(f"  - {model['model_name']} {model['version']}")
                    logger.info(f"    Accuracy: {model['performance']['accuracy']*100:.2f}%")
                    logger.info(f"    F1-Score: {model['performance']['f1_score']:.4f}")
                    logger.info(f"    Meets Target: {'✓' if model['performance']['meets_target'] else '✗'}")
        
        logger.info("\n" + "="*60)
        logger.info("PHASE 4.3 MODEL REGISTRY COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        
        print("\n" + "="*60)
        print("REGISTRY SUMMARY")
        print("="*60)
        print(f"Total models registered: {registry_index['total_models']}")
        print(f"Development: {registry_index['by_stage']['development']}")
        print(f"Staging: {registry_index['by_stage']['staging']}")
        print(f"Production: {registry_index['by_stage']['production']}")
        print(f"\nRegistry location: {registry.registry_path}")
        print(f"Registry index: {registry.registry_path / 'registry_index.json'}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Registry setup failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
