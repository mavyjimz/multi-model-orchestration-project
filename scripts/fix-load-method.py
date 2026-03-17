#!/usr/bin/env python3
"""Replace the broken load method in p4.4-inference-api.py with correct indentation"""

def fix_load_method(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # The correctly indented load method
    correct_load_method = '''    def load(self) -> bool:
        """Load model, vectorizer, and metadata"""
        try:
            logger.info("Loading model and vectorizer...")
            
            # Load model
            model_path = Config.MODEL_PHASE4_PATH / Config.MODEL_FILE
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            with open(model_path, 'rb') as f:
                loaded = pickle.load(f)
            
            # Extract model from dictionary wrapper if present
            if isinstance(loaded, dict):
                self.model = loaded.get("model", loaded)
                logger.info("  Model extracted from dictionary wrapper")
            else:
                self.model = loaded
            logger.info(f"✓ Model loaded: {model_path}")
            logger.info(f"  Model type: {self.model.__class__.__name__}")
            logger.info(f"  Model classes: {getattr(self.model, 'classes_', 'N/A')}")
            
            # Load vectorizer
            vectorizer_path = Config.EMBEDDING_PATH / Config.VECTORIZER_FILE
            if not vectorizer_path.exists():
                raise FileNotFoundError(f"Vectorizer not found: {vectorizer_path}")
            
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            logger.info(f"✓ Vectorizer loaded: {vectorizer_path}")
            logger.info(f"  Features: {self.vectorizer.get_feature_names_out().shape[0]}")
            
            # Load metadata
            metadata_path = Config.MODEL_PHASE4_PATH / "model_manifest_v1.0.1.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"✓ Metadata loaded: {metadata_path}")
            
            # Extract classes safely - MUST BE INSIDE TRY BLOCK
            if hasattr(self.model, "classes_"):
                self.classes = self.model.classes_.tolist()
            else:
                self.classes = []
                logger.warning("Model has no classes_ attribute")
            logger.info(f"✓ Classes loaded: {len(self.classes)} classes")
            
            self.is_loaded = True
            self.load_time = datetime.utcnow()
            
            logger.info("✓ Model loading complete")
            return True
            
        except Exception as e:
            logger.error(f"✗ Model loading failed: {str(e)}", exc_info=True)
            return False'''
    
    # Find and replace the load method
    import re
    # Pattern matches from "def load(self)" to the next method definition at same indent level
    pattern = r'    def load\(self\) -> bool:.*?(?=\n    def [a-z_]+\(|\nclass |\Z)'
    
    new_content = re.sub(pattern, correct_load_method, content, flags=re.DOTALL)
    
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print(f"Fixed load method in {file_path}")

if __name__ == '__main__':
    fix_load_method('scripts/p4.4-inference-api.py')
