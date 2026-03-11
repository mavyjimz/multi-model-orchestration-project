#!/usr/bin/env python3
"""
Phase 3.4: Interactive Query Interface
Provides CLI interface for real-time similarity search queries against FAISS index.

Requirements:
- Load trained FAISS index and vectorizer
- CLI interface for user queries
- Support text and embedding queries
- Display top-k results with confidence scores
- Show retrieved document metadata
- Multi-turn query support
- Error handling for edge cases

Success Criteria:
- Real-time query response (<100ms)
- User-friendly output formatting
- Support for multi-turn queries
- Error handling for edge cases
"""

import os
import sys
import json
import time
import pickle
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import faiss
except ImportError:
    print("Error: faiss package not installed. Run: pip install faiss-cpu")
    sys.exit(1)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    print("Error: scikit-learn not installed. Run: pip install scikit-learn")
    sys.exit(1)


class InteractiveQueryInterface:
    """
    Interactive CLI interface for FAISS-based similarity search.
    Supports multi-turn queries with context tracking.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the query interface with FAISS index and vectorizer.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.vectorizer = None
        self.faiss_index = None
        self.metadata = None
        self.query_history = []
        self.results_history = []
        self.is_loaded = False
        
        # Load models and index
        self._load_resources()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except ImportError:
            # Fallback to default config if yaml not available
            print("Warning: PyYAML not installed, using default configuration")
            return self._get_default_config()
        except FileNotFoundError:
            print(f"Warning: Config file not found at {config_path}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "vector_db": {
                "index_path": "data/vector_db/faiss_index_v1.0/index.faiss",
                "metadata_path": "data/vector_db/faiss_index_v1.0/index_metadata.json",
                "vectorizer_path": "data/final/vectorizer.pkl"
            },
            "query": {
                "top_k": 5,
                "confidence_threshold": 0.5,
                "max_history": 10
            },
            "display": {
                "show_metadata": True,
                "show_scores": True,
                "max_content_length": 500
            }
        }
    
    def _load_resources(self) -> None:
        """Load FAISS index, vectorizer, and metadata."""
        print("=" * 60)
        print("PHASE 3.4: Interactive Query Interface")
        print("=" * 60)
        
        # Resolve paths relative to project root
        project_root = Path(__file__).parent.parent
        vector_db_config = self.config.get("vector_db", {})
        
        index_path = project_root / vector_db_config.get("index_path", 
                      "data/vector_db/faiss_index_v1.0/index.faiss")
        metadata_path = project_root / vector_db_config.get("metadata_path",
                         "data/vector_db/faiss_index_v1.0/index_metadata.json")
        vectorizer_path = project_root / vector_db_config.get("vectorizer_path",
                           "data/final/vectorizer.pkl")
        
        # Load vectorizer
        print(f"\n[1/3] Loading vectorizer from: {vectorizer_path}")
        if not vectorizer_path.exists():
            raise FileNotFoundError(f"Vectorizer not found at {vectorizer_path}")
        
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        print(f"      Vectorizer loaded: {self.vectorizer.get_feature_names_out().size} features")
        
        # Load FAISS index
        print(f"\n[2/3] Loading FAISS index from: {index_path}")
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        
        self.faiss_index = faiss.read_index(str(index_path))
        print(f"      FAISS index loaded: {self.faiss_index.ntotal} vectors")
        
        # Load metadata
        print(f"\n[3/3] Loading metadata from: {metadata_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        print(f"      Metadata loaded: {len(self.metadata)} documents")
        
        self.is_loaded = True
        print("\n" + "=" * 60)
        print("Resources loaded successfully. Ready for queries.")
        print("=" * 60)
    
    def _text_to_embedding(self, text: str) -> np.ndarray:
        """
        Convert text query to TF-IDF embedding.
        
        Args:
            text: Input text query
            
        Returns:
            TF-IDF embedding vector
        """
        if self.vectorizer is None:
            raise RuntimeError("Vectorizer not loaded")
        
        # Transform text using fitted vectorizer
        embedding = self.vectorizer.transform([text])
        return embedding.toarray().astype(np.float32)
    
    def _search(self, query_embedding: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform FAISS similarity search.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            
        Returns:
            Tuple of (distances, indices)
        """
        if self.faiss_index is None:
            raise RuntimeError("FAISS index not loaded")
        
        # Ensure embedding is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        
        return distances[0], indices[0]
    
    def _compute_confidence(self, distance: float) -> float:
        """
        Convert FAISS distance to confidence score.
        FAISS uses L2 distance, so lower = more similar.
        
        Args:
            distance: L2 distance from FAISS
            
        Returns:
            Confidence score between 0 and 1
        """
        # Convert L2 distance to similarity score
        # Using exponential decay: confidence = exp(-distance)
        confidence = np.exp(-distance)
        return float(confidence)
    
    def _get_document_info(self, index: int) -> Dict[str, Any]:
        """
        Retrieve document metadata by index.
        
        Args:
            index: Document index in FAISS
            
        Returns:
            Document metadata dictionary
        """
        if self.metadata is None or index >= len(self.metadata):
            return {"error": "Document not found"}
        
        return self.metadata[index]
    
    def _format_result(self, rank: int, index: int, distance: float, 
                       metadata: Dict[str, Any]) -> str:
        """
        Format a single search result for display.
        
        Args:
            rank: Result rank (1-based)
            index: FAISS index
            distance: L2 distance
            metadata: Document metadata
            
        Returns:
            Formatted result string
        """
        confidence = self._compute_confidence(distance)
        display_config = self.config.get("display", {})
        max_content_length = display_config.get("max_content_length", 500)
        
        # Build result string
        lines = []
        lines.append(f"\n  [{'=' * 50}]")
        lines.append(f"  Rank #{rank} | Index: {index} | Confidence: {confidence:.4f}")
        lines.append(f"  [{'=' * 50}]")
        
        # Show metadata
        if display_config.get("show_metadata", True):
            if "intent" in metadata:
                lines.append(f"  Intent: {metadata['intent']}")
            if "source" in metadata:
                lines.append(f"  Source: {metadata['source']}")
            if "doc_id" in metadata:
                lines.append(f"  Doc ID: {metadata['doc_id']}")
        
        # Show content preview
        if "content" in metadata:
            content = metadata["content"]
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."
            lines.append(f"\n  Content Preview:")
            lines.append(f"  {'-' * 40}")
            for line in content.split('\n')[:5]:
                lines.append(f"  {line}")
        
        return '\n'.join(lines)
    
    def query(self, text: str, top_k: int = None, show_results: bool = True) -> List[Dict[str, Any]]:
        """
        Execute a query and return results.
        
        Args:
            text: Query text
            top_k: Number of results (default from config)
            show_results: Whether to print results to console
            
        Returns:
            List of result dictionaries
        """
        if not self.is_loaded:
            raise RuntimeError("Resources not loaded")
        
        start_time = time.time()
        
        # Get top_k from config if not specified
        if top_k is None:
            top_k = self.config.get("query", {}).get("top_k", 5)
        
        # Convert text to embedding
        query_embedding = self._text_to_embedding(text)
        
        # Search
        distances, indices = self._search(query_embedding, top_k)
        
        # Build results
        results = []
        for rank, (idx, dist) in enumerate(zip(indices, distances), 1):
            if idx == -1:  # FAISS returns -1 for no match
                continue
            
            metadata = self._get_document_info(int(idx))
            confidence = self._compute_confidence(dist)
            
            result = {
                "rank": rank,
                "index": int(idx),
                "distance": float(dist),
                "confidence": confidence,
                "metadata": metadata
            }
            results.append(result)
            
            if show_results:
                print(self._format_result(rank, int(idx), dist, metadata))
        
        # Track query history
        elapsed_time = (time.time() - start_time) * 1000  # ms
        self.query_history.append({
            "query": text,
            "timestamp": time.time(),
            "latency_ms": elapsed_time,
            "num_results": len(results)
        })
        
        # Limit history size
        max_history = self.config.get("query", {}).get("max_history", 10)
        if len(self.query_history) > max_history:
            self.query_history = self.query_history[-max_history:]
        
        if show_results:
            print(f"\n  [{'=' * 50}]")
            print(f"  Query completed in {elapsed_time:.2f}ms")
            print(f"  Retrieved {len(results)} results")
            print(f"  [{'=' * 50}]")
        
        return results
    
    def run_interactive_mode(self) -> None:
        """
        Run interactive CLI loop for continuous queries.
        """
        print("\n" + "=" * 60)
        print("INTERACTIVE QUERY MODE")
        print("=" * 60)
        print("\nCommands:")
        print("  [query text]  - Search for similar documents")
        print("  /k [number]   - Set top-k results (default: 5)")
        print("  /history      - Show query history")
        print("  /clear        - Clear query history")
        print("  /stats        - Show session statistics")
        print("  /export       - Export results to JSON")
        print("  /quit         - Exit interactive mode")
        print("=" * 60)
        
        current_top_k = self.config.get("query", {}).get("top_k", 5)
        
        while True:
            try:
                # Get user input
                user_input = input("\n[Query]> ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    command = user_input.split()[0].lower()
                    args = user_input.split()[1:]
                    
                    if command in ["/quit", "/exit", "/q"]:
                        print("\nExiting interactive mode...")
                        break
                    
                    elif command == "/k":
                        if args and args[0].isdigit():
                            current_top_k = int(args[0])
                            print(f"Top-k set to: {current_top_k}")
                        else:
                            print(f"Current top-k: {current_top_k}")
                            print("Usage: /k [number]")
                    
                    elif command == "/history":
                        if not self.query_history:
                            print("No query history")
                        else:
                            print(f"\nQuery History ({len(self.query_history)} queries):")
                            for i, q in enumerate(self.query_history[-5:], 1):
                                print(f"  {i}. {q['query'][:50]}... ({q['latency_ms']:.2f}ms)")
                    
                    elif command == "/clear":
                        self.query_history = []
                        self.results_history = []
                        print("Query history cleared")
                    
                    elif command == "/stats":
                        self._print_stats()
                    
                    elif command == "/export":
                        self._export_results()
                    
                    else:
                        print(f"Unknown command: {command}")
                        print("Type /quit to exit, or enter a query")
                
                else:
                    # Execute query
                    print(f"\nSearching for: '{user_input}'")
                    results = self.query(user_input, top_k=current_top_k, show_results=True)
                    self.results_history.extend(results)
            
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type /quit to exit.")
            except Exception as e:
                print(f"\nError: {str(e)}")
                print("Please try again with a different query")
    
    def _print_stats(self) -> None:
        """Print session statistics."""
        if not self.query_history:
            print("No queries executed yet")
            return
        
        latencies = [q["latency_ms"] for q in self.query_history]
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        print("\n" + "-" * 40)
        print("Session Statistics:")
        print("-" * 40)
        print(f"  Total queries: {len(self.query_history)}")
        print(f"  Avg latency: {avg_latency:.2f}ms")
        print(f"  Min latency: {min_latency:.2f}ms")
        print(f"  Max latency: {max_latency:.2f}ms")
        print(f"  Total results: {len(self.results_history)}")
        print("-" * 40)
    
    def _export_results(self, filename: str = "query_results_export.json") -> None:
        """Export query history and results to JSON."""
        export_data = {
            "query_history": self.query_history,
            "results": self.results_history,
            "session_stats": {
                "total_queries": len(self.query_history),
                "total_results": len(self.results_history)
            }
        }
        
        output_path = Path(filename)
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"Results exported to: {output_path.absolute()}")
    
    def run_single_query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Run a single query without interactive mode.
        Useful for scripting and automation.
        
        Args:
            query_text: Query string
            top_k: Number of results
            
        Returns:
            List of result dictionaries
        """
        return self.query(query_text, top_k=top_k, show_results=True)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Phase 3.4: Interactive Query Interface for FAISS Similarity Search"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Single query text (non-interactive mode)"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--export", "-e",
        action="store_true",
        help="Export results after query"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize interface
        interface = InteractiveQueryInterface(config_path=args.config)
        
        if args.query:
            # Single query mode
            print(f"\nExecuting single query: '{args.query}'")
            results = interface.run_single_query(args.query, top_k=args.top_k)
            
            if args.export:
                interface._export_results()
        else:
            # Interactive mode
            interface.run_interactive_mode()
    
    except FileNotFoundError as e:
        print(f"\nError: Required file not found - {e}")
        print("\nPlease ensure the following files exist:")
        print("  - data/final/vectorizer.pkl")
        print("  - data/vector_db/faiss_index_v1.0/index.faiss")
        print("  - data/vector_db/faiss_index_v1.0/index_metadata.json")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
