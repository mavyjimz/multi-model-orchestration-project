"""
Traffic Router Middleware
Routes requests between baseline and candidate models based on canary configuration
"""

import random
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib


@dataclass
class RoutingDecision:
    """Represents a routing decision"""
    request_id: str
    timestamp: str
    selected_model: str
    selected_version: str
    traffic_percentage: int
    routing_reason: str


class TrafficRouter:
    """
    Routes incoming requests between baseline and candidate models
    
    Uses consistent hashing for session stickiness and configurable
    traffic splitting percentages.
    """
    
    def __init__(self, deployment_id: str, config_path: str = None):
        self.deployment_id = deployment_id
        self.config_path = config_path
        
        self.baseline_model = "intent-classifier-sgd"
        self.baseline_version = "v1.0.2"
        self.candidate_model = "intent-classifier-sgd"
        self.candidate_version = "v1.0.3"
        self.candidate_traffic_percentage = 10  # Default 10%
        
        self.metrics_dir = Path(f"results/phase11/canary_deployments/metrics/{deployment_id}")
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        self.request_log: list = []
    
    def set_traffic_split(self, candidate_percentage: int):
        """Update the traffic split percentage"""
        if not 0 <= candidate_percentage <= 100:
            raise ValueError("Traffic percentage must be between 0 and 100")
        self.candidate_traffic_percentage = candidate_percentage
    
    def route_request(
        self,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
        sticky: bool = True
    ) -> RoutingDecision:
        """
        Route a request to either baseline or candidate model
        
        Args:
            request_id: Unique request identifier
            session_id: Session identifier for sticky routing
            sticky: Whether to use session-based sticky routing
        
        Returns:
            RoutingDecision with selected model information
        """
        if request_id is None:
            request_id = self._generate_request_id()
        
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        # Determine which model to use
        if sticky and session_id:
            # Use session-based sticky routing
            selected = self._sticky_route(session_id)
            reason = "sticky_session"
        else:
            # Use random routing based on traffic percentage
            selected = self._random_route()
            reason = "random_split"
        
        decision = RoutingDecision(
            request_id=request_id,
            timestamp=timestamp,
            selected_model=selected["model"],
            selected_version=selected["version"],
            traffic_percentage=self.candidate_traffic_percentage,
            routing_reason=reason
        )
        
        # Log the routing decision
        self._log_routing_decision(decision)
        
        return decision
    
    def _sticky_route(self, session_id: str) -> Dict[str, str]:
        """Route based on session ID for consistency"""
        # Hash session ID to get consistent routing
        hash_value = int(hashlib.md5(session_id.encode()).hexdigest(), 16)
        bucket = hash_value % 100
        
        if bucket < self.candidate_traffic_percentage:
            return {
                "model": self.candidate_model,
                "version": self.candidate_version
            }
        else:
            return {
                "model": self.baseline_model,
                "version": self.baseline_version
            }
    
    def _random_route(self) -> Dict[str, str]:
        """Route based on random percentage"""
        if random.randint(1, 100) <= self.candidate_traffic_percentage:
            return {
                "model": self.candidate_model,
                "version": self.candidate_version
            }
        else:
            return {
                "model": self.baseline_model,
                "version": self.baseline_version
            }
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        random_suffix = random.randint(1000, 9999)
        return f"req-{timestamp}-{random_suffix}"
    
    def _log_routing_decision(self, decision: RoutingDecision):
        """Log routing decision to file"""
        log_entry = {
            "request_id": decision.request_id,
            "timestamp": decision.timestamp,
            "deployment_id": self.deployment_id,
            "selected_model": decision.selected_model,
            "selected_version": decision.selected_version,
            "traffic_percentage": decision.traffic_percentage,
            "routing_reason": decision.routing_reason
        }
        
        # Append to daily log file
        date_str = datetime.utcnow().strftime("%Y%m%d")
        log_file = self.metrics_dir / f"routing_log_{date_str}.jsonl"
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")
        
        self.request_log.append(log_entry)
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        total = len(self.request_log)
        
        if total == 0:
            return {
                "total_requests": 0,
                "baseline_count": 0,
                "candidate_count": 0,
                "baseline_percentage": 0.0,
                "candidate_percentage": 0.0
            }
        
        candidate_count = sum(
            1 for log in self.request_log
            if log["selected_version"] == self.candidate_version
        )
        baseline_count = total - candidate_count
        
        return {
            "total_requests": total,
            "baseline_count": baseline_count,
            "candidate_count": candidate_count,
            "baseline_percentage": round(baseline_count / total * 100, 2),
            "candidate_percentage": round(candidate_count / total * 100, 2),
            "configured_candidate_percentage": self.candidate_traffic_percentage
        }
    
    def export_metrics(self) -> str:
        """Export routing metrics to JSON file"""
        stats = self.get_routing_stats()
        stats["deployment_id"] = self.deployment_id
        stats["exported_at"] = datetime.utcnow().isoformat() + "Z"
        
        metrics_file = self.metrics_dir / f"routing_stats_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(metrics_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return str(metrics_file)


if __name__ == "__main__":
    # Test the traffic router
    router = TrafficRouter(deployment_id="test-deployment-001")
    
    print("Traffic Router Test")
    print("=" * 50)
    
    # Simulate some requests
    for i in range(100):
        decision = router.route_request(session_id=f"session-{i % 10}")
    
    # Get stats
    stats = router.get_routing_stats()
    print(f"Total requests: {stats['total_requests']}")
    print(f"Baseline: {stats['baseline_count']} ({stats['baseline_percentage']}%)")
    print(f"Candidate: {stats['candidate_count']} ({stats['candidate_percentage']}%)")
    
    # Export metrics
    metrics_file = router.export_metrics()
    print(f"\nMetrics exported to: {metrics_file}")
