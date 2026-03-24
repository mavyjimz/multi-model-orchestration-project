# Business Continuity Plan

## Objective
Maintain critical operations during system outages.

## Degraded Operation Modes
1. **Offline Mode**: Use rule-based fallback (scripts/continuity/offline-mode.py)
2. **Cached Responses**: Serve last known good predictions for 1 hour
3. **Manual Processing**: Queue requests for manual review during extended outages

## Activation Criteria
- API unavailable for > 15 minutes
- Model registry corruption detected
- Infrastructure failure in primary region

## Recovery Steps
1. Activate offline mode script
2. Notify stakeholders via escalation matrix
3. Restore from backup (see DR Runbook)
4. Validate system health
5. Deactivate offline mode
