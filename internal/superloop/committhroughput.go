package superloop

import "github.com/anthony-chaudhary/fak/internal/fleetmetrics"

const CommitRecoveryRef = "fleet-commit-throughput"

// GateCommitThroughput makes zero landed commits a top-level recovery member
// whenever a fleet is active. It runs after the normal walk, so it cannot hide
// concrete member debt; it prevents an otherwise-clean active loop from calling
// itself healthy while trunk output is stalled.
func GateCommitThroughput(decision DriveDecision, metric fleetmetrics.CommitThroughput, activeWorkers int) DriveDecision {
	health := metric.Health(activeWorkers)
	if health.Healthy || decision.Enter || !decision.Satisfied {
		return decision
	}
	decision.Enter = true
	decision.Satisfied = false
	decision.Member = Member{Kind: KindSurface, Ref: CommitRecoveryRef, Enter: health.NextAction}
	decision.Action = health.NextAction
	decision.Reason = health.Reason
	return decision
}
