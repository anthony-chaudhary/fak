package superloop

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/fleetmetrics"
)

func TestGateCommitThroughputMakesActiveZeroRateTopLevelWork(t *testing.T) {
	clean := DriveDecision{Intent: "night", Satisfied: true, Reason: "members clean"}
	got := GateCommitThroughput(clean, fleetmetrics.CommitThroughput{Measured: true}, 6)
	if !got.Enter || got.Satisfied || got.Member.Ref != CommitRecoveryRef || got.Action == "" {
		t.Fatalf("decision=%+v", got)
	}
}

func TestGateCommitThroughputAcceptsPositiveRate(t *testing.T) {
	clean := DriveDecision{Intent: "night", Satisfied: true, Reason: "members clean"}
	got := GateCommitThroughput(clean, fleetmetrics.CommitThroughput{Measured: true, Current: 1}, 6)
	if got.Enter || !got.Satisfied || got.Reason != clean.Reason {
		t.Fatalf("decision=%+v, want unchanged clean decision", got)
	}
}

func TestGateCommitThroughputDoesNotDisplaceConcreteDebt(t *testing.T) {
	debt := DriveDecision{Intent: "night", Enter: true, Member: Member{Ref: "real-work"}, Reason: "member debt"}
	got := GateCommitThroughput(debt, fleetmetrics.CommitThroughput{Measured: true}, 6)
	if got.Member.Ref != "real-work" {
		t.Fatalf("decision=%+v", got)
	}
}
