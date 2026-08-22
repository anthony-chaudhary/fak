package ultracodebench

import (
	"strings"
	"testing"
)

func TestJoinActivationKeepsUnverifiedEvidenceHonest(t *testing.T) {
	r, err := BeforeSpawn(BeforeSpawnInput{RunID: "run", ChildID: "active", Harness: "codex", Requested: SettingOn, Resolved: SettingOn, Injected: true})
	if err != nil {
		t.Fatal(err)
	}
	r, err = Acknowledge(r, ObservableActive, SourceRuntimeObservation)
	if err != nil {
		t.Fatal(err)
	}
	digest := "sha256:" + strings.Repeat("a", 64)
	rows, err := JoinActivation([]ActivationReceipt{r}, []DownstreamEvidence{
		{RunID: "run", ChildID: "active", Outcome: "accepted", Usage: UsageEvidence{BilledTokens: 10, SpendUSD: .01}, TrajectoryDigest: digest},
		{RunID: "run", ChildID: "missing", Outcome: "accepted", Usage: UsageEvidence{BilledTokens: 11}, TrajectoryDigest: digest},
	})
	if err != nil {
		t.Fatal(err)
	}
	if rows[0].Attribution != AttributionVerified || rows[0].ActivationState != ActivationActive {
		t.Fatalf("active attribution=%+v", rows[0])
	}
	if rows[1].Attribution != AttributionUnverified || rows[1].ActivationState != ActivationUnknown || rows[1].Activation != nil {
		t.Fatalf("missing attribution=%+v", rows[1])
	}
}

func TestJoinActivationRejectsTrajectoryPaths(t *testing.T) {
	_, err := JoinActivation(nil, []DownstreamEvidence{{RunID: "run", ChildID: "child", Outcome: "accepted", TrajectoryDigest: `C:\\private\\session.jsonl`}})
	if err == nil {
		t.Fatal("trajectory path was retained instead of requiring a digest")
	}
}
