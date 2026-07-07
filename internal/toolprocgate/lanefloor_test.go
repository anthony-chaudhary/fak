package toolprocgate

import (
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestSubagentSiblingLaneTouchRefusedByFloor is the #2891 witness: a child
// spawned under a lane-scoped grant that tries to touch a SIBLING lane's
// files is refused by the inherited capability floor with a structured
// reason from the closed vocabulary — not by an env-var convention. The
// grant's env deliberately carries a Hermes-style decoy variable claiming
// the sibling lane; the floor adjudicates the GRANT, so the refusal stands.
func TestSubagentSiblingLaneTouchRefusedByFloor(t *testing.T) {
	broker := NewSpawnBroker()
	grant, err := broker.Admit(SpawnAttempt{
		AgentRunID:   "agent-1",
		ParentRunID:  "parent-1",
		ToolCallID:   "tool-1",
		PolicyDigest: "sha256:policy",
		Argv:         []string{"agent"},
		// The convention (an env var the child could read around) claims the
		// gateway lane; the adjudicated floor below says otherwise and wins.
		Env:     []EnvVar{{Name: "FAK_LANE_TREE", Value: "internal/gateway/**"}},
		CWD:     t.TempDir(),
		Backend: "guard",
		Envelope: CapabilityEnvelope{
			Capabilities: []abi.Capability{CapAgentRunSpawn},
			LaneTree:     []string{"internal/toolprocgate/**"},
		},
	})
	if err != nil {
		t.Fatalf("Admit: %v", err)
	}
	if len(grant.Envelope.LaneTree) != 1 || grant.Envelope.LaneTree[0] != "internal/toolprocgate/**" {
		t.Fatalf("grant lane tree = %v, want the granted floor on the grant itself", grant.Envelope.LaneTree)
	}

	denied := broker.AdmitFSTouch(grant, FSTouch{
		ToolCallID: "tool-1-write",
		Path:       "internal/gateway/gateway.go",
		AtMS:       1_000,
	})
	if denied.Allowed() || denied.Reason != ReasonSiblingLaneTouch {
		t.Fatalf("sibling-lane touch decision = %+v, want deny with SIBLING_LANE_TOUCH", denied)
	}
	if denied.LeakEvent == nil || denied.LeakEvent.Action != LeakFSDenied {
		t.Fatalf("sibling-lane refusal carried no fs_denied leak event: %+v", denied.LeakEvent)
	}
	if err := ValidateLeakEvent(*denied.LeakEvent); err != nil {
		t.Fatalf("refusal leak event failed the closed-vocabulary validator: %v", err)
	}
	if strings.Contains(denied.LeakEvent.BoundedRef.Digest, "gateway.go") {
		t.Fatalf("leak event carried the raw touched path, want a digest: %+v", denied.LeakEvent.BoundedRef)
	}

	allowed := broker.AdmitFSTouch(grant, FSTouch{
		ToolCallID: "tool-1-read",
		Path:       "internal/toolprocgate/output.go",
		AtMS:       1_001,
	})
	if !allowed.Allowed() || allowed.Reason != "" || allowed.LeakEvent != nil {
		t.Fatalf("in-lane touch decision = %+v, want silent allow", allowed)
	}

	var fsDenied []LeakEvent
	for _, ev := range broker.LeakEvents() {
		if ev.Action == LeakFSDenied {
			fsDenied = append(fsDenied, ev)
		}
	}
	if len(fsDenied) != 1 || fsDenied[0].Reason != ReasonSiblingLaneTouch || fsDenied[0].SourceChannel != "fs" {
		t.Fatalf("leak stream fs_denied rows = %+v, want exactly the sibling-lane refusal", fsDenied)
	}
	rep := LeakReportFromEvents(broker.LeakEvents())
	if rep.Counts.ByReason[ReasonSiblingLaneTouch] != 1 || rep.Denied < 1 {
		t.Fatalf("leak report did not witness the refusal: %+v", rep.Counts)
	}
}

// TestLaneFloorFailsClosedWithoutGrantedTree pins the anti-convention floor:
// a grant that carries NO lane tree admits nothing. The absence of a scope
// (Hermes' unset env var) is a structured refusal, not a wildcard.
func TestLaneFloorFailsClosedWithoutGrantedTree(t *testing.T) {
	broker := NewSpawnBroker()
	grant, err := broker.Admit(SpawnAttempt{
		AgentRunID:   "agent-1",
		ToolCallID:   "tool-1",
		PolicyDigest: "sha256:policy",
		Argv:         []string{"agent"},
		CWD:          t.TempDir(),
		Backend:      "guard",
		Envelope:     CapabilityEnvelope{Capabilities: []abi.Capability{CapAgentRunSpawn}},
	})
	if err != nil {
		t.Fatalf("Admit: %v", err)
	}
	d := broker.AdmitFSTouch(grant, FSTouch{ToolCallID: "tool-1-write", Path: "internal/toolprocgate/output.go", AtMS: 1})
	if d.Allowed() || d.Reason != ReasonMissingLaneFloor {
		t.Fatalf("unscoped-grant touch decision = %+v, want deny with MISSING_LANE_FLOOR", d)
	}
	if d.LeakEvent == nil {
		t.Fatal("unscoped-grant refusal carried no leak event")
	}
	if err := ValidateLeakEvent(*d.LeakEvent); err != nil {
		t.Fatalf("refusal leak event failed the closed-vocabulary validator: %v", err)
	}
}

// TestLaneFloorRefusesWorkspaceEscape pins that a touch which leaves the
// lane-tree coordinate system entirely — traversal, absolute, or the
// workspace root itself — is refused as LANE_PATH_ESCAPE even when a lane
// tree was granted.
func TestLaneFloorRefusesWorkspaceEscape(t *testing.T) {
	floor := SpawnGrant{
		AgentRunID: "agent-1", ToolCallID: "tool-1", PolicyDigest: "sha256:policy", Backend: "guard",
		Envelope: CapabilityEnvelope{LaneTree: []string{"internal/toolprocgate/**"}},
	}.LaneFloor()
	for _, p := range []string{
		"../fak-private/secrets.txt",
		"internal/toolprocgate/../../cmd/fak/main.go",
		// Would lexically clean back in-lane, but ".." is refused outright:
		// lexical cleaning does not match filesystem resolution under symlinks.
		"internal/toolprocgate/sub/../output.go",
		"/etc/passwd",
		`C:\work\other\file.go`,
		".",
		"",
	} {
		d := floor.AdmitTouch(FSTouch{ToolCallID: "tool-1-write", Path: p, AtMS: 1})
		if d.Allowed() || d.Reason != ReasonLanePathEscape {
			t.Fatalf("escape path %q decision = %+v, want deny with LANE_PATH_ESCAPE", p, d)
		}
	}
}

// TestSpawnDeniesMalformedLaneTree pins fail-closed at the OTHER end of the
// wire: a floor that cannot be trusted (traversing or absolute pattern) is
// never granted — the spawn itself is denied with a structured reason.
func TestSpawnDeniesMalformedLaneTree(t *testing.T) {
	for _, tree := range [][]string{
		{"../sibling/**"},
		{"/abs/**"},
		{" "},
	} {
		broker := NewSpawnBroker()
		_, err := broker.Admit(SpawnAttempt{
			AgentRunID:   "agent-1",
			ToolCallID:   "tool-1",
			PolicyDigest: "sha256:policy",
			Argv:         []string{"agent"},
			CWD:          t.TempDir(),
			Backend:      "guard",
			Envelope: CapabilityEnvelope{
				Capabilities: []abi.Capability{CapAgentRunSpawn},
				LaneTree:     tree,
			},
		})
		var denied SpawnDeniedError
		if !errors.As(err, &denied) || denied.Audit.Reason != "INVALID_LANE_TREE" {
			t.Fatalf("Admit with lane tree %v = %v, want SpawnDeniedError INVALID_LANE_TREE", tree, err)
		}
	}
}
