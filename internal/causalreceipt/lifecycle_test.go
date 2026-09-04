package causalreceipt

import (
	"testing"
	"time"
)

// Invariant: Causal receipts must preserve deterministic incident attribution across all phase transitions.
// Guard: Validate enforces parent-child causal phase links and lifecycle release consistency.

func TestCausalReceiptLifecycleValidation(t *testing.T) {
	t.Parallel()

	now := time.Now().UTC()
	r := Receipt{
		Schema: Schema,
		IDs: IDs{
			Work:         "work-test-lifecycle",
			Turn:         "turn-1",
			Graph:        "graph-1",
			Request:      "req-1",
			ModelSession: "sess-1",
		},
		Phases: []Phase{
			{
				ID:           "p1",
				Kind:         "agent",
				Engine:       "fak-native",
				Backend:      "offline",
				Outcome:      "completed",
				Started:      now,
				Ended:        now.Add(10 * time.Millisecond),
				OperationIDs: []string{"op-1"},
			},
			{
				ID:              "p2",
				ParentID:        "p1",
				Kind:            "model",
				Engine:          "fak-native",
				Backend:         "metal",
				Outcome:         "completed",
				Started:         now.Add(10 * time.Millisecond),
				Ended:           now.Add(20 * time.Millisecond),
				Tokens:          128,
				Bytes:           1024,
				CacheReuseBytes: 512,
				QueueNS:         100,
				LoadNS:          200,
				VerificationNS:  300,
				ResourceIDs:     []string{"res-1"},
			},
		},
		Resources: []Resource{
			{
				ID:              "res-1",
				Kind:            "model_weights",
				State:           "released",
				PlannedLocality: "device",
				ActualLocality:  "device",
				Bytes:           1024,
				Released:        true,
			},
		},
		Decisions: []Decision{
			{
				Kind:    "route",
				ID:      "d-1",
				Reason:  "resident",
				Planned: "metal",
				Actual:  "metal",
			},
		},
	}

	if err := Validate(r); err != nil {
		t.Fatalf("Validate failed for clean receipt: %v", err)
	}

	m, err := DeriveMetrics(r)
	if err != nil {
		t.Fatalf("DeriveMetrics failed: %v", err)
	}
	if m.Tokens != 128 || m.Bytes != 1024 || m.CacheReuseBytes != 512 {
		t.Fatalf("unexpected metrics values: %+v", m)
	}
}
