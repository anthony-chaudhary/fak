package lifecycle

import "testing"

func TestOperatorControlsByPhase(t *testing.T) {
	tests := []struct {
		name, control, stage, wantStage string
		members                         []Member
		wantActions                     int
		hold                            bool
	}{
		{"cancel prepare", "cancel", "prepare", "cancelled", nil, 0, false},
		{"cancel partial pause", "cancel", "partial_pause", "cancelled", []Member{{ID: "b", State: "paused"}}, 1, false},
		{"cancel host action", "cancel", "host_action", "operator_hold", []Member{{ID: "a", State: "paused"}}, 1, true},
		{"cancel partial restore", "cancel", "partial_restore", "operator_hold", []Member{{ID: "a", State: "missing"}}, 1, true},
		{"resume verifies ready", "resume", "partial_restore", "ready", []Member{{ID: "a", State: "restored", Readback: true}}, 1, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, err := Control(Transaction{ID: "tx-1", Stage: tt.stage, Members: tt.members}, tt.control)
			if err != nil {
				t.Fatal(err)
			}
			if p.NextStage != tt.wantStage || len(p.Actions) != tt.wantActions || p.OperatorHold != tt.hold {
				t.Fatalf("preview=%+v", p)
			}
		})
	}
}
func TestRollbackRequiresIndependentReadbackAndIsIdempotent(t *testing.T) {
	tx := Transaction{ID: "tx-1", Stage: "partial_restore", Members: []Member{{ID: "a", State: "restored"}}}
	p, _ := Control(tx, "rollback")
	if p.Outcome != "" || !p.OperatorHold {
		t.Fatalf("unwitnessed rollback claimed success: %+v", p)
	}
	tx.Members[0].Readback = true
	p, _ = Control(tx, "rollback")
	if p.Outcome != "rolled_back" {
		t.Fatalf("witnessed rollback=%+v", p)
	}
	tx = Apply(tx, p)
	again, _ := Control(tx, "rollback")
	if len(again.Actions) != 0 || again.Outcome != "rolled_back" {
		t.Fatalf("idempotent rollback=%+v", again)
	}
}
