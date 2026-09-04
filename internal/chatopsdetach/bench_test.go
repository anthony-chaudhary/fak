package chatopsdetach

import (
	"testing"
	"time"
)

// BenchmarkChatOpsDetach exercises the core pure decision folds in a loop.
func BenchmarkChatOpsDetach(b *testing.B) {
	cmd := Command{
		Nonce:   "ts-bench-100",
		Verb:    VerbDispatch,
		Target:  "#2265",
		Channel: "C1",
		User:    "u1",
	}
	adm := Admission{
		Admitted: true,
		RunID:    "run-bench",
		Lane:     "chatops",
	}
	prior := Record{
		Nonce: "ts-bench-100",
		RunID: "run-bench",
		Lane:  "chatops",
	}
	stall := Stall{
		RunID:     "run-bench",
		Issue:     "#2265",
		SilentFor: 15 * time.Minute,
		Budget:    10 * time.Minute,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Exercise dispatch admission fold
		d1 := Decide(cmd, adm, Record{})
		if d1.Action != Dispatch {
			b.Fatalf("unexpected action: %v", d1.Action)
		}

		// Exercise idempotent re-ack fold
		d2 := Decide(cmd, adm, prior)
		if d2.Action != ReAck {
			b.Fatalf("unexpected action: %v", d2.Action)
		}

		// Exercise stall escalation judgment
		esc := JudgeStall(stall)
		if !esc.Escalate {
			b.Fatalf("unexpected stall verdict: %+v", esc)
		}
	}
}
