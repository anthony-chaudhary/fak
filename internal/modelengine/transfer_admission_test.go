package modelengine

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// These tests pin the two-budget admission contract adapted from vLLM
// (vllm/config/scheduler.py and
// vllm/distributed/kv_transfer/kv_connector/v1/base.py at
// 56e993a68990cd3bb0719d5d91dab549ce0d5a79, Apache-2.0): execution slots and
// in-flight KV-transfer slots are INDEPENDENT ceilings. A full compute batch
// still admits a bounded transfer start, the transfer cap strictly bounds
// concurrent starts, and releasing a transfer slot admits the next waiter.
// The tests fail if the two caps are conflated.

// fak-test:runtime fast est=10ms lane=default
func TestTransferAdmission(t *testing.T) {
	t.Run("config rejects non-positive budgets", func(t *testing.T) {
		for _, tc := range []struct{ exec, transfer int }{
			{0, 1}, {1, 0}, {-1, 1}, {1, -1}, {0, 0},
		} {
			if _, err := NewTransferAdmission(tc.exec, tc.transfer); !errors.Is(err, ErrTransferAdmissionConfig) {
				t.Fatalf("NewTransferAdmission(%d,%d) err=%v, want ErrTransferAdmissionConfig", tc.exec, tc.transfer, err)
			}
		}
	})

	t.Run("transfer budget exhaustion defers admission", func(t *testing.T) {
		a, err := NewTransferAdmission(8, 2)
		if err != nil {
			t.Fatal(err)
		}
		// Two transfer-needing requests fit the transfer budget.
		for i := 0; i < 2; i++ {
			d := a.Admit(TransferRequest{ID: "t", NeedsTransfer: true})
			if d.Verdict != cachemeta.AdmissionAllow || !d.TransferSlot {
				t.Fatalf("transfer %d: verdict=%q slot=%v, want allow+slot", i, d.Verdict, d.TransferSlot)
			}
		}
		// The third is deferred: the transfer budget, not the execution budget,
		// is what refuses it (execution cap is 8 and only 2 are used).
		d := a.Admit(TransferRequest{ID: "t", NeedsTransfer: true})
		if d.Verdict != cachemeta.AdmissionDefer {
			t.Fatalf("third transfer verdict=%q, want defer", d.Verdict)
		}
		if d.TransferSlot || d.ExecutionSlot {
			t.Fatalf("deferred request consumed a slot: %+v", d)
		}
		st := a.Stats()
		if st.TransferUsed != 2 || st.TransferCap != 2 || st.Waiters != 1 || st.Deferred != 1 {
			t.Fatalf("stats after defer = %+v", st)
		}
	})

	t.Run("execution slots and transfer budget are independently capped", func(t *testing.T) {
		// A LARGE execution cap must not raise the transfer cap.
		a, err := NewTransferAdmission(100, 1)
		if err != nil {
			t.Fatal(err)
		}
		if d := a.Admit(TransferRequest{ID: "t0", NeedsTransfer: true}); d.Verdict != cachemeta.AdmissionAllow {
			t.Fatalf("first transfer verdict=%q, want allow", d.Verdict)
		}
		if d := a.Admit(TransferRequest{ID: "t1", NeedsTransfer: true}); d.Verdict != cachemeta.AdmissionDefer {
			t.Fatalf("second transfer verdict=%q, want defer despite exec cap 100", d.Verdict)
		}
		// A transfer start holds a transfer slot only: it must not draw down the
		// execution budget it is not yet using.
		if st := a.Stats(); st.TransferUsed != 1 || st.ExecutionUsed != 0 {
			t.Fatalf("conflated caps: %+v", st)
		}

		// Conversely, a FULL compute batch must still admit a bounded transfer
		// start: the transfer budget is what gates transfers, not the exec cap.
		b, err := NewTransferAdmission(1, 4)
		if err != nil {
			t.Fatal(err)
		}
		if d := b.Admit(TransferRequest{ID: "e0"}); d.Verdict != cachemeta.AdmissionAllow || !d.ExecutionSlot {
			t.Fatalf("exec fill verdict=%q slot=%v, want allow+exec", d.Verdict, d.ExecutionSlot)
		}
		// Execution budget is now full (1/1). A transfer-needing request is
		// still admitted because the transfer budget is separate.
		d := b.Admit(TransferRequest{ID: "t0", NeedsTransfer: true})
		if d.Verdict != cachemeta.AdmissionAllow || !d.TransferSlot {
			t.Fatalf("transfer over full batch verdict=%q slot=%v, want allow+transfer", d.Verdict, d.TransferSlot)
		}
		// It did NOT take an execution slot, proving the budgets are not the
		// same counter.
		if d.ExecutionSlot {
			t.Fatalf("transfer request took an execution slot from a full batch: %+v", d)
		}
		if st := b.Stats(); st.ExecutionUsed != 1 || st.TransferUsed != 1 {
			t.Fatalf("stats = %+v, want exec 1/1 transfer 1/4", st)
		}
	})

	t.Run("transfer starts cannot starve a runnable request of the batch", func(t *testing.T) {
		a, err := NewTransferAdmission(2, 4)
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 4; i++ {
			d := a.Admit(TransferRequest{ID: "loading", NeedsTransfer: true})
			if d.Verdict != cachemeta.AdmissionAllow || !d.TransferSlot || d.ExecutionSlot {
				t.Fatalf("transfer start %d = %+v, want a transfer slot and no execution slot", i, d)
			}
		}
		// Four loads are in flight against an execution cap of two; the batch
		// must still be empty for requests whose KV is already resident.
		for _, id := range []string{"ready-0", "ready-1"} {
			d := a.Admit(TransferRequest{ID: id})
			if d.Verdict != cachemeta.AdmissionAllow || !d.ExecutionSlot || d.TransferSlot || d.ID != id {
				t.Fatalf("runnable %s = %+v, want an execution slot behind four in-flight loads", id, d)
			}
		}
		if st := a.Stats(); st.ExecutionUsed != 2 || st.TransferUsed != 4 {
			t.Fatalf("stats = %+v, want exec 2/2 transfer 4/4", st)
		}
		// A finished load hands back its transfer slot and queues for the batch
		// like any runnable request.
		if _, ok := a.ReleaseTransfer(); ok {
			t.Fatal("ReleaseTransfer promoted a waiter with an empty queue")
		}
		if d := a.Admit(TransferRequest{ID: "loaded"}); d.Verdict != cachemeta.AdmissionDefer {
			t.Fatalf("loaded request over a full batch = %+v, want defer", d)
		}
		d, ok := a.ReleaseExecution()
		if !ok || d.ID != "loaded" || !d.ExecutionSlot || d.TransferSlot {
			t.Fatalf("ReleaseExecution promoted %+v ok=%v, want the loaded request into the batch", d, ok)
		}
	})

	t.Run("releasing a transfer slot admits the next waiter", func(t *testing.T) {
		a, err := NewTransferAdmission(4, 1)
		if err != nil {
			t.Fatal(err)
		}
		if d := a.Admit(TransferRequest{ID: "t0", NeedsTransfer: true}); d.Verdict != cachemeta.AdmissionAllow {
			t.Fatalf("first verdict=%q, want allow", d.Verdict)
		}
		if d := a.Admit(TransferRequest{ID: "t1", NeedsTransfer: true}); d.Verdict != cachemeta.AdmissionDefer {
			t.Fatalf("second verdict=%q, want defer", d.Verdict)
		}
		if d := a.Admit(TransferRequest{ID: "t2", NeedsTransfer: true}); d.Verdict != cachemeta.AdmissionDefer || d.ID != "t2" {
			t.Fatalf("third decision=%+v, want t2 deferred", d)
		}
		if st := a.Stats(); st.Waiters != 2 {
			t.Fatalf("waiters=%d, want 2", st.Waiters)
		}
		d, ok := a.ReleaseTransfer()
		if !ok {
			t.Fatal("ReleaseTransfer promoted no waiter")
		}
		if d.Verdict != cachemeta.AdmissionAllow || !d.TransferSlot || d.ID != "t1" {
			t.Fatalf("promoted %+v, want the first waiter t1 with a transfer slot", d)
		}
		if st := a.Stats(); st.Waiters != 1 || st.TransferUsed != 1 || st.Admitted != 2 {
			t.Fatalf("stats after promote = %+v", st)
		}
		if d, ok := a.ReleaseTransfer(); !ok || d.ID != "t2" {
			t.Fatalf("second release promoted %+v ok=%v, want t2", d, ok)
		}
		// Nothing left to promote.
		if _, ok := a.ReleaseTransfer(); ok {
			t.Fatal("ReleaseTransfer promoted a waiter with an empty queue")
		}
	})

	t.Run("release promotes the first waiter its own budget allows", func(t *testing.T) {
		// Execution cap 1, transfer cap 1. Fill both, then queue a
		// non-transfer waiter (blocked on exec) ahead of a transfer waiter
		// (blocked on transfer). Releasing the transfer slot must skip the
		// exec-blocked waiter and promote the transfer waiter.
		a, err := NewTransferAdmission(1, 1)
		if err != nil {
			t.Fatal(err)
		}
		if d := a.Admit(TransferRequest{ID: "e0"}); d.Verdict != cachemeta.AdmissionAllow {
			t.Fatalf("exec fill verdict=%q", d.Verdict)
		}
		if d := a.Admit(TransferRequest{ID: "t0", NeedsTransfer: true}); d.Verdict != cachemeta.AdmissionAllow {
			t.Fatalf("transfer fill verdict=%q", d.Verdict)
		}
		if d := a.Admit(TransferRequest{ID: "e1"}); d.Verdict != cachemeta.AdmissionDefer {
			t.Fatalf("exec waiter verdict=%q, want defer", d.Verdict)
		}
		if d := a.Admit(TransferRequest{ID: "t1", NeedsTransfer: true}); d.Verdict != cachemeta.AdmissionDefer {
			t.Fatalf("transfer waiter verdict=%q, want defer", d.Verdict)
		}
		d, ok := a.ReleaseTransfer()
		if !ok || d.Verdict != cachemeta.AdmissionAllow || !d.TransferSlot || d.ID != "t1" {
			t.Fatalf("ReleaseTransfer promoted %+v ok=%v, want the transfer waiter t1", d, ok)
		}
		if st := a.Stats(); st.Waiters != 1 || st.TransferUsed != 1 || st.ExecutionUsed != 1 {
			t.Fatalf("stats = %+v, want the exec waiter still queued", st)
		}
	})
}
