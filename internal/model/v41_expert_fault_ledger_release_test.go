package model

import (
	"runtime"
	"testing"
	"weak"
)

// A V4.1 forward installs a fault ledger for its model; the registry must not
// keep that model reachable once every other reference is gone, or every
// reloaded serving model and every test fixture stays resident for the life of
// the process.
// fak-test:runtime fast est=100ms lane=default
func TestV41ExpertFaultLedgerDoesNotRetainModel(t *testing.T) {
	m := &Model{}
	m.v41SetExpertFaultPhase(V41PhasePrefill)
	key := weak.Make(m)
	if v41ExpertFaultLedgerOf(m, false) == nil {
		t.Fatal("setting a phase did not install the model's ledger")
	}
	m = nil
	for i := 0; i < 50; i++ {
		runtime.GC()
		if _, ok := v41ExpertFaultLedgers.Load(key); !ok && key.Value() == nil {
			return
		}
		runtime.Gosched()
	}
	_, held := v41ExpertFaultLedgers.Load(key)
	t.Fatalf("released model still reachable=%v, ledger entry held=%v", key.Value() != nil, held)
}
