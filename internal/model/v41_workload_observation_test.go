package model

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime medium est=3s lane=default
// Synthetic CPU fixture through real producers and the existing serialized
// attribution record. Not a Vulkan, performance or restoration witness.
func TestV41WorkloadProducerToSerializedRecord(t *testing.T) {
	t.Setenv(v41SuffixUbatchEnv, "0")
	t.Setenv(v41SuffixRefoldMaxHistoryEnv, "0")
	m := v41IncrementalPlainModel(t, 2)
	s := m.NewSession()
	prefix, _ := v41IncrementalPrefix(m.Cfg, 2)
	s.Prefill(prefix)
	s.Prefill(v41SuffixTail(m.Cfg, 2, 3))
	s.Step(v41SuffixTail(m.Cfg, 5, 1)[0])
	at := m.V41ExpertFaultAttribution()
	want := V41WorkloadLifetime{
		ColdFull:     V41WorkloadRows{ObservedOperations: 1, ExecutedRows: 2, AppendedRows: 2},
		SuffixSerial: V41WorkloadRows{ObservedOperations: 3, ExecutedRows: 3, AppendedRows: 3},
		SerialDecode: V41WorkloadRows{ObservedOperations: 1, ExecutedRows: 1, AppendedRows: 1},
	}
	if at.Workload != want {
		t.Fatalf("observed routes: %+v, want %+v", at.Workload, want)
	}
	if at.Prefill.Tokens != 5 || at.Decode.Tokens != 1 {
		t.Fatal("legacy phase denominator changed")
	}
	if at.WorkloadCoverage.RestoredPrefixTokens != nil {
		t.Fatal("nonempty history fabricated restored count")
	}
	if at.WorkloadCoverage.CounterBinding != "not_bound_to_expert_or_device_timing" {
		t.Fatal("lifetime work became request timing")
	}
	data, err := json.Marshal(at)
	if err != nil {
		t.Fatal(err)
	}
	for _, required := range []string{`"cold_full_prefill"`, `"suffix_incremental_prefill"`, `"serial_incremental_decode"`, `"restored_prefix_tokens":null`, `"failed_partial_work":"unavailable_except_committed_cohort_lanes"`, `"effective_timestamp_mode":"unavailable_at_model_ledger"`} {
		if !strings.Contains(string(data), required) {
			t.Fatalf("missing %s in %s", required, data)
		}
	}
	// Fail a real cold forward after its attention/projection phase work, before
	// any successful route note. Missing success must not erase coverage.
	failedModel := v41IncrementalPlainModel(t, 2)
	sentinel := errors.New("workload observation witness route failure")
	failedModel.SetExpertRouteEventObserver(func(ExpertRouteEvent) { panic(sentinel) })
	failedSession := failedModel.NewSession()
	if err := panicAsError(func() { failedSession.Prefill(prefix) }); !errors.Is(err, sentinel) {
		t.Fatalf("failed forward lost sentinel: %v", err)
	}
	failed := failedModel.V41ExpertFaultAttribution()
	if failed.Prefill == (V41ExpertFaultAttributionPhase{}) {
		t.Fatal("failure fixture did not observe phase work")
	}
	if failed.Workload != (V41WorkloadLifetime{}) || failed.WorkloadCoverage.FailedPartialWork != "unavailable_except_committed_cohort_lanes" {
		t.Fatal("failed forward fabricated success or lost missingness")
	}
	ledger := v41ExpertFaultLedgerOf(m, false)
	ledger.mu.Lock()
	busy := m.V41ExpertFaultAttribution()
	ledger.mu.Unlock()
	if busy.WorkloadCoverage.Scope != "unavailable" || busy.WorkloadCoverage.Basis != "ledger_busy" {
		t.Fatal("busy snapshot fabricated fresh zero")
	}
}
