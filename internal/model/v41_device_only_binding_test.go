package model

import (
	"errors"
	"slices"
	"testing"
)

func v41HostOpNames(ops []v41HostOp) []string {
	names := make([]string, len(ops))
	for i, op := range ops {
		names[i] = op.name
	}
	return names
}

// Ref #13668. An empty static inventory is not admission: a backend that
// leaves a used op's seam unbound would run it on the host, so device-only
// refuses with the typed inventory error naming that op. Today the Vulkan
// indexer score capability is hard-off, which this pins as a refusal.
// fak-test:runtime medium est=2s lane=default
func TestV41DeviceOnlyRefusesUnboundIndexerScore(t *testing.T) {
	saved := v41DeviceOnlyHostOps
	t.Cleanup(func() { v41DeviceOnlyHostOps = saved })
	v41DeviceOnlyHostOps = nil

	m := v41IndexerTestFixture(t)
	if !v41UsesIndexer(m.Cfg) || !v41UsesCompressedKV(m.Cfg) {
		t.Fatal("fixture must run the compressed indexer path")
	}
	without := v41DenseTestSession(t, m, newV41DenseTestBackend())
	if !slices.Contains(v41HostOpNames(without.v41DeviceOnlyUnboundOps()), "attention-indexer-score") {
		t.Fatalf("unbound ops %v omit the indexer score", v41HostOpNames(without.v41DeviceOnlyUnboundOps()))
	}
	with := v41DenseTestSession(t, m, newV41ScoreTestBackend())
	if slices.Contains(v41HostOpNames(with.v41DeviceOnlyUnboundOps()), "attention-indexer-score") {
		t.Fatal("a backend that admits the indexer score is still refused for it")
	}

	without.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
	err := recoverError(func() { without.validateDeviceOnlyExecution("decode") })
	var refused *BackendForwardOperationError
	var hostOps *v41DeviceOnlyHostOpError
	if !errors.As(err, &refused) || refused.Path != "device-only" || !errors.Is(err, errV41DeviceOnlyHostOp) || !errors.As(err, &hostOps) {
		t.Fatalf("device-only admission = %v, want typed host-op refusal", err)
	}
	if !slices.Contains(v41HostOpNames(hostOps.ops), "attention-indexer-score") || !without.BackendSessionClosed() {
		t.Fatalf("refusal %v does not name the unbound indexer score or left the session open", v41HostOpNames(hostOps.ops))
	}
}

// fak-test:runtime fast est=1s lane=default
func TestV41DeviceOnlyAdmissionJoinsInventoryAndBindings(t *testing.T) {
	savedOps, savedBindings := v41DeviceOnlyHostOps, v41DeviceOnlyBindings
	t.Cleanup(func() { v41DeviceOnlyHostOps, v41DeviceOnlyBindings = savedOps, savedBindings })

	seen := map[string]bool{}
	for _, b := range savedBindings {
		if b.op.name == "" || b.op.site == "" || b.bound == nil || seen[b.op.name] {
			t.Fatalf("binding %+v is blank or duplicated", b.op)
		}
		seen[b.op.name] = true
	}

	s := &Session{M: &Model{Cfg: Config{}}}
	bound := func(*Session) bool { return true }
	unbound := func(*Session) bool { return false }
	never := func(Config) bool { return false }
	v41DeviceOnlyHostOps = nil
	v41DeviceOnlyBindings = []v41DeviceOnlyBinding{
		{op: v41HostOp{name: "a", site: "A"}, bound: bound},
		{op: v41HostOp{name: "unused", site: "U"}, uses: never, bound: unbound},
	}
	if err := s.v41DeviceOnlyAdmissionRefusal(); err != nil {
		t.Fatalf("every used op bound and inventory empty: %v, want admission", err)
	}

	v41DeviceOnlyHostOps = []v41HostOp{{name: "static", site: "S"}, {name: "dup", site: "S2"}}
	v41DeviceOnlyBindings = append(v41DeviceOnlyBindings,
		v41DeviceOnlyBinding{op: v41HostOp{name: "missing", site: "M"}, bound: unbound},
		v41DeviceOnlyBinding{op: v41HostOp{name: "dup", site: "D"}, bound: unbound})
	err := s.v41DeviceOnlyAdmissionRefusal()
	var typed *v41DeviceOnlyHostOpError
	if !errors.Is(err, errV41DeviceOnlyHostOp) || !errors.As(err, &typed) {
		t.Fatalf("refusal = %v, want typed host-op error", err)
	}
	if got := v41HostOpNames(typed.ops); !slices.Equal(got, []string{"static", "dup", "missing"}) {
		t.Fatalf("refused ops = %v, want static inventory then unbound ops, deduplicated", got)
	}
}
