package allinone

import (
	"context"
	"errors"
	"strings"
	"testing"
)

// TestDefaultGuardrailsValidate pins the default-real contract: the zero
// Config never defaults to a mock - real mode requires a lock or bundle plus
// an explicitly configured engine, mock mode requires the explicit opt-in,
// and a contradictory mock selection is refused.
func TestDefaultGuardrailsValidate(t *testing.T) {
	lockFile := writeContractLock(t, "[]", "")
	cases := []struct {
		name    string
		cfg     Config
		wantErr bool
	}{
		{"zero config refuses", Config{}, true},
		{"real engine without lock refuses", Config{Engine: DefaultEngine}, true},
		{"lock without engine refuses", Config{LockPath: lockFile}, true},
		{"lock plus real engine admits", Config{LockPath: lockFile, Engine: DefaultEngine}, false},
		{"lock plus driver admits", Config{LockPath: lockFile, EngineDriver: contractStubEngine{}}, false},
		{"bare mock flag admits", Config{Mock: true}, false},
		{"bare mock engine admits", Config{Engine: MockEngineID}, false},
		{"mock flag plus mock engine admits", Config{LockPath: lockFile, Mock: true, Engine: MockEngineID}, false},
		{"mock flag plus real engine refuses", Config{LockPath: lockFile, Mock: true, Engine: DefaultEngine}, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.cfg.Validate()
			if tc.wantErr && err == nil {
				t.Fatalf("Validate() = nil, want error")
			}
			if !tc.wantErr && err != nil {
				t.Fatalf("Validate() = %v, want nil", err)
			}
		})
	}
	if err := (Config{LockPath: lockFile}).Validate(); !errors.Is(err, ErrEngineUnconfigured) {
		t.Fatalf("lock without engine err = %v, want ErrEngineUnconfigured", err)
	}
}

// TestDefaultGuardrailsLabels pins the resolved identities: the real default
// is inkernel, the only mock selector is the explicit opt-in, and an empty
// real-mode config reports unconfigured instead of silently becoming a mock.
func TestDefaultGuardrailsLabels(t *testing.T) {
	if DefaultEngine != "inkernel" {
		t.Fatalf("DefaultEngine = %q, want inkernel", DefaultEngine)
	}
	if MockEngineID != "mock" {
		t.Fatalf("MockEngineID = %q, want mock", MockEngineID)
	}
	if got := (Config{}).ResolvedEngine(); got != "unconfigured" {
		t.Fatalf("empty ResolvedEngine = %q, want unconfigured", got)
	}
	if got := (Config{Mock: true}).ResolvedEngine(); got != "mock" {
		t.Fatalf("mock-flag ResolvedEngine = %q, want mock", got)
	}
	if got := (Config{Engine: DefaultEngine}).ResolvedEngine(); got != DefaultEngine {
		t.Fatalf("real ResolvedEngine = %q, want %q", got, DefaultEngine)
	}
	if got := (Config{EngineDriver: contractStubEngine{}}).ResolvedEngine(); got != "custom" {
		t.Fatalf("driver ResolvedEngine = %q, want custom", got)
	}
	if (Config{}).IsMock() {
		t.Fatal("zero Config IsMock = true, want false: mock must never be the default")
	}
	if !(Config{Mock: true}).IsMock() || !(Config{Engine: MockEngineID}).IsMock() {
		t.Fatal("explicit mock opt-in IsMock = false, want true")
	}
	if (Config{Engine: DefaultEngine}).IsMock() {
		t.Fatal("real engine IsMock = true, want false")
	}
}

// TestDefaultGuardrailsContradictoryMockRefused pins the fail-fast refusal:
// a mock flag combined with a real engine is rejected by both DryRunTopology
// and Start, and nothing is bound.
func TestDefaultGuardrailsContradictoryMockRefused(t *testing.T) {
	lockFile := writeContractLock(t, "[]", "")
	cfg := Config{LockPath: lockFile, Addr: "127.0.0.1:0", Mock: true, Engine: DefaultEngine}
	sup, err := NewSupervisor(cfg)
	if err != nil {
		t.Fatalf("NewSupervisor: %v", err)
	}
	// DryRunTopology reports truthfully without enforcing; Start refuses.
	spec, err := sup.DryRunTopology()
	if err != nil {
		t.Fatalf("DryRunTopology: %v, want truthful report", err)
	}
	if spec.Engine != DefaultEngine {
		t.Fatalf("DryRunTopology Engine = %q, want %q", spec.Engine, DefaultEngine)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := sup.Start(ctx); err == nil || !strings.Contains(err.Error(), "cfg.Mock") {
		t.Fatalf("Start err = %v, want cfg.Mock contradiction", err)
	}
	if addr := sup.Addr(); addr != "" {
		t.Fatalf("supervisor listens on %q after refused Start; want nothing bound", addr)
	}
	_ = sup.Shutdown(context.Background())
}
