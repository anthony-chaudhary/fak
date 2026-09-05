package sandbox

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// ---------------------------------------------------------------------------
// MOCK IMPLEMENTATIONS FOR INTERFACE CONFORMANCE
// ---------------------------------------------------------------------------

type mockSnapshotHandle struct {
	id       string
	restored bool
	released bool
}

func (m *mockSnapshotHandle) ID() string { return m.id }
func (m *mockSnapshotHandle) Restore(ctx context.Context) error {
	m.restored = true
	return nil
}
func (m *mockSnapshotHandle) Release() error {
	m.released = true
	return nil
}

type mockInstance struct {
	spec    Spec
	closed  bool
	reset   bool
	execLog []ExecutionRequest
}

func (m *mockInstance) Execute(ctx context.Context, req ExecutionRequest) (ExecutionResult, error) {
	m.execLog = append(m.execLog, req)
	stdout := []byte("hello from " + m.spec.WorkspaceDir + "\r\n\x1b[32mSUCCESS\x1b[0m  ")
	return NewExecutionResult(0, stdout, nil, m.spec.WorkspaceDir, 42, 100, 1024*1024), nil
}

func (m *mockInstance) Reset(ctx context.Context) error {
	m.reset = true
	return nil
}

func (m *mockInstance) Close() error {
	m.closed = true
	return nil
}

func (m *mockInstance) Spec() Spec {
	return m.spec
}

func (m *mockInstance) Snapshot(ctx context.Context) (SnapshotHandle, error) {
	return &mockSnapshotHandle{id: "snap-001"}, nil
}

type mockProvider struct {
	tier      Tier
	available bool
}

func (m *mockProvider) Name() string    { return "mock-" + string(m.tier) }
func (m *mockProvider) Tier() Tier      { return m.tier }
func (m *mockProvider) Available() bool { return m.available }
func (m *mockProvider) Create(ctx context.Context, spec Spec) (Instance, error) {
	if !m.available {
		return nil, NewSandboxError(ErrSandboxUnavailable, "mock provider offline")
	}
	if err := spec.Validate(); err != nil {
		return nil, err
	}
	return &mockInstance{spec: spec}, nil
}

// Compile-time interface checks.
var (
	_ Instance             = (*mockInstance)(nil)
	_ SnapshotableInstance = (*mockInstance)(nil)
	_ SnapshotHandle       = (*mockSnapshotHandle)(nil)
	_ Provider             = (*mockProvider)(nil)
)

// ---------------------------------------------------------------------------
// CONFORMANCE TEST SUITE
// ---------------------------------------------------------------------------

func TestSandboxSpecConformance(t *testing.T) {
	t.Run("tier definitions and helpers", func(t *testing.T) {
		tiers := []struct {
			tier       Tier
			str        string
			level      int
			valid      bool
			virtualize bool
		}{
			{TierL0Wasm, "l0_wasm", 0, true, false},
			{TierL1NativeOS, "l1_native_os", 1, true, false},
			{TierL2Virtual, "l2_virtual", 2, true, true},
			{Tier("invalid_tier"), "invalid_tier", -1, false, false},
			{Tier(""), "", -1, false, false},
		}

		for _, tc := range tiers {
			if got := tc.tier.String(); got != tc.str {
				t.Errorf("Tier.String() = %q, want %q", got, tc.str)
			}
			if got := tc.tier.Valid(); got != tc.valid {
				t.Errorf("Tier.Valid() for %q = %v, want %v", tc.tier, got, tc.valid)
			}
			if got := tc.tier.IsolationLevel(); got != tc.level {
				t.Errorf("Tier.IsolationLevel() for %q = %d, want %d", tc.tier, got, tc.level)
			}
			if got := tc.tier.RequiresVirtualization(); got != tc.virtualize {
				t.Errorf("Tier.RequiresVirtualization() for %q = %v, want %v", tc.tier, got, tc.virtualize)
			}
		}

		// ParseTier checks
		parsed, err := ParseTier("l0_wasm")
		if err != nil || parsed != TierL0Wasm {
			t.Errorf("ParseTier(l0_wasm) = %v, %v", parsed, err)
		}
		parsed, err = ParseTier(" l1_native_os \n")
		if err != nil || parsed != TierL1NativeOS {
			t.Errorf("ParseTier(trimmed l1_native_os) = %v, %v", parsed, err)
		}
		parsed, err = ParseTier("l2_virtual")
		if err != nil || parsed != TierL2Virtual {
			t.Errorf("ParseTier(l2_virtual) = %v, %v", parsed, err)
		}
		if _, err := ParseTier("unknown_rung"); err == nil {
			t.Errorf("ParseTier(unknown_rung) expected error, got nil")
		}
	})

	t.Run("refusal constants and typed errors", func(t *testing.T) {
		consts := map[string]string{
			"ErrSandboxUnavailable":        ErrSandboxUnavailable,
			"ErrLanePathEscape":            ErrLanePathEscape,
			"ErrEgressBlocked":             ErrEgressBlocked,
			"ErrSiblingLaneTouch":          ErrSiblingLaneTouch,
			"ErrSecretExfiltrationAttempt": ErrSecretExfiltrationAttempt,
		}

		expected := map[string]string{
			"ErrSandboxUnavailable":        "SANDBOX_UNAVAILABLE",
			"ErrLanePathEscape":            "LANE_PATH_ESCAPE",
			"ErrEgressBlocked":             "EGRESS_BLOCKED",
			"ErrSiblingLaneTouch":          "SIBLING_LANE_TOUCH",
			"ErrSecretExfiltrationAttempt": "SECRET_EXFILTRATION_ATTEMPT",
		}

		for k, v := range expected {
			if consts[k] != v {
				t.Errorf("refusal constant %s = %q, want %q", k, consts[k], v)
			}
		}

		// SandboxError construction & wrapping
		rawErr := errors.New("underlying kernel socket permission denied")
		se := WrapSandboxError(ErrEgressBlocked, "outbound connection refused by policy", rawErr)

		if se.Token != ErrEgressBlocked {
			t.Errorf("Token = %q, want %q", se.Token, ErrEgressBlocked)
		}
		if !strings.Contains(se.Error(), ErrEgressBlocked) || !strings.Contains(se.Error(), "underlying kernel socket") {
			t.Errorf("Error() = %q, missing token or message", se.Error())
		}
		if !errors.Is(se, rawErr) {
			t.Errorf("errors.Is(se, rawErr) = false, want true")
		}
		if !IsSandboxError(se, ErrEgressBlocked) {
			t.Errorf("IsSandboxError(se, ErrEgressBlocked) = false, want true")
		}
		if IsSandboxError(se, ErrLanePathEscape) {
			t.Errorf("IsSandboxError(se, ErrLanePathEscape) = true, want false")
		}

		plainSe := NewSandboxError(ErrLanePathEscape, "traversal outside boundary")
		if plainSe.Unwrap() != nil {
			t.Errorf("plainSe.Unwrap() = %v, want nil", plainSe.Unwrap())
		}
		if !IsSandboxError(plainSe, ErrLanePathEscape) {
			t.Errorf("IsSandboxError(plainSe, ErrLanePathEscape) = false, want true")
		}
	})

	t.Run("spec validation and JSON roundtrip", func(t *testing.T) {
		spec := Spec{
			Tier:             TierL1NativeOS,
			Rootfs:           "/var/lib/fak/rootfs/alpine-3.20",
			WorkspaceDir:     "/work/repo",
			LaneTree:         []string{"internal/sandbox/**", "docs/architecture/**"},
			ReadOnlyPaths:    []string{"/usr", "/lib", "/bin"},
			WritablePaths:    []string{"/tmp", "/work/repo/scratch"},
			MemoryLimitBytes: 1024 * 1024 * 1024,
			CPULimitPercent:  80,
			FuelLimit:        500000,
			TimeoutMS:        30000,
			Env:              []string{"LANG=C.UTF-8", "FAK_SANDBOX=1"},
			EgressPolicy:     EgressBlocked,
			Capabilities:     []abi.Capability{"fs.read", "fs.write", "proc.exec"},
		}

		if err := spec.Validate(); err != nil {
			t.Fatalf("spec.Validate() failed: %v", err)
		}

		data, err := json.Marshal(spec)
		if err != nil {
			t.Fatalf("json.Marshal(spec) failed: %v", err)
		}

		var roundtrip Spec
		if err := json.Unmarshal(data, &roundtrip); err != nil {
			t.Fatalf("json.Unmarshal(spec) failed: %v", err)
		}

		if !reflect.DeepEqual(spec, roundtrip) {
			t.Errorf("Spec roundtrip mismatch:\ngot  %+v\nwant %+v", roundtrip, spec)
		}

		// Validation edge cases
		badSpecs := []struct {
			name string
			mod  func(s *Spec)
		}{
			{"invalid tier", func(s *Spec) { s.Tier = "bad_tier" }},
			{"empty workspace", func(s *Spec) { s.WorkspaceDir = "" }},
			{"negative memory", func(s *Spec) { s.MemoryLimitBytes = -1 }},
			{"negative cpu percent", func(s *Spec) { s.CPULimitPercent = -5 }},
			{"excessive cpu percent", func(s *Spec) { s.CPULimitPercent = 101 }},
			{"negative fuel", func(s *Spec) { s.FuelLimit = -10 }},
			{"negative timeout", func(s *Spec) { s.TimeoutMS = -100 }},
			{"missing egress policy", func(s *Spec) { s.EgressPolicy = "" }},
		}

		for _, tc := range badSpecs {
			copied := spec
			tc.mod(&copied)
			if err := copied.Validate(); err == nil {
				t.Errorf("spec.Validate() for %q expected error, got nil", tc.name)
			}
		}
	})

	t.Run("capability envelope", func(t *testing.T) {
		env := CapabilityEnvelope{
			LaneTree:     []string{"internal/sandbox/**"},
			Capabilities: []abi.Capability{"fs.read", "fs.write"},
			AllowNetwork: false,
			AllowIPC:     false,
		}

		if !env.HasCapability("fs.read") {
			t.Errorf("env.HasCapability(fs.read) = false, want true")
		}
		if env.HasCapability("net.dial") {
			t.Errorf("env.HasCapability(net.dial) = true, want false")
		}
	})

	t.Run("execution request and result JSON roundtrip", func(t *testing.T) {
		req := ExecutionRequest{
			Command:    "go",
			Argv:       []string{"test", "./..."},
			Stdin:      []byte("input-stream"),
			Env:        []string{"CGO_ENABLED=0"},
			TimeoutMS:  15000,
			WorkingDir: "/work/repo",
		}

		reqBytes, err := json.Marshal(req)
		if err != nil {
			t.Fatalf("json.Marshal(req) failed: %v", err)
		}

		var reqRoundtrip ExecutionRequest
		if err := json.Unmarshal(reqBytes, &reqRoundtrip); err != nil {
			t.Fatalf("json.Unmarshal(req) failed: %v", err)
		}
		if !reflect.DeepEqual(req, reqRoundtrip) {
			t.Errorf("ExecutionRequest roundtrip mismatch:\ngot  %+v\nwant %+v", reqRoundtrip, req)
		}

		res := ExecutionResult{
			ExitCode:         0,
			Stdout:           []byte("ok\n"),
			Stderr:           []byte(""),
			NormalizedStdout: []byte("ok\n"),
			NormalizedStderr: []byte(""),
			DurationMS:       120,
			FuelUsed:         450,
			MemoryBytes:      64 * 1024 * 1024,
			Audits: []Audit{
				{TimestampMS: 1725500000000, Type: "net_deny", Message: "dial blocked by policy"},
			},
		}

		resBytes, err := json.Marshal(res)
		if err != nil {
			t.Fatalf("json.Marshal(res) failed: %v", err)
		}

		var resRoundtrip ExecutionResult
		if err := json.Unmarshal(resBytes, &resRoundtrip); err != nil {
			t.Fatalf("json.Unmarshal(res) failed: %v", err)
		}
		if !reflect.DeepEqual(res, resRoundtrip) {
			t.Errorf("ExecutionResult roundtrip mismatch:\ngot  %+v\nwant %+v", resRoundtrip, res)
		}
	})

	t.Run("output normalization helper", func(t *testing.T) {
		raw := []byte("\x1b[31;1mError in /srv/fak/workspace/pkg/test.go:42\x1b[0m\r\nFailed to compile:   \r\n\x1b[33mWarning\x1b[0m  \r")
		ws := "/srv/fak/workspace"

		got := string(NormalizeOutput(raw, ws))
		want := "Error in /workspace/pkg/test.go:42\nFailed to compile:\nWarning"

		if got != want {
			t.Errorf("NormalizeOutput mismatch:\ngot:  %q\nwant: %q", got, want)
		}

		// Windows backslash workspace path normalization test
		rawWin := []byte("Build output: C:\\work\\repo\\workspace\\cmd\\main.go:10\r\nDone.  ")
		wsWin := "C:\\work\\repo\\workspace"
		gotWin := string(NormalizeOutput(rawWin, wsWin))
		wantWin := "Build output: /workspace\\cmd\\main.go:10\nDone."
		if gotWin != wantWin {
			t.Errorf("NormalizeOutput (Windows path) mismatch:\ngot:  %q\nwant: %q", gotWin, wantWin)
		}

		// Empty raw input
		if len(NormalizeOutput(nil, ws)) != 0 {
			t.Errorf("NormalizeOutput(nil) should return empty bytes")
		}

		// NewExecutionResult helper
		execRes := NewExecutionResult(1, raw, []byte("stderr\r\n"), ws, 200, 50, 1024)
		if string(execRes.NormalizedStdout) != want {
			t.Errorf("NewExecutionResult NormalizedStdout = %q, want %q", string(execRes.NormalizedStdout), want)
		}
		if string(execRes.NormalizedStderr) != "stderr" {
			t.Errorf("NewExecutionResult NormalizedStderr = %q, want %q", string(execRes.NormalizedStderr), "stderr")
		}
	})

	t.Run("instance lifecycle and mock execution", func(t *testing.T) {
		ctx := context.Background()
		prov := &mockProvider{tier: TierL1NativeOS, available: true}

		spec := Spec{
			Tier:             TierL1NativeOS,
			WorkspaceDir:     "/work/sandbox",
			EgressPolicy:     EgressBlocked,
			MemoryLimitBytes: 256 * 1024 * 1024,
			CPULimitPercent:  50,
			TimeoutMS:        5000,
		}

		inst, err := prov.Create(ctx, spec)
		if err != nil {
			t.Fatalf("prov.Create failed: %v", err)
		}

		if inst.Spec().WorkspaceDir != "/work/sandbox" {
			t.Errorf("inst.Spec().WorkspaceDir = %q, want /work/sandbox", inst.Spec().WorkspaceDir)
		}

		res, err := inst.Execute(ctx, ExecutionRequest{Command: "echo", Argv: []string{"hi"}})
		if err != nil {
			t.Fatalf("inst.Execute failed: %v", err)
		}
		if res.ExitCode != 0 {
			t.Errorf("res.ExitCode = %d, want 0", res.ExitCode)
		}
		if !strings.Contains(string(res.NormalizedStdout), "/workspace") {
			t.Errorf("res.NormalizedStdout = %q, expected /workspace substitution", string(res.NormalizedStdout))
		}

		if err := inst.Reset(ctx); err != nil {
			t.Fatalf("inst.Reset failed: %v", err)
		}

		// Snapshot check
		snapInst, ok := inst.(SnapshotableInstance)
		if !ok {
			t.Fatalf("inst does not implement SnapshotableInstance")
		}
		snap, err := snapInst.Snapshot(ctx)
		if err != nil {
			t.Fatalf("snapInst.Snapshot failed: %v", err)
		}
		if snap.ID() != "snap-001" {
			t.Errorf("snap.ID() = %q, want snap-001", snap.ID())
		}
		if err := snap.Restore(ctx); err != nil {
			t.Fatalf("snap.Restore failed: %v", err)
		}
		if err := snap.Release(); err != nil {
			t.Fatalf("snap.Release failed: %v", err)
		}

		if err := inst.Close(); err != nil {
			t.Fatalf("inst.Close failed: %v", err)
		}

		// Unavailable provider error
		unavailProv := &mockProvider{tier: TierL2Virtual, available: false}
		_, err = unavailProv.Create(ctx, spec)
		if err == nil || !IsSandboxError(err, ErrSandboxUnavailable) {
			t.Errorf("unavailProv.Create err = %v, expected ErrSandboxUnavailable", err)
		}
	})
}
