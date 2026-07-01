package policy

import (
	"strings"
	"testing"
)

// TestIsolationTrustLevelSelectsBackendDeterministically is the #2013
// acceptance line: setting a trust level selects the backend
// deterministically — same level, same backend, every call.
func TestIsolationTrustLevelSelectsBackendDeterministically(t *testing.T) {
	rt, err := ParseRuntime([]byte(`{
		"version": "fak-policy/v1",
		"allow": ["Read"],
		"isolation": {
			"backends": ["goroutine", "subprocess", "container"],
			"trust": {"trusted": "goroutine", "vetted": "subprocess", "untrusted": "container"}
		}
	}`))
	if err != nil {
		t.Fatalf("ParseRuntime: %v", err)
	}
	if rt.Isolation == nil {
		t.Fatal("Runtime.Isolation is nil for a manifest that declares the dial")
	}
	want := map[string]string{"trusted": "goroutine", "vetted": "subprocess", "untrusted": "container"}
	for level, backend := range want {
		for i := 0; i < 3; i++ { // deterministic across repeated calls
			got, err := rt.Isolation.BackendFor(level)
			if err != nil {
				t.Fatalf("BackendFor(%q) call %d: %v", level, i, err)
			}
			if got != backend {
				t.Fatalf("BackendFor(%q) call %d = %q, want %q", level, i, got, backend)
			}
		}
	}
	// Level names normalize: "  Trusted " dials the same as "trusted".
	if got, err := rt.Isolation.BackendFor("  Trusted "); err != nil || got != "goroutine" {
		t.Fatalf("BackendFor(\"  Trusted \") = %q, %v; want goroutine, nil", got, err)
	}
}

// TestIsolationUnknownAndUntrustedDefaultToStrongest is the #2013 fail-closed
// default: an unknown trust level, an empty one, and an undeclared "untrusted"
// all resolve to the STRONGEST configured backend (gvisor here), never a
// weaker one — regardless of declaration order in the manifest.
func TestIsolationUnknownAndUntrustedDefaultToStrongest(t *testing.T) {
	rt, err := ParseRuntime([]byte(`{
		"version": "fak-policy/v1",
		"isolation": {
			"backends": ["subprocess", "gvisor", "goroutine"],
			"trust": {"trusted": "goroutine"}
		}
	}`))
	if err != nil {
		t.Fatalf("ParseRuntime: %v", err)
	}
	for _, level := range []string{"never-heard-of-it", TrustUntrusted, ""} {
		got, err := rt.Isolation.BackendFor(level)
		if err != nil {
			t.Fatalf("BackendFor(%q): %v", level, err)
		}
		if got != "gvisor" {
			t.Fatalf("BackendFor(%q) = %q, want gvisor (the strongest configured)", level, got)
		}
	}
}

// TestIsolationNeverGoroutineForUntrusted pins the invariant on both rungs:
// at LOAD an explicit untrusted→goroutine dial is refused, and at RESOLUTION a
// deployment whose strongest backend is goroutine refuses unknown/untrusted
// work instead of running it in-process.
func TestIsolationNeverGoroutineForUntrusted(t *testing.T) {
	_, err := ParseRuntime([]byte(`{
		"version": "fak-policy/v1",
		"isolation": {
			"backends": ["goroutine", "container"],
			"trust": {"untrusted": "goroutine"}
		}
	}`))
	if err == nil || !strings.Contains(err.Error(), "never") {
		t.Fatalf("untrusted→goroutine must be refused at load, got err=%v", err)
	}

	rt, err := ParseRuntime([]byte(`{
		"version": "fak-policy/v1",
		"isolation": {
			"backends": ["goroutine"],
			"trust": {"trusted": "goroutine"}
		}
	}`))
	if err != nil {
		t.Fatalf("ParseRuntime: %v", err)
	}
	// Declared trusted work may still run in-process...
	if got, err := rt.Isolation.BackendFor("trusted"); err != nil || got != "goroutine" {
		t.Fatalf("BackendFor(trusted) = %q, %v; want goroutine, nil", got, err)
	}
	// ...but unknown/untrusted must be refused when nothing stronger exists.
	for _, level := range []string{TrustUntrusted, "unknown-level"} {
		if got, err := rt.Isolation.BackendFor(level); err == nil {
			t.Fatalf("BackendFor(%q) = %q, want refusal (strongest available is goroutine)", level, got)
		}
	}
}

// TestIsolationValidationFailsLoud proves the block's load-time validation:
// every malformed dial is a hard error at parse, never a silent misplacement.
func TestIsolationValidationFailsLoud(t *testing.T) {
	cases := []struct {
		name, manifest, want string
	}{
		{"unknown backend", `{"isolation":{"backends":["chroot"]}}`, "unknown backend"},
		{"empty backends", `{"isolation":{"backends":[]}}`, "at least one"},
		{"block without backends", `{"isolation":{"trust":{"trusted":"goroutine"}}}`, "at least one"},
		{"trust to undeclared backend", `{"isolation":{"backends":["goroutine"],"trust":{"trusted":"container"}}}`, "not declared in isolation.backends"},
		{"duplicate backend", `{"isolation":{"backends":["goroutine","goroutine"]}}`, "duplicate"},
		{"misspelled key", `{"isolation":{"backend":["goroutine"]}}`, "invalid manifest"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ParseRuntime([]byte(tc.manifest))
			if err == nil {
				t.Fatalf("manifest %s must fail to load", tc.manifest)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error %q does not name the defect (want substring %q)", err, tc.want)
			}
		})
	}
}

// TestIsolationAbsentFailsClosed: no isolation block means NO dial — the
// runtime field stays nil and resolution refuses, rather than defaulting any
// work (least of all untrusted) onto goroutine. Also pins the operator
// summary so `fak policy --check` surfaces the dial state either way.
func TestIsolationAbsentFailsClosed(t *testing.T) {
	rt, err := ParseRuntime([]byte(`{"version":"fak-policy/v1","allow":["Read"]}`))
	if err != nil {
		t.Fatalf("ParseRuntime: %v", err)
	}
	if rt.Isolation != nil {
		t.Fatalf("Runtime.Isolation = %+v, want nil for a manifest with no isolation block", rt.Isolation)
	}
	if got, err := rt.Isolation.BackendFor("trusted"); err == nil {
		t.Fatalf("BackendFor on a nil dial = %q, want fail-closed refusal", got)
	}
	if s := SummaryRuntime(rt); !strings.Contains(s, "isolation          : (none") {
		t.Fatalf("SummaryRuntime should flag the unset dial:\n%s", s)
	}

	withDial, err := ParseRuntime([]byte(`{
		"version": "fak-policy/v1",
		"isolation": {"backends": ["goroutine", "container"], "trust": {"trusted": "goroutine"}}
	}`))
	if err != nil {
		t.Fatalf("ParseRuntime: %v", err)
	}
	s := SummaryRuntime(withDial)
	if !strings.Contains(s, "isolation backends : goroutine, container (strongest: container)") ||
		!strings.Contains(s, "isolation trust    : trusted -> goroutine") {
		t.Fatalf("SummaryRuntime should surface the declared dial:\n%s", s)
	}
}
