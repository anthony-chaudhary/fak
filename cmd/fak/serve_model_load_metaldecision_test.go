package main

import (
	"runtime"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// TestServeMetalSkipReasonFrom is the pure matrix over the (available, compiled)
// probe pair. The helper is only consulted when live==false, meaning
// available==false; the first two rows are the consulted domain. The available==true
// rows merely document the deterministic answers the helper defines outside its
// consulted domain — no caller reaches them because a live Metal session never
// asks for a skip reason.
func TestServeMetalSkipReasonFrom(t *testing.T) {
	tests := []struct {
		name      string
		available bool
		compiled  bool
		want      serveMetalSkipReason
	}{
		{name: "not compiled is the build skip reason", available: false, compiled: false, want: skipReasonNotCompiled},
		{name: "compiled but no device is the device skip reason", available: false, compiled: true, want: skipReasonNoDevice},
		{name: "outside consulted domain: available+compiled", available: true, compiled: true, want: skipReasonNoDevice},
		{name: "outside consulted domain: available uncompiled", available: true, compiled: false, want: skipReasonNotCompiled},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := serveMetalSkipReasonFrom(tt.available, tt.compiled); got != tt.want {
				t.Fatalf("serveMetalSkipReasonFrom(available=%t, compiled=%t) = %q, want %q", tt.available, tt.compiled, got, tt.want)
			}
		})
	}
}

// TestResolveServeMetalDecisionWrapper pins the twin contract: the decision wrapper
// mirrors resolveServeMetal's (live, error) behavior exactly for the auto-select and
// fail-loud cases, propagates resolver errors untouched, and shades the skip reason
// from the pure Compiled/Available rule only when Metal was declined without error.
func TestResolveServeMetalDecisionWrapper(t *testing.T) {
	t.Setenv("FAK_BACKEND", "")
	tests := []struct {
		name       string
		flag       bool
		env        bool
		backend    string
		wantErrSub string
	}{
		{name: "auto case mirrors resolver", flag: false, env: false, backend: ""},
		{name: "fail-loud requested case mirrors resolver", flag: true, env: false, backend: ""},
		{name: "mutual exclusion error propagates", flag: true, env: false, backend: "auto", wantErrSub: "--backend"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resolved, resolverErr := resolveServeMetal(tt.flag, tt.env, tt.backend)
			decision, decisionErr := resolveServeMetalDecision(tt.flag, tt.env, tt.backend)
			if (resolverErr != nil) != (decisionErr != nil) {
				t.Fatalf("error presence mismatch: resolver err=%v, decision err=%v", resolverErr, decisionErr)
			}
			if resolverErr != nil && decisionErr != nil && resolverErr.Error() != decisionErr.Error() {
				t.Fatalf("error text mismatch: resolver %q, decision %q", resolverErr, decisionErr)
			}
			if decision.live != resolved {
				t.Fatalf("live = %t, want resolver result %t", decision.live, resolved)
			}
			if resolverErr != nil {
				if tt.wantErrSub != "" && !strings.Contains(decisionErr.Error(), tt.wantErrSub) {
					t.Fatalf("error = %v, want substring %q", decisionErr, tt.wantErrSub)
				}
				if decision.skippedBecause != "" {
					t.Fatalf("skippedBecause = %q on error, want empty", decision.skippedBecause)
				}
				return
			}
			if decision.live {
				if decision.skippedBecause != "" {
					t.Fatalf("skippedBecause = %q on a live Metal session, want empty", decision.skippedBecause)
				}
				return
			}
			wantReason := skipReasonNoDevice
			if !metalgemm.Compiled() {
				wantReason = skipReasonNotCompiled
			}
			if decision.skippedBecause != wantReason {
				t.Fatalf("skippedBecause = %q, want %q (compiled=%t available=%t)", decision.skippedBecause, wantReason, metalgemm.Compiled(), metalgemm.Available())
			}
		})
	}
}

// TestResolveServeMetalDecisionMutualExclusionMessages mirrors
// TestServeBackendAutoMetalRejectsExplicitAuto through the decision wrapper:
// an explicit request colliding with an --backend selection is rejected with the
// same mutually-exclusive error text the resolver emits.
func TestResolveServeMetalDecisionMutualExclusionMessages(t *testing.T) {
	t.Setenv("FAK_BACKEND", "")
	if _, err := resolveServeMetalDecision(true, false, "auto"); err == nil || !strings.Contains(err.Error(), "mutually exclusive") {
		t.Fatalf("resolveServeMetalDecision explicit auto error = %v, want mutually exclusive conflict", err)
	}
}

// TestResolveServeMetalDecisionDarwinOnlyReason pins the GOOS gate: the skip
// reason is shaded only when the running GOOS is darwin, because only the darwin
// auto-selection targets Metal. On other GOOS the resolver's false means "Metal
// was never the target", so the decision must carry an empty reason and callers
// must not stamp a Metal line. On non-darwin the empty-reason branch is the
// asserted behavior; on darwin the reason is non-empty exactly when
// metalgemm.Compiled() is true (the live device probe decides which shade).
func TestResolveServeMetalDecisionDarwinOnlyReason(t *testing.T) {
	t.Setenv("FAK_BACKEND", "")
	decision, err := resolveServeMetalDecision(false, false, "")
	if err != nil {
		t.Fatalf("auto decision returned error: %v", err)
	}
	if runtime.GOOS == "darwin" {
		if !metalgemm.Available() {
			if decision.skippedBecause == "" {
				t.Fatalf("darwin decision shades no skip reason; want %q or %q", skipReasonNotCompiled, skipReasonNoDevice)
			}
		} else if !decision.live || decision.skippedBecause != "" {
			t.Fatalf("darwin decision with a usable device: live=%v reason=%q; want live=true reason empty", decision.live, decision.skippedBecause)
		}
		return
	}
	if decision.skippedBecause != "" {
		t.Fatalf("non-darwin GOOS %q shaded skip reason %q; want empty (Metal was not the auto-target)", runtime.GOOS, decision.skippedBecause)
	}
}
