package ablate

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/bench"
)

// SessionCaveats is the acceptance-#3 surfacing helper for #1846: an in-kernel-only
// feature (radix, ctxplan_seam) replayed against a captured session's proxy engine
// reads as a no-op, and the report must SAY so, not silently show a ~0 delta.

func TestSessionCaveatsNilOnSuiteTrace(t *testing.T) {
	// Not a captured session (sliceIsSession=false) — no caveat, even with an
	// in-kernel-only feature in the sweep, since a checked-in suite replayed
	// against "mock" is the documented/expected shape already.
	got := SessionCaveats([]string{FeatureRadix}, "mock", false)
	if got != nil {
		t.Errorf("SessionCaveats(suite trace) = %v, want nil", got)
	}
}

func TestSessionCaveatsNilOnInKernelEngine(t *testing.T) {
	// A session replayed against the REAL in-kernel engine would actually exercise
	// radix/ctxplan_seam — no caveat needed even though sliceIsSession is true.
	got := SessionCaveats([]string{FeatureRadix, FeatureCtxplanSeam}, "inkernel", true)
	if got != nil {
		t.Errorf("SessionCaveats(inkernel engine) = %v, want nil", got)
	}
}

func TestSessionCaveatsNilWhenNoInKernelOnlyFeatureSwept(t *testing.T) {
	// A session sweep of vdso only never touches an in-kernel-only feature, so no
	// caveat is warranted regardless of engine.
	got := SessionCaveats([]string{FeatureVDSO}, "session:abcd1234", true)
	if got != nil {
		t.Errorf("SessionCaveats(vdso-only sweep) = %v, want nil", got)
	}
}

func TestSessionCaveatsFiresForRadixOnCapturedSession(t *testing.T) {
	got := SessionCaveats([]string{FeatureVDSO, FeatureRadix}, "session:abcd1234", true)
	if len(got) != 1 {
		t.Fatalf("SessionCaveats = %v, want exactly 1 caveat", got)
	}
	msg := got[0]
	for _, want := range []string{"radix", "NO-OP", "session:abcd1234", "#1846"} {
		if !contains(msg, want) {
			t.Errorf("caveat message missing %q:\n%s", want, msg)
		}
	}
	// vdso (not in-kernel-only) must NOT be named in the caveat.
	if contains(msg, "vdso") {
		t.Errorf("caveat wrongly names vdso (not an in-kernel-only feature):\n%s", msg)
	}
}

func TestSessionCaveatsFiresForBothInKernelOnlyFeaturesSortedAndDeduped(t *testing.T) {
	got := SessionCaveats([]string{FeatureCtxplanSeam, FeatureRadix}, "session:deadbeef", true)
	if len(got) != 1 {
		t.Fatalf("SessionCaveats = %v, want exactly 1 caveat", got)
	}
	// FeatureRadix ("radix") sorts before FeatureCtxplanSeam ("ctxplan_seam")? No —
	// alphabetically "ctxplan_seam" < "radix". Assert sorted order explicitly rather
	// than assuming input order.
	msg := got[0]
	ci := indexOf(msg, "ctxplan_seam")
	ri := indexOf(msg, "radix")
	if ci < 0 || ri < 0 {
		t.Fatalf("caveat missing one of the two features:\n%s", msg)
	}
	if ci > ri {
		t.Errorf("caveat lists features out of sorted order (want ctxplan_seam before radix):\n%s", msg)
	}
}

func TestReportSessionCaveatsSurfaceOnSessionReplay(t *testing.T) {
	tr := &bench.Trace{SliceID: "session:abcd1234"}
	configs := []FeatureConfig{
		{Name: "all-off", EnvFeatures: map[string]string{FeatureRadix: "off"}},
		{Name: "radix", EnvFeatures: map[string]string{FeatureRadix: "on"}},
	}
	got := reportSessionCaveats(tr, "mock", configs)
	if len(got) != 1 || !strings.Contains(got[0], "NO-OP") || !strings.Contains(got[0], "#1846") {
		t.Fatalf("reportSessionCaveats = %v, want one surfaced no-op caveat", got)
	}
	rep := &Report{WorkloadHash: "hash", Runs: []AblationRun{{ArmID: "all-off", WorkloadHash: "hash"}}, Caveats: got}
	if !strings.Contains(string(rep.JSON()), `"caveats"`) {
		t.Fatalf("report JSON did not surface caveats: %s", rep.JSON())
	}
}

func contains(s, substr string) bool { return indexOf(s, substr) >= 0 }

func indexOf(s, substr string) int {
	n, m := len(s), len(substr)
	if m == 0 {
		return 0
	}
	for i := 0; i+m <= n; i++ {
		if s[i:i+m] == substr {
			return i
		}
	}
	return -1
}
