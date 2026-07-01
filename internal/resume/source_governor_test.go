package resume

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// fixedNow is a stable wall clock for the pure-decision tests.
func fixedNow() time.Time { return time.Unix(1_000_000, 0).UTC() }

func TestAdmitSourceZeroPolicyAlwaysAdmits(t *testing.T) {
	// Even a saturated-looking host admits when nothing is configured: every gate is
	// opt-in, so the permissive zero value behaves as before a policy existed.
	snap := SourceSnapshot{
		LiveResumeCount: 99,
		LaunchUnixTimes: []int64{fixedNow().Unix(), fixedNow().Unix()},
		LastLaunchUnix:  fixedNow().Unix(),
	}
	d := AdmitSource(snap, SourcePolicy{}, fixedNow())
	if !d.Admit || d.Reason != ReasonSourceAdmitted {
		t.Fatalf("zero policy must admit, got admit=%v reason=%q", d.Admit, d.Reason)
	}
}

func TestAdmitSourceLiveCeiling(t *testing.T) {
	pol := SourcePolicy{MaxLiveResumes: 4}
	now := fixedNow()

	// Under the ceiling (3 of 4) ⇒ admit.
	if d := AdmitSource(SourceSnapshot{LiveResumeCount: 3}, pol, now); !d.Admit {
		t.Fatalf("under ceiling must admit, got reason=%q", d.Reason)
	}
	// At the ceiling (4 of 4) ⇒ refuse SOURCE_SATURATED.
	if d := AdmitSource(SourceSnapshot{LiveResumeCount: 4}, pol, now); d.Admit || d.Reason != ReasonSourceSaturated {
		t.Fatalf("at ceiling must refuse SOURCE_SATURATED, got admit=%v reason=%q", d.Admit, d.Reason)
	}
	// Over the ceiling (6 of 4) ⇒ refuse SOURCE_SATURATED, and the live count is echoed.
	if d := AdmitSource(SourceSnapshot{LiveResumeCount: 6}, pol, now); d.Admit || d.Reason != ReasonSourceSaturated || d.LiveResumes != 6 {
		t.Fatalf("over ceiling must refuse with echoed live count, got admit=%v reason=%q live=%d", d.Admit, d.Reason, d.LiveResumes)
	}
	// Disabled (0) ⇒ admit even when many are live.
	if d := AdmitSource(SourceSnapshot{LiveResumeCount: 50}, SourcePolicy{}, now); !d.Admit {
		t.Fatalf("ceiling unset must admit, got reason=%q", d.Reason)
	}
}

func TestAdmitSourceRateWindow(t *testing.T) {
	now := fixedNow()
	pol := SourcePolicy{MaxLaunchesPerWindow: 3, WindowSeconds: 300}

	// 2 launches in the window (cap 3) ⇒ admit; only in-window launches count.
	inWindow := []int64{now.Unix() - 10, now.Unix() - 100}
	older := []int64{now.Unix() - 400, now.Unix() - 9999} // outside the 300s window
	snapUnder := SourceSnapshot{LaunchUnixTimes: append(append([]int64{}, inWindow...), older...)}
	if d := AdmitSource(snapUnder, pol, now); !d.Admit || d.WindowLaunches != 2 {
		t.Fatalf("under rate must admit with 2 in-window, got admit=%v reason=%q n=%d", d.Admit, d.Reason, d.WindowLaunches)
	}

	// 3 launches in the window (cap 3) ⇒ refuse LAUNCH_RATE_EXCEEDED with a retry_after
	// of the oldest in-window launch + the window.
	oldest := now.Unix() - 250
	snapAt := SourceSnapshot{LaunchUnixTimes: []int64{now.Unix() - 5, now.Unix() - 50, oldest}}
	d := AdmitSource(snapAt, pol, now)
	if d.Admit || d.Reason != ReasonLaunchRate {
		t.Fatalf("at rate must refuse LAUNCH_RATE_EXCEEDED, got admit=%v reason=%q", d.Admit, d.Reason)
	}
	if want := oldest + 300; d.RetryAfterUnix != want {
		t.Fatalf("retry_after = %d, want oldest+window = %d", d.RetryAfterUnix, want)
	}

	// Window unset (either field 0) ⇒ rate gate disabled.
	if d := AdmitSource(snapAt, SourcePolicy{MaxLaunchesPerWindow: 3}, now); !d.Admit {
		t.Fatalf("window unset must disable the rate gate, got reason=%q", d.Reason)
	}
}

func TestAdmitSourceSpacingFloor(t *testing.T) {
	now := fixedNow()
	pol := SourcePolicy{MinLaunchSpacingSeconds: 8}

	// Last launch 3s ago, floor 8s ⇒ refuse LAUNCH_SPACING_FLOOR, retry at last+floor.
	last := now.Unix() - 3
	d := AdmitSource(SourceSnapshot{LastLaunchUnix: last}, pol, now)
	if d.Admit || d.Reason != ReasonLaunchSpacing {
		t.Fatalf("inside spacing floor must refuse, got admit=%v reason=%q", d.Admit, d.Reason)
	}
	if want := last + 8; d.RetryAfterUnix != want {
		t.Fatalf("retry_after = %d, want last+floor = %d", d.RetryAfterUnix, want)
	}

	// Last launch 20s ago, floor 8s ⇒ admit (outside the floor).
	if d := AdmitSource(SourceSnapshot{LastLaunchUnix: now.Unix() - 20}, pol, now); !d.Admit {
		t.Fatalf("outside spacing floor must admit, got reason=%q", d.Reason)
	}
	// No prior launch (0) ⇒ admit; nothing to space against.
	if d := AdmitSource(SourceSnapshot{LastLaunchUnix: 0}, pol, now); !d.Admit {
		t.Fatalf("no prior launch must admit, got reason=%q", d.Reason)
	}
}

func TestAdmitSourceGateOrder(t *testing.T) {
	now := fixedNow()
	// A host that trips ALL THREE gates at once. The live ceiling is checked first, so
	// it wins deterministically over the rate and spacing refusals.
	snap := SourceSnapshot{
		LiveResumeCount: 9,
		LaunchUnixTimes: []int64{now.Unix() - 1, now.Unix() - 2, now.Unix() - 3},
		LastLaunchUnix:  now.Unix() - 1,
	}
	pol := SourcePolicy{
		MaxLiveResumes:          4,
		MaxLaunchesPerWindow:    3,
		WindowSeconds:           300,
		MinLaunchSpacingSeconds: 8,
	}
	if d := AdmitSource(snap, pol, now); d.Reason != ReasonSourceSaturated {
		t.Fatalf("live ceiling must win over rate+spacing, got %q", d.Reason)
	}

	// Remove the live pressure: now the rate window wins over the spacing floor.
	snap.LiveResumeCount = 0
	if d := AdmitSource(snap, pol, now); d.Reason != ReasonLaunchRate {
		t.Fatalf("rate window must win over spacing, got %q", d.Reason)
	}

	// Remove the rate pressure too (one in-window launch, cap 3): spacing floor decides.
	snap.LaunchUnixTimes = []int64{now.Unix() - 1}
	if d := AdmitSource(snap, pol, now); d.Reason != ReasonLaunchSpacing {
		t.Fatalf("spacing floor must decide last, got %q", d.Reason)
	}
}

func TestLoadSourcePolicyMissingFileIsPermissive(t *testing.T) {
	// A path that does not exist ⇒ empty permissive set, NO error (fail-open).
	missing := filepath.Join(t.TempDir(), "nope.json")
	p, err := LoadSourcePolicy(missing)
	if err != nil {
		t.Fatalf("missing file must not error, got %v", err)
	}
	if (p != SourcePolicies{}) {
		t.Fatalf("missing file must yield the empty permissive set, got %+v", p)
	}
	// Empty path is also permissive.
	if p, err := LoadSourcePolicy("   "); err != nil || (p != SourcePolicies{}) {
		t.Fatalf("empty path must be permissive, got %+v err=%v", p, err)
	}
}

func TestLoadSourcePolicyRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "policy.json")
	body := `{"schema":"fak.resume-source-policy.v1","default":{"max_live_resumes":4,"max_launches_per_window":10,"window_seconds":300,"min_launch_spacing_seconds":8}}`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	p, err := LoadSourcePolicy(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	d := p.Default
	if d.MaxLiveResumes != 4 || d.MaxLaunchesPerWindow != 10 || d.WindowSeconds != 300 || d.MinLaunchSpacingSeconds != 8 {
		t.Fatalf("round-trip mismatch: %+v", d)
	}
}

func TestLoadSourcePolicyRejectsBadSchemaAndJSON(t *testing.T) {
	dir := t.TempDir()
	bad := filepath.Join(dir, "badschema.json")
	if err := os.WriteFile(bad, []byte(`{"schema":"fak.other.v9","default":{}}`), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	if _, err := LoadSourcePolicy(bad); err == nil {
		t.Fatalf("a wrong schema tag must be a loud error, got nil")
	}
	malformed := filepath.Join(dir, "malformed.json")
	if err := os.WriteFile(malformed, []byte(`{not json`), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	if _, err := LoadSourcePolicy(malformed); err == nil {
		t.Fatalf("malformed JSON must be a loud error, got nil")
	}
	// An empty schema tag is tolerated (an operator may omit it), matching LoadPolicies.
	noschema := filepath.Join(dir, "noschema.json")
	if err := os.WriteFile(noschema, []byte(`{"default":{"max_live_resumes":2}}`), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	p, err := LoadSourcePolicy(noschema)
	if err != nil || p.Default.MaxLiveResumes != 2 {
		t.Fatalf("empty schema must load, got %+v err=%v", p, err)
	}
}
