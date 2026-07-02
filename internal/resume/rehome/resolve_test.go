package rehome

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

const testProject = "C--work-fak" // ProjectSlug(`C:\work\fak`)
const testCWD = `C:\work\fak`

func TestResolveNotFound(t *testing.T) {
	home := t.TempDir()
	got := Resolve(ResolveInput{SID: "ghost", Home: home, CWD: testCWD})
	if got.OK || got.Action != "NOT_FOUND" {
		t.Fatalf("Resolve = %+v, want NOT_FOUND", got)
	}
}

func TestResolvePinsAvailableOwner(t *testing.T) {
	home := t.TempDir()
	sid := "s1"
	writeTranscript(t, home, ".claude-owner", testProject, sid, time.Now(), 10)
	got := Resolve(ResolveInput{
		SID: sid, Home: home, CWD: testCWD,
		OwnerStatus: &OwnerStatus{Available: true},
		RehomeFn:    RehomeTranscript,
	})
	if got.Action != "PIN" {
		t.Fatalf("action = %q, want PIN", got.Action)
	}
	if got.PinConfigDir != filepath.Join(home, ".claude-owner") {
		t.Fatalf("pin dir = %q, want owner dir", got.PinConfigDir)
	}
	if got.Rehomed {
		t.Fatal("PIN should not rehome")
	}
}

func TestResolvePinMirrorsIntoCwdSlug(t *testing.T) {
	home := t.TempDir()
	sid := "s2"
	writeTranscript(t, home, ".claude-owner", testProject, sid, time.Now(), 10)
	// Resume launches from a DIFFERENT directory than the session's birth slug.
	got := Resolve(ResolveInput{
		SID: sid, Home: home, CWD: `C:\work\slack-helpers`,
		OwnerStatus: &OwnerStatus{Available: true},
		RehomeFn:    RehomeTranscript,
	})
	if got.Action != "PIN" {
		t.Fatalf("action = %q, want PIN", got.Action)
	}
	if got.MirroredToCwd != "C--work-slack-helpers" {
		t.Fatalf("mirrored_to_cwd_slug = %q, want C--work-slack-helpers", got.MirroredToCwd)
	}
	mirror := filepath.Join(home, ".claude-owner", "projects", "C--work-slack-helpers", sid+".jsonl")
	if _, err := os.Stat(mirror); err != nil {
		t.Fatalf("cwd-slug mirror not written: %v", err)
	}
}

func TestResolveCarriedThrottleReProbePins(t *testing.T) {
	home := t.TempDir()
	sid := "s3"
	writeTranscript(t, home, ".claude-owner", testProject, sid, time.Now(), 10)
	got := Resolve(ResolveInput{
		SID: sid, Home: home, CWD: testCWD, ProbeOwner: true,
		OwnerStatus: &OwnerStatus{Available: false, BlockKind: "usage", StatusSource: "registry", BlockReason: "usage limit; resets 3pm"},
		ProbeFn: func(account, _ string) *ProbeResult {
			return &ProbeResult{Available: true, StatusSource: "probe"}
		},
		RehomeFn: RehomeTranscript,
	})
	if got.Action != "PIN" {
		t.Fatalf("action = %q, want PIN (stale carried throttle cleared by probe)", got.Action)
	}
	if got.OwnerProbe == nil || !got.OwnerProbe.Available {
		t.Fatalf("owner_probe = %+v, want available", got.OwnerProbe)
	}
}

func TestResolveRehomesBlockedOwner(t *testing.T) {
	home := t.TempDir()
	sid := "s4"
	writeTranscript(t, home, ".claude-owner", testProject, sid, time.Now(), 10)
	healthyCfg := filepath.Join(home, ".claude-healthy")

	got := Resolve(ResolveInput{
		SID: sid, Home: home, CWD: testCWD, ProbeOwner: false,
		OwnerStatus:  &OwnerStatus{Available: false, BlockKind: "usage", StatusSource: "registry", BlockReason: "usage limit"},
		Availability: []Target{{Account: ".claude-healthy", Available: true, LiveSessions: 0, ConfigDir: healthyCfg}},
		RehomeFn:     RehomeTranscript,
	})
	if got.Action != "REHOME" {
		t.Fatalf("action = %q, want REHOME", got.Action)
	}
	if got.PinAccount != ".claude-healthy" {
		t.Fatalf("pin account = %q, want .claude-healthy", got.PinAccount)
	}
	if !got.Rehomed {
		t.Fatal("REHOME should set rehomed=true")
	}
	moved := filepath.Join(healthyCfg, "projects", testProject, sid+".jsonl")
	if _, err := os.Stat(moved); err != nil {
		t.Fatalf("transcript not re-homed onto target: %v", err)
	}
}

func TestResolvePinBlockedNoHealthyTarget(t *testing.T) {
	home := t.TempDir()
	sid := "s5"
	writeTranscript(t, home, ".claude-owner", testProject, sid, time.Now(), 10)
	got := Resolve(ResolveInput{
		SID: sid, Home: home, CWD: testCWD, ProbeOwner: false,
		OwnerStatus:  &OwnerStatus{Available: false, BlockKind: "usage", StatusSource: "registry", BlockReason: "usage limit"},
		Availability: []Target{}, // no healthy worker
		RehomeFn:     RehomeTranscript,
	})
	if got.Action != "PIN_BLOCKED" {
		t.Fatalf("action = %q, want PIN_BLOCKED", got.Action)
	}
	if got.PinConfigDir != filepath.Join(home, ".claude-owner") {
		t.Fatalf("pin dir = %q, want owner (best effort)", got.PinConfigDir)
	}
}

// TestResolveWaitsForImminentOwnerReset: a blocked owner whose reset is minutes away is
// worth waiting for — the verdict is WAIT_RESET pinned to the owner with a machine-checkable
// countdown, and NO transcript copy lands on the healthy sibling (the silent-re-home this
// verdict exists to prevent).
func TestResolveWaitsForImminentOwnerReset(t *testing.T) {
	home := t.TempDir()
	sid := "s-wait"
	writeTranscript(t, home, ".claude-owner", testProject, sid, time.Now(), 10)
	healthyCfg := filepath.Join(home, ".claude-healthy")
	now := int64(1_700_000_000)

	got := Resolve(ResolveInput{
		SID: sid, Home: home, CWD: testCWD, NowUnix: now,
		OwnerStatus:  &OwnerStatus{Available: false, BlockKind: "usage", BlockReason: "usage limit; resets 7:10pm", ResetUnix: now + 180},
		Availability: []Target{{Account: ".claude-healthy", Available: true, ConfigDir: healthyCfg}},
		RehomeFn:     RehomeTranscript,
	})
	if got.Action != "WAIT_RESET" {
		t.Fatalf("action = %q, want WAIT_RESET", got.Action)
	}
	if got.PinAccount != ".claude-owner" {
		t.Errorf("pin account = %q, want the owner (the seat worth waiting for)", got.PinAccount)
	}
	if got.WaitSeconds != 180 || got.ResetUnix != now+180 {
		t.Errorf("wait/reset = %d/%d, want 180/%d", got.WaitSeconds, got.ResetUnix, now+180)
	}
	if _, err := os.Stat(filepath.Join(healthyCfg, "projects", testProject, sid+".jsonl")); err == nil {
		t.Error("WAIT_RESET must not copy the transcript onto the sibling")
	}
}

// TestResolveWaitHorizon: the wait verdict is bounded — a reset beyond the horizon (or an
// already-expired one, or none at all) re-homes exactly as before, and NoWait forces the
// copy even when the reset is imminent.
func TestResolveWaitHorizon(t *testing.T) {
	now := int64(1_700_000_000)
	cases := []struct {
		name   string
		reset  int64
		noWait bool
		want   string
	}{
		{"imminent reset waits", now + WaitResetHorizonSeconds, false, "WAIT_RESET"},
		{"distant reset rehomes", now + WaitResetHorizonSeconds + 1, false, "REHOME"},
		{"expired reset rehomes", now - 30, false, "REHOME"},
		{"unknown reset rehomes", 0, false, "REHOME"},
		{"no-wait forces rehome", now + 60, true, "REHOME"},
	}
	for _, tc := range cases {
		home := t.TempDir()
		sid := "s-horizon"
		writeTranscript(t, home, ".claude-owner", testProject, sid, time.Now(), 10)
		got := Resolve(ResolveInput{
			SID: sid, Home: home, CWD: testCWD, NowUnix: now, NoWait: tc.noWait,
			OwnerStatus:  &OwnerStatus{Available: false, BlockKind: "usage", BlockReason: "usage limit", ResetUnix: tc.reset},
			Availability: []Target{{Account: ".claude-healthy", Available: true, ConfigDir: filepath.Join(home, ".claude-healthy")}},
			RehomeFn:     RehomeTranscript,
		})
		if got.Action != tc.want {
			t.Errorf("%s: action = %q, want %q", tc.name, got.Action, tc.want)
		}
	}
}

// TestResolveProbeResetFeedsWait: when the owner's carried throttle is re-probed and the
// probe confirms the block WITH a fresher reset window, the wait verdict uses the probe's
// instant, not the carried one.
func TestResolveProbeResetFeedsWait(t *testing.T) {
	home := t.TempDir()
	sid := "s-probe-wait"
	writeTranscript(t, home, ".claude-owner", testProject, sid, time.Now(), 10)
	now := int64(1_700_000_000)

	got := Resolve(ResolveInput{
		SID: sid, Home: home, CWD: testCWD, ProbeOwner: true, NowUnix: now,
		OwnerStatus: &OwnerStatus{Available: false, BlockKind: "usage", StatusSource: "registry",
			BlockReason: "usage limit; resets 3pm", ResetUnix: now + WaitResetHorizonSeconds + 600},
		ProbeFn: func(account, _ string) *ProbeResult {
			return &ProbeResult{Available: false, BlockReason: "usage limit; resets 7:10pm", ResetUnix: now + 120}
		},
		Availability: []Target{{Account: ".claude-healthy", Available: true, ConfigDir: filepath.Join(home, ".claude-healthy")}},
		RehomeFn:     RehomeTranscript,
	})
	if got.Action != "WAIT_RESET" {
		t.Fatalf("action = %q, want WAIT_RESET (probe's fresher reset is imminent)", got.Action)
	}
	if got.ResetUnix != now+120 {
		t.Errorf("reset_unix = %d, want the probe's %d", got.ResetUnix, now+120)
	}
}
