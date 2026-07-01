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
