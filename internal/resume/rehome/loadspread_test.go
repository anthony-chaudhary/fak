package rehome

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestResolveLoadSpreadsOffLoadedOwner: the owner is AVAILABLE but already carries
// the fleet burst cap of live sessions while a near-idle healthy seat exists -> the
// resume re-homes to the less-loaded seat instead of piling onto the loaded owner
// (the july7 429 pile-up shape). Safe by default; FAK_LOAD_REHOME=0 disables.
func TestResolveLoadSpreadsOffLoadedOwner(t *testing.T) {
	t.Setenv("FAK_REHOME_CAP", "")
	t.Setenv(LoadSpreadEnv, "")
	home := t.TempDir()
	sid := "spread1"
	writeTranscript(t, home, ".claude-owner", testProject, sid, time.Now(), 10)
	idle := filepath.Join(home, ".claude-idle")
	got := Resolve(ResolveInput{
		SID: sid, Home: home, CWD: testCWD,
		OwnerStatus: &OwnerStatus{Available: true},
		RehomeFn:    RehomeTranscript,
		Availability: []Target{
			{Account: ".claude-owner", Available: true, LiveSessions: DefaultRehomeCap, ConfigDir: filepath.Join(home, ".claude-owner")},
			{Account: ".claude-idle", Available: true, LiveSessions: 1, ConfigDir: idle},
		},
	})
	if got.Action != "REHOME" || got.PinAccount != ".claude-idle" {
		t.Fatalf("Resolve = %+v, want REHOME onto .claude-idle", got)
	}
	if got.LoadSpread == nil || got.LoadSpread.OwnerLive != DefaultRehomeCap || got.LoadSpread.RehomeCap != DefaultRehomeCap {
		t.Fatalf("load_spread = %+v, want owner_live=%d cap=%d recorded", got.LoadSpread, DefaultRehomeCap, DefaultRehomeCap)
	}
	if !got.Rehomed {
		t.Fatalf("expected a real transcript copy: %+v", got)
	}
	copied := filepath.Join(idle, "projects", testProject, sid+".jsonl")
	if _, err := os.Stat(copied); err != nil {
		t.Fatalf("re-homed transcript missing at %s: %v", copied, err)
	}
}

// TestResolveLoadSpreadKeepsPinWithoutFreerSeat: every other seat is itself at the
// cap, so there is nowhere better to go -> the pin to the available owner stands
// (a lone resume is never stranded by the spread).
func TestResolveLoadSpreadKeepsPinWithoutFreerSeat(t *testing.T) {
	t.Setenv("FAK_REHOME_CAP", "")
	t.Setenv(LoadSpreadEnv, "")
	home := t.TempDir()
	sid := "spread2"
	writeTranscript(t, home, ".claude-owner", testProject, sid, time.Now(), 10)
	got := Resolve(ResolveInput{
		SID: sid, Home: home, CWD: testCWD,
		OwnerStatus: &OwnerStatus{Available: true},
		RehomeFn:    RehomeTranscript,
		Availability: []Target{
			{Account: ".claude-owner", Available: true, LiveSessions: DefaultRehomeCap + 2, ConfigDir: filepath.Join(home, ".claude-owner")},
			{Account: ".claude-alsofull", Available: true, LiveSessions: DefaultRehomeCap, ConfigDir: filepath.Join(home, ".claude-alsofull")},
		},
	})
	if got.Action != "PIN" || got.PinAccount != ".claude-owner" {
		t.Fatalf("Resolve = %+v, want PIN to owner (no under-cap seat to spread to)", got)
	}
	if got.LoadSpread != nil {
		t.Fatalf("no spread should be recorded on a kept pin: %+v", got)
	}
}

// TestResolveLoadSpreadUnderCapOwnerPins: an owner below the burst cap keeps the
// historical pin even when an idle seat exists — the spread needs positive
// overload evidence, not merely a freer sibling.
func TestResolveLoadSpreadUnderCapOwnerPins(t *testing.T) {
	t.Setenv("FAK_REHOME_CAP", "")
	t.Setenv(LoadSpreadEnv, "")
	home := t.TempDir()
	sid := "spread3"
	writeTranscript(t, home, ".claude-owner", testProject, sid, time.Now(), 10)
	got := Resolve(ResolveInput{
		SID: sid, Home: home, CWD: testCWD,
		OwnerStatus: &OwnerStatus{Available: true},
		RehomeFn:    RehomeTranscript,
		Availability: []Target{
			{Account: ".claude-owner", Available: true, LiveSessions: DefaultRehomeCap - 1, ConfigDir: filepath.Join(home, ".claude-owner")},
			{Account: ".claude-idle", Available: true, LiveSessions: 0, ConfigDir: filepath.Join(home, ".claude-idle")},
		},
	})
	if got.Action != "PIN" || got.PinAccount != ".claude-owner" || got.LoadSpread != nil {
		t.Fatalf("Resolve = %+v, want plain PIN to under-cap owner", got)
	}
}

// TestResolveLoadSpreadKillSwitch: FAK_LOAD_REHOME=0 restores the historical
// pin-home-whenever-available behavior even for a heavily loaded owner.
func TestResolveLoadSpreadKillSwitch(t *testing.T) {
	t.Setenv("FAK_REHOME_CAP", "")
	t.Setenv(LoadSpreadEnv, "0")
	home := t.TempDir()
	sid := "spread4"
	writeTranscript(t, home, ".claude-owner", testProject, sid, time.Now(), 10)
	got := Resolve(ResolveInput{
		SID: sid, Home: home, CWD: testCWD,
		OwnerStatus: &OwnerStatus{Available: true},
		RehomeFn:    RehomeTranscript,
		Availability: []Target{
			{Account: ".claude-owner", Available: true, LiveSessions: DefaultRehomeCap + 4, ConfigDir: filepath.Join(home, ".claude-owner")},
			{Account: ".claude-idle", Available: true, LiveSessions: 0, ConfigDir: filepath.Join(home, ".claude-idle")},
		},
	})
	if got.Action != "PIN" || got.PinAccount != ".claude-owner" {
		t.Fatalf("Resolve = %+v, want PIN (FAK_LOAD_REHOME=0 restores pin-home)", got)
	}
}
