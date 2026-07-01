package rehome

import (
	"testing"
	"time"
)

// probeMap builds a ProbeFunc from an account->available map. A missing account
// returns nil (unprobeable).
func probeMap(m map[string]bool) ProbeFunc {
	return func(account, _ string) *ProbeResult {
		avail, ok := m[account]
		if !ok {
			return nil
		}
		return &ProbeResult{Available: avail, StatusSource: "probe"}
	}
}

func TestReselectSingleMatchKeeps(t *testing.T) {
	home := t.TempDir()
	sid := "one"
	writeTranscript(t, home, ".claude-a", "C--work-fak", sid, time.Now(), 100)
	got := ReselectDuplicateOwner(sid, home, probeMap(map[string]bool{".claude-a": false}))
	if got.Mode != ReselectKeep {
		t.Fatalf("mode = %v, want Keep for single match", got.Mode)
	}
}

func TestReselectFreshestServesKeeps(t *testing.T) {
	home := t.TempDir()
	sid := "dup"
	base := time.Now().Add(-time.Hour)
	writeTranscript(t, home, ".claude-fresh", "C--work-fak", sid, base.Add(10*time.Minute), 100)
	writeTranscript(t, home, ".claude-old", "C--work-fak", sid, base, 100)
	got := ReselectDuplicateOwner(sid, home, probeMap(map[string]bool{
		".claude-fresh": true, ".claude-old": true,
	}))
	if got.Mode != ReselectKeep {
		t.Fatalf("mode = %v, want Keep when freshest serves", got.Mode)
	}
}

func TestReselectParitySiblingPins(t *testing.T) {
	home := t.TempDir()
	sid := "dup"
	base := time.Now().Add(-time.Hour)
	// freshest is walled; older sibling is at content parity (same size) and serves.
	writeTranscript(t, home, ".claude-day24", "C--work-fak", sid, base.Add(10*time.Minute), 100)
	writeTranscript(t, home, ".claude-q", "C--work-fak", sid, base, 100)
	got := ReselectDuplicateOwner(sid, home, probeMap(map[string]bool{
		".claude-day24": false, ".claude-q": true,
	}))
	if got.Mode != ReselectPin {
		t.Fatalf("mode = %v, want Pin", got.Mode)
	}
	if got.Owner == nil || got.Owner.Account != ".claude-q" {
		t.Fatalf("pin owner = %+v, want .claude-q", got.Owner)
	}
	if got.Owner.DupCount != 2 {
		t.Fatalf("dup_count = %d, want 2", got.Owner.DupCount)
	}
}

func TestReselectBehindSiblingRehomes(t *testing.T) {
	home := t.TempDir()
	sid := "dup"
	base := time.Now().Add(-time.Hour)
	// freshest walled with 100 bytes; only serving sibling is content-behind (50).
	writeTranscript(t, home, ".claude-day24", "C--work-fak", sid, base.Add(10*time.Minute), 100)
	writeTranscript(t, home, ".claude-q", "C--work-fak", sid, base, 50)
	got := ReselectDuplicateOwner(sid, home, probeMap(map[string]bool{
		".claude-day24": false, ".claude-q": true,
	}))
	if got.Mode != ReselectRehome {
		t.Fatalf("mode = %v, want Rehome", got.Mode)
	}
	if got.Source == nil || got.Source.Account != ".claude-day24" {
		t.Fatalf("rehome source = %+v, want .claude-day24 (freshest content)", got.Source)
	}
	if got.Target == nil || got.Target.Account != ".claude-q" {
		t.Fatalf("rehome target = %+v, want .claude-q (serving sibling)", got.Target)
	}
}

func TestReselectAllSiblingsBlockedKeeps(t *testing.T) {
	home := t.TempDir()
	sid := "dup"
	base := time.Now().Add(-time.Hour)
	writeTranscript(t, home, ".claude-day24", "C--work-fak", sid, base.Add(10*time.Minute), 100)
	writeTranscript(t, home, ".claude-q", "C--work-fak", sid, base, 50)
	got := ReselectDuplicateOwner(sid, home, probeMap(map[string]bool{
		".claude-day24": false, ".claude-q": false,
	}))
	if got.Mode != ReselectKeep {
		t.Fatalf("mode = %v, want Keep when every sibling is blocked", got.Mode)
	}
}
