package rehome

import (
	"reflect"
	"testing"
)

func accounts(ts []Target) []string {
	out := make([]string, len(ts))
	for i, t := range ts {
		out[i] = t.Account
	}
	return out
}

func TestRehomeTargetsRanksLeastLoadedFirst(t *testing.T) {
	avail := []Target{
		{Account: ".claude-b", Available: true, LiveSessions: 3},
		{Account: ".claude-a", Available: true, LiveSessions: 1},
		{Account: ".claude-c", Available: true, LiveSessions: 2},
	}
	got := accounts(RehomeTargets(avail, "", nil, RehomeCap()))
	want := []string{".claude-a", ".claude-c", ".claude-b"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ranking = %v, want %v", got, want)
	}
}

func TestRehomeTargetsExcludesOwnerAndOpencode(t *testing.T) {
	avail := []Target{
		{Account: ".claude-owner", Available: true, LiveSessions: 0},
		{Account: ".config-opencode-x", Available: true, LiveSessions: 0}, // not .claude*
		{Account: ".claude-healthy", Available: true, LiveSessions: 0},
		{Account: ".claude-blocked", Available: false, LiveSessions: 0},
	}
	got := accounts(RehomeTargets(avail, ".claude-owner", nil, RehomeCap()))
	want := []string{".claude-healthy"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("filtered = %v, want %v", got, want)
	}
}

func TestRehomeTargetsCapDropsOverloaded(t *testing.T) {
	avail := []Target{
		{Account: ".claude-full", Available: true, LiveSessions: 4}, // >= cap 4
		{Account: ".claude-room", Available: true, LiveSessions: 3},
	}
	got := accounts(RehomeTargets(avail, "", nil, DefaultRehomeCap))
	want := []string{".claude-room"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("capped = %v, want %v", got, want)
	}
	// Uncapped relief: a single interactive resume admits the over-cap account too,
	// still least-loaded first.
	got = accounts(RehomeTargets(avail, "", nil, CapUnbounded))
	want = []string{".claude-room", ".claude-full"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("uncapped = %v, want %v", got, want)
	}
}

func TestRehomeTargetsAssignedFoldsIntoLoad(t *testing.T) {
	avail := []Target{
		{Account: ".claude-a", Available: true, LiveSessions: 1},
		{Account: ".claude-b", Available: true, LiveSessions: 2},
	}
	// a already took 2 this pass -> effective load 3 vs b's 2 -> b first.
	got := accounts(RehomeTargets(avail, "", map[string]int{".claude-a": 2}, CapUnbounded))
	want := []string{".claude-b", ".claude-a"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("assigned-folded = %v, want %v", got, want)
	}
}

func TestRehomeTargetsPositiveEvidenceTieBreak(t *testing.T) {
	avail := []Target{
		{Account: ".claude-carried", Available: true, LiveSessions: 1, VerdictSource: "carried"},
		{Account: ".claude-probed", Available: true, LiveSessions: 1, VerdictSource: "probe"},
	}
	// Equal load -> proven (probe) sorts ahead of carried.
	got := accounts(RehomeTargets(avail, "", nil, RehomeCap()))
	want := []string{".claude-probed", ".claude-carried"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("evidence tie-break = %v, want %v", got, want)
	}
}

func TestRehomeCapEnvOverride(t *testing.T) {
	t.Setenv("FAK_REHOME_CAP", "7")
	if got := RehomeCap(); got != 7 {
		t.Fatalf("RehomeCap() = %d, want 7", got)
	}
	t.Setenv("FAK_REHOME_CAP", "notanint")
	if got := RehomeCap(); got != DefaultRehomeCap {
		t.Fatalf("RehomeCap() on bad env = %d, want %d", got, DefaultRehomeCap)
	}
}
