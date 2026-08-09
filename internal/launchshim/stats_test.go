package launchshim

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestStatsAggregateAndLeakFence(t *testing.T) {
	path := filepath.Join(t.TempDir(), "stats.json")
	t.Setenv("FAK_LAUNCH_STATS", path)
	secret := "PROMPT_SECRET_ARG_CANARY"
	t.Setenv("IGNORED_PROMPT", secret)
	if err := Record("explicit", "claude", "guarded", "success", 200*time.Millisecond); err != nil {
		t.Fatal(err)
	}
	if err := Record("explicit", "claude", "guarded", "success", 300*time.Millisecond); err != nil {
		t.Fatal(err)
	}
	if err := Record("shim", "private-provider-name", "direct", "provider_exit", 2*time.Second); err != nil {
		t.Fatal(err)
	}
	s, err := ReadStats()
	if err != nil {
		t.Fatal(err)
	}
	if len(s.Counters) != 2 || s.Counters[0].Count != 2 || s.Counters[1].Provider != "custom" {
		t.Fatalf("stats=%+v", s)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{secret, "private-provider-name", t.TempDir()} {
		if strings.Contains(string(b), forbidden) {
			t.Fatalf("ledger leaked %q: %s", forbidden, b)
		}
	}
	if err := ResetStats(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("reset left ledger: %v", err)
	}
}

func TestStatsRejectOpenDimensions(t *testing.T) {
	t.Setenv("FAK_LAUNCH_STATS", filepath.Join(t.TempDir(), "s"))
	if err := Record("prompt text", "claude", "guarded", "success", 0); err == nil {
		t.Fatal("open surface accepted")
	}
}
