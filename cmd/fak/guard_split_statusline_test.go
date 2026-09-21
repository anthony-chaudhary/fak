package main

import (
	"errors"
	"strings"
	"testing"
)

func TestGuardSplitBannerEnabled(t *testing.T) {
	tests := []struct {
		val  string
		want bool
	}{
		{"", true},
		{"1", true},
		{"on", true},
		{"0", false},
		{"off", false},
		{"false", false},
		{"no", false},
		{"FALSE", false},
		{" off ", false},
	}
	for _, tt := range tests {
		t.Run("val="+tt.val, func(t *testing.T) {
			getenv := func(string) string { return tt.val }
			if got := guardSplitBannerEnabled(getenv); got != tt.want {
				t.Fatalf("guardSplitBannerEnabled(%q) = %v, want %v", tt.val, got, tt.want)
			}
		})
	}
}

func TestGuardSplitMilestoneNotify(t *testing.T) {
	found := func(string) (string, error) { return "/usr/bin/osascript", nil }
	missing := func(string) (string, error) { return "", errors.New("not found") }

	t.Run("darwin with osascript calls run", func(t *testing.T) {
		var gotName string
		var gotArgs []string
		called := 0
		run := func(name string, args ...string) error {
			called++
			gotName = name
			gotArgs = args
			return nil
		}
		ok := guardSplitMilestoneNotify("darwin", found, run, "fak guard", "session live")
		if !ok {
			t.Fatal("expected notify to report attempted=true")
		}
		if called != 1 {
			t.Fatalf("run called %d times, want 1", called)
		}
		if gotName != "/usr/bin/osascript" || len(gotArgs) != 2 || gotArgs[0] != "-e" {
			t.Fatalf("unexpected exec: %s %q", gotName, gotArgs)
		}
		want := `display notification "session live" with title "fak guard"`
		if gotArgs[1] != want {
			t.Fatalf("script mismatch\n got: %q\nwant: %q", gotArgs[1], want)
		}
	})

	t.Run("darwin missing osascript is a quiet no-op", func(t *testing.T) {
		called := 0
		run := func(string, ...string) error { called++; return nil }
		if ok := guardSplitMilestoneNotify("darwin", missing, run, "t", "m"); ok {
			t.Fatal("expected false when osascript is missing")
		}
		if called != 0 {
			t.Fatalf("run must not be called, got %d calls", called)
		}
	})

	t.Run("linux is a no-op", func(t *testing.T) {
		called := 0
		run := func(string, ...string) error { called++; return nil }
		if ok := guardSplitMilestoneNotify("linux", found, run, "t", "m"); ok {
			t.Fatal("expected false on linux")
		}
		if called != 0 {
			t.Fatalf("run must not be called on linux, got %d calls", called)
		}
	})

	t.Run("windows is a no-op", func(t *testing.T) {
		if ok := guardSplitMilestoneNotify("windows", found, func(string, ...string) error { return nil }, "t", "m"); ok {
			t.Fatal("expected false on windows")
		}
	})

	t.Run("hostile message is escaped safely", func(t *testing.T) {
		var script string
		run := func(_ string, args ...string) error {
			script = args[1]
			return nil
		}
		msg := `he said "drop table" and a \ path`
		title := `t"t\le`
		if ok := guardSplitMilestoneNotify("darwin", found, run, title, msg); !ok {
			t.Fatal("expected notify attempted")
		}
		// 4 literal quotes delimiting the two string literals + 3 escaped ones
		// (2 in the message, 1 in the title) — a hostile quote must be a backslash-escape.
		if got := strings.Count(script, `"`); got != 7 {
			t.Fatalf("quote count = %d, want 7 (escaped, not raw): %q", got, script)
		}
		want := `display notification "he said \"drop table\" and a \\ path" with title "t\"t\\le"`
		if script != want {
			t.Fatalf("escape mismatch\n got: %q\nwant: %q", script, want)
		}
		if strings.Contains(script, `"drop table"`) {
			t.Fatalf("raw quote survived escaping: %q", script)
		}
	})
}
