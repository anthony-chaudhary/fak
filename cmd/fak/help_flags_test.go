package main

import (
	"bytes"
	"flag"
	"strings"
	"testing"
)

func TestConciseFlagDefaultsShortenExplanationsByDefault(t *testing.T) {
	fs := flag.NewFlagSet("example", flag.ContinueOnError)
	fs.String("long", "", "First sentence tells the operator what matters. Second sentence is implementation detail that belongs below the default help surface and should not make scanning harder.")
	var out bytes.Buffer
	printFlagDefaults(&out, fs, true)
	got := out.String()
	if !strings.Contains(got, "First sentence tells the operator what matters.") {
		t.Fatalf("concise defaults lost the leading explanation: %s", got)
	}
	if strings.Contains(got, "implementation detail") {
		t.Fatalf("concise defaults leaked depth into the default surface: %s", got)
	}
	for _, line := range strings.Split(got, "\n") {
		if len([]rune(line)) > 110 {
			t.Fatalf("default flag help line is too long: %s", line)
		}
	}
}

func TestConciseFlagDefaultsBoundLongSummaries(t *testing.T) {
	for _, tc := range []struct {
		name, usage, want string
	}{
		{"at limit", strings.Repeat("a", 104), strings.Repeat("a", 104)},
		{"long word", strings.Repeat("a", 105), strings.Repeat("a", 103) + "…"},
		{"unicode", strings.Repeat("界", 105), strings.Repeat("界", 103) + "…"},
		{"word boundary", strings.Repeat("word ", 30), strings.TrimSpace(strings.Repeat("word ", 20)) + "…"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs := flag.NewFlagSet("example", flag.ContinueOnError)
			fs.String("sample", "", tc.usage)
			var out bytes.Buffer
			printFlagDefaults(&out, fs, true)
			if got, want := out.String(), "  --sample string\n      "+tc.want+"\n"; got != want {
				t.Fatalf("concise defaults = %q, want %q", got, want)
			}
			for _, line := range strings.Split(out.String(), "\n") {
				if len([]rune(line)) > 110 {
					t.Fatalf("default flag help line is too long: %s", line)
				}
			}
		})
	}
}

func TestFlagDefaultsPreserveOperationalDescriptions(t *testing.T) {
	for _, tc := range []struct {
		verb, name, usage string
	}{
		{"commit", "build-check-timeout", "maximum duration for prospective validation (default 4m); controls prospective validation, not advisory-lock waiting or earlier build/materialization phases"},
		{"sync", "goal", "reconcile: target goal: publish (default publish HEAD), publish <sha>, integrate (default integrate origin/dev)"},
		{"sync", "source-commit", "push: exact ancestor commit for an explicit handoff/... branch; never grants publication authority"},
		{"resume", "state", "drive-state to record: paused (reversible hold), draining, or stopped (terminal intent — never auto-resume)"},
		{"info", "token-destination-max-age", "maximum age of --token-destination-snapshot before the TUI labels the retained mix STALE (must be positive when a snapshot path is set)"},
		{"merge", "apply", "auto-resolve the trivial superset case (merged tree == HEAD) with a textless `-s ours` merge; defer any real conflict to hand-merge"},
	} {
		t.Run(tc.verb+"/"+tc.name, func(t *testing.T) {
			fs := flag.NewFlagSet(tc.verb, flag.ContinueOnError)
			fs.String(tc.name, "", tc.usage)
			arg, usage := flag.UnquoteUsage(fs.Lookup(tc.name))
			var out bytes.Buffer
			printFlagDefaults(&out, fs, false)
			if got, want := out.String(), "  --"+tc.name+" "+arg+"\n      "+usage+"\n"; got != want {
				t.Fatalf("operational flag help = %q, want %q", got, want)
			}
			out.Reset()
			printFlagDefaults(&out, fs, true)
			if got, want := out.String(), "  --"+tc.name+" "+arg+"\n      "+conciseFlagSummary(usage)+"\n"; got != want {
				t.Fatalf("concise flag help = %q, want %q", got, want)
			}
		})
	}
}
