package main

import (
	"bytes"
	"flag"
	"strings"
	"testing"
)

func TestServeHelpDefaultsToCategories(t *testing.T) {
	fs, _ := newServeFlagSet()
	var out bytes.Buffer
	printServeHelp(&out, fs, "")
	got := out.String()
	for _, want := range []string{"fak serve help <category>", "start", "native", "context", "help all"} {
		if !strings.Contains(got, want) {
			t.Fatalf("compact help missing %q:\n%s", want, got)
		}
	}
	if strings.Contains(got, "JOIN THE FLEET CONTROL BUS") || strings.Count(got, "\n") > 30 {
		t.Fatalf("default serve help regressed into the detailed flag wall (%d lines)", strings.Count(got, "\n"))
	}
}

func TestServeHelpCategoryIsConciseAndScoped(t *testing.T) {
	fs, _ := newServeFlagSet()
	var out bytes.Buffer
	printServeHelp(&out, fs, "native")
	got := out.String()
	for _, want := range []string{"--native", "--native-max-turns", "--native-code-workspace"} {
		if !strings.Contains(got, want) {
			t.Fatalf("native help missing %q:\n%s", want, got)
		}
	}
	if strings.Contains(got, "--fleet-bus") || strings.Contains(got, "--policy-check") {
		t.Fatalf("native help leaked unrelated categories:\n%s", got)
	}
	for _, line := range strings.Split(got, "\n") {
		if len([]rune(line)) > 110 {
			t.Fatalf("category explanation is not concise (%d runes): %s", len([]rune(line)), line)
		}
	}
}

func TestEveryServeFlagIsShiftedIntoAHelpCategory(t *testing.T) {
	fs, _ := newServeFlagSet()
	categorized := make(map[string]string)
	for _, category := range serveHelpCategories {
		for _, name := range category.flags {
			if prior := categorized[name]; prior != "" {
				t.Fatalf("serve flag --%s appears in both %s and %s", name, prior, category.name)
			}
			categorized[name] = category.name
		}
	}
	fs.VisitAll(func(f *flag.Flag) {
		if categorized[f.Name] == "" {
			t.Errorf("new serve flag --%s needs a task category before it lands", f.Name)
		}
	})
	for name, category := range categorized {
		if fs.Lookup(name) == nil {
			t.Errorf("%s category names removed serve flag --%s", category, name)
		}
	}
}

func TestServeHelpTopicForms(t *testing.T) {
	for _, tc := range []struct {
		argv  []string
		topic string
		ok    bool
	}{
		{[]string{"--help"}, "", true},
		{[]string{"help"}, "", true},
		{[]string{"help", "cache"}, "cache", true},
		{[]string{"cache", "--help"}, "cache", true},
		{[]string{"--addr", ":0"}, "", false},
	} {
		got, ok := serveHelpTopic(tc.argv)
		if got != tc.topic || ok != tc.ok {
			t.Errorf("serveHelpTopic(%q) = %q, %v; want %q, %v", tc.argv, got, ok, tc.topic, tc.ok)
		}
	}
}

func TestServePolicyCanaryFlagDefaultsOffAndParses(t *testing.T) {
	fs, sf := newServeFlagSet()
	if sf.policyCanaryTurns == nil || *sf.policyCanaryTurns != 0 {
		t.Fatalf("policy canary default = %v, want initialized zero", sf.policyCanaryTurns)
	}
	if err := fs.Parse([]string{"--policy-canary-turns", "3"}); err != nil {
		t.Fatal(err)
	}
	if got := *sf.policyCanaryTurns; got != 3 {
		t.Fatalf("policy canary turns = %d, want 3", got)
	}
	var out bytes.Buffer
	printServeHelp(&out, fs, "policy")
	if !strings.Contains(out.String(), "--policy-canary-turns") {
		t.Fatalf("policy help omits canary flag:\n%s", out.String())
	}
}
