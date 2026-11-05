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
	printConciseFlagDefaults(&out, fs)
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
			printConciseFlagDefaults(&out, fs)
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
