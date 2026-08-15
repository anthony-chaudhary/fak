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
