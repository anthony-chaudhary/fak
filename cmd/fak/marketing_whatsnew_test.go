package main

import (
	"bytes"
	"strings"
	"testing"
	"time"
)

func TestWhatsNewFreshnessBoundsAgeAndVelocity(t *testing.T) {
	for _, tc := range []struct {
		name                           string
		age, behind, maxAge, maxBehind int
		want                           bool
	}{
		{name: "at both ceilings", age: 7, behind: 250, maxAge: 7, maxBehind: 250},
		{name: "date stale", age: 8, behind: 0, maxAge: 7, maxBehind: 250, want: true},
		{name: "high velocity stale", age: 0, behind: 251, maxAge: 7, maxBehind: 250, want: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := whatsNewStale(tc.age, tc.behind, tc.maxAge, tc.maxBehind); got != tc.want {
				t.Fatalf("whatsNewStale(%d, %d, %d, %d) = %t, want %t",
					tc.age, tc.behind, tc.maxAge, tc.maxBehind, got, tc.want)
			}
		})
	}
}

func TestWhatsNewAgeUsesCommitDatesNotWallClock(t *testing.T) {
	anchor := time.Date(2026, 8, 1, 23, 0, 0, 0, time.UTC)
	if got := whatsNewAgeDays(anchor, anchor.Add(47*time.Hour)); got != 1 {
		t.Fatalf("age = %d, want 1 complete day", got)
	}
	if got := whatsNewAgeDays(anchor, anchor.Add(-time.Hour)); got != 0 {
		t.Fatalf("backward commit date age = %d, want 0", got)
	}
}

func TestWhatsNewReportNamesBothFreshnessCeilings(t *testing.T) {
	var out bytes.Buffer
	res := whatsNewCheck{
		Path: "docs/whats-new.md", Verdict: "stale", AnchorSHA: strings.Repeat("a", 40),
		AgeDays: 1, MaxAgeDays: 7, CommitsBehind: 251, MaxCommitsBehind: 250,
	}
	if code := whatsNewReport(&out, res, false); code != 1 {
		t.Fatalf("code = %d, want 1", code)
	}
	for _, want := range []string{"age 1/7", "behind 251/250", "docs/whats-new.md: stale"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("report missing %q: %s", want, out.String())
		}
	}
}
