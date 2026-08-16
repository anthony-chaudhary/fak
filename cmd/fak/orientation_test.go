package main

import (
	"bytes"
	"strings"
	"testing"
	"time"
)

func TestOrientationCLIRendersCurrentAndStale(t *testing.T) {
	for _, tc := range []struct{ asOf, want string }{{"2026-08-15", "FAK ORIENTATION — CURRENT"}, {"2026-10-16", "FAK ORIENTATION — STALE"}} {
		var out, stderr bytes.Buffer
		if err := runOrientation([]string{"--as-of", tc.asOf}, &out, &stderr, time.Now); err != nil {
			t.Fatalf("%s: %v (%s)", tc.asOf, err, stderr.String())
		}
		if !strings.Contains(out.String(), tc.want) || !strings.Contains(out.String(), "Owned seam:") {
			t.Fatalf("%s output:\n%s", tc.asOf, out.String())
		}
	}
}

func TestOrientationCLIJSON(t *testing.T) {
	var out bytes.Buffer
	if err := runOrientation([]string{"--as-of", "2026-10-05", "--json"}, &out, &bytes.Buffer{}, time.Now); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"schema": "fak-orientation/1"`, `"freshness": "due-soon"`, `"id": "performance-cache"`, `"id": "harness-bindings"`} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("missing %s:\n%s", want, out.String())
		}
	}
}
