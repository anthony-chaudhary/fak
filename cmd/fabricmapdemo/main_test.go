package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/fabricmap"
)

func TestSelfcheckCapturedOutput(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := run(&stdout, &stderr, []string{"-selfcheck"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var route fabricmap.Route
	if err := json.Unmarshal(stdout.Bytes(), &route); err != nil {
		t.Fatalf("decode captured route: %v\n%s", err, stdout.String())
	}
	if route.From != "L3" || route.To != "L1" || len(route.Links) != 1 {
		t.Fatalf("unexpected route: %#v", route)
	}
	link := route.Links[0]
	if link.ID != "ssd-to-gpu-direct" || link.CPUPath != "bypass" || link.Labels["gpu-direct"] != "yes" {
		t.Fatalf("unexpected proof link: %#v", link)
	}
	if !strings.Contains(stderr.String(), "PASS: L3 is a user label") {
		t.Fatalf("missing selfcheck verdict: %s", stderr.String())
	}
}

func TestUsageRequiresAnInputMode(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := run(&stdout, &stderr, nil); code != 2 {
		t.Fatalf("code=%d, want usage exit 2", code)
	}
	if stdout.Len() != 0 || !strings.Contains(stderr.String(), "usage: fabricmapdemo") {
		t.Fatalf("stdout=%q stderr=%q", stdout.String(), stderr.String())
	}
}
