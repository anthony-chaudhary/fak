package main

import (
	"bytes"
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestQuantwatchOfflineNamedWitness(t *testing.T) {
	var stdout, stderr bytes.Buffer
	fixture := filepath.Join("..", "..", "internal", "quantwatch", "testdata", "ranking-v1.json")
	code := runQuantwatch(&stdout, &stderr, []string{"--snapshot", fixture}, http.DefaultClient, func() time.Time { return time.Time{} })
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, stderr.String())
	}
	for _, want := range []string{`"outcome": "ranked"`, `"deduplicated": 1`, `"hardware_envelope"`, `"evidence_boundary"`, `"query_time"`, `"sources"`} {
		if !bytes.Contains(stdout.Bytes(), []byte(want)) {
			t.Errorf("output missing %s\n%s", want, stdout.String())
		}
	}
}

func TestQuantwatchUnknownVersionExitIsTyped(t *testing.T) {
	var stdout, stderr bytes.Buffer
	fixture := filepath.Join("..", "..", "internal", "quantwatch", "testdata", "unknown-version.json")
	code := runQuantwatch(&stdout, &stderr, []string{"--snapshot", fixture}, http.DefaultClient, time.Now)
	if code != 3 {
		t.Fatalf("exit=%d stderr=%s", code, stderr.String())
	}
	if !bytes.Contains(stdout.Bytes(), []byte(`"reason": "unknown_version"`)) || !bytes.Contains(stdout.Bytes(), []byte(`"abstained": true`)) {
		t.Fatalf("output=%s", stdout.String())
	}
}

func TestMainDispatchIncludesQuantwatch(t *testing.T) {
	raw, err := os.ReadFile("main.go")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(raw, []byte(`case "quantwatch":`)) || !bytes.Contains(raw, []byte(`cmdQuantwatch(args)`)) {
		t.Fatal("quantwatch implementation exists but top-level fak dispatch is missing")
	}
}
