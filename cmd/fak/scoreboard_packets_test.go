package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/issuehygiene"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

func TestRunScoreboardIssuePacketsDefaultBound(t *testing.T) {
	path := filepath.Join(t.TempDir(), "card.json")
	body, err := json.Marshal(scorecard.Payload{Schema: issuehygiene.Schema, KPIs: []scorecard.KPI{
		{Key: "class", Defects: []string{"#9 missing class", "#3 missing class"}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, body, 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if code := runScoreboardIssuePackets(&stdout, &stderr, []string{"--from", path}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var plan issuehygiene.PacketPlan
	if err := json.Unmarshal(stdout.Bytes(), &plan); err != nil {
		t.Fatal(err)
	}
	if plan.PacketSize != 1 || len(plan.Packets) != 2 || plan.Packets[0].Issues[0] != 3 {
		t.Fatalf("unexpected default plan: %+v", plan)
	}
}

func TestRunScoreboardIssuePacketsRejectsOversizedWithoutGate(t *testing.T) {
	path := filepath.Join(t.TempDir(), "card.json")
	body := `{"schema":"` + issuehygiene.Schema + `","kpis":[{"defects":["#1 a","#2 b"]}]}`
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := runScoreboardIssuePackets(&stdout, &stderr, []string{"--from", path, "--packet-size", "2"})
	if code == 0 || !strings.Contains(stderr.String(), "--unsafe-oversized") {
		t.Fatalf("code=%d stderr=%q", code, stderr.String())
	}
}
