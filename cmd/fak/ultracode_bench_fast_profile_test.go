package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/fastintent"
	"github.com/anthony-chaudhary/fak/internal/ultracodebench"
)

func TestRunUltracodeBenchFastProfile(t *testing.T) {
	b := ultracodebench.FastProfileBundle{Schema: ultracodebench.FastProfileSchema, Scenario: "fast-profile"}
	raw, _ := json.Marshal(b)
	p := filepath.Join(t.TempDir(), "pair.json")
	if err := os.WriteFile(p, raw, 0600); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	if code := runUltracodeBench(&out, &errOut, []string{"--scenario", "fast-profile", "--pair", p, "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var r ultracodebench.FastProfileReport
	if err := json.Unmarshal(out.Bytes(), &r); err != nil {
		t.Fatal(err)
	}
	if r.Scenario != "fast-profile" || r.Verdict != "ABSTAIN" {
		t.Fatalf("report=%+v", r)
	}

	// Also verify that a fastintent.ReplayBundle JSON can be passed as --pair to fast-profile.
	replay := fastintent.ReplayBundle{
		Schema:     fastintent.Schema,
		Evaluation: r,
		Verdict:    r.Verdict,
	}
	rawReplay, _ := json.Marshal(replay)
	pReplay := filepath.Join(t.TempDir(), "replay.json")
	if err := os.WriteFile(pReplay, rawReplay, 0600); err != nil {
		t.Fatal(err)
	}
	out.Reset()
	errOut.Reset()
	if code := runUltracodeBench(&out, &errOut, []string{"--scenario", "fast-profile", "--pair", pReplay, "--json"}); code != 0 {
		t.Fatalf("replay pair: code=%d stderr=%s", code, errOut.String())
	}
	var r2 ultracodebench.FastProfileReport
	if err := json.Unmarshal(out.Bytes(), &r2); err != nil {
		t.Fatal(err)
	}
	if r2.Scenario != "fast-profile" || r2.Verdict != "ABSTAIN" {
		t.Fatalf("replay report=%+v", r2)
	}
}
