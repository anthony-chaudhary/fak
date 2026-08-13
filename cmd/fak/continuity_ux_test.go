package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestContinuityUXTaskAliases(t *testing.T) {
	cases := map[string]string{"backup": "export", "restore": "apply", "share": "export", "publish": "export", "recover": "rollback"}
	for in, want := range cases {
		got, _ := continuityCanonicalTask(in)
		if got != want {
			t.Errorf("%s=%s want %s", in, got, want)
		}
	}
	if _, ch := continuityCanonicalTask("share"); ch != "organization" {
		t.Fatalf("share channel=%q", ch)
	}
	if _, ch := continuityCanonicalTask("publish"); ch != "public" {
		t.Fatalf("publish channel=%q", ch)
	}
}

func TestContinuityUXReasonCodesBoundNextActions(t *testing.T) {
	for _, code := range continuityReasonCodes() {
		r := continuityReasons[code]
		if r.Code == "" || r.Meaning == "" || len(r.Next) < 1 || len(r.Next) > 2 {
			t.Fatalf("%s invalid: %#v", code, r)
		}
		var out bytes.Buffer
		if got := runContinuityExplain(&out, []string{"--reason", code, "--json"}); got != 0 {
			t.Fatalf("code=%d", got)
		}
		var decoded continuityReason
		if err := json.Unmarshal(out.Bytes(), &decoded); err != nil {
			t.Fatal(err)
		}
		if decoded.Code != code {
			t.Fatalf("got %#v", decoded)
		}
	}
}

func TestContinuityUXWidthsAreBoundedAndMeaningSurvivesWithoutColor(t *testing.T) {
	for _, width := range []int{40, 80, 120} {
		for _, line := range continuityTaskLines(width) {
			if len(line) > width {
				t.Fatalf("width %d: %q", width, line)
			}
		}
	}
	joined := strings.Join(continuityTaskLines(80), "\n")
	for _, term := range []string{"Object", "Collection", "Context", "Package", "Channel", "Transaction"} {
		if !strings.Contains(joined, term) {
			t.Fatalf("missing %s", term)
		}
	}
}

func TestContinuityReceiptsAreSortedAndMachineReadable(t *testing.T) {
	home := t.TempDir()
	dir := filepath.Join(home, "receipts")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"z.json", "a.json", "ignore.txt"} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("{}"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	var out, stderr bytes.Buffer
	if code := runContinuity(&out, &stderr, []string{"receipts", "--home", home, "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var got []continuityReceiptSummary
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0].ID != "a" || got[1].ID != "z" {
		t.Fatalf("receipts=%#v", got)
	}
}

func TestContinuityUXSelfcheckHumanAndJSON(t *testing.T) {
	var human bytes.Buffer
	if code := runContinuityUXSelfcheck(&human, false); code != 0 {
		t.Fatalf("code=%d", code)
	}
	for _, want := range []string{"PASS explainable continuity UX", "concepts 10 -> 6", "decisions 8 -> 3", "expert controls 8 -> 8", "40 80 120"} {
		if !strings.Contains(human.String(), want) {
			t.Fatalf("missing %q in %s", want, human.String())
		}
	}
	var machine bytes.Buffer
	runContinuityUXSelfcheck(&machine, true)
	var got continuityUXCheck
	if err := json.Unmarshal(machine.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Result != "PASS" || got.Candidate.Concepts >= got.Baseline.Concepts || got.Candidate.Decisions >= got.Baseline.Decisions || got.Candidate.ExpertControls < got.Baseline.ExpertControls {
		t.Fatalf("regression: %#v", got)
	}
}

func TestContinuityHelpProgressivelyDisclosesTaskThenExpertControls(t *testing.T) {
	var out bytes.Buffer
	continuityHelp(&out)
	s := out.String()
	for _, want := range []string{"backup", "restore", "switch", "share", "publish", "status", "preview", "explain", "receipts", "recover", "rollback", "--select", "--channel", "policy/precedence"} {
		if !strings.Contains(s, want) {
			t.Fatalf("missing %q", want)
		}
	}
	if strings.Index(s, "Start with one task") > strings.Index(s, "Foundation/expert tasks") {
		t.Fatal("expert controls shown before ordinary tasks")
	}
}
