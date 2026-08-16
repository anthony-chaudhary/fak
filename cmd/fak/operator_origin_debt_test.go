package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/operatorbrief"
)

func TestOperatorBriefDebtWitnessesCommandInput(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "debt.json")
	payload := map[string]any{"records": []operatorbrief.DebtWitnessRecord{
		{Source: "task", ID: "T-1", Debt: operatorbrief.DebtCaughtAtOrigin, Detail: "fixture drift", Evidence: "gate receipt"},
		{Source: "session", ID: "S-2", Debt: operatorbrief.DebtFoundLate, Detail: "missing origin check", Evidence: "audit receipt"},
	}}
	data, _ := json.Marshal(payload)
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errOut bytes.Buffer
	code := runOperatorBrief(&out, &errOut, []string{"--workspace", dir, "--debt-witnesses", path, "--json"})
	if code != 0 && code != 1 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var got operatorbrief.Report
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("decode: %v\n%s", err, out.String())
	}
	if len(got.OriginDebt) != 1 || got.OriginDebt[0].ID != "T-1" {
		t.Fatalf("origin debt = %+v", got.OriginDebt)
	}
	if len(got.LateFoundDebt) != 1 || got.LateFoundDebt[0].ID != "S-2" {
		t.Fatalf("late debt = %+v", got.LateFoundDebt)
	}
	rendered := operatorbrief.Render(got)
	if !strings.Contains(rendered, "origin_debt:") || !strings.Contains(rendered, "late_found_debt:") {
		t.Fatalf("render missing split sections:\n%s", rendered)
	}
}

func TestOperatorBriefDebtWitnessesRejectMalformedInput(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.json")
	if err := os.WriteFile(path, []byte("{"), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	if code := runOperatorBrief(&out, &errOut, []string{"--debt-witnesses", path}); code != 2 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
}
