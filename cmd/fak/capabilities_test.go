package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/capindex"
)

func TestRunCapabilitiesExposesTurnControl(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := runCapabilities(&out, &errOut, []string{"turn control", "--limit", "3"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	got := out.String()
	for _, want := range []string{"Control a live session out of band", "Avoid unnecessary model turns"} {
		if !strings.Contains(got, want) {
			t.Errorf("output missing %q:\n%s", want, got)
		}
	}
	if strings.Index(got, "supporting capability floor") >= 0 {
		t.Errorf("turn-control query unexpectedly led with security floor:\n%s", got)
	}
}

func TestRunCapabilitiesJSONCarriesWitnessAndCommand(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := runCapabilities(&out, &errOut, []string{"--json", "token savings", "--limit", "2"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var got struct {
		Query    string                    `json:"query"`
		Outcomes []capindex.ProductOutcome `json:"outcomes"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Query != "token savings" || len(got.Outcomes) != 2 {
		t.Fatalf("response=%#v", got)
	}
	for _, outcome := range got.Outcomes {
		if len(outcome.Command) == 0 || outcome.Witness == "" || outcome.Detail == "" {
			t.Errorf("ungrounded outcome: %#v", outcome)
		}
	}
}

func TestRunCapabilitiesIndexesOnDemandFleetCommitCheck(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := runCapabilities(&out, &errOut, []string{"--json", "commits per 10 minutes", "--limit", "1"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var got struct {
		Outcomes []capindex.ProductOutcome `json:"outcomes"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Outcomes) != 1 || got.Outcomes[0].ID != "fleet-commit-health" {
		t.Fatalf("response=%#v", got)
	}
	if command := strings.Join(got.Outcomes[0].Command, " "); command != "fak fleet health --json" {
		t.Fatalf("command=%q", command)
	}
}
