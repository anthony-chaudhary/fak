package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestSelfcheckProvesTrustBoundary(t *testing.T) {
	var out, errw bytes.Buffer
	if code := run(&out, &errw, []string{"-selfcheck"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errw.String())
	}
	for _, want := range []string{"PASS fak-extension-seams/1", "custom linters isolated", "improvements witness-gated", "in-process code trusted-compiled"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("output %q missing %q", out.String(), want)
		}
	}
}

func TestJSONCatalogCarriesActionableContracts(t *testing.T) {
	var out, errw bytes.Buffer
	if code := run(&out, &errw, []string{"-json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errw.String())
	}
	var got report
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if err := validate(got); err != nil {
		t.Fatal(err)
	}
	byName := map[string]seam{}
	for _, s := range got.Seams {
		byName[s.Name] = s
	}
	if got := byName["agent-hook"]; got.Attachment != "out-of-process" || got.Trust != "untrusted" {
		t.Fatalf("agent-hook = %#v", got)
	}
	if got := byName["improvement-proposal"]; !strings.Contains(got.Failure, "no witness means no keep") {
		t.Fatalf("improvement-proposal = %#v", got)
	}
}

func TestValidateRefusesUntrustedInProcessCode(t *testing.T) {
	r := buildReport()
	r.Seams[0].Attachment = "in-process"
	r.Seams[0].Trust = "untrusted"
	if err := validate(r); err == nil || !strings.Contains(err.Error(), "not marked trusted-compiled") {
		t.Fatalf("validate error = %v", err)
	}
}
