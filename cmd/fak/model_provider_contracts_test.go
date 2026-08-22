package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

func TestRunModelProviderContractsJSON(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runModelProviderContracts(&stdout, &stderr, []string{"--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var contracts []modelroute.ProviderContract
	if err := json.Unmarshal(stdout.Bytes(), &contracts); err != nil {
		t.Fatalf("invalid JSON: %v\n%s", err, stdout.String())
	}
	if len(contracts) != 2 || contracts[0].Provider != "anthropic" || contracts[1].Provider != "openai" {
		t.Fatalf("contracts=%+v", contracts)
	}
}

func TestRunModelProviderContractsInspect(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runModelProviderContracts(&stdout, &stderr, nil); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	for _, want := range []string{"PROVIDER", "anthropic", "openai"} {
		if !strings.Contains(stdout.String(), want) {
			t.Errorf("inspect output missing %q:\n%s", want, stdout.String())
		}
	}
}
