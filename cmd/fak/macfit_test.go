package main

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/macfit"
)

func TestRunMacFitJSONWorkedExample(t *testing.T) {
	var out, errOut bytes.Buffer
	if rc := runMacFit(&out, &errOut, []string{"--json"}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errOut.String())
	}
	var got macfit.Result
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("JSON: %v\n%s", err, out.String())
	}
	if got.Schema != "fak-macfit/1" || got.OffAgentsThatFit != 14 || got.OnAgentsThatFit != 57 || got.ExtraAgents != 43 {
		t.Fatalf("unexpected result: %+v", got)
	}
}
