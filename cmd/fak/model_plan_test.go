package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelloadplan"
)

func TestRunModelPlanJSONCapturesHarnessGoalAndSelection(t *testing.T) {
	var out, errOut bytes.Buffer
	code := runModelPlan(&out, &errOut, []string{"qwen38:27b", "--setup", "shared", "--goal", "quality", "--local", "require", "--device-gib", "27", "--disk-gib", "27", "--json"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var p modelloadplan.Plan
	if err := json.Unmarshal(out.Bytes(), &p); err != nil {
		t.Fatal(err)
	}
	if p.Schema != modelloadplan.Schema || p.Request.Setup != "shared" || p.Selected == nil || p.Selected.Quantization != "Q5_K_M" {
		t.Fatalf("plan = %#v", p)
	}
}

func TestRunModelPlanTextMakesHostedFallbackRunnable(t *testing.T) {
	var out, errOut bytes.Buffer
	code := runModelPlan(&out, &errOut, []string{"--setup", "shared"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	for _, want := range []string{"selected: hosted/openrouter", "qwen/qwen3.8-27b", "OPENROUTER_API_KEY"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("output missing %q:\n%s", want, out.String())
		}
	}
}
