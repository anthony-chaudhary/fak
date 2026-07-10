package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

// footprint_heldaccuracy_test.go pins the #3533 CLI: the JSON contract, the held-accuracy
// pair, the gate-holds verdict + zero exit, and the honesty label (no live-accuracy claim).

func TestFootprintHeldAccuracyJSONContract(t *testing.T) {
	var out, errw bytes.Buffer
	if code := runFootprintHeldAccuracy(&out, &errw, true); code != 0 {
		t.Fatalf("gate should hold (exit 0), got exit=%d stderr=%s\n%s", code, errw.String(), out.String())
	}
	var got map[string]any
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("--held-accuracy --json is not valid JSON: %v\n%s", err, out.String())
	}
	if got["schema"] != "fak-footprint-held-accuracy/1" {
		t.Fatalf("schema=%v, want fak-footprint-held-accuracy/1", got["schema"])
	}
	if got["mode"] != "deterministic-faultin-sim" {
		t.Fatalf("mode=%v, want deterministic-faultin-sim", got["mode"])
	}
	if got["live_accuracy_claim_allowed"] != false {
		t.Fatalf("live_accuracy_claim_allowed=%v, want false (that number is #3536)", got["live_accuracy_claim_allowed"])
	}
	if got["gate_holds"] != true {
		t.Fatalf("gate_holds=%v, want true", got["gate_holds"])
	}
	armed, _ := got["armed_pass"].(float64)
	ablated, _ := got["ablated_pass"].(float64)
	total, _ := got["total"].(float64)
	if total < 3 {
		t.Fatalf("total=%v, want >=3 tasks", total)
	}
	if armed < ablated {
		t.Fatalf("armed %v < ablated %v — held-accuracy regression", armed, ablated)
	}
	if armed != total {
		t.Fatalf("armed %v != total %v — a cold tool was lost under deferral", armed, total)
	}
}

func TestFootprintHeldAccuracyHumanRowAndVerb(t *testing.T) {
	var out, errw bytes.Buffer
	if code := runFootprintHeldAccuracy(&out, &errw, false); code != 0 {
		t.Fatalf("exit=%d", code)
	}
	s := out.String()
	for _, want := range []string{"held-accuracy", "gate HOLDS", "deterministic-faultin-sim", "not a live-model"} {
		if !strings.Contains(s, want) {
			t.Fatalf("human row missing %q:\n%s", want, s)
		}
	}
	// routes through the `fak footprint` verb.
	var vout, verr bytes.Buffer
	if code := runMCPFootprint(&vout, &verr, []string{"--held-accuracy", "--json"}); code != 0 {
		t.Fatalf("fak footprint --held-accuracy --json exit=%d: %s", code, verr.String())
	}
	if !strings.Contains(vout.String(), "fak-footprint-held-accuracy/1") {
		t.Fatalf("verb did not route to the held-accuracy eval:\n%s", vout.String())
	}
}
