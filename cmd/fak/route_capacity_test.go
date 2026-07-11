package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRouteJSONExposesCapacityReroute(t *testing.T) {
	p := filepath.Join(t.TempDir(), "targets.json")
	if err := os.WriteFile(p, []byte(`[{"name":"fleet-gpu","model":"70b","pool":"fleet-gpu","model_b":70,"available":true}]`), 0644); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	code := runRoute(&out, &errOut, []string{"--json", "--capacity-reason", "MODEL_CAPACITY_CEILING", "--capacity-from", "local-7b", "--required-model-b", "14", "--local-model-b", "7", "--capacity-targets", p})
	if code != 0 {
		t.Fatalf("code=%d err=%s", code, errOut.String())
	}
	for _, want := range []string{`"capacity_reroute"`, `"rerouted": true`, `"from": "local-7b"`, `"name": "fleet-gpu"`} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("missing %s in %s", want, out.String())
		}
	}
}
func TestRouteJSONReportsNoEligibleCapacityTarget(t *testing.T) {
	var out, errOut bytes.Buffer
	code := runRoute(&out, &errOut, []string{"--json", "--capacity-reason", "MODEL_CONTEXT_WINDOW_CEILING", "--capacity-from", "local", "--required-context", "200000"})
	if code != 0 {
		t.Fatalf("code=%d err=%s", code, errOut.String())
	}
	if !strings.Contains(out.String(), `"rerouted": false`) || !strings.Contains(out.String(), `MODEL_CONTEXT_WINDOW_CEILING`) {
		t.Fatalf("out=%s", out.String())
	}
}
