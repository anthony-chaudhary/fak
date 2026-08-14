package portabilitylab

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestRunIsHermeticCompleteAndFailClosed(t *testing.T) {
	root := t.TempDir()
	r, err := Run(root)
	if err != nil {
		t.Fatal(err)
	}
	if r.Verdict != "pass" || !r.Hermetic || r.ActiveState != "none" {
		t.Fatalf("bad gate: %+v", r)
	}
	if r.CredentialsUsed || r.HostedServicesUsed {
		t.Fatal("lab used external authority")
	}
	seen := map[string]Status{}
	for _, w := range r.Coverage {
		seen[w.ID] = w.Status
		if w.Status != Proven {
			t.Fatalf("%s=%s", w.ID, w.Status)
		}
		if len(w.API) == 0 {
			t.Fatalf("%s lacks API", w.ID)
		}
	}
	for _, id := range requiredIDs() {
		if seen[id] != Proven {
			t.Fatalf("missing %s", id)
		}
	}
	if len(r.FailureArtifacts) < 3 {
		t.Fatalf("failure artifacts=%v", r.FailureArtifacts)
	}
	for _, p := range r.FailureArtifacts {
		if _, e := os.ReadFile(filepath.Join(root, p)); e != nil {
			t.Fatalf("artifact %s: %v", p, e)
		}
	} // paths are additionally exercised by Run
}

func TestCoverageMatrixHasFourTypedStates(t *testing.T) {
	vals := []Status{Proven, Partial, Unsupported, Untested}
	b, err := json.Marshal(vals)
	if err != nil {
		t.Fatal(err)
	}
	if string(b) != `["proven","partial","unsupported","untested"]` {
		t.Fatalf("states=%s", b)
	}
}

func TestDigestStableShape(t *testing.T) {
	r, err := Run(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Digest) != 71 || r.Digest[:7] != "sha256:" {
		t.Fatalf("digest=%q", r.Digest)
	}
}
