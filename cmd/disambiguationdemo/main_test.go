package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestSelfcheckCapturedOutput(t *testing.T) {
	var out, errout bytes.Buffer
	if code := run(&out, &errout, []string{"-selfcheck"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errout.String())
	}
	got := out.String()
	for _, want := range []string{"QUERY runtime: ambiguous", "SCOPE runtime=gateway-serving -> runtime", "CONTRAST ", "no model, key, GPU, network, or private data"} {
		if !strings.Contains(got, want) {
			t.Fatalf("missing %q in %s", want, got)
		}
	}
}
func TestSelfcheckJSON(t *testing.T) {
	var out, errout bytes.Buffer
	if code := run(&out, &errout, []string{"-selfcheck", "-json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errout.String())
	}
	var r receipt
	if err := json.Unmarshal(out.Bytes(), &r); err != nil {
		t.Fatal(err)
	}
	if !r.Offline || r.SearchVerdict != "ambiguous" || len(r.Choices) != 5 || r.CanonicalTerm != "runtime" {
		t.Fatalf("receipt=%#v", r)
	}
}
