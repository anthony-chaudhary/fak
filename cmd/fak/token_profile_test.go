package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/tokenprofile"
)

func TestTokenProfileHaloCapturedOutput(t *testing.T) {
	var out, errOut bytes.Buffer
	if rc := runTokenProfile(&out, &errOut, []string{"--halo"}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errOut.String())
	}
	for _, want := range []string{"HALO:", "input.uncached", "input.cached", "output.reserved", "DOMINANCE cost=input.uncached load=input.cached", "SHIFT LEFT:"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("missing %q in:\n%s", want, out.String())
		}
	}
}

func TestTokenProfileJSONContract(t *testing.T) {
	var out, errOut bytes.Buffer
	if rc := runTokenProfile(&out, &errOut, []string{"--halo", "--json"}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errOut.String())
	}
	var got tokenprofile.Report
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != tokenprofile.Schema || got.DominantLoadClass != tokenprofile.InputCached {
		t.Fatalf("report=%+v", got)
	}
}
