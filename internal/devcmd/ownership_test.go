package devcmd

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/devindex"
)

func TestRunOwnershipJSON(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := RunOwnership(&out, &errOut, devindex.FindRoot("."), true); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var got devindex.OwnershipReport
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != "fak-command-ownership/1" || len(got.Commands) == 0 || got.Graph.PackageCount == 0 {
		t.Fatalf("incomplete report: %+v", got)
	}
}

func TestRunOwnershipText(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := RunOwnership(&out, &errOut, devindex.FindRoot("."), false); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	for _, want := range []string{"command ownership: runtime=", "runtime graph: packages=", "dev-leaks=0"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("output missing %q:\n%s", want, out.String())
		}
	}
}
