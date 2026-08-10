package devcmd

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/selfquery"
)

func TestRunFeatureUsesInjectedDevCatalog(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := RunFeature(&out, &errOut, []string{"query", "gateway", "--root", devindexRoot(), "--plane", "dev", "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var resp selfquery.Response
	if err := json.Unmarshal(out.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if len(resp.Cards) == 0 {
		t.Fatalf("no dev cards: %s", out.String())
	}
	found := false
	for _, c := range resp.Cards {
		if len(c.Request.Command) > 0 && c.Request.Command[0] == "fak-dev" {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("dev cards do not route through fak-dev: %+v", resp.Cards)
	}
}

func TestRunCapabilitiesAdvertisesFakDevIndex(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := RunCapabilities(&out, &errOut, []string{"index", "--root", devindexRoot(), "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	if !strings.Contains(out.String(), "fak-dev") || strings.Contains(out.String(), `["fak","index"`) {
		t.Fatalf("wrong capability route: %s", out.String())
	}
}
