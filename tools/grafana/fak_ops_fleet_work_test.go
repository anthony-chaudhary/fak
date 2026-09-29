package grafana

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// fakOpsFleetWorkPath is the hand-authored Fleet Work board (uid
// fak-ops-fleet-work) rendered by the fak_ops exporter scrape job. No generator
// owns this artifact, so this test is its contract.
const fakOpsFleetWorkPath = "dashboards/fak-ops-fleet-work.json"

// loadFakOpsFleetWork parses the board relative to this test file.
func loadFakOpsFleetWork(t *testing.T) fakOpsDashboard {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	path := filepath.Join(filepath.Dir(file), filepath.FromSlash(fakOpsFleetWorkPath))
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var d fakOpsDashboard
	if err := json.Unmarshal(raw, &d); err != nil {
		t.Fatalf("%s is not well-formed dashboard JSON: %v", path, err)
	}
	if d.UID != "fak-ops-fleet-work" {
		t.Fatalf("expected uid fak-ops-fleet-work, got %q", d.UID)
	}
	return d
}

// TestFleetWorkBoard_QueriesBothTicketPopulations pins that the board surfaces
// BOTH ticket populations the exporter folds. There are two independent
// populations -- the native git-ref store and the git-tracked docs/tickets
// corpus -- and a board that shows one without the other invites an operator to
// sum them into a backlog that does not exist. The native store is further split
// by KIND (mirror vs native-authored), because a native-authored record
// correctly has no upstream and must not read as an "unclassified" failure.
func TestFleetWorkBoard_QueriesBothTicketPopulations(t *testing.T) {
	d := loadFakOpsFleetWork(t)

	var exprs []string
	for _, p := range d.Panels {
		for _, tgt := range p.Targets {
			if e := strings.TrimSpace(tgt.Expr); e != "" {
				exprs = append(exprs, e)
			}
		}
	}
	if len(exprs) == 0 {
		t.Fatalf("%s queries nothing", fakOpsFleetWorkPath)
	}

	for _, family := range []string{
		"fak_ops_tickets_total",
		"fak_ops_tickets_by_kind",
		"fak_ops_tickets_docs_total",
	} {
		found := false
		for _, e := range exprs {
			if strings.Contains(e, family) {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("%s queries no panel for %q; both ticket populations (native store + docs corpus) and the native KIND split must be visible", fakOpsFleetWorkPath, family)
		}
	}
}

// TestFleetWorkBoard_NoStaleVisibilityOnlyLabels pins that no panel still
// describes the ticket row as a plain public/private split. The native store's
// `unknown` visibility bucket is dominated by native-authored tickets (which
// have no forge repository), so a bare "Unclassified tickets" stat misreports a
// normal kind as a failure. The kind axis is the corrective.
func TestFleetWorkBoard_NoStaleVisibilityOnlyLabels(t *testing.T) {
	d := loadFakOpsFleetWork(t)
	for _, p := range d.Panels {
		switch p.Title {
		case "Unclassified tickets", "Public tickets", "Private tickets", "Tickets total", "Tickets present":
			t.Errorf("%s still carries the pre-kind ticket label %q; it must name the population/kind it counts", fakOpsFleetWorkPath, p.Title)
		}
	}
}
