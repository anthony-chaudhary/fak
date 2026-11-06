package grafanacontract

import (
	"os"
	"strings"
	"testing"
)

// routerAlertContract pins the fak_router alert group added for the router
// Prometheus KPI ranking (2026-11-03, "Next steps" item 1). The public repo
// owns the rule file; the metric semantics live in the companion private repo,
// so this test is the seam: deleting or renaming any router-plane alert fails
// here instead of silently disarming the routing availability tier. It reuses
// the ops contract's scanner and spec shape on purpose -- one parser, one
// pinning idiom, per plane.
var routerAlertContract = []opsAlertSpec{
	{
		Group: "fak_router",
		Alerts: []string{
			"RouterAllBackendsFailing",
			"RouterBackendConsecutiveFailures",
			"RouterJournalDegraded",
			"RouterAdmissionRejecting",
			"RouterDurableStoreHigh",
		},
	},
}

// routerExprContract pins the exact PromQL of the four T1 KPI alerts. Each
// expression encodes a misread trap from the KPI ranking: outcome="all_backends_failed"
// (a closed outcome label, not a status filter); the backend-fault state gauge
// (state, not a counter rate, and caller-fault classes excluded by
// construction); the 1h increase window on the durability meter; and the
// always-emitted admission counter summed over kinds. A threshold or label
// change is a semantic change and must fail this pin.
var routerExprContract = map[string]string{
	"RouterAllBackendsFailing":         `sum(rate(fak_router_requests_total{outcome="all_backends_failed"}[5m])) > 0`,
	"RouterBackendConsecutiveFailures": `max(fak_router_backend_consecutive_failures) >= 3`,
	"RouterJournalDegraded":            `increase(fak_router_journal_degraded_total[1h]) > 0`,
	"RouterAdmissionRejecting":         `sum(rate(fak_router_admission_rejected_total[5m])) > 0`,
}

// TestRouterAlertGroupExists pins the fak_router group and every alert in it,
// mirroring TestOpsAlertGroupsExist for the routing plane.
func TestRouterAlertGroupExists(t *testing.T) {
	raw, err := os.ReadFile(prometheusAlertsPath)
	if err != nil {
		t.Fatalf("read %s: %v", prometheusAlertsPath, err)
	}
	groups := ParseAlertGroups(string(raw))

	for _, spec := range routerAlertContract {
		g, ok := groupByName(groups, spec.Group)
		if !ok {
			t.Errorf("alert group %q not found in %s (was it deleted or renamed?)", spec.Group, prometheusAlertsPath)
			continue
		}
		got := map[string]bool{}
		for _, a := range g.Alerts {
			got[a.Name] = true
		}
		for _, want := range spec.Alerts {
			if !got[want] {
				t.Errorf("alert %q missing from group %q", want, spec.Group)
			}
		}
	}
}

// TestRouterAlertRulesAreWellFormed asserts every alert in the fak_router group
// carries the fields a rule needs to actually fire and to be readable when it
// does: a non-empty expr, a severity in the closed {info,warning,critical}
// set, and non-empty summary/description annotations.
func TestRouterAlertRulesAreWellFormed(t *testing.T) {
	raw, err := os.ReadFile(prometheusAlertsPath)
	if err != nil {
		t.Fatalf("read %s: %v", prometheusAlertsPath, err)
	}
	groups := ParseAlertGroups(string(raw))

	validSeverity := map[string]bool{"info": true, "warning": true, "critical": true}
	for _, spec := range routerAlertContract {
		g, ok := groupByName(groups, spec.Group)
		if !ok {
			t.Errorf("alert group %q not found", spec.Group)
			continue
		}
		for _, a := range g.Alerts {
			if strings.TrimSpace(a.Expr) == "" {
				t.Errorf("%s/%s: empty expr", spec.Group, a.Name)
			}
			if !validSeverity[a.Severity] {
				t.Errorf("%s/%s: severity %q not in {info,warning,critical}", spec.Group, a.Name, a.Severity)
			}
			if strings.TrimSpace(a.Summary) == "" {
				t.Errorf("%s/%s: empty annotations.summary", spec.Group, a.Name)
			}
			if strings.TrimSpace(a.Description) == "" {
				t.Errorf("%s/%s: empty annotations.description", spec.Group, a.Name)
			}
		}
	}
}

// TestRouterAlertExprsMatchKPIs pins the exact expr of each T1 KPI alert, so a
// silent threshold or label drift (e.g. loosening the consecutive-failures
// floor, or summing caller-fault failure classes into a backend-fault alert)
// fails the build instead of quietly changing what the alert means.
func TestRouterAlertExprsMatchKPIs(t *testing.T) {
	raw, err := os.ReadFile(prometheusAlertsPath)
	if err != nil {
		t.Fatalf("read %s: %v", prometheusAlertsPath, err)
	}
	groups := ParseAlertGroups(string(raw))

	for _, spec := range routerAlertContract {
		g, ok := groupByName(groups, spec.Group)
		if !ok {
			t.Errorf("alert group %q not found", spec.Group)
			continue
		}
		for _, a := range g.Alerts {
			want, pinned := routerExprContract[a.Name]
			if !pinned {
				continue
			}
			if a.Expr != want {
				t.Errorf("%s expr = %q, want pinned KPI expression %q", a.Name, a.Expr, want)
			}
		}
	}
}

// TestRouterAlertNegativeControl proves the pin catches regressions: the real
// rule file with the group renamed reads as ABSENT, and the same file with the
// consecutive-failures threshold drifted stops matching its pinned expr.
func TestRouterAlertNegativeControl(t *testing.T) {
	raw, err := os.ReadFile(prometheusAlertsPath)
	if err != nil {
		t.Fatalf("read %s: %v", prometheusAlertsPath, err)
	}
	text := string(raw)

	renamed := strings.Replace(text, "- name: fak_router", "- name: fak_router_renamed", 1)
	if renamed == text {
		t.Fatal("fixture rename did not apply; - name: fak_router line moved")
	}
	if _, ok := groupByName(ParseAlertGroups(renamed), "fak_router"); ok {
		t.Error("renamed fak_router group still found by the parser")
	}

	const pinnedThreshold = "max(fak_router_backend_consecutive_failures) >= 3"
	drifted := strings.Replace(text, pinnedThreshold, "max(fak_router_backend_consecutive_failures) >= 7", 1)
	if drifted == text {
		t.Fatal("fixture threshold drift did not apply; pinned expr line moved")
	}
	groups := ParseAlertGroups(drifted)
	g, ok := groupByName(groups, "fak_router")
	if !ok {
		t.Fatal("fak_router group lost after threshold drift")
	}
	matched := 0
	for _, a := range g.Alerts {
		if want, pinned := routerExprContract[a.Name]; pinned && a.Expr == want {
			matched++
		}
	}
	if matched != len(routerExprContract)-1 {
		t.Errorf("drifted fixture: %d/%d pinned exprs still match, want %d (only the drifted alert may stop matching)",
			matched, len(routerExprContract), len(routerExprContract)-1)
	}
}
