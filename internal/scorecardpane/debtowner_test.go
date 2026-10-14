package scorecardpane

// debtowner_test.go — the #1974 registry gates: every folded debt key has an
// owner, every named lane is a real dos.toml declaration, and the folded
// control-pane JSON actually carries owner_leaf/lane/pathspec for a debt > 0.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestDebtOwnerRegistryCoversEveryDebtKey is the #1974 coverage gate: every debt
// key the control pane folds (Cards) or catalogs (DebtCategories) MUST resolve to
// an owner. A missing row is the exact regression the issue targets — a debt key
// with no owning leaf/lane/pathspec is un-dispatchable.
func TestDebtOwnerRegistryCoversEveryDebtKey(t *testing.T) {
	for _, c := range Cards {
		if _, ok := OwnerForDebt(c.Debt); !ok {
			t.Errorf("debt key %q (card %q) has no DebtOwners row — a regressed row "+
				"would fold with no owning leaf/lane/pathspec (#1974)", c.Debt, c.Key)
		}
	}
	for _, c := range DebtCategories {
		if _, ok := OwnerForDebt(c.DebtKey); !ok {
			t.Errorf("debt key %q (category %q) has no DebtOwners row (#1974)", c.DebtKey, c.Key)
		}
	}
}

// TestDebtOwnerRowsAreComplete proves a registered row names a real destination:
// a non-empty leaf, lane, and pathspec. A half-filled row is a silent hole.
func TestDebtOwnerRowsAreComplete(t *testing.T) {
	for debt, o := range DebtOwners {
		if strings.TrimSpace(o.Leaf) == "" {
			t.Errorf("debt key %q owner has empty leaf", debt)
		}
		if strings.TrimSpace(o.Lane) == "" {
			t.Errorf("debt key %q owner has empty lane", debt)
		}
		if strings.TrimSpace(o.Pathspec) == "" {
			t.Errorf("debt key %q owner has empty pathspec", debt)
		}
	}
}

// TestDebtOwnerLanesAreDeclared proves every lane a row names is a real dos.toml
// [lanes] roster entry. The registry cannot bind a phantom lane (the same failure
// class lane_coverage_test.go guards for leaves): if a lane is renamed or retired,
// this reds and the row must be reconciled.
func TestDebtOwnerLanesAreDeclared(t *testing.T) {
	roster := readControlpaneLaneRoster(t)
	for debt, o := range DebtOwners {
		if !roster[o.Lane] {
			t.Errorf("debt key %q names lane %q which no dos.toml [lanes] roster entry declares",
				debt, o.Lane)
		}
	}
}

// TestDebtOwnerLaneMatchesLeafForGoBackedCards pins the Go-backed convention: a
// card whose customer payload comes from internal/<leaf> names that same leaf as
// its lane (the internal package is its own lane root), so the ownership row can
// never drift from where the detector actually lives.
func TestDebtOwnerLaneMatchesLeafForGoBackedCards(t *testing.T) {
	for _, c := range Cards {
		if !goBackedKey(c.Key) {
			continue
		}
		o, ok := OwnerForDebt(c.Debt)
		if !ok {
			continue // coverage gate owns the missing-row report
		}
		// A Go-backed card may legitimately be owned by the coarse "cmd" lane when
		// its detector has no dedicated internal leaf; every other lane must equal
		// the leaf (one lane per internal leaf).
		if o.Lane == "cmd" {
			continue
		}
		if o.Leaf != o.Lane {
			t.Errorf("Go-backed card %q (debt %q): owner leaf %q != lane %q — an "+
				"internal/<leaf> detector is owned by its own lane", c.Key, c.Debt, o.Leaf, o.Lane)
		}
	}
}

// TestMetricFromPayloadStampsOwnerIntoJSON is the #1974 done-condition witness: a
// metric with debt > 0 folds owner_leaf and lane into the machine-readable
// control-pane JSON (the same shape the folded Payload carries).
func TestMetricFromPayloadStampsOwnerIntoJSON(t *testing.T) {
	card := Card{Key: "flow", Debt: "flow_debt", Label: "flow-metrics",
		Cmd: "go run ./cmd/fak score flow --json"}
	m := MetricFromPayload(card, map[string]any{
		"schema": "flow-scorecard/1",
		"corpus": map[string]any{"flow_debt": float64(3), "grade": "C"},
	}, "")
	if m.Debt == nil || *m.Debt != 3 {
		t.Fatalf("flow metric debt = %v, want 3", m.Debt)
	}
	if m.OwnerLeaf == "" || m.OwnerLane == "" || m.OwnerPathspec == "" {
		t.Fatalf("flow metric missing owner: leaf=%q lane=%q pathspec=%q",
			m.OwnerLeaf, m.OwnerLane, m.OwnerPathspec)
	}
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal metric: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatalf("unmarshal metric: %v", err)
	}
	if got["owner_leaf"] != "flowmetrics" || got["lane"] != "flowmetrics" {
		t.Fatalf("folded JSON owner_leaf/lane = %v/%v, want flowmetrics/flowmetrics\n%s",
			got["owner_leaf"], got["lane"], b)
	}
}

// TestMetricOwnerOmitsForUnregisteredKey anchors the omit-empty contract: an
// unregistered debt key must NOT fabricate an owner — the fields stay absent.
func TestMetricOwnerOmitsForUnregisteredKey(t *testing.T) {
	m := MetricFromPayload(Card{Key: "unknown", Debt: "totally_unknown_debt", Label: "unknown"},
		map[string]any{"totally_unknown_debt": float64(1)}, "")
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(b), "owner_leaf") || strings.Contains(string(b), `"lane"`) {
		t.Fatalf("unregistered debt key fabricated an owner: %s", b)
	}
}

// readControlpaneLaneRoster parses the [lanes] concurrent + exclusive roster names
// out of dos.toml. It is a deliberately tiny line scan (the repo carries no TOML
// dependency), matching lane_coverage_test.go's own reader.
func readControlpaneLaneRoster(t *testing.T) map[string]bool {
	t.Helper()
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatalf("resolve repo root: %v", err)
	}
	src, err := os.ReadFile(filepath.Join(root, "dos.toml"))
	if err != nil {
		t.Fatalf("read dos.toml: %v", err)
	}
	names := map[string]bool{}
	inLanes := false
	for _, line := range strings.Split(string(src), "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "[lanes]") {
			inLanes = true
			continue
		}
		if strings.HasPrefix(trimmed, "[") && inLanes {
			break // left the [lanes] table
		}
		if !inLanes {
			continue
		}
		for _, tok := range quotedStrings(trimmed) {
			names[tok] = true
		}
	}
	if len(names) < 30 {
		t.Fatalf("parsed only %d lane names from dos.toml — roster scan drifted", len(names))
	}
	return names
}

// quotedStrings returns every "..." token on a line.
func quotedStrings(line string) []string {
	var out []string
	rest := line
	for {
		i := strings.IndexByte(rest, '"')
		if i < 0 {
			return out
		}
		rest = rest[i+1:]
		j := strings.IndexByte(rest, '"')
		if j < 0 {
			return out
		}
		out = append(out, rest[:j])
		rest = rest[j+1:]
	}
}
