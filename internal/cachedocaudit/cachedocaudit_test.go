package cachedocaudit

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

const (
	stem   = "widget"
	docRel = "docs/widget.md"
)

func goodTriple() (string, Manifest, map[string]any) {
	doc := "# Widget\n\n" +
		"| metric | value | provenance | field |\n" +
		"|---|---|---|---|\n" +
		"| Fires | 100 fires over 40 sessions | WITNESSED | `m.fires` |\n" +
		"| Cache read | 1.23M | OBSERVED | `m.big` |\n" +
		"| Avoided | **$60.00 (50.0% reduction)** | OBSERVED | `m.avoided` |\n" +
		"| Sessions | 10 discovered — 3 priced, 7 held out (5 synthetic, 2 other) | WITNESSED | `m.disc` |\n"

	snapshot := map[string]any{
		"m": map[string]any{
			"fires":     100,
			"sessions":  40,
			"big":       1234567,
			"counter":   120.0,
			"actual":    60.0,
			"avoided":   60.0,
			"reduction": 50.0,
			"disc":      10,
			"priced":    3,
			"syn":       5,
		},
	}

	staleDays := 30
	formulaTol := 0.001
	formulaExpect := 60.0
	totalSessions := 10.0

	manifest := Manifest{
		Schema:         Schema,
		Doc:            docRel,
		SnapshotDir:    filepath.ToSlash(filepath.Join("tools", "docnumbers", "snapshots", stem)),
		SnapshotDate:   time.Now().UTC().Format("2006-01-02"),
		StaleAfterDays: &staleDays,
		Sources: map[string]Source{
			"main.json": {
				Cmd:    nil,
				Window: "live_snapshot",
			},
		},
		Claims: []Claim{
			{
				ID:         "fires",
				AppearsAs:  "100 fires over 40 sessions",
				Source:     "main.json",
				Provenance: "WITNESSED",
				Numbers: []ClaimNumber{
					{Display: "100", Field: "m.fires", Expected: 100},
					{Display: "40", Field: "m.sessions", Expected: 40},
				},
			},
			{
				ID:         "big",
				AppearsAs:  "1.23M",
				Source:     "main.json",
				Provenance: "OBSERVED",
				Numbers: []ClaimNumber{
					{Display: "1.23M", Field: "m.big", Expected: 1234567},
				},
			},
			{
				ID:         "avoided",
				AppearsAs:  "**$60.00 (50.0% reduction)**",
				Source:     "main.json",
				Provenance: "OBSERVED",
				Numbers: []ClaimNumber{
					{Display: "$60.00", Field: "m.avoided", Expected: 60.0},
					{Display: "50.0%", Field: "m.reduction", Expected: 50.0},
				},
			},
			{
				ID:         "sessions",
				AppearsAs:  "10 discovered — 3 priced, 7 held out (5 synthetic, 2 other)",
				Source:     "main.json",
				Provenance: "WITNESSED",
				Numbers: []ClaimNumber{
					{Display: "10", Field: "m.disc", Expected: 10},
					{Display: "3", Field: "m.priced", Expected: 3},
					{Display: "5", Field: "m.syn", Expected: 5},
				},
			},
		},
		Invariants: []Invariant{
			{
				Kind:  "sum",
				Label: "sessions",
				Total: &totalSessions,
				Parts: []float64{3, 5, 2},
			},
			{
				Kind:   "formula",
				Label:  "avoided",
				Expr:   "120.0 - 60.0",
				Expect: &formulaExpect,
				Tol:    &formulaTol,
			},
		},
	}

	return doc, manifest, snapshot
}

func runTestTriple(t *testing.T, doc string, manifest Manifest, snapshot map[string]any) (int, []Finding, []Finding) {
	t.Helper()
	tempDir := t.TempDir()

	docPath := filepath.Join(tempDir, manifest.Doc)
	if err := os.MkdirAll(filepath.Dir(docPath), 0755); err != nil {
		t.Fatalf("failed to create doc dir: %v", err)
	}
	if err := os.WriteFile(docPath, []byte(doc), 0644); err != nil {
		t.Fatalf("failed to write doc: %v", err)
	}

	snapDir := filepath.Join(tempDir, manifest.SnapshotDir)
	if err := os.MkdirAll(snapDir, 0755); err != nil {
		t.Fatalf("failed to create snap dir: %v", err)
	}
	snapBytes, err := json.Marshal(snapshot)
	if err != nil {
		t.Fatalf("failed to marshal snapshot: %v", err)
	}
	if err := os.WriteFile(filepath.Join(snapDir, "main.json"), snapBytes, 0644); err != nil {
		t.Fatalf("failed to write snap: %v", err)
	}

	manDir := filepath.Join(tempDir, "tools", "docnumbers")
	if err := os.MkdirAll(manDir, 0755); err != nil {
		t.Fatalf("failed to create man dir: %v", err)
	}
	manBytes, err := json.Marshal(manifest)
	if err != nil {
		t.Fatalf("failed to marshal manifest: %v", err)
	}
	manifestPath := filepath.Join(manDir, stem+".json")
	if err := os.WriteFile(manifestPath, manBytes, 0644); err != nil {
		t.Fatalf("failed to write manifest: %v", err)
	}

	var stdout, stderr bytes.Buffer
	code := Run(&stdout, &stderr, AuditOptions{
		Root:  tempDir,
		Today: time.Now().UTC(),
	})

	loaded, err := LoadManifests(tempDir, "")
	if err != nil || len(loaded) == 0 {
		t.Fatalf("failed to load manifest back: %v", err)
	}
	fails, warns := AuditManifest(tempDir, loaded[0], time.Now().UTC())

	return code, fails, warns
}

func checkSet(findings []Finding) map[string]bool {
	set := make(map[string]bool)
	for _, f := range findings {
		set[f.Check] = true
	}
	return set
}

func TestGoodPasses(t *testing.T) {
	doc, man, snap := goodTriple()
	code, fails, _ := runTestTriple(t, doc, man, snap)
	if code != 0 {
		t.Fatalf("clean triple must PASS with exit code 0, got %d (fails: %v)", code, fails)
	}
	if len(fails) != 0 {
		t.Fatalf("expected 0 fails, got %v", fails)
	}
}

func TestBindingFail(t *testing.T) {
	doc, man, snap := goodTriple()
	doc = strings.Replace(doc, "100 fires over 40 sessions", "lots of fires", 1)
	code, fails, _ := runTestTriple(t, doc, man, snap)
	if code != 1 {
		t.Fatalf("expected code 1, got %d", code)
	}
	if !checkSet(fails)["binding"] {
		t.Fatalf("expected binding failure, got %v", fails)
	}
}

func TestSnapshotFail(t *testing.T) {
	doc, man, snap := goodTriple()
	snap["m"].(map[string]any)["fires"] = 999
	code, fails, _ := runTestTriple(t, doc, man, snap)
	if code != 1 {
		t.Fatalf("expected code 1, got %d", code)
	}
	if !checkSet(fails)["snapshot"] {
		t.Fatalf("expected snapshot failure, got %v", fails)
	}
}

func TestRoundingFail(t *testing.T) {
	doc, man, snap := goodTriple()
	doc = strings.Replace(doc, "$60.00", "$61.00", 1)
	for i := range man.Claims {
		if man.Claims[i].ID == "avoided" {
			man.Claims[i].AppearsAs = strings.Replace(man.Claims[i].AppearsAs, "$60.00", "$61.00", 1)
			man.Claims[i].Numbers[0].Display = "$61.00"
		}
	}
	code, fails, _ := runTestTriple(t, doc, man, snap)
	if code != 1 {
		t.Fatalf("expected code 1, got %d", code)
	}
	if !checkSet(fails)["rounding"] {
		t.Fatalf("expected rounding failure, got %v", fails)
	}
}

func TestSumInvariantFail(t *testing.T) {
	doc, man, snap := goodTriple()
	for i := range man.Invariants {
		if man.Invariants[i].Kind == "sum" {
			man.Invariants[i].Parts = []float64{3, 5}
		}
	}
	code, fails, _ := runTestTriple(t, doc, man, snap)
	if code != 1 {
		t.Fatalf("expected code 1, got %d", code)
	}
	if !checkSet(fails)["invariants"] {
		t.Fatalf("expected invariants failure, got %v", fails)
	}
}

func TestFormulaInvariantFail(t *testing.T) {
	doc, man, snap := goodTriple()
	wrongExpect := 99.0
	for i := range man.Invariants {
		if man.Invariants[i].Kind == "formula" {
			man.Invariants[i].Expect = &wrongExpect
		}
	}
	code, fails, _ := runTestTriple(t, doc, man, snap)
	if code != 1 {
		t.Fatalf("expected code 1, got %d", code)
	}
	if !checkSet(fails)["invariants"] {
		t.Fatalf("expected invariants failure, got %v", fails)
	}
}

func TestProvenanceMissingWarnsNotFails(t *testing.T) {
	doc, man, snap := goodTriple()
	doc = strings.Replace(doc, "| 100 fires over 40 sessions | WITNESSED |", "| 100 fires over 40 sessions | |", 1)
	code, fails, warns := runTestTriple(t, doc, man, snap)
	if code != 0 {
		t.Fatalf("a missing provenance label is a WARN, not a FAIL; got code %d, fails %v", code, fails)
	}
	if !checkSet(warns)["provenance"] {
		t.Fatalf("expected provenance warning, got %v", warns)
	}
}

func TestStalenessWarnsNotFails(t *testing.T) {
	doc, man, snap := goodTriple()
	man.SnapshotDate = "2000-01-01"
	staleDays := 7
	man.StaleAfterDays = &staleDays
	code, fails, warns := runTestTriple(t, doc, man, snap)
	if code != 0 {
		t.Fatalf("a stale snapshot is a WARN, not a FAIL; got code %d, fails %v", code, fails)
	}
	if !checkSet(warns)["staleness"] {
		t.Fatalf("expected staleness warning, got %v", warns)
	}
}

func TestApproxRenderOk(t *testing.T) {
	doc, man, snap := goodTriple()
	doc = strings.Replace(doc, "| Fires |", "| About | ≈$60 saved | WITNESSED | `m.avoided` |\n| Fires |", 1)
	man.Claims = append(man.Claims, Claim{
		ID:         "approx",
		AppearsAs:  "≈$60 saved",
		Source:     "main.json",
		Provenance: "WITNESSED",
		Numbers: []ClaimNumber{
			{Display: "≈$60", Field: "m.avoided", Expected: 60.0},
		},
	})
	code, fails, _ := runTestTriple(t, doc, man, snap)
	if code != 0 {
		t.Fatalf("approx render must PASS; code=%d, fails=%v", code, fails)
	}
}

func TestParseDisplayVariants(t *testing.T) {
	p1, err := ParseDisplay("62.6M")
	if err != nil || math.Abs(p1.Value-62.6e6) > 1e-3 {
		t.Fatalf("62.6M: got %v, err: %v", p1, err)
	}

	p2, err := ParseDisplay("$1,181.37")
	if err != nil || math.Abs(p2.Value-1181.37) > 1e-3 {
		t.Fatalf("$1,181.37: got %v, err: %v", p2, err)
	}

	p3, err := ParseDisplay("89.3%")
	if err != nil || math.Abs(p3.Value-89.3) > 1e-3 {
		t.Fatalf("89.3%%: got %v, err: %v", p3, err)
	}

	p4, err := ParseDisplay("≈34,000")
	if err != nil || !p4.Approx {
		t.Fatalf("≈34,000: expected Approx true, got %v, err: %v", p4, err)
	}

	p5, err := ParseDisplay("2,347")
	if err != nil || p5.Approx {
		t.Fatalf("2,347: expected Approx false, got %v, err: %v", p5, err)
	}
}

func TestRoundingBoundaries(t *testing.T) {
	if ok, _ := RoundingOK("$18.64", 18.635071500000002); !ok {
		t.Errorf("expected $18.64 rounding to pass")
	}
	if ok, _ := RoundingOK("89.3%", 89.34669014110735); !ok {
		t.Errorf("expected 89.3%% rounding to pass")
	}
	if ok, _ := RoundingOK("$18.70", 18.635071500000002); ok {
		t.Errorf("expected $18.70 rounding to fail")
	}
	if ok, _ := RoundingOK("≈20%", 20.307425929069925); !ok {
		t.Errorf("expected ≈20%% approx rounding to pass")
	}
}

func TestSafeEvalRejectsNonArithmetic(t *testing.T) {
	_, err := SafeEval("__import__('os').system('echo hi')")
	if err == nil {
		t.Fatalf("expected SafeEval to reject dangerous input")
	}
}

func TestSafeEvalCalculations(t *testing.T) {
	v1, err := SafeEval("120.0 - 60.0")
	if err != nil || math.Abs(v1-60.0) > 1e-6 {
		t.Fatalf("120.0 - 60.0: got %v, err %v", v1, err)
	}

	v2, err := SafeEval("100 * (1181.3689650000001 - 375.5363760000001) / 1181.3689650000001")
	if err != nil || math.Abs(v2-68.2117621906548) > 1e-4 {
		t.Fatalf("formula calculation: got %v, err %v", v2, err)
	}
}

func TestRealRepoManifest(t *testing.T) {
	// Discover repo root
	root := filepath.Join("..", "..")
	manifestPath := filepath.Join(root, "tools", "docnumbers", "fable5-more-usage-for-free.json")
	if _, err := os.Stat(manifestPath); os.IsNotExist(err) {
		t.Skip("repository manifest not found; skipping real repo test")
	}

	manifests, err := LoadManifests(root, "fable5-more-usage-for-free.json")
	if err != nil || len(manifests) == 0 {
		t.Fatalf("failed to load real repo manifest: %v", err)
	}

	fixedToday, _ := time.Parse("2006-01-02", "2026-07-15")
	fails, warns := AuditManifest(root, manifests[0], fixedToday)
	if len(fails) > 0 {
		t.Fatalf("real repo manifest failed audit: %v", fails)
	}
	_ = warns
}

func TestCLIJSONOutput(t *testing.T) {
	doc, man, snap := goodTriple()
	tempDir := t.TempDir()

	docPath := filepath.Join(tempDir, man.Doc)
	_ = os.MkdirAll(filepath.Dir(docPath), 0755)
	_ = os.WriteFile(docPath, []byte(doc), 0644)

	snapDir := filepath.Join(tempDir, man.SnapshotDir)
	_ = os.MkdirAll(snapDir, 0755)
	snapBytes, _ := json.Marshal(snap)
	_ = os.WriteFile(filepath.Join(snapDir, "main.json"), snapBytes, 0644)

	manDir := filepath.Join(tempDir, "tools", "docnumbers")
	_ = os.MkdirAll(manDir, 0755)
	manBytes, _ := json.Marshal(man)
	_ = os.WriteFile(filepath.Join(manDir, stem+".json"), manBytes, 0644)

	var stdout, stderr bytes.Buffer
	code := Run(&stdout, &stderr, AuditOptions{
		Root:   tempDir,
		AsJSON: true,
		Today:  time.Now().UTC(),
	})
	if code != 0 {
		t.Fatalf("expected code 0, got %d", code)
	}

	var res AuditResult
	if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
		t.Fatalf("failed to unmarshal JSON output: %v", err)
	}
	if !res.OK || len(res.Fails) != 0 {
		t.Fatalf("expected clean JSON audit result, got %+v", res)
	}
}

func TestLoadManifestsFilter(t *testing.T) {
	_, man, snap := goodTriple()
	tempDir := t.TempDir()

	snapDir := filepath.Join(tempDir, man.SnapshotDir)
	_ = os.MkdirAll(snapDir, 0755)
	snapBytes, _ := json.Marshal(snap)
	_ = os.WriteFile(filepath.Join(snapDir, "main.json"), snapBytes, 0644)

	manDir := filepath.Join(tempDir, "tools", "docnumbers")
	_ = os.MkdirAll(manDir, 0755)
	manBytes, _ := json.Marshal(man)
	_ = os.WriteFile(filepath.Join(manDir, "widget.json"), manBytes, 0644)
	_ = os.WriteFile(filepath.Join(manDir, "other.json"), manBytes, 0644)

	loadedAll, err := LoadManifests(tempDir, "")
	if err != nil || len(loadedAll) != 2 {
		t.Fatalf("expected 2 manifests, got %d (err: %v)", len(loadedAll), err)
	}

	loadedWidget, err := LoadManifests(tempDir, "widget.json")
	if err != nil || len(loadedWidget) != 1 {
		t.Fatalf("expected 1 manifest filtered by widget.json, got %d", len(loadedWidget))
	}

	loadedNone, err := LoadManifests(tempDir, "nonexistent.json")
	if err != nil || len(loadedNone) != 0 {
		t.Fatalf("expected 0 manifests for nonexistent filter, got %d", len(loadedNone))
	}
}

func TestNoManifestsFound(t *testing.T) {
	tempDir := t.TempDir()
	var stdout, stderr bytes.Buffer
	code := Run(&stdout, &stderr, AuditOptions{Root: tempDir})
	if code != 2 {
		t.Fatalf("expected exit code 2 when no manifests found, got %d", code)
	}
}

func TestNestFields(t *testing.T) {
	src := map[string]any{
		"a": map[string]any{
			"b": 10,
			"c": map[string]any{
				"d": "nested",
			},
		},
		"x": 42,
	}

	nested := nestFields([]string{"a.b", "a.c.d"}, src)
	aMap, ok := nested["a"].(map[string]any)
	if !ok {
		t.Fatalf("expected nested 'a' to be a map, got %T", nested["a"])
	}
	if aMap["b"] != 10 {
		t.Fatalf("expected a.b == 10, got %v", aMap["b"])
	}
	cMap, ok := aMap["c"].(map[string]any)
	if !ok {
		t.Fatalf("expected a.c to be a map, got %T", aMap["c"])
	}
	if cMap["d"] != "nested" {
		t.Fatalf("expected a.c.d == 'nested', got %v", cMap["d"])
	}
	if _, exists := nested["x"]; exists {
		t.Fatalf("field 'x' should not be present in nested result")
	}
}
