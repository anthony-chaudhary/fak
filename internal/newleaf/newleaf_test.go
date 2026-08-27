package newleaf

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const dosTomlFixture = `[lanes]
concurrent = [
  "foo",
  # new-leaf:lane
]
autopick = [
  "foo",
  # new-leaf:lane
]
[lanes.trees]
foo = ["internal/foo/**"]
# new-leaf:tree
cmd = ["cmd/**"]
`

// A line that MENTIONS the marker is not a marker. The real dos.toml carries a
// prose comment inside [lanes].concurrent explaining why some lanes landed by
// hand — and that sentence quotes `# new-leaf:lane`. A substring match treated it
// as a third marker and inserted the generated leaf into `concurrent` TWICE, so
// `fak new-leaf` reliably handed back a tree that failed the roster gate
// (CONCURRENT_LANES_OVERLAP) it had just written.
func TestAddLeafLaneIgnoresProseThatMentionsTheMarker(t *testing.T) {
	fixture := `[lanes]
concurrent = [
  "foo",
  # They landed by hand, so the ` + "`# new-leaf:lane`" + ` auto-insert never fired.
  "bar",
  # new-leaf:lane
]
autopick = [
  "foo",
  # new-leaf:lane
]
[lanes.trees]
foo = ["internal/foo/**"]
# new-leaf:tree
cmd = ["cmd/**"]
`
	out, err := AddLeafLane(fixture, "fedtrust")
	if err != nil {
		t.Fatalf("AddLeafLane: %v", err)
	}
	// Exactly two lane entries: one per REAL marker (concurrent, autopick).
	if got := strings.Count(out, `  "fedtrust",`); got != 2 {
		t.Fatalf("lane inserted %d time(s), want exactly 2 (one per real marker):\n%s", got, out)
	}
	// And the duplicate specifically must not be inside `concurrent`.
	concurrent := out[strings.Index(out, "concurrent = ["):strings.Index(out, "autopick = [")]
	if got := strings.Count(concurrent, `"fedtrust"`); got != 1 {
		t.Fatalf("[lanes].concurrent declares the leaf %d time(s), want exactly 1:\n%s", got, concurrent)
	}
	if !strings.Contains(out, `fedtrust = ["internal/fedtrust/**"]`) {
		t.Errorf("lane tree missing:\n%s", out)
	}
}

func TestInsertBeforeMarkerRequiresTheMarkerToOpenTheLine(t *testing.T) {
	text := "a\n  # see `# new-leaf:tree` for how this works\nb\n# new-leaf:tree\nc\n"
	out, err := InsertBeforeMarker(text, TreeMarker, "INSERTED\n")
	if err != nil {
		t.Fatalf("InsertBeforeMarker: %v", err)
	}
	if strings.Count(out, "INSERTED") != 1 {
		t.Fatalf("inserted %d times, want 1:\n%s", strings.Count(out, "INSERTED"), out)
	}
	if !strings.Contains(out, "b\nINSERTED\n# new-leaf:tree") {
		t.Fatalf("insert did not land above the real marker:\n%s", out)
	}
	if _, err := InsertBeforeMarker("only a mention of # new-leaf:tree mid-line\n", TreeMarker, "x\n"); err == nil {
		t.Error("a mere mention was accepted as a marker; it must report the marker as not found")
	}
}

func TestDocGoCarriesTierAndPackage(t *testing.T) {
	out := DocGo("fedtrust", "composer", 3, "a federated trust gate")
	if !strings.Contains(out, "package fedtrust") {
		t.Fatalf("doc missing package: %s", out)
	}
	if !strings.Contains(out, "Tier: composer (3)") {
		t.Fatalf("doc missing tier: %s", out)
	}
	if !strings.Contains(out, "import only") {
		t.Fatalf("doc missing import contract: %s", out)
	}
}

func TestImplGoRegisterAddsABIAndInit(t *testing.T) {
	reg := ImplGo("fedtrust", true)
	if !strings.Contains(reg, `import "github.com/anthony-chaudhary/fak/internal/abi"`) {
		t.Fatalf("registered impl missing ABI import: %s", reg)
	}
	if !strings.Contains(reg, "func init()") {
		t.Fatalf("registered impl missing init: %s", reg)
	}
	plain := ImplGo("fedtrust", false)
	if strings.Contains(plain, "abi") {
		t.Fatalf("plain impl unexpectedly mentions abi: %s", plain)
	}
	if !strings.Contains(plain, "func Ready() bool") {
		t.Fatalf("plain impl missing Ready: %s", plain)
	}
}

func TestTestGoScaffoldsWorkingSpine(t *testing.T) {
	want := `package fedtrust

import "testing"

// TestSpine drives the generated leaf's real surface end to end. Keep this
// representative path working while the proof envelope expands around it.
func TestSpine(t *testing.T) {
	if !Ready() {
		t.Fatal("generated leaf spine did not reach Ready")
	}
}
`
	got := TestGo("fedtrust")
	if got != want {
		t.Fatalf("generated spine scaffold drifted:\ngot:\n%s\nwant:\n%s", got, want)
	}
	if strings.Contains(got, "TestTODO") {
		t.Fatal("generated spine regressed to a disconnected TestTODO")
	}
}

func TestInsertBeforeMarker(t *testing.T) {
	out, err := InsertBeforeMarker("a\n# MARK\nb\n", "# MARK", "X\n")
	if err != nil {
		t.Fatalf("InsertBeforeMarker returned error: %v", err)
	}
	if out != "a\nX\n# MARK\nb\n" {
		t.Fatalf("out = %q", out)
	}
	if _, err := InsertBeforeMarker("a\nb\n", "# NOPE", "X\n"); err == nil {
		t.Fatal("InsertBeforeMarker missing marker returned nil error")
	}
}

func TestInsertBeforeAllMarkersHitsEveryMarker(t *testing.T) {
	out, err := InsertBeforeAllMarkers("# M\nx\n# M\n", "# M", "Y\n")
	if err != nil {
		t.Fatalf("InsertBeforeAllMarkers returned error: %v", err)
	}
	if strings.Count(out, "Y\n") != 2 {
		t.Fatalf("inserted count = %d, want 2: %q", strings.Count(out, "Y\n"), out)
	}
}

func TestAddRegistrationIdempotent(t *testing.T) {
	reg := "import (\n\t_ \"x/foo\"\n)\n"
	once, err := AddRegistration(reg, "bar")
	if err != nil {
		t.Fatalf("AddRegistration returned error: %v", err)
	}
	if !strings.Contains(once, `_ "github.com/anthony-chaudhary/fak/internal/bar"`) {
		t.Fatalf("registration missing blank import: %s", once)
	}
	twice, err := AddRegistration(once, "bar")
	if err != nil {
		t.Fatalf("AddRegistration second call returned error: %v", err)
	}
	if twice != once {
		t.Fatalf("AddRegistration is not idempotent:\nonce=%s\ntwice=%s", once, twice)
	}
}

func TestAddLeafLaneRealLayoutNotFakPrefix(t *testing.T) {
	out, err := AddLeafLane(dosTomlFixture, "bar")
	if err != nil {
		t.Fatalf("AddLeafLane returned error: %v", err)
	}
	if !strings.Contains(out, `bar = ["internal/bar/**"]`) {
		t.Fatalf("lane tree missing real layout: %s", out)
	}
	if strings.Contains(out, "fak/internal/bar") {
		t.Fatalf("lane tree contains stale fak prefix: %s", out)
	}
	if strings.Count(out, "  \"bar\",\n") != 2 {
		t.Fatalf("lane name count = %d, want 2: %s", strings.Count(out, "  \"bar\",\n"), out)
	}
	twice, err := AddLeafLane(out, "bar")
	if err != nil {
		t.Fatalf("AddLeafLane second call returned error: %v", err)
	}
	if twice != out {
		t.Fatal("AddLeafLane is not idempotent")
	}
}

func TestAddLeafLaneLegacyPrefixTreatedAsPresent(t *testing.T) {
	legacy := dosTomlFixture + `baz = ["fak/internal/baz/**"]` + "\n"
	out, err := AddLeafLane(legacy, "baz")
	if err != nil {
		t.Fatalf("AddLeafLane returned error: %v", err)
	}
	if out != legacy {
		t.Fatalf("legacy lane changed:\n%s", out)
	}
}

func TestTiersAndNameRE(t *testing.T) {
	for tier, want := range map[string]int{"primitive": 1, "foundation-composite": 2, "mechanism": 3, "composer": 4, "integrator": 5} {
		if got := Tiers[tier]; got != want {
			t.Fatalf("%s tier = %d, want %d", tier, got, want)
		}
	}
	if !NameRE.MatchString("fedtrust") {
		t.Fatal("fedtrust should match NameRE")
	}
}

func TestApplyDryRunDoesNotWrite(t *testing.T) {
	root := newLeafWorkspace(t)
	report, err := Apply(Options{Root: root, Name: "fedtrust", Tier: "composer", DryRun: true})
	if err != nil {
		t.Fatalf("Apply dry-run returned error: %v", err)
	}
	if !report.DryRun {
		t.Fatal("report DryRun = false, want true")
	}
	if _, err := os.Stat(filepath.Join(root, "internal", "fedtrust")); !os.IsNotExist(err) {
		t.Fatalf("leaf dir exists after dry-run, err=%v", err)
	}
}

func TestApplyDryRunOrdersSpineThenFanout(t *testing.T) {
	root := newLeafWorkspace(t)
	report, err := Apply(Options{Root: root, Name: "fedtrust", Tier: "primitive", DryRun: true})
	if err != nil {
		t.Fatalf("Apply dry-run returned error: %v", err)
	}
	want := []string{
		"1. name the end-to-end outcome and implement its smallest applied path through the real seam in internal/fedtrust/fedtrust.go",
		"2. go test ./internal/fedtrust ./internal/architest to capture the working spine test or command that drives the real fedtrust object end to end",
		"3. fak-dev issue fanout --title \"<feature>\" --leaf fedtrust --spine <commit|command|doc> --json",
		"4. expand the exhaustive proof envelope around that spine: failure paths, edge cases, platforms, concurrency, and soak",
		"5. measure the end-to-end baseline before optimizing, then keep only a net-true gain",
	}
	if strings.Join(report.NextSteps, "\n") != strings.Join(want, "\n") {
		t.Fatalf("dry-run next_steps drifted:\ngot:  %#v\nwant: %#v", report.NextSteps, want)
	}
}

func TestApplyCreatesLeafAndUpdatesTables(t *testing.T) {
	root := newLeafWorkspace(t)
	report, err := Apply(Options{
		Root:     root,
		Name:     "fedtrust",
		Tier:     "composer",
		Register: true,
		Summary:  "a federated trust gate",
	})
	if err != nil {
		t.Fatalf("Apply returned error: %v", err)
	}
	for _, rel := range []string{
		"internal/fedtrust/doc.go",
		"internal/fedtrust/fedtrust.go",
		"internal/fedtrust/fedtrust_test.go",
	} {
		if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(rel))); err != nil {
			t.Fatalf("%s was not written: %v", rel, err)
		}
	}
	arch := readNewLeafFile(t, root, "internal/architest/architest_test.go")
	if !strings.Contains(arch, `"fedtrust": 4,`) {
		t.Fatalf("architest tier missing fedtrust: %s", arch)
	}
	reg := readNewLeafFile(t, root, "internal/registrations/registrations.go")
	if !strings.Contains(reg, `_ "github.com/anthony-chaudhary/fak/internal/fedtrust"`) {
		t.Fatalf("registration missing fedtrust: %s", reg)
	}
	dos := readNewLeafFile(t, root, "dos.toml")
	if !strings.Contains(dos, `fedtrust = ["internal/fedtrust/**"]`) || strings.Count(dos, "  \"fedtrust\",\n") != 2 {
		t.Fatalf("dos lane missing fedtrust: %s", dos)
	}
	if len(report.Edits) == 0 || len(report.NextSteps) == 0 {
		t.Fatalf("report missing edits/next steps: %+v", report)
	}
}

func TestParseTierTableReadsRows(t *testing.T) {
	arch := "package architest\n\nvar tier = map[string]int{\n\t\"abi\": 0,\n\t\"foo\": 1,\n\t\"engine\": 2,\n\t// new-leaf:tier\n}\n"
	tierOf := ParseTierTable(arch)
	if tierOf["abi"] != 0 || tierOf["foo"] != 1 || tierOf["engine"] != 2 {
		t.Fatalf("parsed tier table = %v", tierOf)
	}
	if len(tierOf) != 3 {
		t.Fatalf("expected 3 rows, got %d: %v", len(tierOf), tierOf)
	}
}

func TestMinTierDistinguishesPrimitiveFromComposite(t *testing.T) {
	tierOf := map[string]int{"abi": 0, "foo": 1, "engine": 3, "recall": 4}
	if got, gov := MinTier(nil, tierOf); got != 1 || gov != "" {
		t.Fatalf("empty deps: level=%d gov=%q", got, gov)
	}
	if got, gov := MinTier([]string{"abi"}, tierOf); got != 1 || gov != "" {
		t.Fatalf("abi deps: level=%d gov=%q", got, gov)
	}
	if got, gov := MinTier([]string{"foo"}, tierOf); got != 2 || gov != "foo" {
		t.Fatalf("primitive dep: level=%d gov=%q", got, gov)
	}
	if got, gov := MinTier([]string{"foo", "engine"}, tierOf); got != 3 || gov != "engine" {
		t.Fatalf("mechanism dep: level=%d gov=%q", got, gov)
	}
}

func TestTierAdvisoryFiresOnMismatch(t *testing.T) {
	tierOf := map[string]int{"abi": 0, "foo": 1, "engine": 3}
	if a := TierAdvisory("primitive", nil, tierOf); a != "" {
		t.Fatalf("minimum declaration should not advise: %q", a)
	}
	over := TierAdvisory("composer", nil, tierOf)
	if !strings.Contains(over, "primitive") || !strings.Contains(over, "you declared") {
		t.Fatalf("over-declared advisory = %q", over)
	}
	under := TierAdvisory("primitive", []string{"engine"}, tierOf)
	if !strings.Contains(under, "tier must be") || !strings.Contains(under, "engine") {
		t.Fatalf("under-declared advisory = %q", under)
	}
}

func TestScanInternalDepsSkipsTestsAndStdlib(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "x.go"),
		[]byte("package x\n\nimport (\n\t\"fmt\"\n\t_ \"github.com/anthony-chaudhary/fak/internal/engine\"\n\t_ \"github.com/anthony-chaudhary/fak/internal/foo/sub\"\n)\n\nvar _ = fmt.Sprint\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "x_test.go"),
		[]byte("package x\n\nimport _ \"github.com/anthony-chaudhary/fak/internal/recall\"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	deps, err := ScanInternalDeps(dir)
	if err != nil {
		t.Fatalf("ScanInternalDeps: %v", err)
	}
	// engine + foo (sub collapses to the leaf); stdlib and _test import excluded.
	if strings.Join(deps, ",") != "engine,foo" {
		t.Fatalf("deps = %v, want [engine foo]", deps)
	}
}

func TestSuggestForExistingLeaf(t *testing.T) {
	root := t.TempDir()
	mustWriteNewLeafFile(t, root, "internal/architest/architest_test.go",
		"package architest\n\nvar tier = map[string]int{\n\t\"abi\": 0,\n\t\"engine\": 3,\n\t// new-leaf:tier\n}\n")
	mustWriteNewLeafFile(t, root, "internal/beta/beta.go",
		"package beta\n\nimport _ \"github.com/anthony-chaudhary/fak/internal/engine\"\n")
	s, err := Suggest(root, "beta", "integrator")
	if err != nil {
		t.Fatalf("Suggest: %v", err)
	}
	if s.SuggestedTier != "mechanism" || s.GoverningDep != "engine" {
		t.Fatalf("suggestion = %+v, want mechanism governed by engine", s)
	}
	if s.Advisory == "" {
		t.Fatalf("over-declared integrator should fire an advisory: %+v", s)
	}
}

func newLeafWorkspace(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	mustWriteNewLeafFile(t, root, "internal/architest/architest_test.go", "package architest\n\nvar tier = map[string]int{\n\t\"abi\": 0,\n\t// new-leaf:tier\n}\n")
	mustWriteNewLeafFile(t, root, "internal/registrations/registrations.go", "package registrations\n\nimport (\n\t_ \"github.com/anthony-chaudhary/fak/internal/abi\"\n)\n")
	mustWriteNewLeafFile(t, root, "dos.toml", dosTomlFixture)
	return root
}

func mustWriteNewLeafFile(t *testing.T, root, rel, body string) {
	t.Helper()
	path := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", filepath.Dir(path), err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("write %s: %v", rel, err)
	}
}

func readNewLeafFile(t *testing.T, root, rel string) string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
	if err != nil {
		t.Fatalf("read %s: %v", rel, err)
	}
	return string(raw)
}
