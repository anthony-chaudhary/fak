package hooks

import (
	"strings"
	"testing"
)

// gate_swallowederror_test.go — unit tests for the SWALLOWED_ERROR hygiene gate (issue #2899).
// The gate is fak-specific (no Python oracle to diff against), so these tests pin the exact
// verdict on every corner the detector has to get right: the `_ = <call>()` discard IS flagged,
// the `//nolint:errdiscard` opt-out (same-line and line-above) suppresses it, a discard inside a
// string/comment husk is never a phantom match, test files are not graded, excluded dirs are
// skipped, a plain value discard (`_ = x`) and a multi-assign (`x, _ := f()`) are NOT flagged,
// and the per-file cap holds. treeFromFiles is the shared in-memory TrackedTree builder from
// gate_deadcode_test.go.

// swallowedLines returns the sorted "file:line" keys of a gate's findings.
func swallowedLines(findings []Finding) []string {
	out := make([]string, 0, len(findings))
	for _, f := range findings {
		out = append(out, f.File+":"+itoa(int64(f.Line)))
	}
	return out
}

func TestGateSwallowedError_FlagsBlankCallDiscard(t *testing.T) {
	files := map[string]string{
		"internal/x/a.go": "package x\n\nimport \"os\"\n\nfunc run() {\n\t_ = os.Remove(\"/tmp/x\")\n}\n",
	}
	got, err := gateSwallowedErrorTree(treeFromFiles(files))
	if err != nil {
		t.Fatalf("gate error: %v", err)
	}
	keys := swallowedLines(got)
	if strings.Join(keys, "|") != "internal/x/a.go:6" {
		t.Fatalf("blank call discard must be flagged at line 6, got %v", keys)
	}
}

func TestGateSwallowedError_MethodChainDiscardFlagged(t *testing.T) {
	files := map[string]string{
		"internal/x/a.go": "package x\n\nfunc run(c closer) {\n\t_ = c.Body().Close()\n}\n\ntype closer interface{ Body() closer; Close() error }\n",
	}
	got, _ := gateSwallowedErrorTree(treeFromFiles(files))
	if strings.Join(swallowedLines(got), "|") != "internal/x/a.go:4" {
		t.Fatalf("chained call discard must be flagged, got %v", swallowedLines(got))
	}
}

func TestGateSwallowedError_NolintSameLineOptOut(t *testing.T) {
	files := map[string]string{
		"internal/x/a.go": "package x\n\nimport \"os\"\n\nfunc run() {\n\t_ = os.Remove(\"/tmp/x\") //nolint:errdiscard best-effort cleanup\n}\n",
	}
	got, _ := gateSwallowedErrorTree(treeFromFiles(files))
	if len(got) != 0 {
		t.Fatalf("//nolint:errdiscard on the same line must suppress, got %v", swallowedLines(got))
	}
}

func TestGateSwallowedError_NolintLineAboveOptOut(t *testing.T) {
	files := map[string]string{
		"internal/x/a.go": "package x\n\nimport \"os\"\n\nfunc run() {\n\t//nolint:errdiscard best-effort cleanup\n\t_ = os.Remove(\"/tmp/x\")\n}\n",
	}
	got, _ := gateSwallowedErrorTree(treeFromFiles(files))
	if len(got) != 0 {
		t.Fatalf("//nolint:errdiscard on the line above must suppress, got %v", swallowedLines(got))
	}
}

func TestGateSwallowedError_DiscardInStringNotMatched(t *testing.T) {
	// The literal text `_ = f()` appears only inside a string — the code-only projection blanks
	// the string husk, so it is not a phantom discard.
	files := map[string]string{
		"internal/x/a.go": "package x\n\nfunc doc() string { return \"_ = f() is the pattern\" }\n",
	}
	got, _ := gateSwallowedErrorTree(treeFromFiles(files))
	if len(got) != 0 {
		t.Fatalf("a `_ = f()` inside a string must not match, got %v", swallowedLines(got))
	}
}

func TestGateSwallowedError_DiscardInCommentNotMatched(t *testing.T) {
	files := map[string]string{
		"internal/x/a.go": "package x\n\n// _ = f() would swallow the error\nfunc real() {}\n",
	}
	got, _ := gateSwallowedErrorTree(treeFromFiles(files))
	if len(got) != 0 {
		t.Fatalf("a `_ = f()` inside a comment must not match, got %v", swallowedLines(got))
	}
}

func TestGateSwallowedError_PlainValueDiscardNotFlagged(t *testing.T) {
	// `_ = x` (no call) discards a plain value — there is no error to lose, so it is out of scope.
	files := map[string]string{
		"internal/x/a.go": "package x\n\nfunc run(x int) {\n\t_ = x\n}\n",
	}
	got, _ := gateSwallowedErrorTree(treeFromFiles(files))
	if len(got) != 0 {
		t.Fatalf("a plain value discard `_ = x` must not be flagged, got %v", swallowedLines(got))
	}
}

func TestGateSwallowedError_MultiAssignNotFlagged(t *testing.T) {
	// `v, _ := f()` names the error position as a throwaway but USES the value — this is the
	// errcheck family's domain, not the `_ = <call>()` full-discard idiom this gate grades.
	files := map[string]string{
		"internal/x/a.go": "package x\n\nfunc f() (int, error) { return 1, nil }\n\nfunc run() int {\n\tv, _ := f()\n\treturn v\n}\n",
	}
	got, _ := gateSwallowedErrorTree(treeFromFiles(files))
	if len(got) != 0 {
		t.Fatalf("a multi-assign `v, _ := f()` must not be flagged, got %v", swallowedLines(got))
	}
}

func TestGateSwallowedError_TestFilesNotGraded(t *testing.T) {
	files := map[string]string{
		"internal/x/a_test.go": "package x\n\nimport \"os\"\n\nfunc setup() {\n\t_ = os.Remove(\"/tmp/x\")\n}\n",
	}
	got, _ := gateSwallowedErrorTree(treeFromFiles(files))
	if len(got) != 0 {
		t.Fatalf("_test.go must not be graded, got %v", swallowedLines(got))
	}
}

func TestGateSwallowedError_ExcludedDirsNotGraded(t *testing.T) {
	files := map[string]string{
		"internal/x/testdata/fixture.go": "package fixture\n\nimport \"os\"\n\nfunc f() { _ = os.Remove(\"/x\") }\n",
		"vendor/dep/dep.go":              "package dep\n\nimport \"os\"\n\nfunc f() { _ = os.Remove(\"/x\") }\n",
	}
	got, _ := gateSwallowedErrorTree(treeFromFiles(files))
	if len(got) != 0 {
		t.Fatalf("testdata/vendor must be excluded, got %v", swallowedLines(got))
	}
}

func TestGateSwallowedError_PerFileCap(t *testing.T) {
	// swallowedCapPerFile+3 discards in one file; only the cap is reported.
	var b strings.Builder
	b.WriteString("package x\n\nimport \"os\"\n\nfunc run() {\n")
	for i := 0; i < swallowedCapPerFile+3; i++ {
		b.WriteString("\t_ = os.Remove(\"/tmp/x\")\n")
	}
	b.WriteString("}\n")
	files := map[string]string{"internal/x/a.go": b.String()}
	got, _ := gateSwallowedErrorTree(treeFromFiles(files))
	if len(got) != swallowedCapPerFile {
		t.Fatalf("per-file cap: got %d findings, want %d", len(got), swallowedCapPerFile)
	}
}

func TestGateSwallowedError_RegisteredDefaultOff(t *testing.T) {
	// The gate must be registered in HygieneGates() and ship DefaultOff (a ratchet that lands
	// clean against the tree's pre-existing discards, exactly like DEAD_CODE / BARE_DEV_SPELLING).
	var found bool
	for _, g := range HygieneGates() {
		if g.Name == swallowedErrorGate {
			found = true
			if !g.DefaultOff {
				t.Fatalf("%s must ship DefaultOff (a ratchet), got DefaultOff=false", swallowedErrorGate)
			}
		}
	}
	if !found {
		t.Fatalf("%s not registered in HygieneGates()", swallowedErrorGate)
	}
	// And it must be reachable by name via the --gates filter path.
	if HygieneGateByName(swallowedErrorGate) == nil {
		t.Fatalf("%s not resolvable via HygieneGateByName", swallowedErrorGate)
	}
}
