package benchcatalog

import (
	"strings"
	"testing"
)

func TestVCacheBenchmarkIsDiscoverableOfflineGate(t *testing.T) {
	b, ok := Get("vcache")
	if !ok {
		t.Fatal("vcache benchmark missing from catalog")
	}
	if b.Kind != KindVerb || b.Need != NeedNone {
		t.Fatalf("vcache kind/need = %s/%s, want verb/offline", b.Kind, b.Need)
	}
	if !strings.Contains(b.Run, "fak vcache bench --json") {
		t.Fatalf("vcache run = %q, want vcache bench JSON gate", b.Run)
	}
	for _, want := range []string{"--telemetry", "--anchors-file", "--index-out", "--plan-out", "--two-x"} {
		if !containsFlag(b.Flags, want) {
			t.Fatalf("vcache flags = %v, missing %s", b.Flags, want)
		}
	}
	if !b.Offline() {
		t.Fatal("vcache scorecard must remain zero-asset/offline by default")
	}
}

func TestAblateBenchmarkIsDiscoverableOfflineGate(t *testing.T) {
	b, ok := Get("ablate")
	if !ok {
		t.Fatal("ablate benchmark missing from catalog (it is a `fak` bench verb, epic #607, but had no registry row)")
	}
	if b.Kind != KindVerb || b.Need != NeedNone {
		t.Fatalf("ablate kind/need = %s/%s, want verb/offline", b.Kind, b.Need)
	}
	if !strings.Contains(b.Run, "fak ablate") {
		t.Fatalf("ablate run = %q, want a `fak ablate` invocation", b.Run)
	}
	if !b.Offline() {
		t.Fatal("ablate runs on the offline mock engine by default; it must stay zero-asset/offline")
	}
	if b.Doc == "" {
		t.Fatal("ablate must point at its methodology doc (docs/benchmarks/ABLATE-RESULTS.md)")
	}
}

func TestToolSandboxBenchmarkIsDiscoverableOfflineGate(t *testing.T) {
	b, ok := Get("toolsandboxbench")
	if !ok {
		t.Fatal("toolsandboxbench benchmark missing from catalog (tracked cmd/*bench* mains must have registry rows)")
	}
	if b.Kind != KindCmd || b.Need != NeedNone {
		t.Fatalf("toolsandboxbench kind/need = %s/%s, want cmd/offline", b.Kind, b.Need)
	}
	if !strings.Contains(b.Run, "go run ./cmd/toolsandboxbench") {
		t.Fatalf("toolsandboxbench run = %q, want cmd invocation", b.Run)
	}
	for _, want := range []string{"-suite", "-contract", "-out", "-md"} {
		if !containsFlag(b.Flags, want) {
			t.Fatalf("toolsandboxbench flags = %v, missing %s", b.Flags, want)
		}
	}
	if !b.Offline() {
		t.Fatal("toolsandboxbench uses a committed local smoke suite; it must stay zero-asset/offline")
	}
	if b.Doc != "BENCHMARK-AUTHORITY.md" {
		t.Fatalf("toolsandboxbench doc = %q, want BENCHMARK-AUTHORITY.md", b.Doc)
	}
}

func TestAgenticBenchRollupIsDiscoverableOfflineGate(t *testing.T) {
	b, ok := Get("agenticbench")
	if !ok {
		t.Fatal("agenticbench benchmark missing from catalog")
	}
	if b.Kind != KindCmd || b.Need != NeedNone {
		t.Fatalf("agenticbench kind/need = %s/%s, want cmd/offline", b.Kind, b.Need)
	}
	if !strings.Contains(b.Run, "go run ./cmd/agenticbench") {
		t.Fatalf("agenticbench run = %q, want cmd invocation", b.Run)
	}
	for _, want := range []string{"-root", "-out", "-md", "-strict"} {
		if !containsFlag(b.Flags, want) {
			t.Fatalf("agenticbench flags = %v, missing %s", b.Flags, want)
		}
	}
	if !b.Offline() {
		t.Fatal("agenticbench only reads committed artifacts; it must stay zero-asset/offline")
	}
}

func TestAgentBenchDemoIsDiscoverableOfflineGate(t *testing.T) {
	b, ok := Get("agentbenchdemo")
	if !ok {
		t.Fatal("agentbenchdemo benchmark missing from catalog (tracked cmd/*bench* mains must have registry rows)")
	}
	if b.Kind != KindCmd || b.Need != NeedNone {
		t.Fatalf("agentbenchdemo kind/need = %s/%s, want cmd/offline", b.Kind, b.Need)
	}
	if !strings.Contains(b.Run, "go run ./cmd/agentbenchdemo") {
		t.Fatalf("agentbenchdemo run = %q, want cmd invocation", b.Run)
	}
	for _, want := range []string{"-n", "-json", "-selfcheck"} {
		if !containsFlag(b.Flags, want) {
			t.Fatalf("agentbenchdemo flags = %v, missing %s", b.Flags, want)
		}
	}
	if !b.Offline() {
		t.Fatal("agentbenchdemo uses a deterministic local tool-call plan; it must stay zero-asset/offline")
	}
	if b.Doc != "cmd/agentbenchdemo/README.md" {
		t.Fatalf("agentbenchdemo doc = %q, want cmd/agentbenchdemo/README.md", b.Doc)
	}
}

func TestBrowserActionBenchmarkIsDiscoverableOfflineGate(t *testing.T) {
	b, ok := Get("browseractionbench")
	if !ok {
		t.Fatal("browseractionbench benchmark missing from catalog (tracked cmd/*bench* mains must have registry rows)")
	}
	if b.Kind != KindCmd || b.Need != NeedNone {
		t.Fatalf("browseractionbench kind/need = %s/%s, want cmd/offline", b.Kind, b.Need)
	}
	if !strings.Contains(b.Run, "go run ./cmd/browseractionbench") {
		t.Fatalf("browseractionbench run = %q, want cmd invocation", b.Run)
	}
	for _, want := range []string{"-suite", "-contract", "-out", "-md"} {
		if !containsFlag(b.Flags, want) {
			t.Fatalf("browseractionbench flags = %v, missing %s", b.Flags, want)
		}
	}
	if !b.Offline() {
		t.Fatal("browseractionbench uses a committed local smoke suite; it must stay zero-asset/offline")
	}
	if !strings.Contains(b.Doc, "AGENTIC-BENCHMARK-RUN-PACKETS") {
		t.Fatalf("browseractionbench doc = %q, want run-packet methodology note", b.Doc)
	}
}

func TestTerminalBenchBenchmarkIsDiscoverableOfflineGate(t *testing.T) {
	b, ok := Get("terminalbench")
	if !ok {
		t.Fatal("terminalbench benchmark missing from catalog (tracked cmd/*bench* mains must have registry rows)")
	}
	if b.Kind != KindCmd || b.Need != NeedNone {
		t.Fatalf("terminalbench kind/need = %s/%s, want cmd/offline", b.Kind, b.Need)
	}
	if !strings.Contains(b.Run, "go run ./cmd/terminalbench") {
		t.Fatalf("terminalbench run = %q, want cmd invocation", b.Run)
	}
	for _, want := range []string{"-suite", "-contract", "-out", "-md"} {
		if !containsFlag(b.Flags, want) {
			t.Fatalf("terminalbench flags = %v, missing %s", b.Flags, want)
		}
	}
	if !b.Offline() {
		t.Fatal("terminalbench uses a committed local smoke suite; it must stay zero-asset/offline")
	}
	if !strings.Contains(b.Doc, "AGENTIC-BENCHMARK-RUN-PACKETS") {
		t.Fatalf("terminalbench doc = %q, want run-packet methodology note", b.Doc)
	}
}

func containsFlag(flags []string, want string) bool {
	for _, flag := range flags {
		if strings.Contains(flag, want) {
			return true
		}
	}
	return false
}

// TestAllSortedAndUnique guards the registry's two structural invariants the
// `fak benchmarks` verb and the bench-DX scorecard both rely on: All() is sorted
// by Name, and every Name is unique.
func TestAllSortedAndUnique(t *testing.T) {
	all := All()
	if len(all) == 0 {
		t.Fatal("registry is empty")
	}
	seen := map[string]bool{}
	for i, b := range all {
		if b.Name == "" {
			t.Errorf("entry %d has empty Name", i)
		}
		if seen[b.Name] {
			t.Errorf("duplicate Name %q", b.Name)
		}
		seen[b.Name] = true
		if i > 0 && all[i-1].Name > b.Name {
			t.Errorf("not sorted: %q before %q", all[i-1].Name, b.Name)
		}
	}
}

// TestEveryEntryHasSummaryAndRun checks the two fields the developer reads first:
// every benchmark must say what it measures (Summary) and how to run it (Run),
// and carry a known Need/Kind.
func TestEveryEntryHasSummaryAndRun(t *testing.T) {
	for _, b := range All() {
		if b.Summary == "" {
			t.Errorf("%s: empty Summary (a number with no inline meaning)", b.Name)
		}
		if b.Run == "" {
			t.Errorf("%s: empty Run (no copy-pasteable command)", b.Name)
		}
		switch b.Need {
		case NeedNone, NeedWeights, NeedDataset:
		default:
			t.Errorf("%s: unknown Need %q", b.Name, b.Need)
		}
		switch b.Kind {
		case KindCmd, KindVerb:
		default:
			t.Errorf("%s: unknown Kind %q", b.Name, b.Kind)
		}
	}
}

// TestOfflineSubsetOfAll confirms Offline() returns exactly the NeedNone entries
// and that Get round-trips every registered name.
func TestOfflineSubsetOfAll(t *testing.T) {
	for _, b := range Offline() {
		if b.Need != NeedNone {
			t.Errorf("Offline() returned %s with Need=%q", b.Name, b.Need)
		}
	}
	for _, name := range Names() {
		if _, ok := Get(name); !ok {
			t.Errorf("Get(%q) missing for a registered name", name)
		}
	}
	if _, ok := Get("definitely-not-a-bench"); ok {
		t.Error("Get returned ok for an unregistered name")
	}
}
