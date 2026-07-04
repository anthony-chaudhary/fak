package microagent_test

import (
	"go/parser"
	"go/token"
	"io/fs"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/modelengine"

	// The microagent-minimal set UNDER TEST. This external test package blank-imports
	// ONLY this package, so the abi registries the runtime tests below walk reflect the
	// minimal set in isolation — nothing pulls in the full defconfig. That isolation is
	// what makes "the floor survives under minimal-only" a real witness and not an
	// artifact of some other import wiring the full stack.
	_ "github.com/anthony-chaudhary/fak/internal/registrations/microagent"
)

const modInternal = "github.com/anthony-chaudhary/fak/internal/"

// capabilityFloor is the closed set of leaves the microagent-minimal set MUST keep so
// no adjudication rung is silently dropped: the Ref resolver, the vDSO fast path, the
// pre-flight rung ladder, the DOS reference monitor (with its POLICY_BLOCK floor), the
// write-time result-admission chain (ctxmmu + normgate + ifc), the stewards, and the
// engine seam. Dropping any one from microagent.go reds TestMicroagentMinimalKeepsFloor.
var capabilityFloor = []string{
	"blob", "vdso",
	"grammar", "preflight", "ratelimit",
	"adjudicator", "ctxmmu", "normgate", "ifc", "steward",
	"engine", "modelengine",
}

// leafImports returns the set of internal/<leaf> short-names blank- or named-imported by
// the non-test source in dir (collapsing a sub-package to its first path segment), the
// same derivation architest uses for the request-path closure.
func leafImports(t *testing.T, dir string) map[string]bool {
	t.Helper()
	fset := token.NewFileSet()
	parsed, err := parser.ParseDir(fset, dir, func(fi fs.FileInfo) bool {
		return !strings.HasSuffix(fi.Name(), "_test.go")
	}, parser.ImportsOnly)
	if err != nil {
		t.Fatalf("parse %s: %v", dir, err)
	}
	leaves := map[string]bool{}
	for _, p := range parsed {
		for _, f := range p.Files {
			for _, spec := range f.Imports {
				path := strings.Trim(spec.Path.Value, `"`)
				if !strings.HasPrefix(path, modInternal) {
					continue
				}
				leaves[strings.SplitN(strings.TrimPrefix(path, modInternal), "/", 2)[0]] = true
			}
		}
	}
	return leaves
}

// pkgDirs locates the microagent-minimal dir and the full-defconfig dir from this test
// file's own compiled path (runtime.Caller), so the test is independent of the working
// directory the runner (WSL / CI) launches it from.
func pkgDirs(t *testing.T) (minimal, full string) {
	t.Helper()
	_, self, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed — cannot locate the microagent package dir")
	}
	minimal = filepath.Dir(self)          // internal/registrations/microagent
	full = filepath.Dir(minimal)          // internal/registrations
	return minimal, full
}

// TestMicroagentMinimalIsStrictSubsetOfFullDefconfig is the structural half of issue
// #2009's acceptance item 1 ("a strictly smaller registration set"). It parses the real
// blank-import lines of both packages — not a hand-maintained list that could drift from
// what is actually wired — and asserts every leaf the minimal set imports is also in the
// full defconfig, that the minimal set is non-empty, and that it is strictly smaller. A
// leaf added to microagent.go that the full defconfig does not carry, or a minimal set
// that grew to equal/exceed the full set, reds here.
func TestMicroagentMinimalIsStrictSubsetOfFullDefconfig(t *testing.T) {
	minDir, fullDir := pkgDirs(t)
	minimal := leafImports(t, minDir)
	full := leafImports(t, fullDir)

	if len(minimal) == 0 {
		t.Fatal("microagent-minimal set imports no internal leaf — an empty floor cannot host a microagent")
	}
	for leaf := range minimal {
		if !full[leaf] {
			t.Errorf("microagent-minimal imports internal/%s, which the full defconfig "+
				"(internal/registrations) does not — the minimal set must be a strict SUBSET "+
				"of the full defconfig, never a superset or a divergent list", leaf)
		}
	}
	if len(minimal) >= len(full) {
		t.Errorf("microagent-minimal links %d internal leaves but the full defconfig links %d — "+
			"the minimal set must be STRICTLY smaller (issue #2009 acceptance 1)", len(minimal), len(full))
	} else {
		t.Logf("microagent-minimal links %d internal leaves vs %d in the full defconfig (strictly smaller)", len(minimal), len(full))
	}
}

// TestMicroagentMinimalKeepsFloor is the structural half of acceptance item 2 ("no
// policy/adjudicator floor is lost"): every capability-floor leaf must be present in the
// minimal set's import list. Dropping one from microagent.go reds here, naming the rung.
func TestMicroagentMinimalKeepsFloor(t *testing.T) {
	minDir, _ := pkgDirs(t)
	minimal := leafImports(t, minDir)
	for _, leaf := range capabilityFloor {
		if !minimal[leaf] {
			t.Errorf("microagent-minimal dropped the capability-floor leaf internal/%s — "+
				"the adjudication floor would ship dark on the microagent host (issue #2009)", leaf)
		}
	}
}

// TestMicroagentMinimalFloorIsLiveAtRuntime is the runtime half of acceptance item 2. It
// walks the abi registries linked by the minimal set IN ISOLATION (this test package
// blank-imports only microagent) and proves the adjudication floor is actually enrolled,
// not merely imported: a live Ref resolver, the DOS reference monitor rung, a non-empty
// result-admission chain, and the engine seam. This is the "no adjudication rung silently
// dropped" witness the issue asks internal/architest to guarantee, asserted at the
// defconfig boundary the same way registrations_test.go asserts it for the full set.
func TestMicroagentMinimalFloorIsLiveAtRuntime(t *testing.T) {
	if abi.ActiveResolver() == nil {
		t.Fatal("microagent-minimal linked no Ref resolver — internal/blob is missing from the set; the kernel cannot resolve a single Ref")
	}

	monitorPresent := false
	for _, a := range abi.Adjudicators() {
		if a == adjudicator.Default {
			monitorPresent = true
			break
		}
	}
	if !monitorPresent {
		t.Errorf("microagent-minimal did not enroll the DOS reference monitor (adjudicator.Default) in abi.Adjudicators() — the policy/adjudication floor is lost (issue #2009 acceptance 2)")
	}

	if len(abi.ResultAdmitters()) == 0 {
		t.Error("microagent-minimal enrolled no write-time result admitter (ctxmmu/normgate/ifc) — the quarantine floor is lost")
	}

	ids := map[string]bool{}
	for _, id := range abi.EngineIDs() {
		ids[id] = true
	}
	if !ids["mock"] {
		t.Errorf("microagent-minimal did not enroll the mock engine — the offline engine seam is missing; have: %v", abi.EngineIDs())
	}
	if !ids[modelengine.EngineID] {
		t.Errorf("microagent-minimal did not enroll the %q engine — the in-kernel model seam is missing; have: %v", modelengine.EngineID, abi.EngineIDs())
	}
}

// TestMicroagentMinimalDropsFullKernelOnlyStewards is the runtime mirror of the strict-
// subset claim: the full defconfig enrolls the AgentDojo ASR steward (pinned by
// registrations_test.go TestDefconfigEnrollsAgentDojoSteward), but that leaf is a
// full-kernel-only gate — it is NOT in the microagent floor. Under the minimal set in
// isolation it must be ABSENT, proving the smaller link set is real at runtime, not just
// a smaller import list masking the same enrolled population.
func TestMicroagentMinimalDropsFullKernelOnlyStewards(t *testing.T) {
	const fullOnly = "agentdojo-asr-zero"
	for _, s := range abi.Stewards() {
		if s.Name() == fullOnly {
			t.Fatalf("microagent-minimal enrolled %q, a full-kernel-only steward the minimal floor should not carry — "+
				"the set is not actually smaller at runtime (a floor leaf must be transitively pulling agentdojo in)", fullOnly)
		}
	}
}
