package architest

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// facadeOffList names the blank imports in cmd/fak/registered_leaves.go whose package
// declares NO init() and therefore performs no self-registration when the fak binary
// links it. Each is kept on the reachability roster so `fak version modules` lane
// tooling (and other repo-scoped tooling that reaches a leaf only through the binary's
// link set) can still see the package come alive during a build; the import is kept
// deliberately inert. This is the same consciously-reviewed allow-list shape as
// regOffList and the tier table: a leaf added here is a reviewed "wired nowhere,
// deliberately linked" decision, not an accident.
//
// Derivation contract (issue #12873): this list must stay in lockstep with the tree.
// A leaf that grows a real init() registration graduates off this list automatically by
// the test below re-deriving self-registration from the AST; a NEW leaf that blank-imports
// without init() fails the build until it is listed here for a stated reason.
//
// Populated empirically on the post-#12873 tree: every entry below is blank-imported in
// cmd/fak/registered_leaves.go and declares no func init() in its non-test .go files.
// macromailbox and macrostate are EXCLUDED — both were reaped as inert blank-import
// facades by #12873, and the regression pin below keeps them out.
var facadeOffList = map[string]string{
	"advmodel":                   "reachability-roster entry for fak version modules lanes; no self-registration",
	"agentsched":                 "reachability-roster entry for fak version modules lanes; no self-registration",
	"breathgate":                 "reachability-roster entry for fak version modules lanes; no self-registration",
	"cache":                      "reachability-roster entry for fak version modules lanes; no self-registration",
	"cavemansafety":              "reachability-roster entry for fak version modules lanes; no self-registration",
	"chatopsdetach":              "reachability-roster entry for fak version modules lanes; no self-registration",
	"closurerate":                "reachability-roster entry for fak version modules lanes; no self-registration",
	"codexsession":               "reachability-roster entry for fak version modules lanes; no self-registration",
	"cohort":                     "reachability-roster entry for fak version modules lanes; no self-registration",
	"completiondist":             "reachability-roster entry for fak version modules lanes; no self-registration",
	"composition":                "reachability-roster entry for fak version modules lanes; no self-registration",
	"computeadmit":               "reachability-roster entry for fak version modules lanes; no self-registration (its reason-code registration fires from the Submit-admitter constructor, never from init())",
	"computetune":                "reachability-roster entry for fak version modules lanes; no self-registration",
	"dataslot":                   "reachability-roster entry for fak version modules lanes; no self-registration",
	"deadlineadmit":              "reachability-roster entry for fak version modules lanes; no self-registration",
	"decodemigrate":              "reachability-roster entry for fak version modules lanes; no self-registration",
	"deepseekv4kv":               "reachability-roster entry for fak version modules lanes; no self-registration",
	"dependencyquarantine":       "reachability-roster entry for fak version modules lanes; no self-registration",
	"deployment":                 "reachability-roster entry for fak version modules lanes; no self-registration",
	"docrender":                  "reachability-roster entry for fak version modules lanes; no self-registration",
	"dormancysim":                "reachability-roster entry for fak version modules lanes; no self-registration",
	"dosadapter":                 "reachability-roster entry for fak version modules lanes; no self-registration",
	"dsparity":                   "reachability-roster entry for fak version modules lanes; no self-registration",
	"epochbridge":                "reachability-roster entry for fak version modules lanes; no self-registration",
	"escalation":                 "reachability-roster entry for fak version modules lanes; no self-registration",
	"estimatecal":                "reachability-roster entry for fak version modules lanes; no self-registration",
	"eveimport":                  "reachability-roster entry for fak version modules lanes; no self-registration",
	"eveparity":                  "reachability-roster entry for fak version modules lanes; no self-registration",
	"fakrpc":                     "reachability-roster entry for fak version modules lanes; no self-registration",
	"faultlab":                   "reachability-roster entry for fak version modules lanes; no self-registration",
	"fleetfreeze":                "reachability-roster entry for fak version modules lanes; no self-registration",
	"fleetmemory":                "reachability-roster entry for fak version modules lanes; no self-registration",
	"fleetsim":                   "reachability-roster entry for fak version modules lanes; no self-registration",
	"fleetverify":                "reachability-roster entry for fak version modules lanes; no self-registration",
	"fp4runtime":                 "reachability-roster entry for fak version modules lanes; no self-registration",
	"framebus":                   "reachability-roster entry for fak version modules lanes; no self-registration",
	"gcpgpu":                     "reachability-roster entry for fak version modules lanes; no self-registration",
	"gitresource":                "reachability-roster entry for fak version modules lanes; no self-registration",
	"godfileceiling":             "reachability-roster entry for fak version modules lanes; no self-registration",
	"guardcorpus":                "reachability-roster entry for fak version modules lanes; no self-registration",
	"harnessmodelsetconformance": "reachability-roster entry for fak version modules lanes; no self-registration",
	"harnesswarm":                "reachability-roster entry for fak version modules lanes; no self-registration",
	"humanctl":                   "reachability-roster entry for fak version modules lanes; no self-registration",
	"incidentrsi":                "reachability-roster entry for fak version modules lanes; no self-registration",
	"interactivesession":         "reachability-roster entry for fak version modules lanes; no self-registration",
	"issuecatalog":               "reachability-roster entry for fak version modules lanes; no self-registration",
	"issuecheck":                 "reachability-roster entry for fak version modules lanes; no self-registration",
	"issueownerprompt":           "reachability-roster entry for fak version modules lanes; no self-registration",
	"kimik3page":                 "reachability-roster entry for fak version modules lanes; no self-registration",
	"knownenv":                   "reachability-roster entry for fak version modules lanes; no self-registration",
	"kvint2eval":                 "reachability-roster entry for fak version modules lanes; no self-registration",
	"launchlatency":              "reachability-roster entry for fak version modules lanes; no self-registration",
	"lifebridge":                 "reachability-roster entry for fak version modules lanes; no self-registration",
	"lightgapport":               "reachability-roster entry for fak version modules lanes; no self-registration",
	"localappux":                 "reachability-roster entry for fak version modules lanes; no self-registration",
	"lookahead":                  "reachability-roster entry for fak version modules lanes; no self-registration",
	"loopunblock":                "reachability-roster entry for fak version modules lanes; no self-registration",
	"market":                     "reachability-roster entry for fak version modules lanes; no self-registration",
	"mcpbroker":                  "reachability-roster entry for fak version modules lanes; no self-registration",
	"memorycotravel":             "reachability-roster entry for fak version modules lanes; no self-registration",
	"microscaleeval":             "reachability-roster entry for fak version modules lanes; no self-registration",
	"mixedprecision":             "reachability-roster entry for fak version modules lanes; no self-registration",
	"modelpack":                  "reachability-roster entry for fak version modules lanes; no self-registration",
	"modelsrc":                   "reachability-roster entry for fak version modules lanes; no self-registration",
	"mtpeval":                    "reachability-roster entry for fak version modules lanes; no self-registration",
	"nativeperfartifact":         "reachability-roster entry for fak version modules lanes; no self-registration",
	"nativeperfbackend":          "reachability-roster entry for fak version modules lanes; no self-registration",
	"nativeperfcorrelation":      "reachability-roster entry for fak version modules lanes; no self-registration",
	"nativeperfcoverage":         "reachability-roster entry for fak version modules lanes; no self-registration",
	"nativeperfobscontract":      "reachability-roster entry for fak version modules lanes; no self-registration",
	"nativeperfslo":              "reachability-roster entry for fak version modules lanes; no self-registration",
	"observability":              "reachability-roster entry for fak version modules lanes; no self-registration",
	"ociartifact":                "reachability-roster entry for fak version modules lanes; no self-registration",
	"openaiadapter":              "reachability-roster entry for fak version modules lanes; no self-registration",
	"opensweharder":              "reachability-roster entry for fak version modules lanes; no self-registration",
	"openviking":                 "reachability-roster entry for fak version modules lanes; no self-registration",
	"operatortouches":            "reachability-roster entry for fak version modules lanes; no self-registration",
	"portabilityswitch":          "reachability-roster entry for fak version modules lanes; no self-registration",
	"projectionspine":            "reachability-roster entry for fak version modules lanes; no self-registration",
	"providerjobaccounting":      "reachability-roster entry for fak version modules lanes; no self-registration",
	"qevicteval":                 "reachability-roster entry for fak version modules lanes; no self-registration",
	"quantcompat":                "reachability-roster entry for fak version modules lanes; no self-registration",
	"quantfixture":               "reachability-roster entry for fak version modules lanes; no self-registration",
	"quantlicense":               "reachability-roster entry for fak version modules lanes; no self-registration",
	"quantmatrix":                "reachability-roster entry for fak version modules lanes; no self-registration",
	"quantobs":                   "reachability-roster entry for fak version modules lanes; no self-registration",
	"quantpolicy":                "reachability-roster entry for fak version modules lanes; no self-registration",
	"quantroute":                 "reachability-roster entry for fak version modules lanes; no self-registration",
	"qwen4exp":                   "reachability-roster entry for fak version modules lanes; no self-registration",
	"qwenflashnext":              "reachability-roster entry for fak version modules lanes; no self-registration",
	"qwensemanticstop":           "reachability-roster entry for fak version modules lanes; no self-registration",
	"qwenworkbudget":             "reachability-roster entry for fak version modules lanes; no self-registration",
	"reflexagent":                "reachability-roster entry for fak version modules lanes; no self-registration",
	"region":                     "reachability-roster entry for fak version modules lanes; no self-registration",
	"requanteval":                "reachability-roster entry for fak version modules lanes; no self-registration",
	"residency":                  "reachability-roster entry for fak version modules lanes; no self-registration",
	"residualquant":              "reachability-roster entry for fak version modules lanes; no self-registration",
	"resourcelifecycle":          "reachability-roster entry for fak version modules lanes; no self-registration",
	"resultstier":                "reachability-roster entry for fak version modules lanes; no self-registration",
	"resulttier":                 "reachability-roster entry for fak version modules lanes; no self-registration",
	"rollout":                    "reachability-roster entry for fak version modules lanes; no self-registration",
	"rotationmeta":               "reachability-roster entry for fak version modules lanes; no self-registration",
	"schedcontract":              "reachability-roster entry for fak version modules lanes; no self-registration",
	"schemaadapter":              "reachability-roster entry for fak version modules lanes; no self-registration",
	"scratchmark":                "reachability-roster entry for fak version modules lanes; no self-registration",
	"servingsim":                 "reachability-roster entry for fak version modules lanes; no self-registration",
	"servingsupervision":         "reachability-roster entry for fak version modules lanes; no self-registration",
	"sessionreplay":              "reachability-roster entry for fak version modules lanes; no self-registration",
	"spendrollup":                "reachability-roster entry for fak version modules lanes; no self-registration",
	"stallpage":                  "reachability-roster entry for fak version modules lanes; no self-registration",
	"studydrift":                 "reachability-roster entry for fak version modules lanes; no self-registration",
	"studyreceipt":               "reachability-roster entry for fak version modules lanes; no self-registration",
	"subtractiveprofile":         "reachability-roster entry for fak version modules lanes; no self-registration",
	"sweepconfig":                "reachability-roster entry for fak version modules lanes; no self-registration",
	"taskidentity":               "reachability-roster entry for fak version modules lanes; no self-registration",
	"taskvc":                     "reachability-roster entry for fak version modules lanes; no self-registration",
	"timeaware":                  "reachability-roster entry for fak version modules lanes; no self-registration",
	"toolbound":                  "reachability-roster entry for fak version modules lanes; no self-registration",
	"tracesink":                  "reachability-roster entry for fak version modules lanes; no self-registration",
	"trajctlhook":                "reachability-roster entry for fak version modules lanes; no self-registration",
	"turnkind":                   "reachability-roster entry for fak version modules lanes; no self-registration",
	"ultracodeborrow":            "reachability-roster entry for fak version modules lanes; no self-registration",
	"ultracodecrossover":         "reachability-roster entry for fak version modules lanes; no self-registration",
	"ultracodedogfood":           "reachability-roster entry for fak version modules lanes; no self-registration",
	"ultracodenegcontrol":        "reachability-roster entry for fak version modules lanes; no self-registration",
	"ultracodetokenizer":         "reachability-roster entry for fak version modules lanes; no self-registration",
	"usagepreflight":             "reachability-roster entry for fak version modules lanes; no self-registration",
	"vcacheqa":                   "reachability-roster entry for fak version modules lanes; no self-registration",
	"vcachestar":                 "reachability-roster entry for fak version modules lanes; no self-registration",
	"vllmcompile":                "reachability-roster entry for fak version modules lanes; no self-registration",
	"vllmquant":                  "reachability-roster entry for fak version modules lanes; no self-registration",
	"wavefuel":                   "reachability-roster entry for fak version modules lanes; no self-registration",
	"wiplease":                   "reachability-roster entry for fak version modules lanes; no self-registration",
	"workerenvelope":             "reachability-roster entry for fak version modules lanes; no self-registration",
	"worklog":                    "reachability-roster entry for fak version modules lanes; no self-registration",
	"workspaceslot":              "reachability-roster entry for fak version modules lanes; no self-registration",
}

// reapedFacadeLeaves pins the leaves #12873 deleted as inert blank-import facades.
// Re-adding any of them re-opens the island: a blank import that fires no init() links
// dead code into every fak build while fooling roster tooling into counting the package
// as reached.
var reapedFacadeLeaves = []string{"macromailbox", "macrostate"}

// blankImportLeaves parses cmd/fak/registered_leaves.go and returns the <name> of every
// blank import (_ "github.com/anthony-chaudhary/fak/internal/<name>"), in file order.
// Reading the AST — not a text grep — means a commented-out or renamed import path is
// correctly not counted.
func blankImportLeaves(t *testing.T, internal string) []string {
	t.Helper()
	path := filepath.Join(internal, "..", "cmd", "fak", "registered_leaves.go")
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, path, nil, parser.ImportsOnly)
	if err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
	var out []string
	for _, spec := range f.Imports {
		if spec.Name == nil || spec.Name.Name != "_" {
			continue
		}
		p := strings.Trim(spec.Path.Value, `"`)
		if !strings.HasPrefix(p, modPrefix) {
			continue
		}
		name := strings.SplitN(strings.TrimPrefix(p, modPrefix), "/", 2)[0]
		if name == "" {
			t.Errorf("blank import %q in registered_leaves.go does not name an internal package directly under internal/", p)
			continue
		}
		out = append(out, name)
	}
	return out
}

// declaresInitFunc reports whether DIRECTORY's non-test .go files declare a func init()
// (via AST walk, so a doc comment mentioning init is correctly not counted).
func declaresInitFunc(t *testing.T, internal, pkg string) bool {
	t.Helper()
	dir := filepath.Join(internal, pkg)
	fset := token.NewFileSet()
	parsed, err := parser.ParseDir(fset, dir,
		func(fi os.FileInfo) bool { return !strings.HasSuffix(fi.Name(), "_test.go") }, 0)
	if err != nil {
		t.Fatalf("parse %s: %v", dir, err)
	}
	for _, p := range parsed {
		for _, f := range p.Files {
			found := false
			ast.Inspect(f, func(n ast.Node) bool {
				if fd, ok := n.(*ast.FuncDecl); ok && fd.Recv == nil && fd.Name.Name == "init" {
					found = true
				}
				return true
			})
			if found {
				return true
			}
		}
	}
	return false
}

// TestRegisteredLeavesFacadeImports closes the facade hole in cmd/fak/registered_leaves.go
// (issue #12873): a blank import of a package that declares no init() links NOTHING — the
// package compiles into the binary and roster/reachability tooling counts it as reached,
// but no code ever runs. That is an island masquerading as a connection. #12873 reaped
// macromailbox and macrostate for exactly this; this test makes the ratchet stick:
//
//  1. every blank-imported leaf either self-registers (declares func init()) — OK, or
//  2. is consciously listed in facadeOffList with a reason — OK, or the contributor fixes
//     the linkage (see the failure message below), and
//  3. the reaped macromailbox/macrostate facades stay gone: not blank-imported, not on
//     the off-list, directories deleted.
//
// The self-registration set is re-derived from the AST on every run, so a leaf that grows
// a real init() graduates off facadeOffList automatically, and a new unlinked blank import
// fails the build instead of silently islanding.
func TestRegisteredLeavesFacadeImports(t *testing.T) {
	internal := internalDir(t)

	// Regression pin on the #12873 reap: the deleted facade imports must not return,
	// must not be allow-listed, and their directories must stay deleted. If you are
	// re-adding one of these, you are re-opening the island #12873 closed — add a real
	// init() registration or a production caller instead, or file the reap issue for
	// an honest re-review with evidence the package actually runs.
	for _, reaped := range reapedFacadeLeaves {
		var imported bool
		for _, leaf := range blankImportLeaves(t, internal) {
			if leaf == reaped {
				imported = true
			}
		}
		if imported {
			t.Errorf("facade regression: %q is blank-imported in cmd/fak/registered_leaves.go again. "+
				"It was reaped by issue #12873 as an inert blank-import facade (no init(), no production caller); "+
				"re-adding it re-opens the island. Give internal/%s a real init() registration, wire a production "+
				"caller, or delete it again.", reaped, reaped)
		}
		if reason, ok := facadeOffList[reaped]; ok {
			t.Errorf("facade regression: %q is listed in facadeOffList (reason %q) but was reaped by issue "+
				"#12873 as an inert blank-import facade. Remove it from facadeOffList and keep the package deleted; "+
				"an allow-list entry must name a package that exists and is deliberately linked.", reaped, reason)
		}
		dir := filepath.Join(internal, reaped)
		if _, err := os.Stat(dir); err == nil {
			t.Errorf("facade regression: directory %s exists again. It was deleted by issue #12873 after "+
				"macromailbox/macrostate proved to be inert blank-import facades. Re-adding the directory without "+
				"a real init() registration re-opens the island; delete it or wire it for real.", dir)
		} else if !os.IsNotExist(err) {
			t.Fatalf("stat %s: %v", dir, err)
		}
	}

	var selfRegistered, offListed int
	for _, leaf := range blankImportLeaves(t, internal) {
		if declaresInitFunc(t, internal, leaf) {
			selfRegistered++
			continue
		}
		if _, ok := facadeOffList[leaf]; ok {
			offListed++
			continue
		}
		t.Errorf("blank import _ \"%s%s\" in cmd/fak/registered_leaves.go is a potential facade: internal/%s "+
			"declares no func init(), so the import links the package in without running anything from it "+
			"(an island that roster/reachability tooling counts as reached). Per the wire-or-delete-or-allow-list "+
			"doctrine (internal/unwiredscore/dispatch.go), do ONE of:\n"+
			"    1. give internal/%s a real init() registration so the import self-registers;\n"+
			"    2. wire a production caller so the package is reached without the blank import;\n"+
			"    3. delete internal/%s if nothing reaches it;\n"+
			"    4. consciously append %q to facadeOffList in internal/architest/facade_guard_test.go with a "+
			"one-line reason (same review chokepoint as the tier table and regOffList).",
			modPrefix, leaf, leaf, leaf, leaf, leaf)
	}
	t.Logf("facade ratchet: %d blank imports checked, %d self-register, %d on facadeOffList",
		selfRegistered+offListed, selfRegistered, offListed)
}
