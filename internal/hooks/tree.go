package hooks

import (
	"context"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// tree.go — the --audit-tree twin of the staged-diff reader. `make hygiene` (and the CI step it
// mirrors) historically spawned ELEVEN Python interpreters, each doing sub-millisecond
// regex/path work over the whole git-tracked tree. This is the Go side of collapsing the ported
// gates into ONE `fak hygiene` process: read `git ls-files` ONCE into a TrackedTree, then run
// every tree-mode gate over it — the same move the hooks port made one boundary earlier (#928).
//
// Where StagedDiff scans `git diff --cached`, TrackedTree scans `git ls-files`. The per-item gate
// logic is IDENTICAL between the two modes (a path predicate, a per-line scan, a per-file link
// resolve); only the input set differs — exactly as each tools/check_*.py `--audit-tree` branch
// differs from its `--audit-staged` branch. A gate that has a staged-ONLY sub-rule (DOC_PLACEMENT's
// unindexed-note rule) drops it in tree mode, matching the Python.

// TrackedTree is the whole git-tracked tree read ONCE and shared across every hygiene gate. Paths
// is the `git ls-files` set (sorted, forward-slash); the lazy fileCache caches a gate's file reads.
type TrackedTree struct {
	Root      string
	Paths     []string // `git ls-files` (sorted, forward-slash, NUL-split so odd paths survive)
	fileCache map[string]fileEntry
}

// ReadTrackedTree runs `git ls-files -z` under root and folds the tracked path set into a
// TrackedTree. A git failure returns ErrCouldNotRun so `fak hygiene` fails open (exit 2 → the
// Makefile/CI wrapper falls back to the Python checkers), exactly like ReadStagedDiff.
func ReadTrackedTree(root string) (*TrackedTree, error) {
	return readTrackedTreeWith(context.Background(), realRunner, root)
}

func readTrackedTreeWith(ctx context.Context, run Runner, root string) (*TrackedTree, error) {
	// -z (NUL-delimited) so a path with a space or a quoting-trigger char survives intact — the
	// Python checkers mostly used whitespace .split(), which is fine on this tree (no spaced
	// paths) but the NUL form is strictly safer and verdict-identical.
	out, code, err := run(ctx, root, "ls-files", "-z")
	if err != nil || code != 0 {
		return nil, ErrCouldNotRun
	}
	var paths []string
	for _, p := range strings.Split(out, "\x00") {
		if p != "" {
			paths = append(paths, p)
		}
	}
	sort.Strings(paths)
	return &TrackedTree{Root: root, Paths: paths, fileCache: map[string]fileEntry{}}, nil
}

// FileBytes / Exists / Size mirror the StagedDiff disk-read helpers (both read the working tree).
// Duplicated rather than shared via embedding to keep the proven staged path untouched.
func (t *TrackedTree) FileBytes(rel string) ([]byte, bool) {
	if e, ok := t.fileCache[rel]; ok {
		return e.data, e.exists
	}
	b, err := os.ReadFile(filepath.Join(t.Root, filepath.FromSlash(rel)))
	e := fileEntry{data: b, exists: err == nil}
	t.fileCache[rel] = e
	return e.data, e.exists
}

func (t *TrackedTree) Exists(rel string) bool {
	_, err := os.Stat(filepath.Join(t.Root, filepath.FromSlash(rel)))
	return err == nil
}

func (t *TrackedTree) Size(rel string) (int64, bool) {
	fi, err := os.Stat(filepath.Join(t.Root, filepath.FromSlash(rel)))
	if err != nil {
		return 0, false
	}
	return fi.Size(), true
}

// IndexMD reads the curated INDEX.md the doc/index gates consult.
func (t *TrackedTree) IndexMD() (string, bool) {
	b, ok := t.FileBytes("INDEX.md")
	return string(b), ok
}

// HygieneGate is one whole-tree gate run by `fak hygiene`. Unlike the staged Gate it carries no
// ModeEnv/EscapeEnv: `make hygiene` (and the CI mirror) invoke each Python checker's `--audit-tree`
// branch unconditionally — that branch IGNORES the per-gate FLEET_*_GUARD / ALLOW_* escapes (they
// gate only `--audit-staged`). So every hygiene gate is an always-on HARD gate, matching the Python.
type HygieneGate struct {
	Name  string
	Check func(t *TrackedTree) ([]Finding, error)
	// DefaultOff marks a gate that does NOT run in the default `fak hygiene` / `make ci`
	// sweep — it fires only when named explicitly via `--gates`. A ratchet lands DefaultOff
	// while its migration is in flight (so it never reds the shared trunk against known,
	// not-yet-migrated debt) and flips on with a one-line change once the tree is clean.
	DefaultOff bool
	// PushScoped demotes findings outside the committed push delta to advisory.
	// It is used by TIER_DECLARED so a peer's undeclared leaf cannot wedge all pushes.
	PushScoped bool
}

// HygieneGates returns the tree-mode gates that have a parity-proven Go twin, in the order
// `make hygiene` / `make index-sync` run them. All `make hygiene` checkers now have native
// Go gates here (#928, #10940).
func HygieneGates() []HygieneGate {
	return []HygieneGate{
		{"DOC_PLACEMENT", gateDocPlacementTree, false, false},
		{"BROKEN_LINK", gateBrokenLinkTree, false, false},
		{"FILE_ADMISSION", gateFileAdmissionTree, false, false},
		{"SECRET_SHAPE", gateSecretShapeTree, false, false},
		{"PROVENANCE_LABEL", gateProvenanceLabelTree, false, false},
		{"INDEX_SYNC", gateIndexSyncTree, false, false},
		{"BRAND_CONSISTENCY", gateBrandConsistencyTree, false, false},
		{"TIER_DECLARED", gateTierDeclaredTree, false, true},
		{"NEW_PYTHON_TOOL", gatePythonToolTree, false, false},
		// GOD_FILE_GROWTH ships default-ON like NEW_PYTHON_TOOL: the grandfathered
		// baseline (godfile_baseline.go) freezes today's offenders at-size, so the
		// tree is clean the moment the gate lands — only NEW growth can red it.
		{"GOD_FILE_GROWTH", gateGodFileGrowthTree, false, false},
		{"HARDWARE_TELL", gateHardwareTreeTell, false, false},
		// DEAD_CODE (the slop-prevention twin of NEW_PYTHON_TOOL) is DefaultOff: the tree still
		// carries pre-existing dead unexported symbols the /slop-score loop is retiring, so it
		// would red the shared trunk against known debt. It is the always-available audit sweep
		// (`fak hygiene --gates DEAD_CODE`) that proves the retirement, and flips DefaultOff:false
		// — the enforcement gate that HOLDS the line at zero — once the tree is clean.
		{"DEAD_CODE", gateDeadCodeTree, true, false},
		// SWALLOWED_ERROR (issue #2899, hermes-inspiration epic #2871) is DefaultOff: the tree
		// still carries pre-existing `_ = <call>()` error discards (the Go `except: pass`) that
		// predate the floor, so wiring it always-on would red `make ci` against known debt. It is
		// the always-available audit sweep (`fak hygiene --gates SWALLOWED_ERROR`) that proves the
		// retirement, and flips DefaultOff:false — the enforcement gate that HOLDS the line at zero
		// un-witnessed discards — once the tree is clean.
		{"SWALLOWED_ERROR", gateSwallowedErrorTree, true, false},
		// DEMO_COMMAND (issue #928 A4) ships default-ON: it is a faithful port of
		// tools/demo_command_audit.py, which `make hygiene` already runs green over the real
		// tree, so the gate lands clean and only a NEW stale demo-command reference can red it.
		{"DEMO_COMMAND", gateDemoCommandTree, false, false},
		// BROWSER_CONTRACT (issue #928 A5) ships default-ON: it is a faithful port of
		// tools/demo_browser_contract.py, which `make hygiene` already runs green over the real
		// tree, so the gate lands clean and only NEW browser-demo metadata drift (a moved default
		// port, a dropped demoui helper, a stale run/public doc, a bad lifecycle decision) can red it.
		{"BROWSER_CONTRACT", gateBrowserContractTree, false, false},
		// DEMO_LIVE_LINKS (issue #10940) ships default-ON: it is a faithful port of
		// tools/demo_live_links.py, which `make hygiene` already runs green over the real
		// tree, so the gate lands clean and only stale hosted demo links or metadata drift can red it.
		{"DEMO_LIVE_LINKS", gateDemoLiveLinksTree, false, false},
		// GUARD_MCP_STATUS (issue #10940) ships default-ON: it is a faithful port of
		// tools/guard_mcp_status_audit.py, which `make hygiene` already runs green over the real
		// tree, so the gate lands clean and only broken status packet claims can red it.
		{"GUARD_MCP_STATUS", gateGuardMCPStatusTree, false, false},
	}
}

// HygieneGateByName returns the named gate's Check, or nil — the parity harness and the
// `fak hygiene --gates` filter look gates up by name.
func HygieneGateByName(name string) func(*TrackedTree) ([]Finding, error) {
	for _, g := range HygieneGates() {
		if g.Name == name {
			return g.Check
		}
	}
	return nil
}
