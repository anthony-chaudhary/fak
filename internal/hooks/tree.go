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

// treeReader is the disk-read surface the per-item gate helpers share between StagedDiff and
// TrackedTree — both read the SAME working tree, so one helper implementation serves both modes.
// (Both StagedDiff and TrackedTree satisfy it.)
type treeReader interface {
	FileBytes(rel string) ([]byte, bool)
	Exists(rel string) bool
	Size(rel string) (int64, bool)
}

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
}

// HygieneGates returns the tree-mode gates that have a parity-proven Go twin, in the order
// `make hygiene` / `make index-sync` run them. The remaining `make hygiene` checkers
// (demo_* x3, brand_consistency, scrub_hardware_names, guard_mcp_status_audit) stay on the Python
// path until they are ported under #928 A3/A4/A5; each port appends its gate here.
func HygieneGates() []HygieneGate {
	return []HygieneGate{
		{"DOC_PLACEMENT", gateDocPlacementTree},
		{"BROKEN_LINK", gateBrokenLinkTree},
		{"FILE_ADMISSION", gateFileAdmissionTree},
		{"SECRET_SHAPE", gateSecretShapeTree},
		{"PROVENANCE_LABEL", gateProvenanceLabelTree},
		{"INDEX_SYNC", gateIndexSyncTree},
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
