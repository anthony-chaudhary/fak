// leafdecl.go — the UNDECLARED_LEAF rung (#2082): surface a missing leaf
// declaration at EDIT time instead of at commit time.
//
// A new `internal/<leaf>` tree is only legal once it is declared TWICE: once in
// dos.toml's `[lanes.trees]` (which is what makes the `(fak <leaf>)` commit stamp
// bind to a real lane) and once in the architest tier map (which is what
// TestEveryPackageDeclaresTier reads). Today both facts only surface at COMMIT
// time — many turns after the edit that created the tree, when the agent has
// dropped the context that made the fix cheap, and on the shared trunk a peer may
// already have pushed the local commit past the point where the subject can be
// amended.
//
// So this rung answers the same question the commit gate asks, at the moment the
// file is written. It is ADVISORY by construction (see severity.go): the edit is
// never blocked, the agent just gets the two declarations it still owes and the
// verb that writes them.
//
// PURE, like the rest of the classification core: the taxonomy is parsed from
// readers and injected by the command layer, so the classifier itself never
// touches the filesystem (the ...ForWorkspace helper is the IO seam, mirroring
// LiveMonitorTaskIDsFromJournalFile).
package repoguard

import (
	"io"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"
)

// ReasonUndeclaredLeaf is the structured token for writing into an internal leaf
// that the lane taxonomy and/or the architest tier map does not declare yet.
const ReasonUndeclaredLeaf = "UNDECLARED_LEAF"

// LeafDeclarations is the pre-computed convention taxonomy the rung compares a
// written path against. Lanes comes from dos.toml `[lanes.trees]`; Tiers comes
// from the architest `var tier` map. Both empty means "not loaded" — the rung
// then stays silent rather than guessing every leaf is undeclared.
type LeafDeclarations struct {
	Lanes map[string]bool
	Tiers map[string]bool
}

// Loaded reports whether the taxonomy was resolved well enough to judge a leaf.
// An unreadable dos.toml / architest file must never turn every edit into an
// advisory, so a half-loaded taxonomy only judges the half it actually has.
func (d LeafDeclarations) Loaded() bool {
	return len(d.Lanes) > 0 || len(d.Tiers) > 0
}

// LeafForWritePath returns the `internal/<leaf>` package a Write/Edit file_path
// lands in, relative to workspaceRoot. ok is false for anything outside
// `<workspace>/internal/<leaf>/...` — including the bare `internal/` dir itself.
func LeafForWritePath(filePath, workspaceRoot string) (string, bool) {
	ws := normalize(workspaceRoot)
	if ws == "" {
		return "", false
	}
	absTarget, ok := toAbs(filePath, ws)
	if !ok || !isUnder(absTarget, ws) {
		return "", false
	}
	rel := strings.TrimPrefix(strings.TrimPrefix(absTarget, ws), "/")
	parts := strings.Split(path.Clean(rel), "/")
	if len(parts) < 3 || parts[0] != "internal" || parts[1] == "" || parts[1] == "." {
		return "", false
	}
	return parts[1], true
}

// laneTreeRowRE matches a `[lanes.trees]` row: `name = [...]` (a trailing `#`
// comment is common and irrelevant — only the key names the leaf).
var laneTreeRowRE = regexp.MustCompile(`^([A-Za-z0-9][\w.\-]*)\s*=\s*\[`)

// ParseLaneTrees folds the leaf names declared in dos.toml's `[lanes.trees]`
// section. Section-scoped: `[lanes]` above it lists lane names for the arbiter,
// not trees, and every section after it is unrelated.
func ParseLaneTrees(r io.Reader) map[string]bool {
	out := map[string]bool{}
	inSection := false
	for _, raw := range readLines(r) {
		line := strings.TrimSpace(raw)
		if strings.HasPrefix(line, "[") {
			inSection = line == "[lanes.trees]"
			continue
		}
		if !inSection || line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if m := laneTreeRowRE.FindStringSubmatch(line); m != nil {
			out[strings.ToLower(m[1])] = true
		}
	}
	return out
}

// archTierEntryRE matches one `"leaf": N` entry of the architest tier map.
var archTierEntryRE = regexp.MustCompile(`"([A-Za-z0-9][\w.\-]*)"\s*:\s*\d+`)

// ParseArchTiers folds the leaf names declared in the architest `var tier`
// map — the table TestEveryPackageDeclaresTier fails a new package against.
func ParseArchTiers(r io.Reader) map[string]bool {
	out := map[string]bool{}
	inTable := false
	for _, raw := range readLines(r) {
		line := strings.TrimSpace(raw)
		if !inTable {
			if strings.Contains(line, "var tier = map[string]int{") {
				inTable = true
			}
			continue
		}
		if line == "}" {
			break
		}
		for _, m := range archTierEntryRE.FindAllStringSubmatch(line, -1) {
			out[strings.ToLower(m[1])] = true
		}
	}
	return out
}

// LeafDeclarationsForWorkspace is the IO seam: it reads the two declaration
// files out of a workspace root. Best-effort by design — a missing or unreadable
// file yields an empty half, which keeps the rung silent instead of noisy.
func LeafDeclarationsForWorkspace(workspaceRoot string) LeafDeclarations {
	return LeafDeclarations{
		Lanes: parseFile(filepath.Join(workspaceRoot, "dos.toml"), ParseLaneTrees),
		Tiers: parseFile(filepath.Join(workspaceRoot, "internal", "architest", "architest_test.go"), ParseArchTiers),
	}
}

func parseFile(p string, parse func(io.Reader) map[string]bool) map[string]bool {
	f, err := os.Open(p)
	if err != nil {
		return nil
	}
	defer f.Close()
	return parse(f)
}

func readLines(r io.Reader) []string {
	b, err := io.ReadAll(r)
	if err != nil {
		return nil
	}
	return strings.Split(strings.ReplaceAll(string(b), "\r\n", "\n"), "\n")
}

func classifyUndeclaredLeaf(filePath, workspaceRoot string, decl LeafDeclarations) []Violation {
	if !decl.Loaded() {
		return nil
	}
	leaf, ok := LeafForWritePath(filePath, workspaceRoot)
	if !ok {
		return nil
	}
	key := strings.ToLower(leaf)
	missingLane := len(decl.Lanes) > 0 && !decl.Lanes[key]
	missingTier := len(decl.Tiers) > 0 && !decl.Tiers[key]
	if !missingLane && !missingTier {
		return nil
	}
	return []Violation{{
		Reason:   ReasonUndeclaredLeaf,
		Op:       "write",
		Target:   filePath,
		Resolved: "internal/" + leaf,
		Why:      undeclaredWhy(missingLane, missingTier),
		Fix:      "fak new-leaf " + leaf + " --tier <foundation|mechanism|composer|integrator> --register",
	}}
}

func undeclaredWhy(missingLane, missingTier bool) string {
	switch {
	case missingLane && missingTier:
		return "no dos.toml [lanes.trees] row and no architest tier row"
	case missingLane:
		return "no dos.toml [lanes.trees] row"
	default:
		return "no architest tier row"
	}
}

func renderUndeclaredLeafReason(violations []Violation) string {
	parts := make([]string, len(violations))
	for i, v := range violations {
		parts[i] = v.Target + " -> " + v.Resolved + " (" + v.Why + "; fix: " + v.Fix + ")"
	}
	return ReasonUndeclaredLeaf + ": this edit writes into an internal leaf the repo has not declared yet. " +
		strings.Join(parts, "; ") +
		". Declare it NOW, while the context is cheap: a missing [lanes.trees] row leaves the " +
		"`(fak <leaf>)` commit stamp unbindable, and a missing tier row reds architest " +
		"(TestEveryPackageDeclaresTier) on the pre-commit hook. Advisory only — the edit is allowed."
}
