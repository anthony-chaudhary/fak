// Package modver derives a per-module version stamp from git history — the
// "version everything" spine. The whole-binary version (internal/appversion,
// the root VERSION file) says which *release* you run; modver says how far each
// *module* has moved. A module's version is DERIVED, never declared: on a
// shared multi-session trunk a hand-maintained per-module version file would
// rot within hours, but `rev` — the count of trunk NON-MERGE commits that
// touched the module — is monotonic, conflict-free, and computable from the
// history alone. Merge commits are excluded on purpose (see Snapshot): a merge
// carries no authored module work, so rev is stable across an in-place trunk
// merge and reflects only real commits.
//
// A module version renders as "r<rev>+g<shortsha>" (e.g. r412+g2bc81478):
// the monotonic revision plus the last-touch commit. Successive ledger stamps
// (docs/nightrun/module-versions.jsonl, schema fak-module-versions/1) make
// growth trends queryable, and JoinScores attaches an external module→score
// map so a score series can be read against the version series.
package modver

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// Schema is the ledger row schema tag.
const Schema = "fak-module-versions/1"

// Module is one versioned unit: an internal/<leaf> package, a cmd/<dir> binary,
// a .github/workflows/<file> CI workflow, or a tools/<family> script.
type Module struct {
	Name       string   `json:"module"` // e.g. "internal/modver", "cmd/fak", ".github/workflows/ci.yml", "tools/account_probe"
	Kind       string   `json:"kind"`   // "internal" | "cmd" | "workflow" | "tools"
	Rev        int      `json:"rev"`    // distinct commits touching the module
	LastCommit string   `json:"last_commit"`
	LastDate   string   `json:"last_date"` // committer date (ISO) of the last touch
	Score      *float64 `json:"score,omitempty"`
}

// Version renders the derived module version: the monotonic revision counter
// plus the last-touch commit, semver-build-metadata style.
func (m Module) Version() string {
	return fmt.Sprintf("r%d+g%s", m.Rev, m.LastCommit)
}

// Report is one whole-tree snapshot of module versions.
type Report struct {
	Head       string   `json:"head"`
	AppVersion string   `json:"app_version"`
	Modules    []Module `json:"modules"`
}

// Runner executes git in dir and returns stdout. Failures (nonzero exit)
// surface as errors carrying stderr.
type Runner func(ctx context.Context, dir string, args ...string) ([]byte, error)

// RealRunner shells to git without flashing a console window.
func RealRunner(ctx context.Context, dir string, args ...string) ([]byte, error) {
	cmd := exec.CommandContext(ctx, "git", args...)
	cmd.Dir = dir
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	windowgate.ConfigureBackgroundCommand(cmd)
	if err := cmd.Run(); err != nil {
		var exit *exec.ExitError
		if errors.As(err, &exit) {
			detail := strings.TrimSpace(stderr.String())
			if detail == "" {
				detail = fmt.Sprintf("exit %d", exit.ExitCode())
			}
			return nil, fmt.Errorf("modver: git %s: %s", strings.Join(args[:1], " "), detail)
		}
		return nil, fmt.Errorf("modver: git: %w", err)
	}
	return stdout.Bytes(), nil
}

// trackedRoots are the path prefixes that define the module key space today:
// the internal/ leaves and cmd/ binaries, the .github/workflows/ CI keyspace
// (each workflow file is its own module), and the tools/ script keyspace (each
// top-level script family is a module — see moduleOf). (docs/, skills, policies
// are follow-on key spaces — see the version-everything backlog.)
var trackedRoots = []string{"internal", "cmd", ".github/workflows", "tools"}

// Snapshot computes the module-version report for the repo at dir: one
// `git ls-files` to bound the LIVE module set, one `git log --name-only`
// pass over the history to derive each module's rev and last touch.
func Snapshot(ctx context.Context, dir string, run Runner) (Report, error) {
	if run == nil {
		run = RealRunner
	}
	headOut, err := run(ctx, dir, "rev-parse", "--short=8", "HEAD")
	if err != nil {
		return Report{}, err
	}
	lsArgs := append([]string{"ls-files", "-z", "--"}, trackedRoots...)
	lsOut, err := run(ctx, dir, lsArgs...)
	if err != nil {
		return Report{}, err
	}
	live := liveModules(lsOut)
	// --no-merges pins the rev semantics (#2475): rev counts distinct NON-MERGE
	// commits touching the module. A merge commit carries no authored module
	// work — the file changes live in the non-merge commits it joins, each of
	// which git's DAG walk lists exactly once, so there is no double-counting.
	// git suppresses merge diffs from --name-only by default, but that default is
	// overridable (--diff-merges, log.diffMerges); --no-merges makes the exclusion
	// explicit and independent of git config, so rev is stable across an in-place
	// trunk merge. Deliberately NOT --first-parent: work that reaches the trunk
	// through a merge lives off the first-parent line, and --first-parent would
	// silently undercount it while the merge commit itself contributes no files.
	logArgs := append([]string{"log", "--no-merges", "--pretty=format:%x1e%h%x09%cI", "--name-only", "--"}, trackedRoots...)
	logOut, err := run(ctx, dir, logArgs...)
	if err != nil {
		return Report{}, err
	}
	modules := parseLog(logOut, live)
	appVer := ""
	if v, ok := appversion.FromDir(dir); ok {
		appVer = v
	}
	return Report{
		Head:       strings.TrimSpace(string(headOut)),
		AppVersion: appVer,
		Modules:    modules,
	}, nil
}

// liveModules folds NUL-separated `git ls-files` output into the set of
// modules that exist at HEAD, so history-only (deleted) modules do not ghost
// into the report.
func liveModules(lsFilesOut []byte) map[string]bool {
	live := map[string]bool{}
	for _, p := range bytes.Split(lsFilesOut, []byte{0}) {
		if name, _, ok := moduleOf(string(p)); ok {
			live[name] = true
		}
	}
	return live
}

// moduleOf maps a repo-relative path to its module key: internal/<leaf>/… →
// internal/<leaf>, cmd/<dir>/… → cmd/<dir>. A directory keyspace groups every
// file under a module directory into one module. The .github/workflows/ CI
// keyspace is file-keyed instead: each workflow file (.github/workflows/<file>)
// is its own module, since a workflow's unit of behavior is the file, not a
// directory. The tools/ script keyspace is family-keyed: each top-level script
// (tools/<name>.py|.sh|.ps1) is a module keyed by its family (tools/<name>),
// with a _test sibling folded into the same family, since a de-Python unit is a
// script plus its test (see toolsFamily). Files sitting directly under a
// directory root (no module directory) belong to no module, as do nested paths
// under .github/workflows/ (GitHub Actions does not run workflows in
// subdirectories) or under tools/ (registries, caches, and fixtures — not the
// flat frozen-script inventory the de-Python ratchet tracks).
func moduleOf(path string) (name, kind string, ok bool) {
	path = strings.TrimSpace(strings.ReplaceAll(path, "\\", "/"))
	parts := strings.Split(path, "/")
	if len(parts) < 2 {
		return "", "", false
	}
	switch parts[0] {
	case "internal", "cmd":
		if len(parts) < 3 {
			return "", "", false // a file directly under the root belongs to no module
		}
		return parts[0] + "/" + parts[1], parts[0], true
	case ".github":
		if parts[1] == "workflows" && len(parts) == 3 {
			return path, "workflow", true
		}
	case "tools":
		// Flat script keyspace: only top-level scripts count (len == 2); nested
		// paths are registries/caches, not the frozen-script inventory.
		if len(parts) == 2 {
			if fam, famOK := toolsFamily(parts[1]); famOK {
				return "tools/" + fam, "tools", true
			}
		}
	}
	return "", "", false
}

// toolsScriptExts are the extensions the tools/ keyspace versions: the frozen
// Python scripts the de-Python ratchet tracks (internal/pythongate), plus their
// shell / PowerShell siblings. Data and config files under tools/ (.json, .data,
// .md, …) are fixtures, not modules, so they are excluded.
var toolsScriptExts = map[string]bool{".py": true, ".sh": true, ".ps1": true}

// toolsFamily maps a top-level tools/ filename to its module family: the stem
// with the script extension dropped and a trailing _test folded away, so that
// foo.py and foo_test.py collapse to the one module tools/foo — a script and its
// test are one de-Python unit, mirroring how the internal/ and cmd/ directory
// keyspaces fold a leaf's implementation and tests together. A non-script file
// (unknown extension) or a bare dotfile returns ok=false.
func toolsFamily(file string) (fam string, ok bool) {
	dot := strings.LastIndex(file, ".")
	if dot <= 0 { // no extension, or a leading-dot dotfile: not a tracked script
		return "", false
	}
	if !toolsScriptExts[file[dot:]] {
		return "", false
	}
	stem := strings.TrimSuffix(file[:dot], "_test")
	if stem == "" {
		return "", false
	}
	return stem, true
}

// parseLog folds `git log --pretty=format:%x1e%h%x09%cI --name-only` output
// (newest-first) into per-module revisions. A commit touching several files of
// one module counts once; the first record a module appears in is its last
// touch. Modules not in live are dropped.
func parseLog(logOut []byte, live map[string]bool) []Module {
	byName := map[string]*Module{}
	for _, rec := range bytes.Split(logOut, []byte{0x1e}) {
		lines := strings.Split(strings.TrimSpace(string(rec)), "\n")
		if len(lines) == 0 || strings.TrimSpace(lines[0]) == "" {
			continue
		}
		sha, date, ok := strings.Cut(strings.TrimSpace(lines[0]), "\t")
		if !ok {
			continue
		}
		seen := map[string]bool{}
		for _, file := range lines[1:] {
			name, kind, ok := moduleOf(file)
			if !ok || seen[name] || !live[name] {
				continue
			}
			seen[name] = true
			m := byName[name]
			if m == nil {
				m = &Module{Name: name, Kind: kind, LastCommit: sha, LastDate: date}
				byName[name] = m
			}
			m.Rev++
		}
	}
	out := make([]Module, 0, len(byName))
	for _, m := range byName {
		out = append(out, *m)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// LoadScores decodes a flat {"<module>": <number>} JSON map — the minimal
// score-source shape; scorecard-specific adapters are follow-on work.
func LoadScores(b []byte) (map[string]float64, error) {
	var scores map[string]float64
	if err := json.Unmarshal(b, &scores); err != nil {
		return nil, fmt.Errorf("modver: scores file must be a flat {\"module\": number} JSON map: %w", err)
	}
	return scores, nil
}

// JoinScores attaches scores to matching modules and reports how many matched.
func (r *Report) JoinScores(scores map[string]float64) int {
	matched := 0
	for i := range r.Modules {
		if v, ok := scores[r.Modules[i].Name]; ok {
			s := v
			r.Modules[i].Score = &s
			matched++
		}
	}
	return matched
}

// View returns a display-only projection of the report: the modules filtered to
// those whose name has the `only` prefix (empty = every module), sorted by
// `sortKey`, and truncated to the first `top` after sorting (top <= 0 = all).
// The receiver is not mutated — the returned Report carries a fresh Modules
// slice — so the --stamp path, which needs the full name-ordered report, is
// unaffected by a filtered display view. An unknown sortKey is an error so a
// typo fails loud rather than silently falling back to an arbitrary order.
func (r Report) View(only, sortKey string, top int) (Report, error) {
	less, ok := moduleLess(sortKey)
	if !ok {
		return Report{}, fmt.Errorf("modver: unknown sort key %q (want name|rev|date)", sortKey)
	}
	mods := make([]Module, 0, len(r.Modules))
	for _, m := range r.Modules {
		if only == "" || strings.HasPrefix(m.Name, only) {
			mods = append(mods, m)
		}
	}
	sort.SliceStable(mods, func(i, j int) bool { return less(mods[i], mods[j]) })
	if top > 0 && top < len(mods) {
		mods = mods[:top]
	}
	return Report{Head: r.Head, AppVersion: r.AppVersion, Modules: mods}, nil
}

// moduleLess maps a display sort key to a less-func and whether the key is known.
// "name" is ascending (the stable snapshot default); "rev" and "date" are
// DESCENDING so the most-revised / most-recently-touched modules lead — the rows
// an operator scanning for movement wants first, and the ones a --top view should
// keep. Ties fall back to name ascending so the order is total and deterministic.
// LastDate is an ISO-8601 committer date, so a lexicographic compare is
// chronological.
func moduleLess(sortKey string) (func(a, b Module) bool, bool) {
	switch sortKey {
	case "", "name":
		return func(a, b Module) bool { return a.Name < b.Name }, true
	case "rev":
		return func(a, b Module) bool {
			if a.Rev != b.Rev {
				return a.Rev > b.Rev
			}
			return a.Name < b.Name
		}, true
	case "date":
		return func(a, b Module) bool {
			if a.LastDate != b.LastDate {
				return a.LastDate > b.LastDate
			}
			return a.Name < b.Name
		}, true
	default:
		return nil, false
	}
}

// LedgerRow is one appended line of the module-versions ledger.
type LedgerRow struct {
	Schema     string   `json:"schema"`
	TS         string   `json:"ts"`
	Head       string   `json:"head"`
	AppVersion string   `json:"app_version,omitempty"`
	Module     string   `json:"module"`
	Kind       string   `json:"kind"`
	Rev        int      `json:"rev"`
	Version    string   `json:"version"`
	LastCommit string   `json:"last_commit"`
	LastDate   string   `json:"last_date"`
	Score      *float64 `json:"score,omitempty"`
}

// DeltaRows computes the ledger rows a stamp should append: one row per module
// whose rev (or score) moved since its last row in the existing ledger, so the
// ledger grows proportionally to actual change. Unparseable prior lines are
// skipped, not fatal — an append-only ledger a fleet writes will have scars.
func DeltaRows(rep Report, prevLedger []byte, ts string) []LedgerRow {
	type last struct {
		rev   int
		score *float64
	}
	prev := map[string]last{}
	for _, line := range bytes.Split(prevLedger, []byte{'\n'}) {
		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			continue
		}
		var row LedgerRow
		if err := json.Unmarshal(line, &row); err != nil || row.Module == "" {
			continue
		}
		prev[row.Module] = last{rev: row.Rev, score: row.Score}
	}
	var rows []LedgerRow
	for _, m := range rep.Modules {
		if p, ok := prev[m.Name]; ok && p.rev == m.Rev && scoreEq(p.score, m.Score) {
			continue
		}
		rows = append(rows, LedgerRow{
			Schema:     Schema,
			TS:         ts,
			Head:       rep.Head,
			AppVersion: rep.AppVersion,
			Module:     m.Name,
			Kind:       m.Kind,
			Rev:        m.Rev,
			Version:    m.Version(),
			LastCommit: m.LastCommit,
			LastDate:   m.LastDate,
			Score:      m.Score,
		})
	}
	return rows
}

func scoreEq(a, b *float64) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	return *a == *b
}

// AppendLines renders rows as newline-terminated JSONL bytes for appending.
func AppendLines(rows []LedgerRow) ([]byte, error) {
	var buf bytes.Buffer
	for _, r := range rows {
		b, err := json.Marshal(r)
		if err != nil {
			return nil, err
		}
		buf.Write(b)
		buf.WriteByte('\n')
	}
	return buf.Bytes(), nil
}
