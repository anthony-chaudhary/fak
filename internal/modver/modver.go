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

// ReasonModuleRevStale is the closed refusal token emitted when a recalled
// module analysis is pinned to an older module revision than trunk (dos.toml [reasons.MODULE_REV_STALE]).
const ReasonModuleRevStale = "MODULE_REV_STALE"

// Module is one versioned unit: an internal/<leaf> package, a cmd/<dir> binary,
// a .github/workflows/<file> CI workflow, a tools/<family> script, an
// examples/<file>.json policy manifest, a .claude/skills/<name> agent skill, or a
// docs/ prose page (a top-level docs/<file>.md or a docs/<dir> section).
type Module struct {
	Name       string   `json:"module"` // e.g. "internal/modver", "cmd/fak", ".github/workflows/ci.yml", "tools/account_probe", "examples/repo-guard-policy.json", ".claude/skills/commit-clean", "docs/architecture.md", "docs/fak"
	Kind       string   `json:"kind"`   // "internal" | "cmd" | "workflow" | "tools" | "policy" | "skill" | "docs"
	Rev        int      `json:"rev"`    // distinct commits touching the module
	LastCommit string   `json:"last_commit"`
	LastDate   string   `json:"last_date"` // committer date (ISO) of the last touch
	Score      *float64 `json:"score,omitempty"`
	// ScoreProvenance labels where a joined Score came from — one of the closed
	// set witnessed|observed|modeled, or "" when the score carries no label.
	// Empty is deliberately NOT treated as "witnessed": an unlabeled score must
	// stay visibly distinct so a modeled score never masquerades as a witnessed
	// one when the series is quoted (#2498).
	ScoreProvenance string `json:"score_provenance,omitempty"`
}

// Score provenance is a closed set: the origin-strength of a joined score, so a
// modeled estimate never masquerades as a witnessed measurement in a trend
// chart. An unlabeled score carries the empty string, never a defaulted label.
const (
	ProvenanceWitnessed = "witnessed" // measured first-hand from a real run/artifact
	ProvenanceObserved  = "observed"  // seen indirectly, not first-hand witnessed
	ProvenanceModeled   = "modeled"   // estimated/predicted, not measured
)

// knownProvenance is the closed set LoadScores accepts in the extended shape;
// "" (unlabeled) is also allowed but is tracked separately since it is not a
// positive label.
var knownProvenance = map[string]bool{
	ProvenanceWitnessed: true,
	ProvenanceObserved:  true,
	ProvenanceModeled:   true,
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
// (each workflow file is its own module), the tools/ script keyspace (each
// top-level script family is a module), and the examples/ policy-manifest
// keyspace (each top-level examples/<file>.json is a module — see moduleOf), and
// the .claude/skills/ agent-skill keyspace (each skill directory is a module),
// and the docs/ prose keyspace (each top-level page and each docs/<dir> section
// is a module — #2460).
var trackedRoots = []string{"internal", "cmd", ".github/workflows", "tools", "examples", ".claude/skills", "docs"}

// Snapshot computes the module-version report for the repo at dir: one
// `git ls-files` to bound the LIVE module set, one `git log --name-only`
// pass over the history to derive each module's rev and last touch.
func Snapshot(ctx context.Context, dir string, run Runner) (Report, error) {
	run = gitRunner(run)
	head, err := headShort(ctx, dir, run)
	if err != nil {
		return Report{}, err
	}
	live, logOut, err := liveAndLog(ctx, dir, run, "")
	if err != nil {
		return Report{}, err
	}
	modules := parseLog(logOut, live)
	appVer := ""
	if v, ok := appversion.FromDir(dir); ok {
		appVer = v
	}
	return Report{
		Head:       head,
		AppVersion: appVer,
		Modules:    modules,
	}, nil
}

// SnapshotAt computes the same derived module-version report as Snapshot, but
// pins both the live-file set and the history walk to ref. This is the
// historical provenance seam for receipts that must prove which module
// revision existed at an affected, fixing, tested, or released commit.
func SnapshotAt(ctx context.Context, dir string, run Runner, ref string) (Report, error) {
	run = gitRunner(run)
	ref = strings.TrimSpace(ref)
	if ref == "" {
		return Report{}, fmt.Errorf("modver: ref is required")
	}
	headOut, err := run(ctx, dir, "rev-parse", "--short=8", ref+"^{commit}")
	if err != nil {
		return Report{}, err
	}
	lsArgs := append([]string{"ls-tree", "-r", "-z", "--name-only", ref, "--"}, trackedRoots...)
	lsOut, err := run(ctx, dir, lsArgs...)
	if err != nil {
		return Report{}, err
	}
	logArgs := append([]string{"log", "--no-merges", "--pretty=format:%x1e%h%x09%cI", "--name-only", ref, "--"}, trackedRoots...)
	logOut, err := run(ctx, dir, logArgs...)
	if err != nil {
		return Report{}, err
	}
	modules := parseLog(logOut, liveModules(lsOut))
	return Report{Head: strings.TrimSpace(string(headOut)), Modules: modules}, nil
}

// gitRunner resolves the Runner seam's nil default — "a nil Runner means real git" is
// the contract every modver entry point (Snapshot, Ghosts, DriftSnapshot) opens with,
// and this is the one place that substitution happens.
func gitRunner(run Runner) Runner {
	if run == nil {
		return RealRunner
	}
	return run
}

// headShort resolves HEAD's short (8-char) sha through the Runner seam — the boundary
// stamp both the module report and the drift readout are pinned to. The trim is part of
// the contract: callers store the returned string directly.
func headShort(ctx context.Context, dir string, run Runner) (string, error) {
	out, err := run(ctx, dir, "rev-parse", "--short=8", "HEAD")
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

// liveAndLog runs the two git calls every modver pass is bounded by, and is why the
// three passes cannot drift apart on what a module or a revision IS: one `git ls-files`
// fixing the LIVE module set at HEAD, then one `git log --no-merges --name-only` walk
// over the same trackedRoots, whose raw bytes the caller folds its own way (parseLog,
// parseGhosts, Drift).
//
// rev selects the walk's range: "" for the whole history (Snapshot's rev-since-forever
// and Ghosts' tombstones) or "<tag>..HEAD" for the tag-bounded drift readout, where each
// module's rev IS its commit count since the last tag. The range selector is placed
// BEFORE the "--" separator because git parses everything after "--" as a pathspec.
//
// --no-merges pins the rev semantics (#2475): rev counts distinct NON-MERGE commits
// touching the module. A merge commit carries no authored module work — the file changes
// live in the non-merge commits it joins, each of which git's DAG walk lists exactly
// once, so there is no double-counting, and a merge is likewise never mistaken for a
// ghost's deletion commit. git suppresses merge diffs from --name-only by default, but
// that default is overridable (--diff-merges, log.diffMerges); --no-merges makes the
// exclusion explicit and independent of git config, so rev is stable across an in-place
// trunk merge. Deliberately NOT --first-parent: work that reaches the trunk through a
// merge lives off the first-parent line, and --first-parent would silently undercount it
// while the merge commit itself contributes no files.
func liveAndLog(ctx context.Context, dir string, run Runner, rev string) (map[string]bool, []byte, error) {
	lsArgs := append([]string{"ls-files", "-z", "--"}, trackedRoots...)
	lsOut, err := run(ctx, dir, lsArgs...)
	if err != nil {
		return nil, nil, err
	}
	logArgs := []string{"log", "--no-merges", "--pretty=format:%x1e%h%x09%cI", "--name-only"}
	if rev != "" {
		logArgs = append(logArgs, rev)
	}
	logArgs = append(logArgs, "--")
	logArgs = append(logArgs, trackedRoots...)
	logOut, err := run(ctx, dir, logArgs...)
	if err != nil {
		return nil, nil, err
	}
	return liveModules(lsOut), logOut, nil
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
// script plus its test (see toolsFamily). The examples/ policy-manifest keyspace
// is file-keyed and flat like the workflow keyspace: each top-level manifest
// (examples/<file>.json) is its own module, since a policy's unit of behavior is
// the file; nested examples/<demo>/… paths are runnable demos and their
// fixtures, not deployable capability-floor manifests, so they are excluded.
// The .claude/skills/ agent-skill keyspace is directory-keyed like internal/ and
// cmd/: each .claude/skills/<name>/… path is the module .claude/skills/<name>,
// since a skill's unit of behavior is its whole directory — the SKILL.md steering
// text plus any helper script it ships move together, and a skill is also the
// repo's slash-command definition, so the same key space versions both. Other
// .claude/ subtrees (settings, goal-prompts, the memory mirror) are not skill
// definitions and belong to no module.
// The docs/ prose keyspace (#2460) is the one hybrid, because docs/ is itself a
// hybrid: a top-level page (docs/<file>.md) stands alone, so it is file-keyed
// like the workflow and policy keyspaces, while a docs/<dir>/… page belongs to a
// section that is revised as a unit, so it is directory-keyed like internal/ and
// cmd/ — docs/fak/edge-quickstart.md is the module docs/fak, at any nesting
// depth. ONLY .md prose counts: docs/ also carries generated data and site
// config (the .jsonl nightrun ledgers — including this package's own
// module-versions.jsonl — plus .json, .txt, .svg, _config.yml), and versioning
// those would make a doc "revised" every time a nightly job appended a row.
// Excluding them keeps rev a statement about PROSE, and keeps the ledger
// convergent: stamping the ledger cannot bump the module the ledger lives in.
// Files sitting directly under a directory root (no module directory) belong to
// no module — including .claude/skills/README.md and .claude/skills/.gitignore,
// which document and configure the keyspace rather than steer an agent — as do
// nested paths under .github/workflows/ (GitHub Actions does not run workflows in
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
	case "examples":
		// Flat, file-keyed policy-manifest keyspace: only top-level
		// examples/<file>.json count (len == 2). Nested examples/<demo>/… paths
		// are runnable demos and their fixtures, not deployable manifests —
		// excluded, the same flat rule as tools/.
		if len(parts) == 2 && strings.HasSuffix(parts[1], ".json") {
			return path, "policy", true
		}
	case ".claude":
		// Directory-keyed agent-skill keyspace: .claude/skills/<name>/… is one
		// module, the same rule internal/<leaf> and cmd/<dir> use. len >= 4
		// requires a file INSIDE a skill directory, so a file sitting directly
		// under .claude/skills/ (len == 3: README.md, .gitignore) belongs to no
		// module. Only the skills/ subtree is a keyspace — .claude/settings.json,
		// .claude/goal-prompts/, and .claude/memory/ are not skill definitions.
		if parts[1] == "skills" && len(parts) >= 4 {
			return ".claude/skills/" + parts[2], "skill", true
		}
	case "docs":
		// Hybrid prose keyspace, .md only (see the doc comment): a top-level page
		// is its own module; a page in a section keys the section directory at any
		// depth. Non-.md paths (generated ledgers, site config, images) are data,
		// not prose, and belong to no module.
		if !strings.HasSuffix(parts[len(parts)-1], ".md") {
			return "", "", false
		}
		if len(parts) == 2 {
			return path, "docs", true
		}
		return "docs/" + parts[1], "docs", true
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

// ModulesForPaths maps a set of repo-relative paths to the deduped, sorted set of
// module keys those paths touch — the modules whose rev a commit of exactly these
// paths would bump. It is the cheap, git-free projection of the same keyspace
// moduleOf defines: no Snapshot, no `git log`, no history walk, since which module
// a path belongs to is a pure function of the path string. That is what lets the
// commit-preview advisory (#2495) name the bumped modules within the preview's
// latency budget — it costs a string split per path, nothing more. Paths under no
// tracked keyspace (a file directly under a root, a non-.md docs/ data file,
// nested tools/ or examples/ fixtures) bump no module and are simply skipped; an
// empty or all-untracked path set returns nil.
//
// This is also the RESERVED docfreshrsi integration seam (#2460) — reserved, not
// yet wired: nothing in internal/docfreshrsi calls this today. The intended shape
// is that a docs-freshness pass holding a corpus key ("docs/fak/edge-quickstart.md")
// would call ModulesForPaths to resolve it to its module key ("docs/fak"), then read
// that module's rev from the fak-module-versions/1 ledger
// (docs/nightrun/module-versions.jsonl), making staleness a rev delta against the
// ledger rather than an ad-hoc signal. The mapping is git-free and pure, so such a
// pass would pay no history walk for it.
func ModulesForPaths(paths []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, p := range paths {
		if name, _, ok := moduleOf(p); ok && !seen[name] {
			seen[name] = true
			out = append(out, name)
		}
	}
	sort.Strings(out)
	return out
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

// ScoreEntry is one external score with its optional provenance label — the
// value shape LoadScores yields and JoinScores attaches. Provenance is "" when
// the source did not label the score.
type ScoreEntry struct {
	Score      float64
	Provenance string
}

// LoadScores decodes the scores file, accepting BOTH the flat
// {"<module>": <number>} shape and the extended
// {"<module>": {"score": <number>, "provenance": "<label>"}} shape, mixed
// freely per module. The flat shape is preserved unchanged (back-compat); the
// extended shape carries a provenance label from the closed set
// witnessed|observed|modeled. An unknown non-empty provenance is rejected so a
// typo cannot flow into the ledger wearing the authority of a real label; an
// unlabeled score keeps an empty provenance, never a defaulted "witnessed".
func LoadScores(b []byte) (map[string]ScoreEntry, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(b, &raw); err != nil {
		return nil, fmt.Errorf("modver: scores file must be a JSON map of module -> number or {\"score\":number,\"provenance\":label}: %w", err)
	}
	scores := make(map[string]ScoreEntry, len(raw))
	for name, msg := range raw {
		// Flat shape: a bare number. Try it first — a number never decodes into
		// the extended struct and vice versa, so the order disambiguates cleanly.
		var f float64
		if err := json.Unmarshal(msg, &f); err == nil {
			scores[name] = ScoreEntry{Score: f}
			continue
		}
		var ext struct {
			Score      float64 `json:"score"`
			Provenance string  `json:"provenance"`
		}
		if err := json.Unmarshal(msg, &ext); err != nil {
			return nil, fmt.Errorf("modver: score for %q must be a number or {\"score\":number,\"provenance\":label}: %w", name, err)
		}
		if ext.Provenance != "" && !knownProvenance[ext.Provenance] {
			return nil, fmt.Errorf("modver: score for %q has unknown provenance %q (want witnessed|observed|modeled)", name, ext.Provenance)
		}
		scores[name] = ScoreEntry{Score: ext.Score, Provenance: ext.Provenance}
	}
	return scores, nil
}

// JoinScores attaches scores (and their provenance) to matching modules and
// reports how many matched.
func (r *Report) JoinScores(scores map[string]ScoreEntry) int {
	matched := 0
	for i := range r.Modules {
		if v, ok := scores[r.Modules[i].Name]; ok {
			s := v.Score
			r.Modules[i].Score = &s
			r.Modules[i].ScoreProvenance = v.Provenance
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
	// ScoreProvenance is the label of Score when one was joined — an additive
	// field on the fak-module-versions/1 schema (omitempty keeps existing
	// flat-score rows byte-identical; a breaking row change would be a /2).
	ScoreProvenance string `json:"score_provenance,omitempty"`
}

// parseLedgerRows decodes an append-only module-versions ledger into its rows in
// file order, skipping blank lines and any scarred (unparseable or module-less)
// entry — an append-only ledger a fleet writes will have scars.
func parseLedgerRows(ledger []byte) []LedgerRow {
	var rows []LedgerRow
	for _, line := range bytes.Split(ledger, []byte{'\n'}) {
		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			continue
		}
		var row LedgerRow
		if err := json.Unmarshal(line, &row); err != nil || row.Module == "" {
			continue
		}
		rows = append(rows, row)
	}
	return rows
}

// DeltaRows computes the ledger rows a stamp should append: one row per module
// whose rev (or score) moved since its last row in the existing ledger, so the
// ledger grows proportionally to actual change. Unparseable prior lines are
// skipped, not fatal — an append-only ledger a fleet writes will have scars.
func DeltaRows(rep Report, prevLedger []byte, ts string) []LedgerRow {
	type last struct {
		rev        int
		score      *float64
		provenance string
	}
	prev := map[string]last{}
	for _, row := range parseLedgerRows(prevLedger) {
		prev[row.Module] = last{rev: row.Rev, score: row.Score, provenance: row.ScoreProvenance}
	}
	var rows []LedgerRow
	for _, m := range rep.Modules {
		// A provenance change alone is a real ledger movement — relabeling a
		// score modeled->witnessed must append a row so the corrected label is
		// witnessed in the trend, even when the numeric score is unchanged.
		if p, ok := prev[m.Name]; ok && p.rev == m.Rev && scoreEq(p.score, m.Score) && p.provenance == m.ScoreProvenance {
			continue
		}
		rows = append(rows, LedgerRow{
			Schema:          Schema,
			TS:              ts,
			Head:            rep.Head,
			AppVersion:      rep.AppVersion,
			Module:          m.Name,
			Kind:            m.Kind,
			Rev:             m.Rev,
			Version:         m.Version(),
			LastCommit:      m.LastCommit,
			LastDate:        m.LastDate,
			Score:           m.Score,
			ScoreProvenance: m.ScoreProvenance,
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
