package hooks

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// gate_fileadmission.go — the FILE_ADMISSION gate, a port of tools/check_committed_files.py.
// It refuses files that should never be committed: credentials, private-lab subsystems, random
// shell/PowerShell scripts (prefer Go sub-modules/applications), build junk, regenerable logs/temp,
// and oversized blobs. The CLASSIFICATION ORDER is load-bearing (check_committed_files.py _classify
// L127-150) and is reproduced exactly.

// fileAdmissionMaxBytes is the oversized-blob ceiling: a committed file bigger than this is
// refused as build junk / an accidental blob. Raised from the original 10 MiB to 25 MiB (a
// legitimate committed asset — a model card, a fixture corpus, a demo capture — routinely
// clears 10 MiB, and refusing it was a false block, not a caught mistake) and made
// operator-tunable via FAK_MAX_FILE_BYTES. Kept in lockstep with check_committed_files.py
// DEFAULT_MAX_BYTES so the local Go gate and the CI Python checker never disagree.
var fileAdmissionMaxBytes = int64(gateEnvInt("FAK_MAX_FILE_BYTES", 25*1024*1024))

// commitShippedBudget caps the non-Go bytes one staged commit may ADD to paths that ship in the
// root module zip. Sized from the per-commit distribution on main: ordinary commits sit far
// below it, a bulk import sits far above it. Kept in lockstep with check_committed_files.py.
var commitShippedBudget = int64(gateEnvInt("FAK_COMMIT_SHIPPED_BYTES", 256*1024))

// patternReason pairs a path regex with the human-readable reason it is refused —
// the shared shape of the secretFiles and privateOnly classification tables.
type patternReason struct {
	re  *regexp.Regexp
	why string
}

var secretFiles = []patternReason{
	{regexp.MustCompile(`(^|/)secrets/`), "secrets dir — credentials never belong in git; keep them gitignored / in a secret store"},
	{regexp.MustCompile(`\.sa\.json$`), "GCP service-account key (*.sa.json) — never commit a key; rotate it and keep it gitignored"},
	{regexp.MustCompile(`-(sa|gcp)-key\.json$`), "cloud service-account key — never commit a key; rotate it and keep it gitignored"},
}

var privateOnly = []patternReason{
	{regexp.MustCompile(`^(cmd|internal)/[^/]*dgx[^/]*/`), "private lab GPU-server connection subsystem — belongs in the private repo, not the public tree"},
	{regexp.MustCompile(`^cmd/slackgc/`), "private lab Slack-housekeeping tool — belongs in the private repo, not the public tree"},
}

// disallowedScriptExtensions lists script extensions prohibited for new repository tooling.
var disallowedScriptExtensions = []string{".ps1", ".sh", ".bat", ".cmd", ".bash", ".zsh"}

const scriptRefusalReason = "random shell/powershell script (.ps1/.sh/.bat/.cmd) refused: prefer making Go sub-modules (nested modules under tools/<name>/), Go applications (cmd/), or Go leaves (internal/) rather than shell or PowerShell scripts (AGENTS.md: prefer Go tooling/submodules). Port automation to Go."

func isScriptPath(p string) bool {
	low := strings.ToLower(p)
	for _, ext := range disallowedScriptExtensions {
		if strings.HasSuffix(low, ext) {
			return true
		}
	}
	return false
}

func scriptAdmissionReason(p string) string {
	if !isScriptPath(p) {
		return ""
	}
	if grandfatheredScripts[p] {
		return ""
	}
	return scriptRefusalReason
}

// privateByMachine matches a raw per-machine benchmark run drop under
// experiments/benchmark/runs/by-machine/ (or the fak/-nested superrepo form). These are
// regenerable harness output carrying infra tells (instance names/zones, credential paths,
// hostnames, accelerator SKUs, GPU topology); commit 62bed967e made the whole tree
// private-by-default in .gitignore, and this is the commit-time backstop that ignore rule
// cannot provide (a `git add -f` or a silent ignore-line revert would re-clutter the tree).
//
// STAGED-ONLY, on purpose: it is enforced by gateFileAdmission over d.AddedRenamedPaths
// (the --diff-filter=AR new-additions set) and is NOT part of the shared classifyFileWith,
// so the tree twin (gateFileAdmissionTree) never fires it on the ~50 grandfathered evidence
// files already tracked under by-machine/ (gitignore is inert for tracked paths) — CI stays
// green. Mirrors _is_private_bymachine_addition in check_committed_files.py, applied only on
// the staged-additions branch of main(). The ALLOW_STRAY_FILE escape skips the whole gate at
// the runner (cmd/fak/hooks.go), so deliberate promotion of a scrubbed artifact still works.
var privateByMachine = regexp.MustCompile(`^(fak/)?experiments/benchmark/runs/by-machine/`)

const privateByMachineReason = "raw benchmark run drop under experiments/benchmark/runs/by-machine/ is PRIVATE-BY-DEFAULT (regenerable harness output with infra tells) — promote a scrubbed artifact deliberately with ALLOW_STRAY_FILE=1, or keep it gitignored"

// isPrivateByMachineAddition reports whether a NEWLY-ADDED path is a raw by-machine run drop.
// Consulted only on the staged-additions branch (gateFileAdmission), never by the tree twin.
func isPrivateByMachineAddition(p string) bool { return privateByMachine.MatchString(p) }

var hardJunk = []*regexp.Regexp{
	regexp.MustCompile(`(^|/)__pycache__/`),
	regexp.MustCompile(`(^|/)\.pytest_cache/`),
	regexp.MustCompile(`(^|/)\.ruff_cache/`),
	regexp.MustCompile(`(^|/)node_modules/`),
	regexp.MustCompile(`\.(pyc|pyo|class|o|a|obj)$`),
	regexp.MustCompile(`\.(exe|dll|so|dylib)$`),
	regexp.MustCompile(`^coverage$`),
	regexp.MustCompile(`(^|/)coverage\.out$`),
	regexp.MustCompile(`\.coverprofile$`),
	regexp.MustCompile(`(^|/)\.DS_Store$`),
	regexp.MustCompile(`(^|/)Thumbs\.db$`),
	regexp.MustCompile(`\.(swp|swo)$`),
	regexp.MustCompile(`~$`),
}

var softJunk = []*regexp.Regexp{
	regexp.MustCompile(`\.log$`),
	regexp.MustCompile(`\.tmp$`),
	regexp.MustCompile(`(^|/)(report|agent-report)\.json$`),
}

// exemptDataDirs — SOFT_JUNK is allowed under these prefixes (L62), str.startswith semantics.
var exemptDataDirs = []string{"experiments/", "testdata/", "internal/", "fak/experiments/", "fak/testdata/"}

// generatedControlArtifactNames identifies per-run orchestration fuel. Reusable templates under
// .claude/goal-prompts are valid project infrastructure; issue-numbered recovery/fleet prompts
// are receipts for one worker run and belong in private/scratch storage instead of git history.
var generatedControlArtifactNames = regexp.MustCompile(`(?i)(?:^|/)(?:frontdoor-\d+-recovery|resfleet-\d+|resolve-issue-\d+-continuation)\.md$`)

// keepExceptions — exact-path allowlist; skip junk rules but still apply the size cap (L64-66).
var keepExceptions = map[string]bool{"fak/demorace-err.log": true}

func gateFileAdmission(d *StagedDiff) ([]Finding, error) {
	// check_committed_files.py uses --diff-filter=AR; the scan body is shared with the tree twin.
	seen := map[string]bool{}
	var findings []Finding
	for _, p := range d.AddedRenamedPaths {
		if seen[p] {
			continue
		}
		seen[p] = true
		why := ""
		if strings.HasPrefix(strings.ToLower(p), ".claude/goal-prompts/") && generatedControlArtifactNames.MatchString(p) {
			why = "generated one-run .claude goal prompt; park under ignored scratch/private storage or delete after the run"
		} else {
			why = classifyFileWith(d, p)
		}
		// STAGED-ONLY fallback (mirrors main()'s staged branch in check_committed_files.py):
		// a new raw by-machine run drop the shared classifier admits (it is under the
		// experiments/ data dir, so soft-junk rules don't apply) is still refused here. NOT in
		// classifyFileWith, so the tree twin keeps the grandfathered evidence files green.
		if why == "" && isPrivateByMachineAddition(p) {
			why = privateByMachineReason
		}
		if why != "" {
			findings = append(findings, Finding{Gate: "FILE_ADMISSION", File: p, Detail: why})
		}
	}
	if d.Treeish == ":" {
		findings = append(findings, commitShippedBudgetFindings(d)...)
	}
	// The denominator is the DEDUPED added/renamed set — the paths this gate actually
	// classified, not the raw list it was handed (#5602).
	d.NoteCandidates("FILE_ADMISSION", len(seen), "added/renamed path(s) classified")
	return findings, nil
}

// commitShippedBudgetFindings refuses a staged commit whose added non-Go bytes on module-shipped
// paths exceed commitShippedBudget. A merge commit is skipped: each side was gated at its own
// commit. Every git read on this path fails closed, because a commit whose shipped bytes cannot
// be bounded must not pass.
func commitShippedBudgetFindings(d *StagedDiff) []Finding {
	if d.run == nil {
		return nil
	}
	merging, err := mergeInProgress(d)
	if err != nil {
		return budgetUnboundable("read whether a merge is in progress (" + err.Error() + ")")
	}
	if merging {
		return nil
	}
	changes, err := stagedAddedRenamedPaths(d)
	if err != nil {
		return budgetUnboundable("list the staged names (" + err.Error() + ")")
	}
	var goPaths []string
	for _, p := range changes {
		if strings.HasSuffix(p, ".go") {
			goPaths = append(goPaths, p)
		}
	}
	textGo, err := textBlobPaths(d, goPaths)
	if err != nil {
		return budgetUnboundable("check which staged .go files are text (" + err.Error() + ")")
	}
	var candidates []string
	for _, p := range changes {
		if !textGo[p] {
			candidates = append(candidates, p)
		}
	}
	ignored, err := exportIgnoredPaths(d, candidates)
	if err != nil {
		return budgetUnboundable("read export-ignore for the staged paths (" + err.Error() + ")")
	}
	tracked, err := trackedPaths(d)
	if err != nil {
		return budgetUnboundable("list the tracked paths (" + err.Error() + ")")
	}
	nested := nestedModuleDirs(tracked)
	sizes, err := stagedBlobSizes(d)
	if err != nil {
		return budgetUnboundable("size the staged blobs (" + err.Error() + ")")
	}
	type shipped struct {
		path string
		size int64
	}
	var files []shipped
	var unsized []string
	var total int64
	for _, p := range changes {
		if textGo[p] || ignored[p] || inNestedModule(p, nested) {
			continue
		}
		sz, ok := sizes[p]
		if !ok {
			unsized = append(unsized, p)
			continue
		}
		files = append(files, shipped{p, sz})
		total += sz
	}
	if len(unsized) > 0 {
		return budgetUnboundable("read the staged blob size of " + strings.Join(unsized, ", "))
	}
	if total <= commitShippedBudget {
		return nil
	}
	sort.SliceStable(files, func(i, j int) bool { return files[i].size > files[j].size })
	var top []string
	for i := 0; i < len(files) && i < 3; i++ {
		top = append(top, files[i].path+" ("+kib(files[i].size)+" KiB)")
	}
	return []Finding{{Gate: "FILE_ADMISSION", Detail: "commit adds " + kib(total) +
		" KiB of non-Go bytes to module-shipped paths (budget " + kib(commitShippedBudget) +
		" KiB); largest: " + strings.Join(top, ", ") +
		". fix: put evidence under an export-ignored path (.gitattributes export-ignore), " +
		"or commit a pointer (path, sha256, size) instead of the bytes. A pure move of " +
		"already-shipped files can be admitted for this one commit with FAK_COMMIT_SHIPPED_BYTES=" +
		fmt.Sprint(total) + ". The budget is per commit and accepts these limits: splitting bulk " +
		"across commits, overwriting a tracked file, and text .go of any size; the tree-wide " +
		"ratchet (internal/architest TestRootModuleZipBudget) is the backstop."}}
}

// budgetUnboundable is the fail-closed verdict for a budget read that failed: the commit's
// shipped bytes cannot be bounded, so the commit is refused rather than admitted.
func budgetUnboundable(what string) []Finding {
	return []Finding{{Gate: "FILE_ADMISSION", Detail: "cannot " + what +
		"; refusing (fail closed): the commit's shipped bytes cannot be bounded."}}
}

// gitRead runs one git read on the staged repo and treats any non-zero exit as a failure.
func gitRead(d *StagedDiff, args ...string) (string, error) {
	out, code, err := d.run(d.ctx, d.Root, args...)
	if err != nil {
		return "", err
	}
	if code != 0 {
		return "", fmt.Errorf("git %s exited %d", args[0], code)
	}
	return out, nil
}

// mergeInProgress reports whether MERGE_HEAD exists. Anything but a clean answer is an error, so
// the budget never runs on a guess about what kind of commit this is.
func mergeInProgress(d *StagedDiff) (bool, error) {
	out, err := gitRead(d, "rev-parse", "--git-path", "MERGE_HEAD")
	if err != nil {
		return false, err
	}
	p := strings.TrimSpace(out)
	if p == "" {
		return false, fmt.Errorf("rev-parse printed no MERGE_HEAD path")
	}
	if !filepath.IsAbs(p) {
		p = filepath.Join(d.Root, p)
	}
	_, statErr := os.Stat(p)
	if statErr == nil {
		return true, nil
	}
	if os.IsNotExist(statErr) {
		return false, nil
	}
	return false, statErr
}

// stagedAddedRenamedPaths lists the added and renamed paths NUL-delimited, so names come back
// raw: a C-quoted name would miss every size lookup under core.quotepath. A rename is billed at
// its new path's full size, exactly like an addition.
func stagedAddedRenamedPaths(d *StagedDiff) ([]string, error) {
	out, err := gitRead(d, "diff", "--cached", "-z", "--name-status", "-M", "--diff-filter=AR")
	if err != nil {
		return nil, err
	}
	fields := strings.Split(out, "\x00")
	var paths []string
	for i := 0; i < len(fields) && fields[i] != ""; {
		n := 2
		if strings.HasPrefix(fields[i], "R") {
			n = 3
		}
		if i+n > len(fields) {
			return nil, fmt.Errorf("malformed name-status listing")
		}
		paths = append(paths, fields[i+n-1])
		i += n
	}
	return paths, nil
}

// trackedPaths lists the index NUL-delimited. Nested go.mod detection depends on it, so a
// failed listing must refuse rather than leave nested modules unseen.
func trackedPaths(d *StagedDiff) ([]string, error) {
	out, err := gitRead(d, "ls-files", "-z")
	if err != nil {
		return nil, err
	}
	var paths []string
	for _, p := range strings.Split(out, "\x00") {
		if p != "" {
			paths = append(paths, p)
		}
	}
	return paths, nil
}

// textBlobPaths returns the paths whose staged blob is text: no NUL byte in the first 8000 bytes,
// the test git grep -I applies. A path git grep does not answer is absent from the map and so is
// billed as binary; a failed batch is an error.
func textBlobPaths(d *StagedDiff, paths []string) (map[string]bool, error) {
	text := map[string]bool{}
	const batch = 100
	for start := 0; start < len(paths); start += batch {
		end := min(start+batch, len(paths))
		// The tree-ish must follow the options: git reads every argument after it as a revision.
		args := append([]string{"--literal-pathspecs", "grep", "-I", "-l", "-z", "-F", "-e", "", "--cached", "--"}, paths[start:end]...)
		out, code, err := d.run(d.ctx, d.Root, args...)
		if err != nil {
			return nil, err
		}
		if code != 0 && code != 1 {
			return nil, fmt.Errorf("git grep exited %d", code)
		}
		for _, n := range strings.Split(out, "\x00") {
			if n != "" {
				text[n] = true
			}
		}
	}
	return text, nil
}

// exportIgnoredPaths asks the index which paths carry export-ignore; those never reach
// `git archive`, so never the module zip. --cached reads the .gitattributes being committed, not
// an uncommitted worktree edit. -z is required: without it git C-quotes non-ASCII names, and the
// raw-name lookup misses them.
func exportIgnoredPaths(d *StagedDiff, paths []string) (map[string]bool, error) {
	ignored := map[string]bool{}
	const batch = 200
	for start := 0; start < len(paths); start += batch {
		end := min(start+batch, len(paths))
		args := append([]string{"check-attr", "-z", "--cached", "export-ignore", "--"}, paths[start:end]...)
		out, err := gitRead(d, args...)
		if err != nil {
			return nil, err
		}
		// -z emits NUL-terminated triples: path, attribute, value.
		fields := strings.Split(out, "\x00")
		for i := 0; i+2 < len(fields); i += 3 {
			if fields[i+1] == "export-ignore" && fields[i+2] == "set" {
				ignored[fields[i]] = true
			}
		}
	}
	return ignored, nil
}

// stagedBlobSizes maps each staged path to the byte size of the blob the commit ships. It writes
// the index as a tree and lists that tree once, so the cost is two git processes for the whole
// commit, not one per file. Gitlinks ship no bytes and map to 0; a path absent from the map could
// not be sized.
func stagedBlobSizes(d *StagedDiff) (map[string]int64, error) {
	tree, err := gitRead(d, "write-tree")
	if err != nil {
		return nil, err
	}
	listing, err := gitRead(d, "ls-tree", "-r", "-l", "-z", strings.TrimSpace(tree))
	if err != nil {
		return nil, err
	}
	sizes := map[string]int64{}
	for _, rec := range strings.Split(listing, "\x00") {
		head, path, ok := strings.Cut(rec, "\t")
		if !ok {
			continue
		}
		fields := strings.Fields(head)
		if len(fields) != 4 {
			continue
		}
		switch fields[1] {
		case "commit":
			sizes[path] = 0
		case "blob":
			if n, perr := strconv.ParseInt(fields[3], 10, 64); perr == nil && n >= 0 {
				sizes[path] = n
			}
		}
	}
	return sizes, nil
}

// nestedModuleDirs returns the directory prefixes of nested go.mod files. A nested module is
// its own module, so its tree is not part of the root module zip.
func nestedModuleDirs(paths []string) []string {
	var dirs []string
	for _, p := range paths {
		if strings.HasSuffix(p, "/go.mod") {
			dirs = append(dirs, strings.TrimSuffix(p, "go.mod"))
		}
	}
	return dirs
}

func inNestedModule(p string, dirs []string) bool {
	for _, dir := range dirs {
		if strings.HasPrefix(p, dir) {
			return true
		}
	}
	return false
}

func startsWithAny(s string, prefixes []string) bool {
	for _, p := range prefixes {
		if len(s) >= len(p) && s[:len(p)] == p {
			return true
		}
	}
	return false
}

// largeFileMsg / oversizedBlobMsg — the two DISTINCT wordings the Python keeps (KEEP_EXCEPTIONS
// path says "large file", the general path says "oversized blob").
func largeFileMsg(sz int64) string {
	return "large file (" + kib(sz) + " KiB > " + kib(fileAdmissionMaxBytes) + " KiB)"
}
func oversizedBlobMsg(sz int64) string {
	return "oversized blob (" + kib(sz) + " KiB > " + kib(fileAdmissionMaxBytes) + " KiB)"
}

func kib(b int64) string { return itoa(b / 1024) }
