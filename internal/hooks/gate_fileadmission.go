package hooks

import (
	"regexp"
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
	// The denominator is the DEDUPED added/renamed set — the paths this gate actually
	// classified, not the raw list it was handed (#5602).
	d.NoteCandidates("FILE_ADMISSION", len(seen), "added/renamed path(s) classified")
	return findings, nil
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
