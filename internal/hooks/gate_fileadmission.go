package hooks

import "regexp"

// gate_fileadmission.go — the FILE_ADMISSION gate, a port of tools/check_committed_files.py.
// It refuses files that should never be committed: credentials, private-lab subsystems, build
// junk, regenerable logs/temp, and oversized blobs. The CLASSIFICATION ORDER is load-bearing
// (check_committed_files.py _classify L127-150) and is reproduced exactly.

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

// keepExceptions — exact-path allowlist; skip junk rules but still apply the size cap (L64-66).
var keepExceptions = map[string]bool{"fak/demorace-err.log": true}

func gateFileAdmission(d *StagedDiff) ([]Finding, error) {
	// check_committed_files.py uses --diff-filter=AR; the scan body is shared with the tree twin.
	return classifyPathsFindings(d, d.AddedRenamedPaths), nil
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
