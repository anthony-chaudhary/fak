package hooks

import "regexp"

// gate_fileadmission.go — the FILE_ADMISSION gate, a port of tools/check_committed_files.py.
// It refuses files that should never be committed: credentials, private-lab subsystems, build
// junk, regenerable logs/temp, and oversized blobs. The CLASSIFICATION ORDER is load-bearing
// (check_committed_files.py _classify L127-150) and is reproduced exactly.

const fileAdmissionMaxBytes = 10 * 1024 * 1024 // DEFAULT_MAX_BYTES (10 MiB), L37

var secretFiles = []struct {
	re  *regexp.Regexp
	why string
}{
	{regexp.MustCompile(`(^|/)secrets/`), "secrets dir — credentials never belong in git; keep them gitignored / in a secret store"},
	{regexp.MustCompile(`\.sa\.json$`), "GCP service-account key (*.sa.json) — never commit a key; rotate it and keep it gitignored"},
	{regexp.MustCompile(`-(sa|gcp)-key\.json$`), "cloud service-account key — never commit a key; rotate it and keep it gitignored"},
}

var privateOnly = []struct {
	re  *regexp.Regexp
	why string
}{
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
