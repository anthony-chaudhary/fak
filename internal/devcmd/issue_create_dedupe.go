package devcmd

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/issuededup"
	"github.com/anthony-chaudhary/fak/internal/issuefanout"
)

// issueCreateDedupeReason is the typed refusal token a write-time near-duplicate
// refusal carries. It is deliberately distinct from the discoverability and
// scrub refusal tokens so a caller (or an agent parsing stderr) can tell a
// duplicate apart from a malformed or leaking draft.
const issueCreateDedupeReason = "ISSUE_NEAR_DUPLICATE"

// issueCreateDedupeFunc fetches the bounded open backlog for the write-time
// near-duplicate gate. It returns the raw `gh issue list --json
// number,title,body` bytes; ParseBacklog does the decoding. It is injectable so
// tests can supply a fake backlog without ever invoking real gh - the same
// seam shape as issueCreateRunner.
type issueCreateDedupeFunc func(cap int) ([]byte, error)

// issueCreateDedupeFetcher is the default fetcher: a read-only, bounded, 60s
// `gh issue list --state open --json number,title,body` - the same shape as
// fetchIssueDedupBacklog (issue_dedup.go:92), minus labels (the write-time gate
// never reads them) and with the cap threaded from --dedupe-cap.
var issueCreateDedupeFetcher issueCreateDedupeFunc = fetchIssueCreateDedupeBacklog

// issueCreateDedupeJSON is the --json evidence block for the near-duplicate
// gate. checked says whether the gate actually ran; scanned/cap record the
// bounded backlog read; matches carry the ranked twin verdicts; refused is the
// one-bit gate outcome; error carries the typed refusal message on a refusal.
type issueCreateDedupeJSON struct {
	Checked bool                 `json:"checked"`
	Scanned int                  `json:"scanned"`
	Cap     int                  `json:"cap"`
	Matches []issuededup.Verdict `json:"matches,omitempty"`
	Refused bool                 `json:"refused"`
	Error   string               `json:"error,omitempty"`
}

// fetchIssueCreateDedupeBacklog is the real write-time backlog read. It is
// offline-safe by construction: read-only gh, one bounded page, no writes
// anywhere. The caller decides what a failure means - the create gate FAILS
// OPEN, so an offline host can still file.
func fetchIssueCreateDedupeBacklog(cap int) ([]byte, error) {
	if cap <= 0 {
		cap = issuefanout.DefaultDedupeCap
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "gh", "issue", "list", "--state", "open",
		"--limit", strconv.Itoa(cap), "--json", "number,title,body")
	configureDispatchHelperCommand(cmd)
	b, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("gh issue list failed: %w (%s)", err, strings.TrimSpace(string(b)))
	}
	return b, nil
}

// issueCreateRunDedupeGate is the write-time near-duplicate pre-check. It runs
// only when armed (--dedupe-checked), the body came from --body/--body-file, and
// the caller is not bypassing the contract with --raw-body. It fetches the
// bounded open backlog, indexes it, and Checks the candidate title+body.
//
// Policy: warn-then-refuse. Any verdict at or above the effective threshold is a
// printed warning; when not --dedupe-warn-only it is a typed refusal
// (issueCreateDedupeReason) that returns exit 3. The gate FAILS OPEN on a fetch
// error - a warning, then continue - so a legitimate filing is never blocked
// offline, mirroring fanout --live's read-only, best-effort backlog scan.
//
// It returns (code, refused). code is only meaningful when refused is true.
func issueCreateRunDedupeGate(stdout, stderr io.Writer, result *issueCreateResult, fetch issueCreateDedupeFunc, armed bool, cap int, threshold float64, warnOnly bool, title, body string, asJSON bool) (int, bool) {
	if !armed {
		return 0, false
	}
	effectiveCap := cap
	if effectiveCap <= 0 {
		effectiveCap = issuefanout.DefaultDedupeCap
	}
	effectiveThreshold := threshold
	if effectiveThreshold <= 0 {
		effectiveThreshold = issuededup.DefaultThreshold
	}
	block := issueCreateDedupeJSON{Checked: true, Cap: effectiveCap}
	result.Dedupe = &block

	if fetch == nil {
		fetch = issueCreateDedupeFetcher
	}
	raw, err := fetch(effectiveCap)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev issue create: near-duplicate check skipped (failing open): %v\n", err)
		block.Error = err.Error()
		return 0, false
	}
	issues, err := issuededup.ParseBacklog(raw)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev issue create: near-duplicate check skipped (failing open): %v\n", err)
		block.Error = err.Error()
		return 0, false
	}
	block.Scanned = len(issues)

	verdicts := issuededup.NewIndex(issues).Check(
		issuededup.Candidate{Title: title, Body: body},
		issuededup.DefaultTopK, effectiveThreshold)
	if len(verdicts) == 0 {
		return 0, false
	}
	block.Matches = verdicts
	top := verdicts[0]
	msg := fmt.Sprintf("%s: candidate is a near-twin of open issue #%d (similarity %.2f, matched on %s): %s",
		issueCreateDedupeReason, top.IssueNumber, top.Similarity, top.MatchedOn, top.Title)

	if warnOnly {
		fmt.Fprintf(stderr, "fak-dev issue create: WARNING near-duplicate: %s (continuing: --dedupe-warn-only)\n", msg)
		return 0, false
	}
	fmt.Fprintf(stderr, "fak-dev issue create: %s\n", msg)
	fmt.Fprintln(stderr, "  (repair: reference the matched issue, or pass --dedupe-warn-only to file anyway)")
	block.Refused = true
	block.Error = msg
	if asJSON {
		result.OK = false
		result.Error = msg
		encodeJSONOrFail(stdout, stderr, *result, "fak-dev issue create")
	}
	return 3, true
}

// issueCreateURLRE parses the issue URL gh prints after `gh issue create`:
// the number from /issues/<n> and the owner/repo from the github.com path.
var issueCreateURLRE = regexp.MustCompile(`https?://github\.com/([^/\s]+)/([^/\s]+)/issues/([0-9]+)`)

// issueCreateFiledMarkerRE matches an existing filed-marker line so write-back
// replaces it in place instead of appending a second one.
var issueCreateFiledMarkerRE = regexp.MustCompile(`(?m)^github_issue:.*$`)

// issueCreateWriteBackFiledMarker stamps the source draft with the filed issue
// identity. On a successful create whose body came from --body-file, it appends
// (or replaces) a `github_issue: <owner/repo>#<n>` line so a filed draft is
// distinguishable from an unfiled one. A repo override wins over the URL path.
// Write-back failure must NOT fail the create - the issue is already filed - so
// it only warns to stderr and returns nil error.
func issueCreateWriteBackFiledMarker(stderr io.Writer, bodyFile, repoOverride, url string) {
	if strings.TrimSpace(bodyFile) == "" {
		return
	}
	m := issueCreateURLRE.FindStringSubmatch(strings.TrimSpace(url))
	if m == nil {
		fmt.Fprintf(stderr, "fak-dev issue create: cannot parse issue URL for filed-marker write-back: %q\n", url)
		return
	}
	owner, name, number := m[1], m[2], m[3]
	if strings.TrimSpace(repoOverride) != "" {
		if parts := strings.SplitN(strings.TrimSpace(repoOverride), "/", 2); len(parts) == 2 {
			owner, name = parts[0], parts[1]
		}
	}
	marker := fmt.Sprintf("github_issue: %s/%s#%s", owner, name, number)
	b, err := os.ReadFile(bodyFile)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev issue create: filed-marker write-back skipped for %s: %v\n", bodyFile, err)
		return
	}
	content := string(b)
	var updated string
	if issueCreateFiledMarkerRE.MatchString(content) {
		updated = issueCreateFiledMarkerRE.ReplaceAllString(content, marker)
	} else {
		updated = strings.TrimRight(content, "\r\n") + "\n\n" + marker + "\n"
	}
	if err := os.WriteFile(bodyFile, []byte(updated), 0o644); err != nil {
		fmt.Fprintf(stderr, "fak-dev issue create: filed-marker write-back skipped for %s: %v\n", bodyFile, err)
	}
}
