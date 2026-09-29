package naivecontrol

import (
	"context"
	"fmt"
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/assumecheck"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// GitRunner runs one git subcommand in dir and returns (stdout, exit code, err).
// err is non-nil only when git could not run; a non-zero exit is reported in code.
// It is the repo-wide shape (assumecheck.Runner, witness.Runner,
// commitlifecycle.GitRunner), so one fake serves every seam.
type GitRunner func(ctx context.Context, dir string, args ...string) (stdout string, code int, err error)

// ClaimSource says where a claimed commit came from. Neither source is evidence;
// both are verified the same way.
type ClaimSource string

const (
	// SourceSelfReport is a SHA the arm reported about itself: a worker or harvest
	// receipt on the naive side, a .witness sidecar on the orchestrated side.
	SourceSelfReport ClaimSource = "self_report"
	// SourceGitLog is a SHA this package found by scanning base..HEAD for a
	// subject citing the pick, the same binding key the orchestrated sweep uses.
	SourceGitLog ClaimSource = "git_log"
)

// Claim is one assertion that SHA resolves Issue.
type Claim struct {
	Issue  int         `json:"issue"`
	SHA    string      `json:"sha"`
	Source ClaimSource `json:"source,omitempty"`
}

// Verdict is the git-derived outcome for one claim.
type Verdict string

const (
	// VerdictLanded: the commit exists, is reachable from HEAD, and is not reverted.
	VerdictLanded Verdict = "LANDED"
	// VerdictNotLanded: git answered no. The SHA is malformed, names no commit (a
	// fabricated claim), is not reachable from HEAD, or was reverted.
	VerdictNotLanded Verdict = "NOT_LANDED"
	// VerdictUnverifiable: git could not answer. Never counted either way.
	VerdictUnverifiable Verdict = "UNVERIFIABLE"
	// VerdictOffPick: the claim names an issue this arm was not offered, so it is
	// outside the arm's work and is not verified or counted.
	VerdictOffPick Verdict = "OFF_PICK"
)

// CommitCheck is one verified claim as recorded in a row.
type CommitCheck struct {
	Issue    int         `json:"issue"`
	SHA      string      `json:"sha"`
	Resolved string      `json:"resolved,omitempty"`
	Source   ClaimSource `json:"source"`
	Verdict  Verdict     `json:"verdict"`
	Detail   string      `json:"detail,omitempty"`
}

// Pick is one issue the arm handed to a worker, and the trunk SHA the work started
// from. Base is what lets this package prove a pick did NOT ship: with no base there
// is nothing to scan, so an unshipped pick stays unknown.
type Pick struct {
	Issue int    `json:"issue"`
	Base  string `json:"base_sha,omitempty"`
}

// Verification is the git evidence for one run. Head is the SHA every ancestry check
// ran against; "" means git could not resolve it and nothing below is evidence.
type Verification struct {
	Head    string
	HeadErr string
	Checks  []CommitCheck
	// Scanned holds the issues whose base..HEAD scan completed. A pick absent here
	// had no base, or its scan failed, so an absence of commits proves nothing.
	Scanned map[int]bool
	// ScanErrs records why a pick's scan did not complete.
	ScanErrs map[int]string
}

var shaRE = regexp.MustCompile(`^[0-9a-fA-F]{7,64}$`)

// validSHA gates every claim before it reaches git. A claim of "HEAD", "main" or
// "-x" is not a commit id; passing it to rev-parse would resolve it to trunk and
// mint a landed commit out of a word.
func validSHA(s string) bool { return shaRE.MatchString(s) }

// ResolveHead pins the SHA that ancestry is checked against, so a verdict can be
// replayed later.
func ResolveHead(ctx context.Context, run GitRunner, dir string) (string, string) {
	if run == nil {
		return "", "no git runner"
	}
	out, code, err := run(ctx, dir, "rev-parse", "--verify", "--quiet", "HEAD^{commit}")
	switch {
	case err != nil:
		return "", "git could not run: " + err.Error()
	case code != 0:
		return "", fmt.Sprintf("git rev-parse HEAD exited %d", code)
	}
	head := strings.TrimSpace(out)
	if !validSHA(head) {
		return "", fmt.Sprintf("git rev-parse HEAD returned %q", head)
	}
	return head, ""
}

// Verify gathers the git evidence for a run: it scans each pick's base..HEAD for
// subjects citing the pick, then checks every claim (scanned and self-reported)
// through the same ancestry witness. Claims naming issues outside picks are
// recorded as OFF_PICK and never reach git.
func Verify(ctx context.Context, run GitRunner, dir string, picks []Pick, claims []Claim) Verification {
	v := Verification{Scanned: map[int]bool{}, ScanErrs: map[int]string{}}
	v.Head, v.HeadErr = ResolveHead(ctx, run, dir)

	offered := map[int]bool{}
	for _, p := range picks {
		offered[p.Issue] = true
	}
	all := make([]Claim, 0, len(claims))
	for _, c := range claims {
		if c.Source == "" {
			c.Source = SourceSelfReport
		}
		all = append(all, c)
	}
	if v.Head != "" {
		found, scanned, errs := scanPicks(ctx, run, dir, picks)
		all = append(all, found...)
		v.Scanned, v.ScanErrs = scanned, errs
	} else {
		for _, p := range picks {
			v.ScanErrs[p.Issue] = "HEAD unresolved: " + v.HeadErr
		}
	}

	var driver *assumecheck.GitAncestryDriver
	if run != nil {
		driver = assumecheck.NewGitAncestryDriverWithRunner(assumecheck.Runner(run), dir)
	}
	seen := map[string]bool{}
	for _, c := range all {
		key := fmt.Sprintf("%d|%s|%s", c.Issue, strings.ToLower(strings.TrimSpace(c.SHA)), c.Source)
		if seen[key] {
			continue
		}
		seen[key] = true
		check := CommitCheck{Issue: c.Issue, SHA: strings.TrimSpace(c.SHA), Source: c.Source}
		switch {
		case !offered[c.Issue]:
			check.Verdict = VerdictOffPick
			check.Detail = fmt.Sprintf("issue #%d was not among this arm's picks", c.Issue)
		case v.Head == "":
			check.Verdict = VerdictUnverifiable
			check.Detail = "HEAD unresolved: " + v.HeadErr
		default:
			check = verifyOne(ctx, run, driver, dir, check)
		}
		v.Checks = append(v.Checks, check)
	}
	sort.SliceStable(v.Checks, func(i, j int) bool {
		if v.Checks[i].Issue != v.Checks[j].Issue {
			return v.Checks[i].Issue < v.Checks[j].Issue
		}
		return v.Checks[i].SHA < v.Checks[j].SHA
	})
	return v
}

// verifyOne decides a single claim. The existence probe runs first and on its own
// because it is the only rung that separates a FABRICATED sha (rev-parse exits 1: git
// answered "no such commit") from a git failure (exit 128 and up). assumecheck's
// cat-file rung maps real git's 128 for a missing object to "unverifiable", which
// would let a fabricated claim park a pick at UNKNOWN instead of counting as zero.
// Reachability and the revert scan are then assumecheck.GitAncestryDriver verbatim.
func verifyOne(ctx context.Context, run GitRunner, driver *assumecheck.GitAncestryDriver, dir string, c CommitCheck) CommitCheck {
	if !validSHA(c.SHA) {
		c.Verdict = VerdictNotLanded
		c.Detail = fmt.Sprintf("claimed sha %q is not a commit id", c.SHA)
		return c
	}
	out, code, err := run(ctx, dir, "rev-parse", "--verify", "--quiet", c.SHA+"^{commit}")
	switch {
	case err != nil:
		c.Verdict = VerdictUnverifiable
		c.Detail = "git could not run: " + err.Error()
		return c
	case code == 1:
		c.Verdict = VerdictNotLanded
		c.Detail = fmt.Sprintf("no commit %s exists in this repository", c.SHA)
		return c
	case code != 0:
		c.Verdict = VerdictUnverifiable
		c.Detail = fmt.Sprintf("git rev-parse exited %d for %s", code, c.SHA)
		return c
	}
	full := strings.TrimSpace(out)
	if !validSHA(full) {
		c.Verdict = VerdictUnverifiable
		c.Detail = fmt.Sprintf("git rev-parse returned %q for %s", full, c.SHA)
		return c
	}
	c.Resolved = strings.ToLower(full)
	ev := driver.Gather(ctx, assumecheck.Target{Ref: c.Resolved, Dir: dir})
	switch {
	case !ev.Witnessed:
		c.Verdict = VerdictUnverifiable
	case ev.Holds:
		c.Verdict = VerdictLanded
	default:
		c.Verdict = VerdictNotLanded
	}
	c.Detail = ev.Detail
	return c
}

// scanPicks lists base..HEAD once per distinct base and binds each commit to a pick
// whose number its subject cites. Every citing commit is kept, not only the newest,
// because commits_landed counts all of an arm's landed work.
func scanPicks(ctx context.Context, run GitRunner, dir string, picks []Pick) ([]Claim, map[int]bool, map[int]string) {
	scanned := map[int]bool{}
	errs := map[int]string{}
	byBase := map[string][]int{}
	var bases []string
	for _, p := range picks {
		base := strings.TrimSpace(p.Base)
		if base == "" {
			errs[p.Issue] = "no base sha recorded for this pick"
			continue
		}
		if !validSHA(base) {
			errs[p.Issue] = fmt.Sprintf("base sha %q is not a commit id", base)
			continue
		}
		if _, ok := byBase[base]; !ok {
			bases = append(bases, base)
		}
		byBase[base] = append(byBase[base], p.Issue)
	}
	var found []Claim
	for _, base := range bases {
		issues := byBase[base]
		out, code, err := run(ctx, dir, "log", "--format=%H%x1f%s", base+"..HEAD")
		if err != nil || code != 0 {
			why := fmt.Sprintf("git log %s..HEAD exited %d", shortSHA(base), code)
			if err != nil {
				why = "git could not run: " + err.Error()
			}
			for _, n := range issues {
				errs[n] = why
			}
			continue
		}
		for _, n := range issues {
			scanned[n] = true
		}
		for _, line := range strings.Split(out, "\n") {
			sha, subject, ok := strings.Cut(line, "\x1f")
			sha = strings.TrimSpace(sha)
			if !ok || sha == "" {
				continue
			}
			for _, n := range issues {
				if dispatchtick.SubjectCitesIssue(subject, n) {
					found = append(found, Claim{Issue: n, SHA: sha, Source: SourceGitLog})
				}
			}
		}
	}
	return found, scanned, errs
}

func shortSHA(s string) string {
	if len(s) > 12 {
		return s[:12]
	}
	return s
}
