// Package marketing turns a WITNESSED completion event — a ship-stamped commit, a
// closed epic, a release — into a fresh marketing artifact, with one rung the other
// outbound posters (internal/scoreboard, benchpost, dispatchpost, dojopost) don't have:
// every claim is keyed to a git-witnessed commit sha. A marketing claim is exactly as
// trustworthy as the commit behind it; an unwitnessed boast is refused at construction
// (see claim.go), and a feature still tagged [SIMULATED]/[STUB] in CLAIMS.md is excluded
// (see honesty.go).
//
// The witness rung reuses internal/hooks.StampOf — the SAME grammar the pre-commit lint
// and `dos verify` bind to — so the marketing ledger counts exactly what the referee can
// bind, never a second drifting copy of the regexes. A subject is a marketable Ship iff
// StampOf grades it "trailer" or "direct" (a real per-leaf stamp); a merge/bookkeeping
// subject ("none") and a release-bundle subject ("release") are NOT per-leaf ships and are
// carried only as an honest "other activity" count, never as a claim.
//
// This file is the witnessed-atom layer: the Ship type and the git collection. It is the
// per-ship complement of internal/cadencereport.shipsBySubjects (which folds the same
// predicate into a count); here each ship keeps its (sha, date, subject) so a claim can
// cite it. Tier 1: stdlib + internal/hooks only, so it builds in isolation and is the
// dispatchable spine the rest of the subsystem hangs off.
package marketing

import (
	"os/exec"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/hooks"
)

// Ship is one git-witnessed completion: a commit whose SUBJECT carries a real per-leaf
// ship-stamp (hooks.StampOf kind "trailer" or "direct"). It is the only thing a marketing
// claim may assert as shipped — the fields are the evidence a claim cites, not prose.
type Ship struct {
	SHA     string    // short sha — the non-forgeable witness a claim renders inline
	Leaf    string    // hooks.StampOf leaf (lowercased), e.g. "gateway" — the subsystem
	Subject string    // the conventional-commits subject (the claim text source)
	Kind    string    // "trailer" | "direct" — never "none"/"release" (those aren't ships)
	Date    time.Time // commit author date, for dated/ordered artifacts; zero if unparsed
}

// Activity is the honest counterpart to a Ship: the non-ship commits in a range (merges,
// bookkeeping, release bundles, body-only mentions). It is rendered as "M other commits"
// so an artifact never inflates the ship count, and never claims un-stamped work.
type Activity struct {
	Commits int // total non-merge commits in the range
	Ships   int // of those, the ones that graded as marketable Ships
}

// nullSep / fieldSep are the git-log format delimiters: a NUL between fields (can't appear
// in a sha/date/subject) and a record terminator, so a multi-line body can never split a
// record. Mirrors the %x00 idiom cadencereport uses for the same reason.
const gitLogFormat = "%H%x1f%ad%x1f%s%x1e"

// CollectShips enumerates the non-merge commits in revRange (e.g. "abc123..HEAD", or "" for
// the whole HEAD history) and returns the subset that grade as marketable Ships, plus the
// Activity tally. The ship predicate is hooks.StampOf ∈ {trailer,direct} — identical to
// cadencereport.shipsBySubjects — so this is the SAME witness the referee binds, with the
// per-ship sha/date kept. Ships are returned newest-first.
//
// A git failure returns (nil, zero, err); a clean range with no ships returns (nil, tally,
// nil) — an honest empty result the caller renders as "no witnessed ships", never a boast.
func CollectShips(root, revRange string) ([]Ship, Activity, error) {
	out, err := runGitLog(root, revRange)
	if err != nil {
		return nil, Activity{}, err
	}
	ships, act := parseShipLog(out)
	sort.SliceStable(ships, func(i, j int) bool { return ships[i].Date.After(ships[j].Date) })
	return ships, act, nil
}

// runGitLog is the single git seam (the only impure call), so parseShipLog is unit-testable
// against canned output without a repo. revRange "" lists all of HEAD; a non-empty range is
// passed verbatim (the caller validates it). --no-merges matches the cadencereport predicate
// (a merge subject is never a ship). Dir is root so it works from any cwd.
func runGitLog(root, revRange string) (string, error) {
	args := []string{"log", "--no-merges", "--date=iso-strict", "--format=" + gitLogFormat}
	if strings.TrimSpace(revRange) != "" {
		args = append(args, revRange)
	}
	cmd := exec.Command("git", args...)
	cmd.Dir = root
	b, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// parseShipLog is the pure fold over git-log output: one record per non-merge commit (NUL-
// separated sha/date/subject, record-separator terminated), keeping those whose subject
// grades as a per-leaf ship. Kept pure (no git) so it is the unit-tested core.
func parseShipLog(out string) ([]Ship, Activity) {
	var ships []Ship
	var act Activity
	for _, rec := range strings.Split(out, "\x1e") {
		rec = strings.Trim(rec, "\n")
		if strings.TrimSpace(rec) == "" {
			continue
		}
		act.Commits++
		f := strings.Split(rec, "\x1f")
		if len(f) < 3 {
			continue
		}
		sha, dateStr, subject := strings.TrimSpace(f[0]), strings.TrimSpace(f[1]), strings.TrimSpace(f[2])
		kind, leaf := hooks.StampOf(subject)
		if kind != "trailer" && kind != "direct" {
			continue // merge/bookkeeping/release/body-only — counted as activity, never a ship
		}
		act.Ships++
		ships = append(ships, Ship{
			SHA:     shortSHA(sha),
			Leaf:    leaf,
			Subject: subject,
			Kind:    kind,
			Date:    parseGitDate(dateStr),
		})
	}
	return ships, act
}

// shortSHA trims a full sha to the conventional 8-char short form a commit link uses; a
// shorter input is returned as-is.
func shortSHA(sha string) string {
	if len(sha) > 8 {
		return sha[:8]
	}
	return sha
}

// parseGitDate parses git's --date=iso-strict output (RFC3339). An unparseable value yields
// the zero time rather than an error — a missing date degrades the ordering, it does not
// invalidate the witness (the sha is the witness, not the date).
func parseGitDate(s string) time.Time {
	if s == "" {
		return time.Time{}
	}
	if t, err := time.Parse(time.RFC3339, s); err == nil {
		return t
	}
	return time.Time{}
}
