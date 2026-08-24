package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/marketing"
)

// marketing_whatsnew.go — `fak marketing whats-new`, the committed refresh path behind
// docs/whats-new.md (#6040).
//
// The page is a FOLD, never a hand-written snapshot: it gathers the ship-stamped commits in a
// window, runs them through the same CLAIMS.md honesty gate every other marketing artifact
// uses, groups them into themes, and stamps itself with the commit it is current through. That
// stamp is what makes staleness checkable — `--check` replays the recorded range and compares
// bytes, so a rotted page or a hand edit is REPORTED rather than trusted.
//
//	fak marketing whats-new                       # render the page to stdout (dry by default)
//	fak marketing whats-new --json                # the same fold as JSON evidence
//	fak marketing whats-new --write               # write docs/whats-new.md
//	fak marketing whats-new --days 30 --write     # widen the window
//	fak marketing whats-new --check --json        # freshness verdict; non-zero when stale/drifted
//
// The default is dry on purpose: a docs page is only rewritten when a maintainer asks.

// whatsNewCheck is the freshness verdict, shaped for both a human line and --json. Every
// field a reader would need to argue with the verdict is present, so "is this page stale?"
// is answered with numbers rather than an opinion.
type whatsNewCheck struct {
	Path                string `json:"path"`
	OK                  bool   `json:"ok"`
	Verdict             string `json:"verdict"` // fresh | stale | drifted | missing-anchor | missing-page
	AnchorSHA           string `json:"anchor_sha,omitempty"`
	AnchorDate          string `json:"anchor_date,omitempty"`
	RangeSpec           string `json:"range_spec,omitempty"`
	HeadSHA             string `json:"head_sha"`
	HeadDate            string `json:"head_date"`
	AgeDays             int    `json:"age_days"`
	MaxAgeDays          int    `json:"max_age_days"`
	CommitsBehind       int    `json:"commits_behind"`
	MaxCommitsBehind    int    `json:"max_commits_behind"`
	MatchesRegeneration bool   `json:"matches_regeneration"`
	NextAction          string `json:"next_action"`
}

func runMarketingWhatsNew(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak marketing whats-new", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", ".", "repo root to read git/CLAIMS.md and write the page under")
	days := fs.Int("days", 7, "window in days of repository history to fold")
	rangeFlag := fs.String("range", "", "fold this rev-range verbatim (e.g. abc123..HEAD); wins over --days")
	perTheme := fs.Int("per-theme", 4, "how many changes each theme lists (all are still counted)")
	write := fs.Bool("write", false, "write "+marketing.RecentChangesPath+" (default: print it)")
	check := fs.Bool("check", false, "verify the committed page's freshness instead of rendering; non-zero when stale or drifted")
	maxAge := fs.Int("max-age-days", 7, "with --check: maximum commit-date age before the page is stale")
	maxBehind := fs.Int("max-commits-behind", 250, "with --check: maximum non-merge commits after the anchor before the page is stale")
	asJSON := fs.Bool("json", false, "emit the fold (or the --check verdict) as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if *check {
		if *maxAge < 0 || *maxBehind < 0 {
			fmt.Fprintln(stderr, "fak marketing whats-new: --max-age-days and --max-commits-behind must be non-negative")
			return 2
		}
		return whatsNewCheckRun(stdout, stderr, *root, *maxAge, *maxBehind, *asJSON)
	}

	page, err := whatsNewBuild(*root, *days, *rangeFlag, *perTheme)
	if err != nil {
		fmt.Fprintf(stderr, "fak marketing whats-new: %v\n", err)
		return 1
	}
	if *asJSON {
		if rc := encodeJSONOrFailPrefixed(stdout, stderr, page, "fak marketing whats-new"); rc != 0 {
			return rc
		}
		return 0
	}
	md := page.Markdown()
	if !*write {
		fmt.Fprint(stdout, md)
		return 0
	}
	path := filepath.Join(*root, filepath.FromSlash(marketing.RecentChangesPath))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		fmt.Fprintf(stderr, "fak marketing whats-new: %v\n", err)
		return 1
	}
	if err := os.WriteFile(path, []byte(md), 0o644); err != nil {
		fmt.Fprintf(stderr, "fak marketing whats-new: write: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "wrote %s (%d stamped change(s) across %d commit(s), current through %s)\n",
		path, page.Ships, page.Commits, shortMark(page.AnchorSHA))
	return 0
}

// whatsNewBuild resolves the window against git and folds it. The anchor is HEAD's sha AND
// its commit date (never the wall clock), and the range ends at that sha rather than at the
// symbolic HEAD, so regenerating after the page itself is committed still reproduces the same
// bytes — the property --check depends on.
func whatsNewBuild(root string, days int, rangeFlag string, perTheme int) (marketing.RecentChangesPage, error) {
	headSHA, headDate, err := marketing.RecentAnchorCommit(root)
	if err != nil {
		return marketing.RecentChangesPage{}, fmt.Errorf("resolve HEAD: %w", err)
	}
	rangeSpec := strings.TrimSpace(rangeFlag)
	windowDays := 0
	if rangeSpec == "" {
		if days <= 0 {
			return marketing.RecentChangesPage{}, fmt.Errorf("--days must be positive (got %d)", days)
		}
		windowDays = days
		start, err := marketing.RecentWindowStart(root, headDate, days)
		if err != nil {
			return marketing.RecentChangesPage{}, fmt.Errorf("resolve window start: %w", err)
		}
		if start == "" {
			// The window reaches past the first commit: fold the whole history up to the
			// anchor rather than silently rendering an empty page.
			rangeSpec = headSHA
		} else {
			rangeSpec = start + ".." + headSHA
		}
	}
	version := ""
	if v, ok := marketing.RecentRepositoryVersion(root); ok {
		version = v
	}
	generatorModule, err := marketing.RecentModuleVersion(root, "internal/marketing", headSHA)
	if err != nil {
		return marketing.RecentChangesPage{}, fmt.Errorf("resolve generator module version: %w", err)
	}
	return whatsNewFold(root, marketing.RecentChangesOptions{
		AnchorSHA:       headSHA,
		AnchorDate:      headDate,
		RangeSpec:       rangeSpec,
		RangeLabel:      marketing.RecentRangeLabel(windowDays, rangeSpec, headDate),
		Version:         version,
		GeneratorModule: generatorModule,
		PerTheme:        perTheme,
		Days:            windowDays,
	})
}

// whatsNewFold is the shared gather+build tail: both the render path and the check path use
// it, so a regeneration is produced by exactly the same code that wrote the page.
func whatsNewFold(root string, opt marketing.RecentChangesOptions) (marketing.RecentChangesPage, error) {
	col, err := marketing.Gather(root, opt.RangeSpec)
	if err != nil {
		return marketing.RecentChangesPage{}, err
	}
	return marketing.BuildRecentChanges(col, opt), nil
}

// whatsNewCheckRun answers three separate questions and reports the first failure: does the
// page carry an anchor at all, has HEAD moved further past it than the allowed window, and
// does re-folding the recorded range still produce the committed bytes. The third is what
// catches a hand edit or a CLAIMS.md change that silently invalidated a scope label.
func whatsNewCheckRun(stdout, stderr io.Writer, root string, maxAge, maxBehind int, asJSON bool) int {
	path := filepath.Join(root, filepath.FromSlash(marketing.RecentChangesPath))
	res := whatsNewCheck{
		Path: marketing.RecentChangesPath, MaxAgeDays: maxAge, MaxCommitsBehind: maxBehind,
	}

	headSHA, headDate, err := marketing.RecentAnchorCommit(root)
	if err != nil {
		fmt.Fprintf(stderr, "fak marketing whats-new --check: resolve HEAD: %v\n", err)
		return 2
	}
	res.HeadSHA, res.HeadDate = headSHA, headDate.Format("2006-01-02")

	raw, err := os.ReadFile(path)
	if err != nil {
		res.Verdict = "missing-page"
		res.NextAction = marketing.RecentChangesVerb + " --write"
		return whatsNewReport(stdout, res, asJSON)
	}
	anchor, ok := marketing.ParseRecentChangesAnchor(string(raw))
	if !ok {
		res.Verdict = "missing-anchor"
		res.NextAction = marketing.RecentChangesVerb + " --write"
		return whatsNewReport(stdout, res, asJSON)
	}
	res.AnchorSHA, res.RangeSpec = anchor.SHA, anchor.RangeSpec
	res.AnchorDate = anchor.Date.Format("2006-01-02")
	res.AgeDays = whatsNewAgeDays(anchor.Date, headDate)
	if n, err := marketing.RecentCommitsBetween(root, anchor.SHA+"..HEAD"); err == nil {
		res.CommitsBehind = n
	}

	// Replay the page's own recorded inputs. Anything but a byte match means the file no
	// longer says what the repository says.
	regen, err := whatsNewFold(root, anchor.Options())
	if err != nil {
		fmt.Fprintf(stderr, "fak marketing whats-new --check: regenerate: %v\n", err)
		return 2
	}
	res.MatchesRegeneration = regen.Markdown() == string(raw)

	switch {
	case !res.MatchesRegeneration:
		res.Verdict = "drifted"
		res.NextAction = marketing.RecentChangesVerb + " --write"
	case whatsNewStale(res.AgeDays, res.CommitsBehind, maxAge, maxBehind):
		res.Verdict = "stale"
		res.NextAction = marketing.RecentChangesVerb + " --write"
	default:
		res.Verdict, res.OK = "fresh", true
		res.NextAction = "none"
	}
	return whatsNewReport(stdout, res, asJSON)
}

func whatsNewStale(ageDays, commitsBehind, maxAge, maxBehind int) bool {
	return ageDays > maxAge || commitsBehind > maxBehind
}

// whatsNewAgeDays measures how far HEAD's commit date has moved past the page's anchor date.
// Commit dates, not the clock: a checkout that has not fetched for a week is not a stale page.
func whatsNewAgeDays(anchor, head time.Time) int {
	if anchor.IsZero() || head.IsZero() {
		return 0
	}
	d := head.Sub(anchor).Hours() / 24
	if d <= 0 {
		return 0
	}
	return int(math.Floor(d))
}

func whatsNewReport(stdout io.Writer, res whatsNewCheck, asJSON bool) int {
	if asJSON {
		b, err := json.MarshalIndent(res, "", "  ")
		if err == nil {
			fmt.Fprintln(stdout, string(b))
		}
	} else {
		fmt.Fprintf(stdout, "%s: %s (anchor %s, age %d/%d day(s), behind %d/%d commit(s), regenerates identically: %t)\n",
			res.Path, res.Verdict, shortMark(res.AnchorSHA), res.AgeDays, res.MaxAgeDays,
			res.CommitsBehind, res.MaxCommitsBehind, res.MatchesRegeneration)
		if !res.OK {
			fmt.Fprintf(stdout, "refresh with: %s\n", res.NextAction)
		}
	}
	if res.OK {
		return 0
	}
	return 1
}
