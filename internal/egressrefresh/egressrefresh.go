// Package egressrefresh re-fetches the bundled egress filter lists
// (internal/egresslist/lists) from their recorded provenance URLs, re-normalizes them
// through the SAME ingest path the kernel compiles, and rewrites the checked-in artifact
// plus its pinned checksum.
//
// WHY THIS IS A SEPARATE LEAF. internal/egresslist is a pure, tier-1, stdlib-only
// classifier the adjudicator folds on the live tool-call decision path; its package
// contract is "no net calls, ever". Refreshing is the opposite kind of work — network,
// filesystem, wall-clock. Keeping the two apart is what lets the decide path stay offline
// and deterministic while the lists still track a churning upstream: this package only
// ever runs from an operator's `fak egresslist refresh`, never from a decision.
//
// THE SHAPE: REFRESH IS A REVIEWABLE DIFF, NOT AN AUTO-UPDATE. A run rewrites files in the
// working tree and stops. The checked-in artifact stays the source of truth, a human reads
// the diff, and the commit is the audit record of what the adjudicator's block set became.
// Nothing here is wired to a scheduler or to policy load — an egress block set that
// silently changed itself under a live agent would be an unreviewed capability change.
//
// FAIL CLOSED, ALWAYS. Every refusal path below keeps the PREVIOUSLY-PINNED artifact
// rather than writing a worse one, because the failure mode is asymmetric: a stale block
// list still blocks yesterday's malware, while a truncated or empty one silently blocks
// NOTHING. An empty block list is not a degraded list, it is an all-permissive one. So a
// fetch error, a non-200, an over-size body, an upstream that parses to too few rules, and
// an upstream that collapsed against its pinned rule count are all REFUSALS that leave
// disk untouched and report a reason — never a partial write.
package egressrefresh

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/egresslist"
)

// Default tuning for a refresh run.
const (
	// DefaultMinRules refuses an upstream that parses to fewer rules than this. One is
	// the floor that matters: zero rules is an all-permissive list.
	DefaultMinRules = 1
	// DefaultMinKeepRatio refuses an upstream whose rule count collapsed below this
	// fraction of the pinned count — the "truncated" half of fail-closed. A real feed
	// churns by percents; a halving is a broken fetch (a captive portal, an error page
	// that happens to parse, a partial body), not news.
	DefaultMinKeepRatio = 0.5
	// DefaultMaxBytes caps a fetched body so a hostile or broken upstream cannot make
	// a maintenance verb eat the box.
	DefaultMaxBytes int64 = 64 << 20
	// DefaultTimeout bounds one upstream fetch.
	DefaultTimeout = 60 * time.Second
)

// Status is the outcome vocabulary for refreshing one list. It is closed on purpose: a
// caller (and the verb's exit code) branches on these, never on free text.
type Status string

const (
	// StatusUpdated: upstream fetched, parsed, and the artifact + checksum rewritten.
	StatusUpdated Status = "updated"
	// StatusUnchanged: upstream fetched and parsed to byte-identical canonical text.
	// The artifact is untouched; only last_refreshed advances, which is what makes
	// "we checked and it is current" distinguishable from "nobody has looked since".
	StatusUnchanged Status = "unchanged"
	// StatusSkipped: the list records no provenance URL, so there is nothing to
	// re-fetch. Not an error — a hand-authored or operator-pinned list is legitimate.
	StatusSkipped Status = "skipped"
	// StatusFailed: a fail-closed refusal. The pinned artifact is intact and Reason
	// names what was refused.
	StatusFailed Status = "failed"
)

// Result is the per-list outcome of a refresh run.
type Result struct {
	Name      string `json:"name"`
	Status    Status `json:"status"`
	Reason    string `json:"reason,omitempty"`
	OldSHA256 string `json:"old_sha256,omitempty"`
	NewSHA256 string `json:"new_sha256,omitempty"`
	OldRules  int    `json:"old_rules"`
	NewRules  int    `json:"new_rules"`
}

// Fetcher fetches one upstream list. It is an interface so the refresh logic — every
// fail-closed branch above — is testable without a network, and so an operator-supplied
// transport (a proxy, an offline mirror) can substitute.
type Fetcher interface {
	Fetch(ctx context.Context, url string) ([]byte, error)
}

// HTTPFetcher is the real upstream fetcher: a bounded, timeout-carrying GET.
type HTTPFetcher struct {
	Client   *http.Client
	MaxBytes int64
}

// Fetch GETs url and returns the body. A non-200 is an error, not a body: an upstream
// serving a 404 page or a captive-portal login is exactly the input that must never reach
// the parser and become "the new block list".
func (f HTTPFetcher) Fetch(ctx context.Context, url string) ([]byte, error) {
	client := f.Client
	if client == nil {
		client = &http.Client{Timeout: DefaultTimeout}
	}
	max := f.MaxBytes
	if max <= 0 {
		max = DefaultMaxBytes
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("User-Agent", "fak-egresslist-refresh/1")
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("upstream returned HTTP %d", resp.StatusCode)
	}
	// Read one byte past the cap so a body sitting exactly at the limit is not
	// mistaken for a truncated one.
	body, err := io.ReadAll(io.LimitReader(resp.Body, max+1))
	if err != nil {
		return nil, fmt.Errorf("read body: %w", err)
	}
	if int64(len(body)) > max {
		return nil, fmt.Errorf("body exceeds %d bytes", max)
	}
	return body, nil
}

// Options configures a refresh run.
type Options struct {
	// Dir is the bundled-lists directory holding manifest.json and the *.txt
	// artifacts (repo: internal/egresslist/lists).
	Dir string
	// Names limits the run to these list names; empty means every recorded list.
	Names []string
	// Fetcher fetches upstreams; nil means a default HTTPFetcher.
	Fetcher Fetcher
	// Now stamps last_refreshed; zero means time.Now(). Injectable so a test asserts
	// an exact stamp instead of racing the clock.
	Now time.Time
	// MinRules / MinKeepRatio are the fail-closed floors; zero means the defaults.
	MinRules     int
	MinKeepRatio float64
	// AllowShrink waives the truncation guard for a genuine upstream shrink an
	// operator has actually looked at.
	AllowShrink bool
	// DryRun reports what would change without writing anything.
	DryRun bool
}

// ManifestPath is the manifest inside a bundled-lists directory.
func ManifestPath(dir string) string { return filepath.Join(dir, "manifest.json") }

// artifactPath is the checked-in list text for a name.
func artifactPath(dir, name string) string { return filepath.Join(dir, name+".txt") }

// Refresh re-fetches the selected lists and rewrites the artifacts whose upstream moved.
//
// Contract: Refresh never modifies disk if dry-run is requested, and preserves all
// previously pinned artifacts whenever an upstream fetch fails or violates guard constraints.
//
// Invariant: egress list refresh is fail-closed and provenance-verified.
// Guard: every upstream fetch refusal preserves the previously-pinned artifact and checksum without mutating disk.
//
// The returned error is reserved for a run that could not start or could not be committed
// (an unreadable manifest, an unknown --name, a failed write) — a per-list refusal is NOT
// an error, it is a Result with StatusFailed, because a run over ten feeds where one
// upstream is down should still land the other nine.
func Refresh(ctx context.Context, opts Options) ([]Result, error) {
	if strings.TrimSpace(opts.Dir) == "" {
		return nil, fmt.Errorf("egressrefresh: no lists directory given")
	}
	raw, err := os.ReadFile(ManifestPath(opts.Dir))
	if err != nil {
		return nil, fmt.Errorf("egressrefresh: read manifest: %w", err)
	}
	man, err := egresslist.ParseManifest(raw)
	if err != nil {
		return nil, err
	}
	selected, err := select_(man, opts.Names)
	if err != nil {
		return nil, err
	}

	fetcher := opts.Fetcher
	if fetcher == nil {
		fetcher = HTTPFetcher{}
	}
	now := opts.Now
	if now.IsZero() {
		now = time.Now()
	}
	minRules := opts.MinRules
	if minRules <= 0 {
		minRules = DefaultMinRules
	}
	ratio := opts.MinKeepRatio
	if ratio <= 0 {
		ratio = DefaultMinKeepRatio
	}

	results := make([]Result, 0, len(selected))
	dirty := false
	for _, src := range selected {
		res, updated, ok := refreshOne(ctx, fetcher, opts, src, now, minRules, ratio, &man)
		results = append(results, res)
		if ok && updated {
			dirty = true
		}
	}
	// The manifest is written ONCE, after every list is resolved: a per-list refusal
	// must not leave a half-updated provenance record behind it.
	if dirty && !opts.DryRun {
		out, err := man.Render()
		if err != nil {
			return results, err
		}
		if err := os.WriteFile(ManifestPath(opts.Dir), out, 0o644); err != nil {
			return results, fmt.Errorf("egressrefresh: write manifest: %w", err)
		}
	}
	return results, nil
}

// refreshOne resolves a single list. It returns the result, whether the manifest was
// mutated, and whether the run may continue committing (false only for a real write
// failure the caller must surface).
//
// Every early return below is a fail-closed exit: nothing has been written to disk at that
// point, so the previously-pinned artifact stands.
func refreshOne(ctx context.Context, fetcher Fetcher, opts Options, src egresslist.Source,
	now time.Time, minRules int, ratio float64, man *egresslist.Manifest) (Result, bool, bool) {

	res := Result{Name: src.Name, OldSHA256: src.SHA256, OldRules: src.Rules}

	if !src.Refreshable() {
		res.Status = StatusSkipped
		res.Reason = "no provenance URL recorded - not refreshable from upstream"
		return res, false, true
	}

	body, err := fetcher.Fetch(ctx, src.URL)
	if err != nil {
		res.Status = StatusFailed
		res.Reason = fmt.Sprintf("fetch %s: %v (keeping the pinned artifact)", src.URL, err)
		return res, false, true
	}

	// Re-normalize through the SAME ingest path the kernel compiles, so what we pin is
	// exactly what a decision will see — never a second, drifting parser.
	list := egresslist.NewBuilder().AddFilterText(src.Name, string(body)).Build()
	block, allow := list.Counts()
	res.NewRules = block + allow

	if res.NewRules < minRules {
		res.Status = StatusFailed
		res.Reason = fmt.Sprintf("upstream parsed to %d rules (minimum %d): keeping the pinned artifact - "+
			"an empty block list is all-permissive, not merely stale", res.NewRules, minRules)
		return res, false, true
	}
	if !opts.AllowShrink && src.Rules > 0 && float64(res.NewRules) < ratio*float64(src.Rules) {
		res.Status = StatusFailed
		res.Reason = fmt.Sprintf("upstream collapsed from %d to %d rules (below the %.0f%% truncation guard): "+
			"keeping the pinned artifact - re-run with --allow-shrink if the shrink is real",
			src.Rules, res.NewRules, ratio*100)
		return res, false, true
	}

	artifact := egresslist.RenderArtifact(src, list)
	res.NewSHA256 = egresslist.Checksum(artifact)

	if res.NewSHA256 == src.SHA256 {
		// Upstream is current. The artifact is untouched (byte-identical), but
		// last_refreshed still advances: "checked, unchanged" and "never checked" are
		// different facts to an operator reading staleness.
		res.Status = StatusUnchanged
		if opts.DryRun {
			return res, false, true
		}
		src.LastRefreshed = now.UTC().Format(time.RFC3339)
		man.Set(src)
		return res, true, true
	}

	res.Status = StatusUpdated
	if opts.DryRun {
		return res, false, true
	}
	// Artifact first, then the manifest that pins it: a crash between the two leaves a
	// checksum mismatch the gate SHOUTS about, which is recoverable. The reverse order
	// would leave a manifest vouching for bytes that were never written.
	if err := os.WriteFile(artifactPath(opts.Dir, src.Name), []byte(artifact), 0o644); err != nil {
		res.Status = StatusFailed
		res.Reason = fmt.Sprintf("write artifact: %v", err)
		return res, false, false
	}
	src.SHA256 = res.NewSHA256
	src.Rules = res.NewRules
	src.LastRefreshed = now.UTC().Format(time.RFC3339)
	man.Set(src)
	return res, true, true
}

// select_ resolves the requested names against the manifest. An unknown name is a hard
// error, never a silent no-op: an operator who typos `--name stevenblak` must be told,
// not handed a green "0 lists refreshed".
func select_(man egresslist.Manifest, names []string) ([]egresslist.Source, error) {
	if len(names) == 0 {
		return man.Lists, nil
	}
	out := make([]egresslist.Source, 0, len(names))
	for _, n := range names {
		s, ok := man.Get(n)
		if !ok {
			known := make([]string, 0, len(man.Lists))
			for _, l := range man.Lists {
				known = append(known, l.Name)
			}
			sort.Strings(known)
			return nil, fmt.Errorf("egressrefresh: no bundled list named %q (known: %s)", n, strings.Join(known, ", "))
		}
		out = append(out, s)
	}
	return out, nil
}
