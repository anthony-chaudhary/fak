package session

// compactaudit_corpus.go — walk a Codex rollout corpus and render the compaction-health
// answer (#4763). The scan itself is in compactaudit.go; this is the corpus sweep, the
// scrub that makes an aggregate checkinable, and the human report.
//
// The human report is the point of the issue: it must lead with "compaction fired and
// held" when that is what the evidence says, so a 36 MB append-only rollout is not
// mistaken for an unbounded session.

import (
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// CompactAuditOptions configures a corpus sweep.
type CompactAuditOptions struct {
	// Roots merges multiple rollout corpora into one aggregate. Root remains the
	// compatibility form used when Roots is empty.
	Roots []string
	Root  string
	// Since drops rollouts not modified at/after this instant. Zero = no filter.
	Since time.Time
	// Cwd, when set, keeps only rollouts whose session_meta cwd contains this string —
	// the "just my repo's sessions" filter.
	Cwd string
	// Limit caps the number of rollouts scanned (0 = unbounded), so an operator can
	// smoke the sweep on a big corpus.
	Limit int
	// GuardedOnly keeps only sessions present in the `fak guard` witness ledger — the
	// fak-routed cohort. Without it a sweep over ~/.codex/sessions measures every Codex
	// session on the box, most of which never crossed fak's wire, which makes any
	// gateway-side before/after unfalsifiable (#5254). See compactaudit_provenance.go.
	GuardedOnly bool
	// GuardWitnessDir is the ledger directory GuardedOnly reads (the caller resolves the
	// Codex home; this package does not guess it).
	GuardWitnessDir string
}

// CompactAuditResult is a whole sweep: the per-session reports plus the roll-up.
type CompactAuditResult struct {
	Root      string `json:"root,omitempty"`
	Generated string `json:"generated,omitempty"`
	// Provenance says which corpus subset this sweep measured (#5254). It survives
	// --scrub: it carries no paths, and without it a guarded-only aggregate is
	// indistinguishable from a whole-corpus one.
	Provenance CompactProvenance      `json:"provenance"`
	Aggregate  CompactAggregate       `json:"aggregate"`
	Sessions   []CompactSessionReport `json:"sessions,omitempty"`
}

// AuditCompactCorpus streams every rollout under opts.Root and reports compaction
// health. Files are scanned one at a time and each is streamed head-bounded, so corpus
// size drives wall time, not memory.
func AuditCompactCorpus(opts CompactAuditOptions) (CompactAuditResult, error) {
	roots := append([]string(nil), opts.Roots...)
	if len(roots) == 0 {
		roots = []string{opts.Root}
	}
	res := CompactAuditResult{Root: strings.Join(roots, string(os.PathListSeparator))}
	res.Provenance.GuardedOnly = opts.GuardedOnly

	// Resolve the ledger BEFORE the walk: a guarded-only sweep with no ledger must fail
	// loudly, not scan the corpus and report an empty cohort as a clean result.
	var guarded map[string]struct{}
	if opts.GuardedOnly {
		ids, err := LoadGuardWitnessIDs(opts.GuardWitnessDir)
		if err != nil {
			return res, err
		}
		guarded = ids
		res.Provenance.LedgerSessions = len(ids)
	}

	var paths []string
	for _, root := range roots {
		err := filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
			if err != nil {
				// A corpus is a live directory: a vanished/locked file must not sink the
				// whole sweep.
				if d != nil && d.IsDir() {
					return fs.SkipDir
				}
				return nil
			}
			if d.IsDir() || !strings.HasSuffix(d.Name(), ".jsonl") {
				return nil
			}
			if !opts.Since.IsZero() {
				info, e := d.Info()
				if e != nil || info.ModTime().Before(opts.Since) {
					return nil
				}
			}
			paths = append(paths, p)
			return nil
		})
		if err != nil {
			return res, err
		}
	}
	sort.Strings(paths)

	for _, p := range paths {
		if opts.Limit > 0 && len(res.Sessions) >= opts.Limit {
			break
		}
		rep, e := scanCompactPath(p)
		if e != nil {
			continue // unreadable rollout: skipped, never fabricated
		}
		if opts.Cwd != "" && !strings.Contains(rep.Cwd, opts.Cwd) {
			continue
		}
		if guarded != nil {
			// A rollout whose session_meta id never landed in the ledger is traffic fak
			// did not route, so no gateway-side transform could have touched its bytes.
			if _, ok := guarded[rep.SessionID]; !ok {
				res.Provenance.Unguarded++
				continue
			}
			res.Provenance.Guarded++
		}
		res.Sessions = append(res.Sessions, rep)
	}
	res.Aggregate = AggregateCompactReports(res.Sessions)
	return res, nil
}

func scanCompactPath(p string) (CompactSessionReport, error) {
	f, err := os.Open(p)
	if err != nil {
		return CompactSessionReport{}, err
	}
	defer f.Close()
	var size int64
	if fi, e := f.Stat(); e == nil {
		size = fi.Size()
	}
	return ScanCompactRollout(f, p, size)
}

// ScrubCompactResult strips everything that cannot be checked into a public repo:
// filesystem paths and the session cwd. Session ids (opaque UUIDs) and the numeric
// witnesses survive, which is what reproduces the headline counts. Prompt and
// tool-output bodies never enter a report in the first place — the scanner drops them
// at read time — so there is nothing to scrub there.
func ScrubCompactResult(res CompactAuditResult) CompactAuditResult {
	out := res
	out.Root = ""
	out.Sessions = make([]CompactSessionReport, 0, len(res.Sessions))
	for _, s := range res.Sessions {
		s.Path = ""
		s.Cwd = ""
		out.Sessions = append(out.Sessions, s)
	}
	return out
}

// WriteCompactAuditJSON emits the machine form.
func WriteCompactAuditJSON(w io.Writer, res CompactAuditResult) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(res)
}

// RenderCompactAudit writes the human report. It deliberately prints append-only bytes
// and cumulative tokens NEXT TO peak resident context, labelled, because the whole
// point is that the first two do not answer the compaction question and the third does.
func RenderCompactAudit(w io.Writer, res CompactAuditResult, topN int) {
	a := res.Aggregate
	fmt.Fprintf(w, "compaction health — %d sessions, %s of append-only rollout\n", a.Sessions, humanBytes(a.Bytes))
	fmt.Fprintf(w, "  fires:            %d across %d sessions (%d measurable pre/post pairs)\n",
		a.Fires, a.CompactedSessions, a.MeasuredFires)
	if a.MeasuredFires > 0 {
		fmt.Fprintf(w, "  resident context: median pre-fire %d -> post-fire %d (median shed %d, post/pre %.2f)\n",
			a.MedianPreTokens, a.MedianPostTokens, a.MedianShedTokens, a.MedianResidualRatio)
	}
	fmt.Fprintln(w)
	fmt.Fprintln(w, "  NOTE: rollout bytes are append-only and cumulative provider tokens are monotonic.")
	fmt.Fprintln(w, "  Neither measures resident context. A large rollout with repeated fires is HEALTHY.")
	fmt.Fprintln(w)

	if len(a.VerdictCounts) > 0 {
		fmt.Fprintln(w, "  verdicts:")
		for _, k := range sortedKeys(a.VerdictCounts) {
			fmt.Fprintf(w, "    %-24s %d  %s\n", k, a.VerdictCounts[k], verdictGloss(k))
		}
		fmt.Fprintln(w)
	}
	if len(a.AnomalyCounts) > 0 {
		fmt.Fprintln(w, "  anomalies:")
		for _, k := range sortedKeys(a.AnomalyCounts) {
			fmt.Fprintf(w, "    %-24s %d sessions\n", k, a.AnomalyCounts[k])
		}
		fmt.Fprintln(w)
	} else {
		fmt.Fprintln(w, "  anomalies: none")
		fmt.Fprintln(w)
	}

	// #4768: how fast the window came BACK after each fire, and out of what.
	writeCompactRegrowthSection(w, a.Regrowth, 12)

	if topN <= 0 {
		return
	}
	ranked := append([]CompactSessionReport(nil), res.Sessions...)
	sort.Slice(ranked, func(i, j int) bool {
		if len(ranked[i].Fires) != len(ranked[j].Fires) {
			return len(ranked[i].Fires) > len(ranked[j].Fires)
		}
		return ranked[i].Bytes > ranked[j].Bytes
	})
	if len(ranked) > topN {
		ranked = ranked[:topN]
	}
	fmt.Fprintf(w, "  top %d sessions by fires:\n", len(ranked))
	for _, s := range ranked {
		id := s.SessionID
		if id == "" {
			id = "(unknown)"
		}
		fmt.Fprintf(w, "    %s  %s\n", id, s.Verdict)
		fmt.Fprintf(w, "      %d fires · %s rollout · %d cumulative input tokens · peak RESIDENT %d/%d\n",
			len(s.Fires), humanBytes(s.Bytes), s.CumulativeInputTokens, s.PeakResidentTokens, s.ContextWindow)
		if len(s.Anomalies) > 0 {
			fmt.Fprintf(w, "      anomalies: %s\n", strings.Join(s.Anomalies, ", "))
		}
	}
}

func verdictGloss(v string) string {
	switch v {
	case VerdictFiredAndHeld:
		return "compaction fired and held — resident context was repeatedly reset"
	case VerdictFiredWithAnomaly:
		return "compaction fired, but at least one fire is anomalous"
	case VerdictNoFireBounded:
		return "never fired — resident context stayed bounded on its own"
	case VerdictNoFireAtCeiling:
		return "reached the ceiling and never fired"
	case VerdictTelemetryMissing:
		return "no token telemetry — compaction health is unknown, not failed"
	}
	return ""
}

func sortedKeys(m map[string]int) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func humanBytes(b int64) string {
	const unit = 1024
	if b < unit {
		return fmt.Sprintf("%d B", b)
	}
	div, exp := int64(unit), 0
	for n := b / unit; n >= unit && exp < 3; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(b)/float64(div), "KMGT"[exp])
}
