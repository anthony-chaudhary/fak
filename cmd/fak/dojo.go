package main

// fak dojo -- the prediction-vs-reality GYM. It runs registered levers (token-
// saving optimizations) over a scenario (a corpus of real transcripts today),
// scores each lever's THEORY against the provider's billed reality, and folds the
// episodes into one schema/ok/verdict/finding/reason/next_action envelope. With
// --append-history it appends a dated row to docs/dojo/history.jsonl so the mean
// calibration error is trended across runs -- the closed loop the resume / vcache
// / cadence surfaces each named as missing: not "what did this lever save" but
// "are our predictors getting better calibrated over time".
//
// This is the I/O shell: it scans the corpus and adapts the existing
// resume.Backtest residual into dojo episodes. The scoring/fold/ledger live in
// the pure internal/dojo package.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dojo"
	"github.com/anthony-chaudhary/fak/internal/resume"
)

const dojoUsage = `fak dojo — the prediction-vs-reality gym

usage:
  fak dojo run --corpus DIR [--ttl 5m|1h] [--lever a,b] [--max-files N]
               [--json] [--check] [--append-history] [--ledger FILE]
               [--workspace DIR] [--date YYYY-MM-DD]
  fak dojo list [--json]

run    scores every registered lever's predicted saving against billed reality
       over the corpus, folds the episodes, and (with --append-history) trends
       the mean calibration error across runs in docs/dojo/history.jsonl.
list   shows the registered levers and the metrics each one scores.

A lever's THEORY is its Claimed number; the provider's own usage records are the
ground truth; an episode's verdict says whether reality met the claim
(CALIBRATED), fell short of it (OVER_CLAIM), or beat it (UNDER_CLAIM).

example (score the resume-posture predictor against your real session history):
  fak dojo run --corpus ~/.claude/projects --append-history`

func cmdDojo(argv []string) { os.Exit(runDojo(os.Stdout, os.Stderr, argv)) }

func runDojo(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, dojoUsage)
		return 2
	}
	switch argv[0] {
	case "run":
		return runDojoRun(stdout, stderr, argv[1:])
	case "list":
		return runDojoList(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		fmt.Fprintln(stdout, dojoUsage)
		return 0
	default:
		fmt.Fprintf(stderr, "fak dojo: unknown subcommand %q (want run or list)\n", argv[0])
		return 2
	}
}

func runDojoRun(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dojo run", flag.ContinueOnError)
	fs.SetOutput(stderr)
	corpus := fs.String("corpus", "", "directory of real Claude Code transcripts (.jsonl, scanned recursively) to score the levers against")
	ttlStr := fs.String("ttl", "5m", "provider cache TTL tier the resume-posture lever scores at: 5m (default) or 1h")
	maxFiles := fs.Int("max-files", 0, "cap the number of transcript files scanned (0 = no cap)")
	leverSel := fs.String("lever", "", "comma-separated levers to run (default: all registered; see `fak dojo list`)")
	asJSON := fs.Bool("json", false, "emit the dojo report as JSON instead of the human table")
	check := fs.Bool("check", false, "advisory gate: exit non-zero only if the run could not be measured")
	appendHistory := fs.Bool("append-history", false, "append a dated row to the durable ledger (docs/dojo/history.jsonl)")
	ledger := fs.String("ledger", "", "ledger path override (default: <root>/"+dojo.DefaultLedgerRel+")")
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	date := fs.String("date", "", "snapshot date YYYY-MM-DD (default: today UTC)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *corpus == "" {
		fmt.Fprintln(stderr, "fak dojo run: need --corpus DIR (a directory of .jsonl transcripts)")
		return 2
	}
	ttl, ok := parseResumeTTL(*ttlStr)
	if !ok {
		fmt.Fprintf(stderr, "fak dojo run: bad --ttl %q (want 5m or 1h)\n", *ttlStr)
		return 2
	}

	levers := registerDojoLevers(ttl, *maxFiles)
	if sel := strings.TrimSpace(*leverSel); sel != "" {
		levers = filterDojoLevers(levers, strings.Split(sel, ","))
		if len(levers) == 0 {
			fmt.Fprintf(stderr, "fak dojo run: no lever matched %q (see `fak dojo list`)\n", sel)
			return 2
		}
	}

	scenario := dojo.Scenario{
		Name:   filepath.Base(filepath.Clean(*corpus)),
		Mode:   "offline",
		Corpus: *corpus,
		Note:   "replay of recorded Claude Code transcripts",
	}

	episodes, runErrs := dojo.Run([]dojo.Scenario{scenario}, levers, dojo.DefaultCalibBand())
	dojo.SortEpisodes(episodes)
	for _, re := range runErrs {
		fmt.Fprintf(stderr, "fak dojo run: lever %q on %q: %s\n", re.Lever, re.Scenario, re.Err)
	}

	now := time.Now().UTC()
	snapDate := *date
	if snapDate == "" {
		snapDate = now.Format("2006-01-02")
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	} else if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}

	report := dojo.Fold(episodes, dojo.FoldOpts{
		Workspace:   root,
		Commit:      dojoHeadCommit(root),
		GeneratedAt: now.Format(time.RFC3339),
		Date:        snapDate,
	})

	// Attach the per-tick trend vs the last ledger row (read-only), and -- only
	// under --append-history -- durably append this tick so the trend accrues.
	ledgerPath := *ledger
	if ledgerPath == "" {
		ledgerPath = filepath.Join(root, filepath.FromSlash(dojo.DefaultLedgerRel))
	}
	row := dojo.RowFromReport(report)
	prior := readDojoLedgerRows(ledgerPath)
	trend := dojo.TrendVsLast(row, prior)
	report.Trend = &trend
	if *appendHistory {
		if err := appendDojoLedgerRow(ledgerPath, row); err != nil {
			fmt.Fprintf(stderr, "fak dojo run: append ledger: %v\n", err)
			return 1
		}
		if !*asJSON && !*check {
			rel, _ := filepath.Rel(root, ledgerPath)
			if rel == "" {
				rel = ledgerPath
			}
			fmt.Fprintf(stdout, "appended dojo row -> %s\n", filepath.ToSlash(rel))
		}
	}

	if *check {
		code, message := dojo.CheckGate(report)
		if *asJSON {
			emitDojoJSON(stdout, report.WithGate(code, message))
		} else {
			fmt.Fprintln(stdout, message)
		}
		return code
	}

	if *asJSON {
		emitDojoJSON(stdout, report)
	} else {
		fmt.Fprintln(stdout, dojo.Render(report))
	}
	if report.OK {
		return 0
	}
	return 1
}

func runDojoList(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dojo list", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the registry as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	cat := dojoLeverCatalog()
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		enc.SetEscapeHTML(false)
		_ = enc.Encode(cat)
		return 0
	}
	fmt.Fprintln(stdout, "registered dojo levers (predict a saving -> score it against billed reality):")
	for _, l := range cat {
		fmt.Fprintf(stdout, "  %-15s %s\n", l.Name, l.Summary)
		for _, m := range l.Metrics {
			fmt.Fprintf(stdout, "      - %-28s %s\n", m.Name, m.Theory)
		}
	}
	return 0
}

// --- the lever registry ----------------------------------------------------

type dojoMetricInfo struct {
	Name   string `json:"name"`
	Theory string `json:"theory"`
}

type dojoLeverInfo struct {
	Name    string           `json:"name"`
	Summary string           `json:"summary"`
	Metrics []dojoMetricInfo `json:"metrics"`
}

// dojoLeverCatalog is the static description of the registered levers and the
// metrics each scores, for `fak dojo list`. It is kept in lockstep with the
// episodes the levers actually emit (resumeEpisodesFromBacktest).
func dojoLeverCatalog() []dojoLeverInfo {
	return []dojoLeverInfo{
		{
			Name:    "resume-posture",
			Summary: "the resume cache-posture projection, scored against the provider's billed cache_read/cache_creation",
			Metrics: []dojoMetricInfo{
				{Name: "posture_accuracy", Theory: "the projection's per-boundary cold/warm call is correct (claim 1.0)"},
				{Name: "cold_write_share", Theory: "a cold resume re-prefill rewrites ~85% of the resident at the write premium (claim 0.85)"},
				{Name: "cross_session_warm_hit_rate", Theory: "~17% of large first-turn resumes hit a still-warm cross-session prefix (claim 0.17)"},
			},
		},
		{
			Name:    "compaction",
			Summary: "history compaction's cache-prefix preservation and token shed scored against billed reality",
			Metrics: []dojoMetricInfo{
				{Name: "cache_prefix_preserved", Theory: "a fired compaction keeps the prefix byte-identical so the provider still cache-reads it (claim 1.0)"},
				{Name: "token_shed_ratio", Theory: "the projected shed for the budget matches the billed delta (claim 1.0)"},
			},
		},
	}
}

func registerDojoLevers(ttl resume.CacheTTL, maxFiles int) []dojo.Lever {
	return []dojo.Lever{
		resumePostureLever{ttl: ttl, maxFiles: maxFiles},
		compactionLever{},
	}
}

func filterDojoLevers(all []dojo.Lever, names []string) []dojo.Lever {
	want := map[string]bool{}
	for _, n := range names {
		if t := strings.TrimSpace(n); t != "" {
			want[t] = true
		}
	}
	var out []dojo.Lever
	for _, lv := range all {
		if want[lv.Name()] {
			out = append(out, lv)
		}
	}
	return out
}

// --- the resume-posture lever ----------------------------------------------

// resumePostureLever scores the resume cache-posture projection against billed
// reality. It reuses the exact corpus scan and resume.Backtest residual the
// `fak resume validate` shell uses, then adapts the report into dojo episodes.
type resumePostureLever struct {
	ttl      resume.CacheTTL
	maxFiles int
}

func (resumePostureLever) Name() string { return "resume-posture" }

func (l resumePostureLever) Episodes(s dojo.Scenario) ([]dojo.ScoredInput, error) {
	files, err := findTranscripts(s.Corpus)
	if err != nil {
		return nil, err
	}
	if l.maxFiles > 0 && len(files) > l.maxFiles {
		files = files[:l.maxFiles]
	}
	sessions := make([][]resume.ObservedTurn, 0, len(files))
	for _, p := range files {
		turns := loadTranscriptTurns(p)
		if len(turns) >= 2 { // a session needs at least one adjacent pair to score
			sessions = append(sessions, turns)
		}
	}
	rep := resume.Backtest(sessions, l.ttl, resume.DefaultRecoveryBand())
	return resumeEpisodesFromBacktest(rep), nil
}

// resumeEpisodesFromBacktest adapts a resume.BacktestReport into the dojo's
// (prediction, outcome) pairs. It is pure so the mapping is unit-testable
// without scanning a corpus. Each metric pairs the projection's THEORY (a
// Claimed number) with the provider's OBSERVED reality.
func resumeEpisodesFromBacktest(rep resume.BacktestReport) []dojo.ScoredInput {
	var out []dojo.ScoredInput

	// posture_accuracy — theory: the projection's cold/warm call is right (1.0);
	// reality: the share that agreed with the provider's billed cache_read.
	if rep.Scored > 0 {
		out = append(out, dojo.ScoredInput{
			Prediction: dojo.Prediction{
				Lever: "resume-posture", Metric: "posture_accuracy", Claimed: 1.0, Unit: "fraction",
				Basis: "the resume projection's per-boundary cold/warm posture call assumed correct",
			},
			Outcome: dojo.Outcome{
				Realized: rep.Accuracy, Provenance: dojo.Observed, Measured: true, Sample: rep.Scored,
				Source: "provider cache_read recovery vs projected posture",
			},
		})
	}

	// cold_write_share — theory: a cold resume re-prefill rewrites ~85% of
	// the resident at the write premium (0.85); reality: the measured write share.
	if rep.FirstTurnCold > 0 {
		out = append(out, dojo.ScoredInput{
			Prediction: dojo.Prediction{
				Lever: "resume-posture", Metric: "cold_write_share", Claimed: 0.85, Unit: "fraction",
				Basis: "the projection prices ~85% of the resident at the cold-write premium (share = 0.85)",
			},
			Outcome: dojo.Outcome{
				Realized: rep.FirstTurnColdWriteShareMean, Provenance: dojo.Observed, Measured: true, Sample: rep.FirstTurnCold,
				Source: "provider cache_creation/prompt on cold cross-file resume re-prefills",
			},
		})
	}

	// cross_session_warm_hit_rate — theory: ~17% of large first-turn resumes hit
	// a still-warm cross-session prefix (0.17); reality: the share of large
	// first turns that hit a still-warm cross-session prefix.
	if rep.FirstTurnResumes > 0 {
		warm := float64(rep.FirstTurnWarmHit) / float64(rep.FirstTurnResumes)
		out = append(out, dojo.ScoredInput{
			Prediction: dojo.Prediction{
				Lever: "resume-posture", Metric: "cross_session_warm_hit_rate", Claimed: 0.17, Unit: "fraction",
				Basis: "~17% of large first-turn resumes hit a still-warm cross-session prefix",
			},
			Outcome: dojo.Outcome{
				Realized: warm, Provenance: dojo.Observed, Measured: true, Sample: rep.FirstTurnResumes,
				Source: "provider cache_read>~0 on the first turn of resumed transcripts",
			},
		})
	}
	return out
}

// --- output + durable ledger I/O -------------------------------------------

// dojoHeadCommit returns the short HEAD commit for the durable ledger row, or
// "unknown" when git is unavailable. Kept local so the dojo shell carries no
// dependency on other leaf packages for a one-line git fact.
func dojoHeadCommit(root string) string {
	cmd := exec.Command("git", "-C", root, "rev-parse", "--short", "HEAD")
	out, err := cmd.Output()
	if err != nil {
		return "unknown"
	}
	return strings.TrimSpace(string(out))
}

func emitDojoJSON(w io.Writer, r dojo.Report) {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	enc.SetEscapeHTML(false)
	_ = enc.Encode(r)
}

// readDojoLedgerRows reads the durable ledger if present (absent ledger -> no
// prior rows, the first tick establishes the series).
func readDojoLedgerRows(path string) []dojo.LedgerRow {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	return dojo.ParseLedger(string(raw))
}

// appendDojoLedgerRow appends one JSONL row to the ledger, creating the parent
// directory on first write.
func appendDojoLedgerRow(path string, row dojo.LedgerRow) error {
	line, err := dojo.AppendLedgerLine(row)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.WriteString(line + "\n")
	return err
}

// --- the compaction lever -------------------------------------------------

// CompactionBacktestReport is the residual of compaction theory against billed reality:
// how often compaction preserved the cache prefix (verified by provider cache_read or
// absence of prefix_mismatch bails) and how accurately the projected shed matched the
// billed token delta. Every field is a count or ratio over observed data — no dollar
// here is a fak figure, they are the provider's own token records or fak's own witnessed
// metrics scored against the projection.
type CompactionBacktestReport struct {
	// Cache prefix preservation metrics
	FiredAttempts         int     `json:"fired_attempts"`         // WITNESSED: compactions that fired (body rewritten, protected prefix shipped byte-identical)
	PrefixMismatchBails   int     `json:"prefix_mismatch_bails"`   // WITNESSED: compactions that bailed with prefix_mismatch (fak-fault signal)
	PostFireCacheReadSum  uint64  `json:"post_fire_cache_read_sum"` // OBSERVED: cumulative provider cache_read on compacted turns
	PostFireCacheReadN    int     `json:"post_fire_cache_read_n"`   // N for the sum (turns with post-fire cache_read)

	// Token shed metrics
	ShedTokensSum         uint64  `json:"shed_tokens_sum"`         // WITNESSED: cumulative tokens fak claims to have shed
	InputTokensOffSum     uint64  `json:"input_tokens_off_sum"`     // OBSERVED: cumulative provider input_tokens on compacted turns with compaction OFF
	InputTokensOnSum      uint64  `json:"input_tokens_on_sum"`      // OBSERVED: cumulative provider input_tokens on compacted turns with compaction ON
}

// compactionLever scores history compaction's theory against billed reality.
// It reads access logs or audit journal entries that contain compaction metrics,
// runs a back-test, and adapts the report into dojo episodes.
type compactionLever struct{}

func (compactionLever) Name() string { return "compaction" }

func (compactionLever) Episodes(s dojo.Scenario) ([]dojo.ScoredInput, error) {
	// TODO: Read access logs/audit journal entries from s.Corpus and build a
	// CompactionBacktestReport. The corpus format is TBD (see #953 for the
	// intended shape: access logs containing compaction outcomes and billing data).
	// For now, return empty slices so the lever is registered and discoverable
	// but produces no episodes until the corpus format is defined.
	return nil, nil
}

// compactionEpisodesFromBacktest adapts a CompactionBacktestReport into the dojo's
// (prediction, outcome) pairs. It is pure so the mapping is unit-testable
// without scanning a corpus. Each metric pairs the projection's THEORY (a
// Claimed number) with the OBSERVED or WITNESSED reality.
func compactionEpisodesFromBacktest(rep CompactionBacktestReport) []dojo.ScoredInput {
	var out []dojo.ScoredInput

	// cache_prefix_preserved — theory: a fired compaction keeps the prefix
	// byte-identical so the provider still cache-reads it (claim 1.0); reality:
	// the inverse of the prefix_mismatch bail rate (a single prefix_mismatch
	// drives the verdict to OVER_CLAIM/F).
	if rep.FiredAttempts > 0 {
		prefixMismatchRate := float64(rep.PrefixMismatchBails) / float64(rep.FiredAttempts)
		preserved := 1.0 - prefixMismatchRate
		out = append(out, dojo.ScoredInput{
			Prediction: dojo.Prediction{
				Lever: "compaction", Metric: "cache_prefix_preserved", Claimed: 1.0, Unit: "fraction",
				Basis: "a fired compaction ships the protected prefix byte-identical",
			},
			Outcome: dojo.Outcome{
				Realized: preserved, Provenance: dojo.Witnessed, Measured: true, Sample: rep.FiredAttempts,
				Source: "inverse of prefix_mismatch bail rate over fired attempts (WITNESSED)",
			},
		})
	}

	// token_shed_ratio — theory: the projected shed for the budget (shed_tokens)
	// matches the billed delta (input_tokens off - on); reality: the ratio of
	// billed delta to projected shed. Claim 1.0 = perfect calibration.
	if rep.ShedTokensSum > 0 && rep.InputTokensOnSum > 0 {
		billedDelta := float64(rep.InputTokensOffSum - rep.InputTokensOnSum)
		// Guard against pathological cases: billed delta can't be negative (ON must be
		// <= OFF for a successful compaction). A negative delta is OVER_CLAIM.
		if billedDelta < 0 {
			billedDelta = 0
		}
		ratio := billedDelta / float64(rep.ShedTokensSum)
		// Cap at 1.0: the billed delta cannot exceed the projected shed by more than
		// the compaction's actual savings (the provider might cache-read more than
		// expected). Values > 1.0 are clamped to 1.0 for the OVER_CLAIM/UNDER_CLAIM
		// verdict interpretation.
		if ratio > 1.0 {
			ratio = 1.0
		}
		sample := rep.FiredAttempts
		if sample == 0 {
			sample = 1
		}
		out = append(out, dojo.ScoredInput{
			Prediction: dojo.Prediction{
				Lever: "compaction", Metric: "token_shed_ratio", Claimed: 1.0, Unit: "fraction",
				Basis: "the projected shed (WITNESSED shed_tokens) matches the billed delta",
			},
			Outcome: dojo.Outcome{
				Realized: ratio, Provenance: dojo.Witnessed, Measured: true, Sample: sample,
				Source: "billed input_tokens delta (OFF - ON) divided by projected shed (WITNESSED)",
			},
		})
	}

	return out
}
