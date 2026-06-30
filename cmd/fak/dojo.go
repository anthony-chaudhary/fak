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
	"flag"
	"fmt"
	"io"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/bench"
	"github.com/anthony-chaudhary/fak/internal/dojo"
	"github.com/anthony-chaudhary/fak/internal/dojopost"
	"github.com/anthony-chaudhary/fak/internal/metrics"
	"github.com/anthony-chaudhary/fak/internal/resume"
	"github.com/anthony-chaudhary/fak/internal/vcachecal"
	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const dojoUsage = `fak dojo — the prediction-vs-reality gym

usage:
  fak dojo run    (--corpus DIR | --live) [--ttl 5m|1h] [--lever a,b] [--max-files N]
                  [--json] [--check] [--append-history] [--ledger FILE]
                  [--workspace DIR] [--date YYYY-MM-DD]
  fak dojo board  --corpus DIR [--ttl 5m|1h] [--max-files N] [--json]
  fak dojo ablate [--trace FILE | --suite NAME] [--engine ID] [--json] [--check]
  fak dojo list   [--json]
  fak dojo post   [--rollup latest|trend] [--corpus DIR] [--ledger FILE]
                  [--dry-run] [--channel ID] [--token TOK] [--source WHO]

run    scores every registered lever's predicted saving against billed reality
       over the corpus, folds the episodes, and (with --append-history) trends
       the mean calibration error across runs in docs/dojo/history.jsonl. With
       --live (auto-selected when no --corpus is given and one exists) it
       discovers the .dojo/live-episodes corpus that --dojo writes; those markers
       are start-only today, so it surfaces what it found and reports what is
       missing to score them rather than inventing a calibration.
board  folds the same run into a cross-lever leaderboard: one row per lever
       (verdict distribution, mean calib-err, worst metric, grade), worst-first.
ablate scores the vDSO on-vs-off ablation over a trace as a WITNESSED episode:
       the engine-call elision the fast path actually delivered vs the claim.
list   shows the registered levers and the metrics each one scores.
post   posts a calibration rollup to the dojo Slack channel: --rollup trend
       (default) folds the committed history ledger into the across-tick
       calibration trend (no corpus scan); --rollup latest scores the corpus
       and posts the latest run. Safe by default (--dry-run renders without
       posting); the token falls back to the scoreboard token.

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
	case "ablate":
		return runDojoAblate(stdout, stderr, argv[1:])
	case "board":
		return runDojoBoard(stdout, stderr, argv[1:])
	case "post":
		return runDojoPost(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		fmt.Fprintln(stdout, dojoUsage)
		return 0
	default:
		fmt.Fprintf(stderr, "fak dojo: unknown subcommand %q (want run, list, ablate, board, or post)\n", argv[0])
		return 2
	}
}

func runDojoRun(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dojo run", flag.ContinueOnError)
	fs.SetOutput(stderr)
	corpus := fs.String("corpus", "", "directory of real Claude Code transcripts (.jsonl, scanned recursively) to score the levers against")
	live := fs.Bool("live", false, "score the live-episode corpus `fak guard --dojo` / `fak serve --dojo` write under <root>/"+dojo.LiveEpisodesRel+" instead of --corpus. Auto-selected when neither --corpus nor --live is given and that corpus exists. The markers are start-only today, so this DISCOVERS + surfaces them and reports what is missing to score them (it never fabricates a calibration).")
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

	// Resolve the workspace root early — the live corpus discovery + auto-select
	// below need it, and the fold reuses it.
	root := *workspace
	if root == "" {
		root = repoRoot()
	} else if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}

	// Live-episode path: explicit --live, or (auto) no --corpus AND a live corpus
	// is present. This forward-wires the corpus `--dojo` already writes into the
	// same scoring pipeline; when the markers are start-only it degrades gracefully
	// (surface + explain) rather than crashing or inventing a score. --corpus
	// always wins over auto-select so the documented transcript path is unchanged.
	useLive := *live
	if !useLive && *corpus == "" {
		if lc, err := dojo.ReadLiveCorpus(filepath.Join(root, filepath.FromSlash(dojo.LiveEpisodesRel))); err == nil && lc.Present {
			useLive = true
		}
	}
	if useLive {
		return runDojoLive(stdout, stderr, root, *asJSON, *check)
	}

	if *corpus == "" {
		fmt.Fprintln(stderr, "fak dojo run: need --corpus DIR (a directory of .jsonl transcripts), or --live to discover the .dojo/live-episodes corpus `--dojo` writes")
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

	episodes := runDojoScenario(stderr, *corpus, levers, "run")

	now := time.Now().UTC()
	snapDate := *date
	if snapDate == "" {
		snapDate = now.Format("2006-01-02")
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

// runDojoLive discovers and folds the live-episode corpus that `fak guard --dojo`
// / `fak serve --dojo` write under <root>/.dojo/live-episodes/. It forward-wires
// that corpus (which nothing read before #1093) into the same scoring pipeline
// the --corpus path uses, then folds whatever it can score.
//
// Today the writer is start-only — each marker carries {mode, command, started,
// workspace} and no billed usage — so there is nothing to score yet. Rather than
// crash or silently score zero, it DEGRADES GRACEFULLY: it surfaces the episodes
// it found and reports, in plain words, what is missing to score them. The
// scorable-marker seam (dojo.ScorableLiveEpisodes) is already wired, so when the
// writer side later captures full episodes those flow straight into the fold with
// no change here. Fail-open: a missing/empty corpus is a clean "nothing recorded
// yet", never an error.
func runDojoLive(stdout, stderr io.Writer, root string, asJSON, check bool) int {
	dir := filepath.Join(root, filepath.FromSlash(dojo.LiveEpisodesRel))
	lc, err := dojo.ReadLiveCorpus(dir)
	if err != nil {
		// A genuine read fault (e.g. a permission error on an existing dir) is
		// reported, but a missing/empty corpus already returned (nil) above.
		fmt.Fprintf(stderr, "fak dojo run --live: read live corpus: %v\n", err)
		return 1
	}

	// Score whatever the corpus carries. While the markers are start-only this is
	// empty, so the fold is honestly unmeasured rather than fabricated.
	var episodes []dojo.Episode
	for _, in := range dojo.ScorableLiveEpisodes(lc) {
		episodes = append(episodes, dojo.Score("live-episodes", in.Prediction, in.Outcome, dojo.DefaultCalibBand()))
	}
	dojo.SortEpisodes(episodes)
	now := time.Now().UTC()
	report := dojo.Fold(episodes, dojo.FoldOpts{
		Workspace:   root,
		Commit:      dojoHeadCommit(root),
		GeneratedAt: now.Format(time.RFC3339),
		Date:        now.Format("2006-01-02"),
	})

	if asJSON {
		// Carry the discovery summary alongside the folded report so a JSON
		// consumer sees BOTH the (honest, possibly-empty) score and what was on
		// disk — found count, the markers, and what is missing to score them.
		_ = writeIndentedJSONNoEscape(stdout, dojoLiveJSON{Report: report, Live: lc})
	} else {
		fmt.Fprintln(stdout, dojo.Render(report))
		fmt.Fprintln(stdout, renderLiveCorpus(lc))
	}

	if check {
		// The advisory gate only fails when the run could not be measured. A live
		// corpus with start-only markers is, honestly, unmeasured.
		code, _ := dojo.CheckGate(report)
		return code
	}
	if report.OK {
		return 0
	}
	return 1
}

// dojoLiveJSON is the --live --json envelope: the folded report plus the raw
// live-corpus discovery, so a consumer can see the score AND the disk state in
// one object.
type dojoLiveJSON struct {
	Report dojo.Report     `json:"report"`
	Live   dojo.LiveCorpus `json:"live_corpus"`
}

// renderLiveCorpus is the human discovery block printed under the folded report
// on the live path: how many start-markers were found and what is missing to
// score them, so an operator can see the loop is recording even before the
// full-episode writer lands.
func renderLiveCorpus(lc dojo.LiveCorpus) string {
	var b strings.Builder
	rel := lc.Dir
	if r, err := filepath.Rel(repoRoot(), lc.Dir); err == nil && !strings.HasPrefix(r, "..") {
		rel = filepath.ToSlash(r)
	}
	fmt.Fprintf(&b, "\n  live-episode corpus: %s\n", rel)
	if !lc.Present {
		b.WriteString("    (none recorded yet — enable `fak guard --dojo` / `fak serve --dojo` to start recording)\n")
		return strings.TrimRight(b.String(), "\n")
	}
	fmt.Fprintf(&b, "    found %d start-marker(s); %d scorable\n", lc.Found, lc.Scorable)
	for _, m := range lc.Markers {
		fmt.Fprintf(&b, "      - %s  (%s, started %s)\n", m.File, m.Command, m.Started)
	}
	if lc.Missing != "" {
		fmt.Fprintf(&b, "    missing to score: %s\n", lc.Missing)
	}
	return strings.TrimRight(b.String(), "\n")
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
		_ = writeIndentedJSONNoEscape(stdout, cat)
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

// runDojoAblate scores the vDSO ablation: it replays a trace through the kernel
// with the fast path ON vs OFF and folds the measured engine-call elision into a
// WITNESSED dojo episode. The claim is "vDSO ON elides every call the OFF arm
// sent to the engine"; reality is the measured (off-on)/off engine-call drop.
// CPU-only and corpus-free (the offline mock engine + a built-in suite trace).
func runDojoAblate(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dojo ablate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	tracePath := fs.String("trace", "", "trace file to replay (default: the built-in suite)")
	suite := fs.String("suite", "", "named suite under the trace dir (default: the first suite)")
	engine := fs.String("engine", "mock", "engine id (the offline mock by default)")
	asJSON := fs.Bool("json", false, "emit the dojo report as JSON instead of the human table")
	check := fs.Bool("check", false, "advisory gate: exit non-zero only if the run could not be measured")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	path := *tracePath
	if path == "" {
		path = resolveSuite(traceDir(), *suite)
	}
	tr, err := bench.LoadTrace(path)
	if err != nil {
		fmt.Fprintln(stderr, "fak dojo ablate:", err)
		return 1
	}
	on, err := bench.RunArm(ctx(), tr, *engine, true, "vdso-on")
	if err != nil {
		fmt.Fprintln(stderr, "fak dojo ablate: on arm:", err)
		return 1
	}
	off, err := bench.RunArm(ctx(), tr, *engine, false, "vdso-off")
	if err != nil {
		fmt.Fprintln(stderr, "fak dojo ablate: off arm:", err)
		return 1
	}

	var episodes []dojo.Episode
	for _, in := range ablateEpisodesFromArms(on, off) {
		episodes = append(episodes, dojo.Score("vdso-ablation", in.Prediction, in.Outcome, dojo.DefaultCalibBand()))
	}
	dojo.SortEpisodes(episodes)
	report := dojo.Fold(episodes, dojo.FoldOpts{
		Workspace:   repoRoot(),
		Commit:      dojoHeadCommit(repoRoot()),
		GeneratedAt: time.Now().UTC().Format(time.RFC3339),
		Date:        time.Now().UTC().Format("2006-01-02"),
	})
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

// ablateEpisodesFromArms adapts an ON/OFF arm pair into WITNESSED dojo episodes.
// It is pure so the mapping is unit-testable without running the kernel. The
// engine-call elision is the headline: vDSO theory is it serves every fast-path
// call locally, so the claim is 1.0 (full elision of the OFF arm's engine calls)
// and reality is the measured drop fraction.
func ablateEpisodesFromArms(on, off metrics.Arm) []dojo.ScoredInput {
	var out []dojo.ScoredInput
	if off.EngineCalls > 0 {
		elided := float64(off.EngineCalls-on.EngineCalls) / float64(off.EngineCalls)
		if elided < 0 {
			elided = 0
		}
		out = append(out, dojo.ScoredInput{
			Prediction: dojo.Registry.MustPredict("vdso-ablation", "engine_call_elision", "fraction"),
			Outcome: dojo.Outcome{
				Realized: elided, Provenance: dojo.Witnessed, Measured: true, Sample: int(off.EngineCalls),
				Source: "(off.engine_calls - on.engine_calls) / off.engine_calls over the replayed trace (WITNESSED)",
			},
		})
	}
	return out
}

// runDojoBoard runs the registered levers over the corpus and folds the scored
// episodes into the cross-lever leaderboard (one row per lever, worst-first).
func runDojoBoard(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dojo board", flag.ContinueOnError)
	fs.SetOutput(stderr)
	corpus := fs.String("corpus", "", "directory of real Claude Code transcripts to score the levers against")
	ttlStr := fs.String("ttl", "5m", "provider cache TTL tier the resume-posture lever scores at: 5m or 1h")
	maxFiles := fs.Int("max-files", 0, "cap the number of transcript files scanned (0 = no cap)")
	asJSON := fs.Bool("json", false, "emit the board as JSON instead of the human table")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *corpus == "" {
		fmt.Fprintln(stderr, "fak dojo board: need --corpus DIR (a directory of .jsonl transcripts)")
		return 2
	}
	ttl, ok := parseResumeTTL(*ttlStr)
	if !ok {
		fmt.Fprintf(stderr, "fak dojo board: bad --ttl %q (want 5m or 1h)\n", *ttlStr)
		return 2
	}
	scenario := dojo.Scenario{
		Name:   filepath.Base(filepath.Clean(*corpus)),
		Mode:   "offline",
		Corpus: *corpus,
		Note:   "replay of recorded Claude Code transcripts",
	}
	episodes, runErrs := dojo.Run([]dojo.Scenario{scenario}, registerDojoLevers(ttl, *maxFiles), dojo.DefaultCalibBand())
	for _, re := range runErrs {
		fmt.Fprintf(stderr, "fak dojo board: lever %q on %q: %s\n", re.Lever, re.Scenario, re.Err)
	}
	board := dojo.BoardFromEpisodes(episodes)
	if *asJSON {
		_ = writeIndentedJSONNoEscape(stdout, board)
		return 0
	}
	fmt.Fprintln(stdout, dojo.RenderBoard(board))
	if len(board.Rows) == 0 {
		return 1
	}
	return 0
}

// runDojoPost posts a dojo calibration rollup to the dojo Slack channel. It is the
// outbound dojo-channel surface — the twin of `fak bench post` / `fak scoreboard post`.
//
//	fak dojo post                              # the across-tick calibration trend (default)
//	fak dojo post --rollup latest --corpus DIR # score the corpus and post the latest run
//	fak dojo post --dry-run                    # render the card and print it; do not post
//
// --rollup trend (default) reads the committed history ledger (docs/dojo/history.jsonl)
// and folds it into the across-tick trend WITHOUT a corpus scan, so CI can post it on a
// cadence cheaply. --rollup latest scores the registered levers over --corpus (the same
// scan `fak dojo run` does) and posts the latest folded run. Safe by default: --dry-run
// renders without posting, and the channel/token resolve from FAK_DOJO_* (token falls
// back to the scoreboard token; channel defaults to the public dojo channel).
func runDojoPost(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dojo post", flag.ContinueOnError)
	fs.SetOutput(stderr)
	rollup := fs.String("rollup", "trend", "which rollup: trend (committed history, no scan) | latest (score --corpus)")
	corpus := fs.String("corpus", "", "latest rollup: directory of real Claude Code transcripts to score the levers against")
	ttlStr := fs.String("ttl", "5m", "latest rollup: provider cache TTL tier the resume-posture lever scores at: 5m or 1h")
	maxFiles := fs.Int("max-files", 0, "latest rollup: cap the number of transcript files scanned (0 = no cap)")
	ledger := fs.String("ledger", "", "trend rollup: ledger path override (default: <root>/"+dojo.DefaultLedgerRel+")")
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	n := fs.Int("n", 6, "trend rollup: number of recent ticks to show; latest rollup: number of worst episodes to show")
	source := fs.String("source", "", "who is posting: ci | agent | <hostname> (default: $FAK_SCOREBOARD_SOURCE or hostname)")
	channel := fs.String("channel", "", "override target channel id (default: $FAK_DOJO_CHANNEL / .env.slack.local / the public dojo channel)")
	token := fs.String("token", "", "override bot token (default: $FAK_DOJO_TOKEN, then scoreboard token)")
	dryRun := fs.Bool("dry-run", false, "render the message and print it; do not post to Slack")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	root := *workspace
	if root == "" {
		root = repoRoot()
	} else if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}

	var post dojopost.Post
	switch *rollup {
	case "trend":
		ledgerPath := *ledger
		if ledgerPath == "" {
			ledgerPath = filepath.Join(root, filepath.FromSlash(dojo.DefaultLedgerRel))
		}
		rows := readDojoLedgerRows(ledgerPath)
		post = dojopost.TrendFromLedger(rows, *n)
	case "latest":
		if *corpus == "" {
			fmt.Fprintln(stderr, "fak dojo post --rollup latest: need --corpus DIR (a directory of .jsonl transcripts)")
			return 2
		}
		ttl, ok := parseResumeTTL(*ttlStr)
		if !ok {
			fmt.Fprintf(stderr, "fak dojo post: bad --ttl %q (want 5m or 1h)\n", *ttlStr)
			return 2
		}
		report := foldDojoCorpusRun(*corpus, ttl, *maxFiles, root, stderr)
		// Attach the across-tick trend (read-only) so the latest card also answers
		// "are we improving" — the same trend `fak dojo run` shows.
		ledgerPath := *ledger
		if ledgerPath == "" {
			ledgerPath = filepath.Join(root, filepath.FromSlash(dojo.DefaultLedgerRel))
		}
		row := dojo.RowFromReport(report)
		trend := dojo.TrendVsLast(row, readDojoLedgerRows(ledgerPath))
		report.Trend = &trend
		post = dojopost.RollupFromReport(report, *n)
	default:
		fmt.Fprintf(stderr, "fak dojo post: unknown --rollup %q (want: trend | latest)\n", *rollup)
		return 2
	}
	post.Source = resolveDojoSource(*source)

	return emitDojoPost(stdout, stderr, post, *channel, *token, *dryRun)
}

// foldDojoCorpusRun scores the registered levers over the corpus and folds the
// episodes into one report — the same path `runDojoRun` takes, factored here so
// `fak dojo post --rollup latest` reuses it. Run errors are reported to stderr but do
// not abort (a partial run still folds the episodes it produced, exactly as `fak dojo
// run` does).
// runDojoScenario replays a single offline corpus through the given levers, sorting the
// episodes and draining any per-lever run errors to stderr — the scenario-build + Run +
// sort + error-drain block the `dojo run` and `dojo post` paths share. label tags the
// error line ("run" / "post").
func runDojoScenario(stderr io.Writer, corpus string, levers []dojo.Lever, label string) []dojo.Episode {
	scenario := dojo.Scenario{
		Name:   filepath.Base(filepath.Clean(corpus)),
		Mode:   "offline",
		Corpus: corpus,
		Note:   "replay of recorded Claude Code transcripts",
	}
	episodes, runErrs := dojo.Run([]dojo.Scenario{scenario}, levers, dojo.DefaultCalibBand())
	dojo.SortEpisodes(episodes)
	for _, re := range runErrs {
		fmt.Fprintf(stderr, "fak dojo %s: lever %q on %q: %s\n", label, re.Lever, re.Scenario, re.Err)
	}
	return episodes
}

func foldDojoCorpusRun(corpus string, ttl resume.CacheTTL, maxFiles int, root string, stderr io.Writer) dojo.Report {
	episodes := runDojoScenario(stderr, corpus, registerDojoLevers(ttl, maxFiles), "post")
	now := time.Now().UTC()
	return dojo.Fold(episodes, dojo.FoldOpts{
		Workspace:   root,
		Commit:      dojoHeadCommit(root),
		GeneratedAt: now.Format(time.RFC3339),
		Date:        now.Format("2006-01-02"),
	})
}

// emitDojoPost is the dry-run / post tail: it prints the rendered card on --dry-run, or
// resolves the channel + token and posts to Slack. It reuses the scoreboard transport
// (a plain chat.postMessage client), matching `fak bench post`.
func emitDojoPost(stdout, stderr io.Writer, post dojopost.Post, channel, token string, dryRun bool) int {
	return slackPostTail(stdout, stderr, slackPostSpec{
		card:           post,
		channel:        channel,
		token:          token,
		dryRun:         dryRun,
		label:          "fak dojo post",
		chanEnv:        "FAK_DOJO_CHANNEL",
		resolveChannel: dojopost.ResolveChannel,
		resolveToken:   dojopost.ResolveToken,
	})
}

// resolveDojoSource picks the post source: the flag, else the shared defaultSource
// ($FAK_SCOREBOARD_SOURCE or hostname).
func resolveDojoSource(flagVal string) string {
	if flagVal != "" {
		return flagVal
	}
	return defaultSource()
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
				{Name: "cross_session_warm_hit_rate", Theory: "~0% by default; workload-dependent and bimodal across corpora (observed 0.00→0.65) (claim 0.0)"},
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
		{
			Name:    "vcache-warmth",
			Summary: "the vCache warmth belief's warm/cold prediction scored against the provider's billed cache_read",
			Metrics: []dojoMetricInfo{
				{Name: "false_warm_rate", Theory: "the warmth belief NEVER predicts warm on a call that bills cache_read=0 (claim 0.0 — the lethal class)"},
				{Name: "warm_recall", Theory: "the belief recalls every genuinely-warm read it could have predicted (claim 1.0)"},
			},
		},
	}
}

func registerDojoLevers(ttl resume.CacheTTL, maxFiles int) []dojo.Lever {
	return []dojo.Lever{
		resumePostureLever{ttl: ttl, maxFiles: maxFiles},
		compactionLever{},
		vcacheLever{maxFiles: maxFiles},
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
		// findTranscripts returns paths sorted lexicographically, so a naive
		// prefix slice front-loads whichever project cluster sorts first. That
		// matters for workload-dependent metrics: dogfooding showed the capped
		// warm-hit rate skewed high (0.47–0.58 at small caps vs 0.16 over the
		// full corpus) purely from the path-ordered prefix. Shuffle with a fixed
		// seed before slicing so a cap is a representative-but-reproducible
		// sample of the whole corpus, not a path-biased one.
		r := rand.New(rand.NewSource(1))
		r.Shuffle(len(files), func(i, j int) { files[i], files[j] = files[j], files[i] })
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
			Prediction: dojo.Registry.MustPredict("resume-posture", "posture_accuracy", "fraction"),
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
			Prediction: dojo.Registry.MustPredict("resume-posture", "cold_write_share", "fraction"),
			Outcome: dojo.Outcome{
				Realized: rep.FirstTurnColdWriteShareMean, Provenance: dojo.Observed, Measured: true, Sample: rep.FirstTurnCold,
				Source: "provider cache_creation/prompt on cold cross-file resume re-prefills",
			},
		})
	}

	// cross_session_warm_hit_rate — theory: ~0% of large first-turn resumes hit a still-warm
	// cross-session prefix by default (0.0), but the rate is highly workload-dependent and
	// bimodal across corpora (observed 0.00→0.65); reality: the share of large first turns
	// that hit a still-warm cross-session prefix.
	if rep.FirstTurnResumes > 0 {
		warm := float64(rep.FirstTurnWarmHit) / float64(rep.FirstTurnResumes)
		out = append(out, dojo.ScoredInput{
			Prediction: dojo.Registry.MustPredict("resume-posture", "cross_session_warm_hit_rate", "fraction"),
			Outcome: dojo.Outcome{
				Realized: warm, Provenance: dojo.Observed, Measured: true, Sample: rep.FirstTurnResumes,
				Source: "provider cache_read>~0 on the first turn of resumed transcripts",
			},
		})
	}
	return out
}

// --- the vcache-warmth lever ----------------------------------------------

// vcacheLever scores the vCache warmth belief's warm/cold prediction against the
// provider's billed cache_read, by replaying the corpus transcripts through the
// already-shipped vcacheobserve.Observe (the same fold `fak vcache observe` and
// /metrics use). The two episodes pair the belief's THEORY (it never false-warms;
// it recalls every genuinely-warm read) with the OBSERVED PredictionError.
type vcacheLever struct {
	maxFiles int
}

func (vcacheLever) Name() string { return "vcache-warmth" }

func (l vcacheLever) Episodes(s dojo.Scenario) ([]dojo.ScoredInput, error) {
	files, err := findTranscripts(s.Corpus)
	if err != nil {
		return nil, err
	}
	if l.maxFiles > 0 && len(files) > l.maxFiles {
		files = files[:l.maxFiles]
	}
	var turns []vcacheobserve.Turn
	for _, p := range files {
		ts, err := readObserveTranscript(p)
		if err != nil {
			continue // a malformed transcript is skipped, not fatal (parity with vcache observe)
		}
		turns = append(turns, ts...)
	}
	if len(turns) == 0 {
		return nil, nil
	}
	rep := vcacheobserve.Observe(turns, vcacheobserve.DefaultMultipliers())
	return vcacheEpisodesFromObserve(rep.Prediction), nil
}

// vcacheEpisodesFromObserve adapts a vcachecal.PredictionError into the dojo's
// (prediction, outcome) pairs. It is pure so the mapping is unit-testable without
// scanning a corpus. Each metric pairs the warmth belief's THEORY (a Claimed
// number) with the provider's OBSERVED reality.
func vcacheEpisodesFromObserve(pe vcachecal.PredictionError) []dojo.ScoredInput {
	var out []dojo.ScoredInput

	// false_warm_rate — theory: the warmth belief never predicts warm on a call
	// that bills cache_read=0 (claim 0.0, the lethal class); reality: the measured
	// FalseWarm / (TrueWarm + FalseWarm).
	if predictedWarm := pe.TrueWarm + pe.FalseWarm; predictedWarm > 0 {
		out = append(out, dojo.ScoredInput{
			Prediction: dojo.Registry.MustPredict("vcache-warmth", "false_warm_rate", "fraction"),
			Outcome: dojo.Outcome{
				Realized: pe.FalseWarmRate(), Provenance: dojo.Observed, Measured: true, Sample: predictedWarm,
				Source: "provider cache_read=0 on believed-warm calls / all believed-warm calls",
			},
		})
	}

	// warm_recall — theory: the belief recalls every genuinely-warm read it could
	// have predicted (claim 1.0); reality: TrueWarm / (TrueWarm + FalseCold) — the
	// share of provider-warm reads the belief actually called warm.
	if warmReads := pe.TrueWarm + pe.FalseCold; warmReads > 0 {
		recall := float64(pe.TrueWarm) / float64(warmReads)
		out = append(out, dojo.ScoredInput{
			Prediction: dojo.Registry.MustPredict("vcache-warmth", "warm_recall", "fraction"),
			Outcome: dojo.Outcome{
				Realized: recall, Provenance: dojo.Observed, Measured: true, Sample: warmReads,
				Source: "believed-warm provider-warm reads / all provider-warm reads (TrueWarm/(TrueWarm+FalseCold))",
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
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return "unknown"
	}
	return strings.TrimSpace(string(out))
}

func emitDojoJSON(w io.Writer, r dojo.Report) {
	_ = writeIndentedJSONNoEscape(w, r)
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
	FiredAttempts        int    `json:"fired_attempts"`           // WITNESSED: compactions that fired (body rewritten, protected prefix shipped byte-identical)
	PrefixMismatchBails  int    `json:"prefix_mismatch_bails"`    // WITNESSED: compactions that bailed with prefix_mismatch (fak-fault signal)
	PostFireCacheReadSum uint64 `json:"post_fire_cache_read_sum"` // OBSERVED: cumulative provider cache_read on compacted turns
	PostFireCacheReadN   int    `json:"post_fire_cache_read_n"`   // N for the sum (turns with post-fire cache_read)

	// Token shed metrics
	ShedTokensSum     uint64 `json:"shed_tokens_sum"`      // WITNESSED: cumulative tokens fak claims to have shed
	InputTokensOffSum uint64 `json:"input_tokens_off_sum"` // OBSERVED: cumulative provider input_tokens on compacted turns with compaction OFF
	InputTokensOnSum  uint64 `json:"input_tokens_on_sum"`  // OBSERVED: cumulative provider input_tokens on compacted turns with compaction ON
}

// compactionLever scores history compaction's theory against billed reality.
// It reads access logs or audit journal entries that contain compaction metrics,
// runs a back-test, and adapts the report into dojo episodes.
type compactionLever struct{}

func (compactionLever) Name() string { return "compaction" }

func (compactionLever) Episodes(s dojo.Scenario) ([]dojo.ScoredInput, error) {
	// Compaction requires paired ON/OFF runs with specific metrics that don't exist in
	// standard transcript files. The corpus format is a paired set of access logs or
	// audit journal entries containing:
	//   - WITNESSED compaction outcomes: fired attempts, prefix_mismatch bails, shed_tokens
	//   - OBSERVED provider billing: input_tokens on compacted turns (compaction ON vs OFF),
	//     cache_read on compacted turns after fire
	// This format is not yet standardized — the lever is registered and discoverable
	// but requires a dedicated compaction corpus to run. See #953 for the intended shape.
	//
	// Report that absence HONESTLY as one UNMEASURED episode rather than a hard error.
	// A returned error makes dojo.Run drop the lever to a stderr RunError, so a run is
	// silently a 1-lever gym (no row in the fold, the board, or the durable ledger) — an
	// operator cannot tell "compaction has no ground truth here" from "compaction was
	// never registered". An UNMEASURED outcome (Measured:false, no Realized) counts the
	// lever toward lever_count and renders it UNMEASURED/grade n/a, inventing no number.
	// The pure compactionEpisodesFromBacktest scores the real metric once a paired
	// ON/OFF corpus exists; until then this is the honest empty state.
	return []dojo.ScoredInput{{
		Prediction: dojo.Registry.MustPredict("compaction", "token_shed_ratio", "fraction"),
		Outcome: dojo.Outcome{
			Measured: false,
			Source:   "no paired ON/OFF compaction corpus from standard transcripts (see #953); fak_gateway_compaction_* expose the ON-side counters but not the OFF counterfactual",
		},
	}}, nil
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
			Prediction: dojo.Registry.MustPredict("compaction", "cache_prefix_preserved", "fraction"),
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
		// Convert to int64 before subtraction to avoid uint64 underflow when ON > OFF.
		billedDelta := float64(int64(rep.InputTokensOffSum) - int64(rep.InputTokensOnSum))
		// Guard against pathological cases: billed delta can't be negative (ON must be
		// <= OFF for a successful compaction). A negative delta is OVER_CLAIM.
		if billedDelta < 0 {
			billedDelta = 0
		}
		ratio := billedDelta / float64(rep.ShedTokensSum)
		// Do NOT clamp the over-performance side to the claim. When the billed delta
		// EXCEEDS the projected shed (the projection under-estimated the saving), the
		// raw ratio is > 1.0 and must surface as UNDER_CLAIM — a real, recurring
		// under-projection the model is not crediting. Clamping it to 1.0 hid that as
		// perfect calibration, so token_shed_ratio could structurally never score
		// UNDER_CLAIM. calibErr caps the downstream magnitude at MaxCalibErr, so an
		// unbounded ratio cannot dominate the fold.
		sample := rep.FiredAttempts
		if sample == 0 {
			sample = 1
		}
		out = append(out, dojo.ScoredInput{
			Prediction: dojo.Registry.MustPredict("compaction", "token_shed_ratio", "fraction"),
			Outcome: dojo.Outcome{
				Realized: ratio, Provenance: dojo.Witnessed, Measured: true, Sample: sample,
				Source: "billed input_tokens delta (OFF - ON) divided by projected shed (WITNESSED)",
			},
		})
	}

	return out
}
