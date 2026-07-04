package main

import (
	"flag"
	"fmt"
	"io"
	"text/tabwriter"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

// runLoopEconomics folds the loop ledger into ONE honest loop-economics readout:
// did the loop save tokens, wall time, worker attention, or retries versus the real
// baseline? (#2646). It reads only the hash-chained ledger — the SAME journal `fak
// loop status` folds — so every witnessed figure it prints is already recorded and
// cannot be a worker's self-report.
//
// The report keeps the four savings accounts the loops-user-value doc names strictly
// separate, so an adoption claim can cite the right one:
//
//   - witnessed    — derived from the ledger's own hash-chained events + the dispatch
//                    progress/worker/cooldown metrics carried in them (close rate,
//                    retry/refusal rate, duplicate attempts avoided, effective workers,
//                    wall time, baseline vs observed open count).
//   - provider     — a provider prompt-cache/billing benefit. Observed, NOT owned by
//                    fak; stays not_yet until an operator folds the provider figure.
//   - fak_authored — token-equivalent work fak removed/reused/served itself. Witnessed
//                    only with a proof for that mechanism; not_yet by default.
//   - modeled      — a formula projecting token-equivalents saved from the witnessed
//                    duplicate-attempts-avoided count. Useful for planning, not a
//                    billing claim; not_yet until the operator supplies the per-attempt
//                    assumption.
//
// The token accounts default to not_yet and only populate from EXPLICIT operator
// inputs, so the report never fabricates a saving the ledger cannot witness.
func runLoopEconomics(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("loop economics", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", defaultLoopLedger(), "loop JSONL ledger path")
	loopID := fs.String("loop", "", "fold a single loop id (default: every loop in the ledger)")
	asJSON := fs.Bool("json", false, "emit the loop-economics report as JSON")
	// The three token accounts default to not_yet. Each populates ONLY from an
	// explicit operator figure, so fak never invents a saving the ledger cannot prove.
	providerCacheTokens := fs.Int64("provider-cache-tokens", -1, "provider-reported cache token benefit to fold into the provider account (observed, not owned); default: not_yet")
	fakAuthoredTokens := fs.Int64("fak-authored-tokens", -1, "witnessed fak-authored token-equivalent saving with a proof for that mechanism; default: not_yet")
	modeledTokensPerAvoided := fs.Int64("modeled-tokens-per-avoided", -1, "modeled token-equivalents per duplicate attempt avoided (projection, not a billing claim); default: not_yet")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak loop economics: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	events, err := loopmgr.Load(*ledger)
	if err != nil {
		fmt.Fprintf(stderr, "fak loop economics: %v\n", err)
		return 1
	}

	rep := foldLoopEconomics(events, loopEconomicsOpts{
		LedgerPath:              *ledger,
		LoopID:                  *loopID,
		ProviderCacheTokens:     loopEconOptToken(*providerCacheTokens),
		FakAuthoredTokens:       loopEconOptToken(*fakAuthoredTokens),
		ModeledTokensPerAvoided: loopEconOptToken(*modeledTokensPerAvoided),
	}, time.Now())

	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, rep, "fak loop economics")
	}
	renderLoopEconomics(stdout, rep)
	return 0
}

// loopEconomicsOpts carries the fold inputs: the ledger identity, an optional single
// loop filter, and the three explicit token-account figures. Each token figure is a
// pointer so the zero value of this struct means "operator supplied nothing" — every
// account then stays not_yet — while an explicit *0 can still record an observed zero.
type loopEconomicsOpts struct {
	LedgerPath              string
	LoopID                  string
	ProviderCacheTokens     *int64
	FakAuthoredTokens       *int64
	ModeledTokensPerAvoided *int64
}

// loopEconOptToken lifts a flag value into the opts token pointer: a negative value is
// the "flag not supplied" sentinel (nil → account stays not_yet); >= 0 is an explicit
// operator figure the fold folds into its account.
func loopEconOptToken(v int64) *int64 {
	if v < 0 {
		return nil
	}
	return &v
}

const schemaLoopEconomics = "fak.loop-economics.v1"

type loopEconomicsReport struct {
	Schema        string             `json:"schema"`
	TSUnixNano    int64              `json:"ts_unix_nano"`
	LedgerPath    string             `json:"ledger_path,omitempty"`
	LoopFilter    string             `json:"loop_filter,omitempty"`
	Loops         int                `json:"loops"`
	Window        loopEconWindow     `json:"window"`
	Witnessed     loopEconWitnessed  `json:"witnessed"`
	ProviderCache loopEconAccount    `json:"provider_cache"`
	FakAuthored   loopEconAccount    `json:"fak_authored"`
	Modeled       loopEconModeled    `json:"modeled"`
	NotYet        []string           `json:"not_yet"`
}

// loopEconWindow is the observed span of the folded events — the wall-clock the
// report's per-loop figures were measured over.
type loopEconWindow struct {
	FirstEventUnixNano int64   `json:"first_event_unix_nano,omitempty"`
	LastEventUnixNano  int64   `json:"last_event_unix_nano,omitempty"`
	SpanSeconds        float64 `json:"span_seconds"`
	Events             int     `json:"events"`
}

// loopEconWitnessed holds the figures derived purely from the hash-chained ledger and
// the dispatch progress/worker/cooldown metrics it carries. Every number here is
// recorded evidence, never a self-report.
type loopEconWitnessed struct {
	// Dispatch progress (progress-loop metrics carried in the ledger).
	BaselineOpen       int64    `json:"baseline_open"`
	ObservedOpen       int64    `json:"observed_open"`
	IssuesClosedByLoop int64    `json:"issues_closed_by_loop"`
	WitnessedOpen      int64    `json:"witnessed_open"`
	CloseRate          *float64 `json:"close_rate"` // closed / (closed + observed_open); nil when the denominator is 0

	// Loop lifecycle counts (fold of the ledger event kinds).
	Fires                    int64    `json:"fires"`
	Admitted                 int64    `json:"admitted"`
	Refused                  int64    `json:"refused"`
	Started                  int64    `json:"started"`
	Ended                    int64    `json:"ended"`
	WitnessedDone            int64    `json:"witnessed_done"`
	RetryRate                *float64 `json:"retry_rate"`                  // refused / fires; nil when fires is 0
	DuplicateAttemptsAvoided int64    `json:"duplicate_attempts_avoided"`  // refused admissions — spawns the governor declined

	// Workers (peak observed).
	EffectiveWorkers int64 `json:"effective_workers"`
	WorkerCap        int64 `json:"worker_cap"`

	// Wall time.
	WallTimeSeconds float64 `json:"wall_time_seconds"`
	WallTimeSource  string  `json:"wall_time_source"` // "run_durations" | "window_span"
}

// loopEconAccount is one separated token-saving account (provider or fak-authored).
// Status is "not_yet" until an explicit witness populates it.
type loopEconAccount struct {
	TokenEquivalentSaved int64  `json:"token_equivalent_saved"`
	Status               string `json:"status"` // "not_yet" | "observed" | "witnessed"
	Source               string `json:"source,omitempty"`
	Note                 string `json:"note"`
}

// loopEconModeled is the projected (not billed) token-equivalent saving: the witnessed
// duplicate-attempts-avoided count times an operator-supplied per-attempt assumption.
type loopEconModeled struct {
	TokenEquivalentSaved int64  `json:"token_equivalent_saved"`
	Status               string `json:"status"` // "not_yet" | "modeled"
	TokensPerAvoided     int64  `json:"tokens_per_avoided"`
	Basis                string `json:"basis"`
	Note                 string `json:"note"`
}

// foldLoopEconomics is the pure fold: it walks the ledger events (optionally filtered
// to one loop id) and produces the loop-economics report. It reads only the recorded
// events and the explicit operator token figures — it takes no I/O and appends nothing,
// so the same events + opts always fold to the same report.
func foldLoopEconomics(events []loopmgr.Event, opts loopEconomicsOpts, now time.Time) loopEconomicsReport {
	rep := loopEconomicsReport{
		Schema:     schemaLoopEconomics,
		TSUnixNano: now.UTC().UnixNano(),
		LedgerPath: opts.LedgerPath,
		LoopFilter: opts.LoopID,
	}

	loopIDs := map[string]bool{}
	var w loopEconWitnessed
	var totalDurationMS int64
	var firstTS, lastTS int64
	folded := 0

	for _, ev := range events {
		if opts.LoopID != "" && ev.LoopID != opts.LoopID {
			continue
		}
		folded++
		loopIDs[ev.LoopID] = true
		if firstTS == 0 || ev.TSUnixNano < firstTS {
			firstTS = ev.TSUnixNano
		}
		if ev.TSUnixNano > lastTS {
			lastTS = ev.TSUnixNano
		}

		switch ev.Kind {
		case loopmgr.EventFire:
			w.Fires++
		case loopmgr.EventAdmit:
			switch ev.Status {
			case loopmgr.StatusAdmitted:
				w.Admitted++
			case loopmgr.StatusRefused:
				w.Refused++
			}
		case loopmgr.EventStart:
			w.Started++
		case loopmgr.EventEnd:
			w.Ended++
		case loopmgr.EventWitness:
			if ev.Status == loopmgr.StatusWitnessedDone {
				w.WitnessedDone++
			}
		}

		// Dispatch progress + worker metrics ride in the event metrics map. Baseline,
		// open, and witnessed-open are point-in-time reads — take the most recent
		// (largest-seq) event that carried each. Closed-by-loop is cumulative — take
		// the peak. Live/cap are the peak concurrency the loop actually reached.
		if v, ok := ev.Metrics["baseline_open"]; ok {
			w.BaselineOpen = v
		}
		if v, ok := ev.Metrics["open_now"]; ok {
			w.ObservedOpen = v
		}
		if v, ok := ev.Metrics["witnessed_open"]; ok {
			w.WitnessedOpen = v
		}
		if v, ok := ev.Metrics["closed_by_loop_total"]; ok && v > w.IssuesClosedByLoop {
			w.IssuesClosedByLoop = v
		}
		if v, ok := ev.Metrics["live"]; ok && v > w.EffectiveWorkers {
			w.EffectiveWorkers = v
		}
		if v, ok := ev.Metrics["max_workers"]; ok && v > w.WorkerCap {
			w.WorkerCap = v
		}
		if v, ok := ev.Metrics["duration_ms"]; ok && v > 0 {
			totalDurationMS += v
		}
	}

	// Duplicate/blocked attempts avoided: every refused admission is a spawn the
	// governor declined (weekly-cap, cooldown, lane collision) — real worker+token
	// spend a naive always-spawn loop would have burned.
	w.DuplicateAttemptsAvoided = w.Refused

	if w.Fires > 0 {
		r := float64(w.Refused) / float64(w.Fires)
		w.RetryRate = &r
	}
	if closedPlusOpen := w.IssuesClosedByLoop + w.ObservedOpen; closedPlusOpen > 0 && w.IssuesClosedByLoop > 0 {
		c := float64(w.IssuesClosedByLoop) / float64(closedPlusOpen)
		w.CloseRate = &c
	}

	// Wall time: prefer the summed measured run durations; fall back to the observed
	// event span when no run carried a duration (a read-only progress-only ledger).
	if totalDurationMS > 0 {
		w.WallTimeSeconds = float64(totalDurationMS) / 1000
		w.WallTimeSource = "run_durations"
	} else if lastTS > firstTS {
		w.WallTimeSeconds = float64(lastTS-firstTS) / 1e9
		w.WallTimeSource = "window_span"
	} else {
		w.WallTimeSource = "window_span"
	}

	rep.Loops = len(loopIDs)
	rep.Witnessed = w
	rep.Window = loopEconWindow{
		FirstEventUnixNano: firstTS,
		LastEventUnixNano:  lastTS,
		Events:             folded,
	}
	if lastTS > firstTS {
		rep.Window.SpanSeconds = float64(lastTS-firstTS) / 1e9
	}

	// Provider-cache account: observed, not owned by fak. Populated only from an
	// explicit operator figure.
	rep.ProviderCache = loopEconAccount{
		Status: "not_yet",
		Note:   "provider prompt-cache/billing benefit — observed, not owned by fak; fold a provider figure with --provider-cache-tokens",
	}
	if opts.ProviderCacheTokens != nil {
		rep.ProviderCache.TokenEquivalentSaved = *opts.ProviderCacheTokens
		rep.ProviderCache.Status = "observed"
		rep.ProviderCache.Source = "operator:--provider-cache-tokens"
		rep.ProviderCache.Note = "provider-reported cache benefit; observed, not owned by fak"
	}

	// fak-authored account: witnessed only with a proof for that mechanism.
	rep.FakAuthored = loopEconAccount{
		Status: "not_yet",
		Note:   "token-equivalent work fak removed/reused/served itself — witnessed only with a proof for that mechanism",
	}
	if opts.FakAuthoredTokens != nil {
		rep.FakAuthored.TokenEquivalentSaved = *opts.FakAuthoredTokens
		rep.FakAuthored.Status = "witnessed"
		rep.FakAuthored.Source = "operator:--fak-authored-tokens"
		rep.FakAuthored.Note = "fak-authored token-equivalent saving supplied with a mechanism proof"
	}

	// Modeled account: a projection from the witnessed duplicate-attempts-avoided
	// count. Never a billing claim; not_yet until the per-attempt assumption is given.
	rep.Modeled = loopEconModeled{
		Status: "not_yet",
		Basis:  "duplicate_attempts_avoided * tokens_per_avoided",
		Note:   "projection for planning, not a billing claim; supply --modeled-tokens-per-avoided to populate",
	}
	if opts.ModeledTokensPerAvoided != nil {
		rep.Modeled.TokensPerAvoided = *opts.ModeledTokensPerAvoided
		rep.Modeled.TokenEquivalentSaved = w.DuplicateAttemptsAvoided * *opts.ModeledTokensPerAvoided
		rep.Modeled.Status = "modeled"
	}

	rep.NotYet = loopEconomicsNotYet(rep)
	return rep
}

// loopEconomicsNotYet names every field the fold could NOT witness from its inputs, so
// a consumer never mistakes a real 0 (or an un-owned account) for a proven win.
func loopEconomicsNotYet(rep loopEconomicsReport) []string {
	var out []string
	if rep.Witnessed.CloseRate == nil {
		out = append(out, "witnessed.close_rate")
	}
	if rep.Witnessed.RetryRate == nil {
		out = append(out, "witnessed.retry_rate")
	}
	if rep.ProviderCache.Status == "not_yet" {
		out = append(out, "provider_cache.token_equivalent_saved")
	}
	if rep.FakAuthored.Status == "not_yet" {
		out = append(out, "fak_authored.token_equivalent_saved")
	}
	if rep.Modeled.Status == "not_yet" {
		out = append(out, "modeled.token_equivalent_saved")
	}
	return out
}

// renderLoopEconomics prints the human table: the witnessed block first (the figures
// the ledger proves), then the three separated token accounts with their claim status,
// then the explicit not-yet list.
func renderLoopEconomics(w io.Writer, rep loopEconomicsReport) {
	scope := "all loops"
	if rep.LoopFilter != "" {
		scope = "loop " + rep.LoopFilter
	}
	fmt.Fprintf(w, "fak loop economics — %s over %d loop(s), %d event(s), window %s\n\n",
		scope, rep.Loops, rep.Window.Events, humanCadence(rep.Window.SpanSeconds))

	wit := rep.Witnessed
	tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "WITNESSED (from the hash-chained ledger)\tVALUE")
	fmt.Fprintf(tw, "  baseline open\t%d\n", wit.BaselineOpen)
	fmt.Fprintf(tw, "  observed open (looped result)\t%d\n", wit.ObservedOpen)
	fmt.Fprintf(tw, "  issues closed by loop\t%d\n", wit.IssuesClosedByLoop)
	fmt.Fprintf(tw, "  witnessed-done still open\t%d\n", wit.WitnessedOpen)
	fmt.Fprintf(tw, "  close rate\t%s\n", loopEconRate(wit.CloseRate))
	fmt.Fprintf(tw, "  retry/refusal rate\t%s\n", loopEconRate(wit.RetryRate))
	fmt.Fprintf(tw, "  duplicate attempts avoided\t%d\n", wit.DuplicateAttemptsAvoided)
	fmt.Fprintf(tw, "  effective workers (peak)\t%d of cap %d\n", wit.EffectiveWorkers, wit.WorkerCap)
	fmt.Fprintf(tw, "  wall time\t%s (%s)\n", humanCadence(wit.WallTimeSeconds), wit.WallTimeSource)
	fmt.Fprintf(tw, "  fires / admitted / refused\t%d / %d / %d\n", wit.Fires, wit.Admitted, wit.Refused)
	fmt.Fprintf(tw, "  started / ended / witnessed\t%d / %d / %d\n", wit.Started, wit.Ended, wit.WitnessedDone)
	_ = tw.Flush()

	fmt.Fprintln(w)
	ta := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintln(ta, "TOKEN-EQUIVALENT SAVED (accounts kept separate)\tVALUE\tSTATUS")
	fmt.Fprintf(ta, "  provider cache (observed, not owned)\t%d\t%s\n", rep.ProviderCache.TokenEquivalentSaved, rep.ProviderCache.Status)
	fmt.Fprintf(ta, "  fak-authored (witnessed)\t%d\t%s\n", rep.FakAuthored.TokenEquivalentSaved, rep.FakAuthored.Status)
	fmt.Fprintf(ta, "  modeled (projection, not billed)\t%d\t%s\n", rep.Modeled.TokenEquivalentSaved, rep.Modeled.Status)
	_ = ta.Flush()

	if len(rep.NotYet) > 0 {
		fmt.Fprintf(w, "\nnot yet witnessed: %d field(s) — %v\n", len(rep.NotYet), rep.NotYet)
	}
}

// loopEconRate renders a fraction as a percentage, or "not_yet" when its denominator
// was 0 (nil) — so a missing rate never reads as a real 0%.
func loopEconRate(r *float64) string {
	if r == nil {
		return "not_yet"
	}
	return fmt.Sprintf("%.1f%%", *r*100)
}
