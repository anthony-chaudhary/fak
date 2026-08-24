// Package auditusage folds the durable sinks fak accumulates across a
// session/fleet lifetime into one cross-session usage rollup for
// `fak audit usage` (#1612, child C of epic #1601):
//
//   - the decision journal (internal/journal) — the security gate's own
//     hash-chained tool-verdict trail.
//   - the loop ledger (internal/loopmgr) — hash-chained fire/admit/witness
//     events for durable loops.
//   - the cache-value ledger (internal/cachevalueledger,
//     docs/nightrun/cache-value.jsonl) — a plain, non-chained JSONL ledger.
//   - .dispatch-runs/ worker logs (internal/dispatchaudit) — pattern-classified
//     dispatch-worker outcomes.
//   - the usage-log family (internal/usagelog's usage.jsonl, plus
//     internal/gatewayusageledger's gateway-usage.jsonl) — CLI-invocation and
//     served-turn counter trails.
//
// Fold is a PURE function: every disk read, hash-chain check, and clock read
// happens in the cmd/fak CLI shell, which hands this package already-parsed
// rows/events plus each sink's chain-verification outcome. Same Input always
// yields the same Report — no I/O, no wall-clock read, inside this package.
//
// Honesty fence: a sink whose hash chain fails verification is NEVER silently
// dropped from the rollup — it still contributes whatever rows a tolerant read
// recovered, AND it surfaces as a CHAIN_BROKEN Finding. Every rollup section
// carries a Basis: "witnessed" for a hash-chained, adjudication-time record
// (the decision journal, the loop ledger); "observed" for a self-reported
// counter or pattern-classified log with no adjudication behind it (the usage
// logs, the cache/gateway ledgers, the dispatch-runs classification) — so a
// reader never mistakes an OBSERVED aggregate for a WITNESSED one.
package auditusage

import (
	"fmt"
	"sort"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
	"github.com/anthony-chaudhary/fak/internal/dispatchaudit"
	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/usagelog"
)

// Schema versions the Report shape.
const Schema = "fak-audit-usage-rollup/1"

// SinkKind names one of the five durable sinks this rollup discovers. The
// usage-log family counts as one sink kind per the issue's own framing
// (usage.jsonl + gateway-usage.jsonl, #1601-A) but is reported as two
// SinkHealth rows since they are two independent files that can each be
// present/absent/broken independently.
type SinkKind string

const (
	SinkDecisionJournal SinkKind = "decision_journal"
	SinkLoopLedger      SinkKind = "loop_ledger"
	SinkCacheValue      SinkKind = "cache_value_ledger"
	SinkDispatchRuns    SinkKind = "dispatch_runs"
	SinkUsageLog        SinkKind = "usage_log"
	SinkGatewayUsageLog SinkKind = "gateway_usage_log"
	// SinkVault is the central log-vault (internal/logvault, #2455): the durable
	// backup of every other sink. It is hash-chained AND mirror-verified, so its
	// Chain is verified/broken from a real Verify pass, not presence alone.
	SinkVault SinkKind = "logvault"
)

// Chain describes a sink's tamper-evidence status.
type Chain string

const (
	// ChainVerified: hash-chained and the strict validator confirmed it sound.
	ChainVerified Chain = "verified"
	// ChainBroken: hash-chained but verification failed — a CHAIN_BROKEN Finding
	// accompanies this. Rows recovered before the break still fold into the rollup.
	ChainBroken Chain = "broken"
	// ChainNone: this sink carries no hash chain by design (cache-value ledger,
	// gateway-usage ledger, dispatch-runs logs) — presence/absence is all there is.
	ChainNone Chain = "unchained"
	// ChainAbsent: nothing on disk yet. Not an error — a fresh checkout or a box
	// that never ran the producing subsystem looks like this.
	ChainAbsent Chain = "absent"
)

// SinkHealth is the per-sink discovery + chain-verification result.
type SinkHealth struct {
	Kind         SinkKind `json:"kind"`
	Path         string   `json:"path,omitempty"`
	Present      bool     `json:"present"`
	RowCount     int      `json:"row_count"`
	Chain        Chain    `json:"chain"`
	BrokenReason string   `json:"broken_reason,omitempty"`
}

// Finding surfaces a sink health problem an operator must see. Today the only
// kind is CHAIN_BROKEN.
type Finding struct {
	Kind   string   `json:"kind"`
	Sink   SinkKind `json:"sink"`
	Path   string   `json:"path,omitempty"`
	Reason string   `json:"reason"`
}

// VerbWeek is one (ISO year-week, verb) invocation bucket from the usage log.
type VerbWeek struct {
	Week   string `json:"week"` // e.g. "2026-W26"
	Verb   string `json:"verb"`
	Total  int    `json:"total"`
	Errors int    `json:"errors"`
}

// GuardRollup summarizes the decision journal's tool-verdict mix. WITNESSED:
// every row is the gateway kernel's own real-time adjudication.
type GuardRollup struct {
	Basis     string         `json:"basis"`
	Total     int            `json:"total"`
	ByVerdict map[string]int `json:"by_verdict,omitempty"`
	ByKind    map[string]int `json:"by_kind,omitempty"`
}

// LoopRollup summarizes fire/admit/witness outcomes across all loops in the
// ledger. WITNESSED: a loop-ledger event is itself a witness verb's own record.
type LoopRollup struct {
	Basis              string `json:"basis"`
	Loops              int    `json:"loops"`
	Fires              int64  `json:"fires"`
	Admitted           int64  `json:"admitted"`
	Refused            int64  `json:"refused"`
	Started            int64  `json:"started"`
	Ended              int64  `json:"ended"`
	Witnessed          int64  `json:"witnessed"`
	WitnessRefused     int64  `json:"witness_refused"`
	WitnessUnavailable int64  `json:"witness_unavailable"`
}

// DispatchRollup summarizes .dispatch-runs/ ship/waste outcomes. OBSERVED: the
// outcome is a pattern classification over log text, not an adjudicated verdict.
type DispatchRollup struct {
	Basis     string                        `json:"basis"`
	Workers   int                           `json:"workers"`
	ByBackend []dispatchaudit.BackendRollup `json:"by_backend,omitempty"`
	Findings  int                           `json:"findings"`
}

// CacheRollup summarizes the cache-value ledger. OBSERVED: a plain self-reported
// counter snapshot, no hash chain, no adjudication.
type CacheRollup struct {
	Basis    string `json:"basis"`
	Sessions int    `json:"sessions"`
}

// GatewayRollup summarizes the gateway-usage ledger. OBSERVED, same posture as
// CacheRollup.
type GatewayRollup struct {
	Basis    string                    `json:"basis"`
	Sessions int                       `json:"sessions"`
	Trend    *gatewayusageledger.Trend `json:"trend,omitempty"`
	// SelfHosted answers "what fraction of the tokens we served did we generate
	// ourselves?" — nil only when the window held no gateway rows at all, since
	// with nothing folded there is no question to decline.
	SelfHosted *SelfHostedRollup                    `json:"self_hosted,omitempty"`
	Dashboard  gatewayusageledger.DashboardAdoption `json:"dashboard"`
}

// SelfHostedRollup is the who-served-it slice of the gateway rollup: the report's
// snake_case view of gatewayusageledger.SelfHostedShare, with that type's two
// derived numbers materialized as fields so a JSON consumer sees them at all
// (they are methods there, and methods do not serialize).
//
// It reports a share and the coverage that share was computed over, TOGETHER and
// never separately. "68% self-hosted" over 2% of the volume is a sample someone
// will quote as a fleet number, and the pairing is the only thing standing between
// this rollup and that headline.
type SelfHostedRollup struct {
	// OutputShare is self-hosted output over CLASSIFIED output, in [0,1]. Nil when
	// Reason explains why the corpus did not earn an answer — a nil share and a 0.0
	// share are different findings, and the difference is "nobody measured" versus
	// "we bought every token".
	OutputShare *float64 `json:"output_share,omitempty"`
	// Reason is set exactly when OutputShare is nil.
	Reason string `json:"reason,omitempty"`
	// ClassifiedOutputFraction is how much of the folded output volume had a
	// resolvable side, in [0,1]. This is the qualifier on OutputShare.
	ClassifiedOutputFraction float64 `json:"classified_output_fraction"`

	SelfHostedTurns        uint64 `json:"self_hosted_turns"`
	SelfHostedInputTokens  uint64 `json:"self_hosted_input_tokens"`
	SelfHostedOutputTokens uint64 `json:"self_hosted_output_tokens"`
	VendorTurns            uint64 `json:"vendor_turns"`
	VendorInputTokens      uint64 `json:"vendor_input_tokens"`
	VendorOutputTokens     uint64 `json:"vendor_output_tokens"`
	// OutputTokens is the unsplit total over the same rows — the denominator that
	// makes ClassifiedOutputFraction auditable rather than asserted.
	OutputTokens uint64 `json:"output_tokens"`
	Rows         int    `json:"rows"`
}

// UsageRollup summarizes usage.jsonl (CLI invocations). OBSERVED: a self-report
// of "this verb ran", hash-chained for tamper-evidence but not an adjudicated
// decision.
type UsageRollup struct {
	Basis  string `json:"basis"`
	Total  int    `json:"total"`
	Errors int    `json:"errors"`
}

// VaultRollup summarizes the central log-vault's health (#2455): is my backup
// current and intact? WITNESSED: every number is folded from logvault's own
// hash-chained manifest, and VerifyMismatches / ChainBroken come from re-hashing
// the mirrors against that manifest — not a self-reported "backup ok" flag.
type VaultRollup struct {
	Basis            string `json:"basis"`
	Present          bool   `json:"present"`
	Sources          int    `json:"sources"`
	Files            int    `json:"files"`
	Bytes            int64  `json:"bytes"`
	Rows             int    `json:"rows"`
	Errors           int    `json:"errors"` // skip-error rows: files that could not be read into the vault
	LastCaptureNano  int64  `json:"last_capture_unix_nano"`
	VerifyChecked    int    `json:"verify_checked"`
	VerifyMismatches int    `json:"verify_mismatches"`
	ChainBroken      bool   `json:"chain_broken"`
}

// Report is the full PURE fold output.
type Report struct {
	Schema      string     `json:"schema"`
	GeneratedAt time.Time  `json:"generated_at"`
	Since       *time.Time `json:"since,omitempty"`

	Sinks    []SinkHealth `json:"sinks"`
	Findings []Finding    `json:"findings,omitempty"`

	InvocationsByVerbWeek []VerbWeek `json:"invocations_by_verb_week,omitempty"`

	Guard    GuardRollup    `json:"guard"`
	Loop     LoopRollup     `json:"loop"`
	Dispatch DispatchRollup `json:"dispatch"`
	Cache    CacheRollup    `json:"cache"`
	Gateway  GatewayRollup  `json:"gateway"`
	Usage    UsageRollup    `json:"usage"`
	Vault    VaultRollup    `json:"vault"`
}

// DecisionJournalInput is the shell's already-executed read + chain-verification
// of the decision journal. VerifyErr non-nil means the chain is broken; Rows is
// the TOLERANT read (whatever a torn/broken tail still let through).
type DecisionJournalInput struct {
	Path      string
	Present   bool
	Rows      []journal.Row
	VerifyErr error
}

// UsageLogInput mirrors DecisionJournalInput for usage.jsonl.
type UsageLogInput struct {
	Path      string
	Present   bool
	Rows      []usagelog.Row
	VerifyErr error
}

// GatewayUsageInput is the shell's read of gateway-usage.jsonl (no chain).
type GatewayUsageInput struct {
	Path    string
	Present bool
	Rows    []gatewayusageledger.Row
}

// CacheValueInput is the shell's read of the cache-value ledger (no chain).
type CacheValueInput struct {
	Path    string
	Present bool
	Rows    []cachevalueledger.Row
}

// LoopLedgerInput is the shell's tolerant read of the loop ledger
// (loopmgr.LoadPrefix): Events is the recovered prefix, Integrity carries the
// break descriptor (if any). Fold itself calls loopmgr.Summarize (a pure
// aggregation, not I/O) after applying the Since cutoff to Events, so the loop
// rollup respects --since the same way every other sink does.
type LoopLedgerInput struct {
	Path      string
	Present   bool
	Events    []loopmgr.Event
	Integrity loopmgr.Integrity
}

// DispatchRunsInput is the shell's scan of .dispatch-runs/ (dispatchaudit.ScanDir).
// Not time-filtered by Since: a Worker record is a whole run log, not a single
// timestamped row, so there is no sound per-row cutoff to apply.
type DispatchRunsInput struct {
	RunsDir string
	Present bool
	Workers []dispatchaudit.Worker
}

// VaultInput is the shell's already-executed fold of the log-vault manifest
// (logvault.Footprint) plus a Verify pass. Kept as plain fields so this package
// never imports internal/logvault — the CLI shell owns every disk read, hash,
// and clock, matching the honesty fence the other sinks use. Present is false for
// a box that never captured (valid-empty, not an error).
type VaultInput struct {
	Path             string
	Present          bool
	Sources          int
	Files            int
	Bytes            int64
	Rows             int
	Errors           int
	LastCaptureNano  int64
	VerifyChecked    int
	VerifyMismatches int
	ChainBroken      bool
	ChainErr         string
}

// Input is everything Fold needs, already read from disk by the CLI shell.
type Input struct {
	Now   time.Time
	Since time.Time // zero = no cutoff

	DecisionJournal DecisionJournalInput
	UsageLog        UsageLogInput
	GatewayUsage    GatewayUsageInput
	CacheValue      CacheValueInput
	LoopLedger      LoopLedgerInput
	DispatchRuns    DispatchRunsInput
	Vault           VaultInput
}

func withinSince(ts time.Time, since time.Time) bool {
	return since.IsZero() || !ts.Before(since)
}

// Fold is the PURE entry point: Input in, Report out. Deterministic — same
// Input always yields the same Report (sinks and findings are sorted).
func Fold(in Input) Report {
	rep := Report{Schema: Schema, GeneratedAt: in.Now}
	if !in.Since.IsZero() {
		s := in.Since
		rep.Since = &s
	}

	dj := SinkHealth{Kind: SinkDecisionJournal, Path: in.DecisionJournal.Path, Present: in.DecisionJournal.Present}
	dj.RowCount = len(in.DecisionJournal.Rows)
	switch {
	case !dj.Present:
		dj.Chain = ChainAbsent
	case in.DecisionJournal.VerifyErr != nil:
		dj.Chain = ChainBroken
		dj.BrokenReason = in.DecisionJournal.VerifyErr.Error()
	default:
		dj.Chain = ChainVerified
	}
	rep.Sinks = append(rep.Sinks, dj)
	rep.Guard = foldGuard(in.DecisionJournal.Rows, in.Since)

	ul := SinkHealth{Kind: SinkUsageLog, Path: in.UsageLog.Path, Present: in.UsageLog.Present}
	ul.RowCount = len(in.UsageLog.Rows)
	switch {
	case !ul.Present:
		ul.Chain = ChainAbsent
	case in.UsageLog.VerifyErr != nil:
		ul.Chain = ChainBroken
		ul.BrokenReason = in.UsageLog.VerifyErr.Error()
	default:
		ul.Chain = ChainVerified
	}
	rep.Sinks = append(rep.Sinks, ul)
	rep.Usage, rep.InvocationsByVerbWeek = foldUsage(in.UsageLog.Rows, in.Since)

	gu := SinkHealth{Kind: SinkGatewayUsageLog, Path: in.GatewayUsage.Path, Present: in.GatewayUsage.Present}
	gu.RowCount = len(in.GatewayUsage.Rows)
	if gu.Present {
		gu.Chain = ChainNone
	} else {
		gu.Chain = ChainAbsent
	}
	rep.Sinks = append(rep.Sinks, gu)
	rep.Gateway = foldGateway(in.GatewayUsage.Rows, in.Since)

	cv := SinkHealth{Kind: SinkCacheValue, Path: in.CacheValue.Path, Present: in.CacheValue.Present}
	cv.RowCount = len(in.CacheValue.Rows)
	if cv.Present {
		cv.Chain = ChainNone
	} else {
		cv.Chain = ChainAbsent
	}
	rep.Sinks = append(rep.Sinks, cv)
	rep.Cache = foldCache(in.CacheValue.Rows, in.Since)

	ll := SinkHealth{Kind: SinkLoopLedger, Path: in.LoopLedger.Path, Present: in.LoopLedger.Present}
	ll.RowCount = in.LoopLedger.Integrity.Recovered
	switch {
	case !ll.Present:
		ll.Chain = ChainAbsent
	case in.LoopLedger.Integrity.Broken:
		ll.Chain = ChainBroken
		ll.BrokenReason = in.LoopLedger.Integrity.Reason
	default:
		ll.Chain = ChainVerified
	}
	rep.Sinks = append(rep.Sinks, ll)
	rep.Loop = foldLoop(in.LoopLedger.Events, in.Since, in.Now)

	dr := SinkHealth{Kind: SinkDispatchRuns, Path: in.DispatchRuns.RunsDir, Present: in.DispatchRuns.Present}
	dr.RowCount = len(in.DispatchRuns.Workers)
	if dr.Present {
		dr.Chain = ChainNone
	} else {
		dr.Chain = ChainAbsent
	}
	rep.Sinks = append(rep.Sinks, dr)
	rep.Dispatch = foldDispatch(in.DispatchRuns.Workers)

	vault := SinkHealth{Kind: SinkVault, Path: in.Vault.Path, Present: in.Vault.Present, RowCount: in.Vault.Rows}
	switch {
	case !in.Vault.Present:
		vault.Chain = ChainAbsent
	case in.Vault.ChainBroken:
		vault.Chain = ChainBroken
		vault.BrokenReason = in.Vault.ChainErr
	case in.Vault.VerifyMismatches > 0:
		// A verified chain over corrupted mirrors is still NOT intact — surface it
		// as a finding so a silent backup corruption cannot hide behind a sound chain.
		vault.Chain = ChainBroken
		vault.BrokenReason = fmt.Sprintf("%d vault mirror/anchor mismatch(es) at last verify", in.Vault.VerifyMismatches)
	default:
		vault.Chain = ChainVerified
	}
	rep.Sinks = append(rep.Sinks, vault)
	rep.Vault = foldVault(in.Vault)

	for _, s := range rep.Sinks {
		if s.Chain == ChainBroken {
			rep.Findings = append(rep.Findings, Finding{
				Kind:   "CHAIN_BROKEN",
				Sink:   s.Kind,
				Path:   s.Path,
				Reason: s.BrokenReason,
			})
		}
	}
	sort.Slice(rep.Findings, func(i, j int) bool { return rep.Findings[i].Sink < rep.Findings[j].Sink })
	sort.Slice(rep.Sinks, func(i, j int) bool { return rep.Sinks[i].Kind < rep.Sinks[j].Kind })

	return rep
}

func foldGuard(rows []journal.Row, since time.Time) GuardRollup {
	g := GuardRollup{Basis: "witnessed", ByVerdict: map[string]int{}, ByKind: map[string]int{}}
	for _, r := range rows {
		if !withinSince(time.Unix(0, r.TSUnixNano), since) {
			continue
		}
		g.Total++
		if r.Verdict != "" {
			g.ByVerdict[r.Verdict]++
		}
		if r.Kind != "" {
			g.ByKind[r.Kind]++
		}
	}
	if len(g.ByVerdict) == 0 {
		g.ByVerdict = nil
	}
	if len(g.ByKind) == 0 {
		g.ByKind = nil
	}
	return g
}

func foldUsage(rows []usagelog.Row, since time.Time) (UsageRollup, []VerbWeek) {
	u := UsageRollup{Basis: "observed"}
	type key struct{ week, verb string }
	byBucket := map[key]*VerbWeek{}
	for _, r := range rows {
		ts := time.Unix(0, r.TSUnixNano).UTC()
		if !withinSince(ts, since) {
			continue
		}
		u.Total++
		isErr := r.ExitCode != 0
		if isErr {
			u.Errors++
		}
		year, wk := ts.ISOWeek()
		week := fmt.Sprintf("%04d-W%02d", year, wk)
		verb := r.Verb
		if verb == "" {
			verb = "(none)"
		}
		k := key{week, verb}
		vw := byBucket[k]
		if vw == nil {
			vw = &VerbWeek{Week: week, Verb: verb}
			byBucket[k] = vw
		}
		vw.Total++
		if isErr {
			vw.Errors++
		}
	}
	out := make([]VerbWeek, 0, len(byBucket))
	for _, vw := range byBucket {
		out = append(out, *vw)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Week != out[j].Week {
			return out[i].Week < out[j].Week
		}
		return out[i].Verb < out[j].Verb
	})
	return u, out
}

func foldGateway(rows []gatewayusageledger.Row, since time.Time) GatewayRollup {
	g := GatewayRollup{Basis: "observed"}
	g.Dashboard = gatewayusageledger.FoldDashboardAdoption(rows, since)
	var filtered []gatewayusageledger.Row
	for _, r := range rows {
		if !withinSince(time.UnixMilli(r.UnixMillis).UTC(), since) {
			continue
		}
		filtered = append(filtered, r)
	}
	// A carryforward row (gatewayusageledger.Cut, #3490) is the fold of many
	// pre-cut rows — expand it back to its folded row count so the session total
	// stays true across a ledger cut. FoldTrend skips carryforward rows itself.
	g.Sessions = 0
	for _, r := range filtered {
		if r.Kind == gatewayusageledger.KindCarryforward && r.Carryforward != nil {
			g.Sessions += r.Carryforward.FoldedRows
			continue
		}
		g.Sessions++
	}
	if trend, ok := gatewayusageledger.FoldTrend(filtered); ok {
		g.Trend = &trend
	}
	// The who-served-it fold runs over the SAME filtered rows, carryforward rows
	// included — see FoldSelfHostedShare, which sums them deliberately where
	// FoldTrend skips them. An empty window gets no rollup at all: declining to
	// answer is a statement about instrumentation, and there is nothing to say it
	// about when no session was folded.
	if len(filtered) > 0 {
		sh := gatewayusageledger.FoldSelfHostedShare(filtered)
		g.SelfHosted = &SelfHostedRollup{
			OutputShare:              sh.OutputShare,
			Reason:                   string(sh.Reason),
			ClassifiedOutputFraction: sh.ClassifiedOutputFraction(),

			SelfHostedTurns:        sh.SelfHostedTurns,
			SelfHostedInputTokens:  sh.SelfHostedInputTokens,
			SelfHostedOutputTokens: sh.SelfHostedOutputTokens,
			VendorTurns:            sh.VendorTurns,
			VendorInputTokens:      sh.VendorInputTokens,
			VendorOutputTokens:     sh.VendorOutputTokens,
			OutputTokens:           sh.OutputTokens,
			Rows:                   sh.Rows,
		}
	}
	return g
}

func foldCache(rows []cachevalueledger.Row, since time.Time) CacheRollup {
	c := CacheRollup{Basis: "observed"}
	for _, r := range rows {
		if !withinSince(time.UnixMilli(r.UnixMillis).UTC(), since) {
			continue
		}
		c.Sessions++
	}
	return c
}

func foldLoop(events []loopmgr.Event, since, now time.Time) LoopRollup {
	var filtered []loopmgr.Event
	for _, ev := range events {
		if !withinSince(time.Unix(0, ev.TSUnixNano), since) {
			continue
		}
		filtered = append(filtered, ev)
	}
	status := loopmgr.Summarize(filtered, now)
	l := LoopRollup{Basis: "witnessed", Loops: len(status.Loops)}
	for _, s := range status.Loops {
		l.Fires += int64(s.Fires)
		l.Admitted += int64(s.Admitted)
		l.Refused += int64(s.Refused)
		l.Started += int64(s.Started)
		l.Ended += int64(s.Ended)
		l.Witnessed += int64(s.Witnessed)
		l.WitnessRefused += int64(s.WitnessRefused)
		l.WitnessUnavailable += int64(s.WitnessUnavailable)
	}
	return l
}

// foldVault projects the shell's vault fold into the rollup. WITNESSED: the whole
// vault is fak's own hash-chained manifest re-hashed against its mirrors.
func foldVault(in VaultInput) VaultRollup {
	return VaultRollup{
		Basis:            "witnessed",
		Present:          in.Present,
		Sources:          in.Sources,
		Files:            in.Files,
		Bytes:            in.Bytes,
		Rows:             in.Rows,
		Errors:           in.Errors,
		LastCaptureNano:  in.LastCaptureNano,
		VerifyChecked:    in.VerifyChecked,
		VerifyMismatches: in.VerifyMismatches,
		ChainBroken:      in.ChainBroken,
	}
}

func foldDispatch(workers []dispatchaudit.Worker) DispatchRollup {
	rep := dispatchaudit.Fold(workers, dispatchaudit.DefaultThresholds())
	return DispatchRollup{
		Basis:     "observed",
		Workers:   len(workers),
		ByBackend: rep.Rollups,
		Findings:  len(rep.Findings),
	}
}
