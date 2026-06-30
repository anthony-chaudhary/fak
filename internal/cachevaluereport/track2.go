package cachevaluereport

// Track 2 of epic #1301 — the OBSERVED provider-$ cache economics — and the
// two-track P&L fold (#1304) that puts it SIDE BY SIDE with the WITNESSED kernel
// reuse (Track 1) without ever blending them.
//
// PROVENANCE FENCE (the whole reason the tracks never merge): Track 1 is
// WITNESSED — fak authored the cache bytes, so the reuse is byte-identical by
// construction. Track 2 is OBSERVED — the provider reported cache_read /
// cache_creation and fak only relays it; the $ NET is a cost PROJECTION priced
// from a caller-supplied base $/MTok, never a fak-WITNESSED claim. Mixing a
// projection into a witnessed number would launder the projection's uncertainty
// into the witness, so the fold keeps two separate accounts and emits a NET line
// that is explicitly labelled OBSERVED.
//
// The Track-2 ledger row mirrors the axes named by rung B (#1303): the four
// billable token axes the provider reports (input / cache_read / cache_creation /
// output), the compaction token-shed (`--compact-history-budget`, #745), and the
// $ duals priced from the model's base input/output $/MTok. This package defines
// the row + reader so the fold has a stable shape to read; the LIVE per-session
// append (wiring the guard/serve/run exit points) is rung B's own deliverable.

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
)

// SavingsLedgerSchema tags each durable OBSERVED-$ row (rung B, #1303). It is
// distinct from the kernel ledger schema (Track 1) so a reader can never confuse
// a WITNESSED row for an OBSERVED one.
const SavingsLedgerSchema = "fak-cache-savings-ledger/1"

// DefaultSavingsLedgerRel is the sibling ledger that carries Track 2 (the OBSERVED
// $ economics), written next to the kernel ledger (docs/nightrun/cache-value.jsonl)
// per the #1303 "sibling file" option, so the two provenances never share a row.
const DefaultSavingsLedgerRel = "docs/nightrun/cache-savings.jsonl"

// SavingsRow is one durable, append-only OBSERVED-$ cache-economics row — Track 2
// of epic #1301, the per-session shape rung B (#1303) persists. Every field is
// OBSERVED (provider-relayed or priced from a supplied base rate), NEVER a
// fak-WITNESSED claim.
//
// The token axes are the four the Anthropic usage block reports (mirroring
// gateway.CacheUsage). The $ axes are priced from the session's base input/output
// $/MTok (e.g. Opus 4.8 = {5,25}) so the row stays correct as prices change
// without re-pricing here. CompactionShedTokens is the prompt tokens the
// `--compact-history-budget` (#745) compaction dropped before they were ever sent
// — a real OBSERVED saving on the input axis, valued at the base input price.
type SavingsRow struct {
	Schema      string `json:"schema"`
	Date        string `json:"date"` // YYYY-MM-DD (UTC), the bucketing key
	SessionType string `json:"session_type,omitempty"`
	Context     string `json:"context,omitempty"`
	GeneratedAt string `json:"generated_at,omitempty"`

	// Token axes (OBSERVED, provider-reported).
	InputTokens         uint64 `json:"input_tokens"`
	CacheReadTokens     uint64 `json:"cache_read_tokens"`
	CacheCreationTokens uint64 `json:"cache_creation_tokens"`
	OutputTokens        uint64 `json:"output_tokens"`

	// CompactionShedTokens is the input tokens `--compact-history-budget` (#745)
	// dropped before the turn was sent — a saving on the spend side, OBSERVED.
	CompactionShedTokens uint64 `json:"compaction_shed_tokens,omitempty"`

	// Saved-token-equivalents (OBSERVED), the read rebate minus the write premium,
	// in input-token units — the same quantity vcachegov.TelemetrySavingsProof
	// reports. NetSavedTokenEquiv keeps its sign: a cold-write-only session reads
	// NEGATIVE until reads repay writes (#1303 says do not floor at zero).
	SavedTokenEquiv    float64 `json:"saved_token_equiv,omitempty"`
	NetSavedTokenEquiv float64 `json:"net_saved_token_equiv,omitempty"`

	// $ duals (OBSERVED projection), priced from the base rates below. RebateUSD is
	// the read rebate, WritePremiumUSD the cache-write premium paid, SpendUSD the
	// API dollars still incurred this session, CompactionSavedUSD the value of the
	// shed tokens. NetUSD = RebateUSD + CompactionSavedUSD − WritePremiumUSD −
	// SpendUSD, the single per-session NET the report folds.
	RebateUSD          float64 `json:"rebate_usd"`
	WritePremiumUSD    float64 `json:"write_premium_usd"`
	SpendUSD           float64 `json:"spend_usd"`
	CompactionSavedUSD float64 `json:"compaction_saved_usd,omitempty"`
	NetUSD             float64 `json:"net_usd"`

	InputPerMTokUSD  float64 `json:"input_per_mtok_usd,omitempty"`
	OutputPerMTokUSD float64 `json:"output_per_mtok_usd,omitempty"`
}

// NetUSDComputed re-derives the NET from the row's own component $ fields, so a
// reader can check the stored NetUSD against its parts. It is the byte-for-byte
// identity the recompute test asserts:
//
//	NET = read rebate + compaction saving − write premium − API spend still incurred
func (r SavingsRow) NetUSDComputed() float64 {
	return r.RebateUSD + r.CompactionSavedUSD - r.WritePremiumUSD - r.SpendUSD
}

// SavingsBucket is one period's OBSERVED-$ roll-up (Track 2). It keeps the four
// component dollar accounts separate (so the NET is always re-derivable) plus the
// running cumulative NET, which is what crosses break-even.
type SavingsBucket struct {
	Period   string `json:"period"` // ISO week, e.g. "2026-W26"
	Start    string `json:"start"`  // earliest row date in the bucket (YYYY-MM-DD)
	Sessions int    `json:"sessions"`

	InputTokens          uint64 `json:"input_tokens"`
	CacheReadTokens      uint64 `json:"cache_read_tokens"`
	CacheCreationTokens  uint64 `json:"cache_creation_tokens"`
	OutputTokens         uint64 `json:"output_tokens"`
	CompactionShedTokens uint64 `json:"compaction_shed_tokens"`

	RebateUSD          float64 `json:"rebate_usd"`
	WritePremiumUSD    float64 `json:"write_premium_usd"`
	SpendUSD           float64 `json:"spend_usd"`
	CompactionSavedUSD float64 `json:"compaction_saved_usd"`

	// NetUSD is this period's net: rebate + compaction − writePremium − spend.
	NetUSD float64 `json:"net_usd"`
	// CumulativeNetUSD is the running total through this period — the line that
	// crosses break-even. BrokeEven is true once it first turns non-negative.
	CumulativeNetUSD float64 `json:"cumulative_net_usd"`
	BrokeEven        bool    `json:"broke_even"`
}

// ParseSavingsLedger reads OBSERVED-$ rows (Track 2) from JSONL content. Like the
// kernel-ledger parser it skips blank and unparseable lines rather than failing,
// so a partially-written ledger still folds.
func ParseSavingsLedger(content string) []SavingsRow {
	var rows []SavingsRow
	sc := bufio.NewScanner(strings.NewReader(content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var row SavingsRow
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			continue
		}
		if row.Date == "" {
			continue
		}
		rows = append(rows, row)
	}
	return rows
}

// ReadSavingsLedgerFile reads the Track-2 ledger at path. A missing file folds to
// no rows (the honest empty-Track-2 case), not an error.
func ReadSavingsLedgerFile(path string) []SavingsRow {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	return ParseSavingsLedger(string(b))
}

// AppendSavingsLine renders the JSONL line for one Track-2 row (the durable shape
// rung B appends; provided here so the writer and reader share one encoding).
func AppendSavingsLine(row SavingsRow) (string, error) {
	if row.Schema == "" {
		row.Schema = SavingsLedgerSchema
	}
	b, err := json.Marshal(row)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// foldSavings buckets Track-2 rows by ISO week and computes each period's NET plus
// the running cumulative that crosses break-even. It is pure (no clock, no I/O):
// bucketing comes from each row's own Date. Rows with an unparseable date are
// skipped, mirroring the Track-1 fold.
func foldSavings(rows []SavingsRow) []SavingsBucket {
	type agg struct {
		b     SavingsBucket
		start time.Time
	}
	byPeriod := map[string]*agg{}
	for _, row := range rows {
		d, err := time.Parse("2006-01-02", row.Date)
		if err != nil {
			continue
		}
		key := isoWeek(d)
		a := byPeriod[key]
		if a == nil {
			a = &agg{b: SavingsBucket{Period: key}, start: d}
			byPeriod[key] = a
		}
		if d.Before(a.start) {
			a.start = d
		}
		b := &a.b
		b.Sessions++
		b.InputTokens += row.InputTokens
		b.CacheReadTokens += row.CacheReadTokens
		b.CacheCreationTokens += row.CacheCreationTokens
		b.OutputTokens += row.OutputTokens
		b.CompactionShedTokens += row.CompactionShedTokens
		b.RebateUSD += row.RebateUSD
		b.WritePremiumUSD += row.WritePremiumUSD
		b.SpendUSD += row.SpendUSD
		b.CompactionSavedUSD += row.CompactionSavedUSD
	}

	keys := make([]string, 0, len(byPeriod))
	for k := range byPeriod {
		keys = append(keys, k)
	}
	sort.Strings(keys) // ISO week keys sort chronologically as strings

	buckets := make([]SavingsBucket, 0, len(keys))
	var cumulative float64
	brokeEven := false
	for _, k := range keys {
		a := byPeriod[k]
		b := a.b
		b.Start = a.start.Format("2006-01-02")
		b.NetUSD = b.RebateUSD + b.CompactionSavedUSD - b.WritePremiumUSD - b.SpendUSD
		cumulative += b.NetUSD
		b.CumulativeNetUSD = cumulative
		if !brokeEven && cumulative >= 0 {
			brokeEven = true
		}
		b.BrokeEven = brokeEven
		buckets = append(buckets, b)
	}
	return buckets
}

// TwoTrackReport is the #1304 P&L: Track 1 (WITNESSED kernel reuse) and Track 2
// (OBSERVED $ economics) SIDE BY SIDE, never blended, with the explicit per-period
// NET and the running cumulative that crosses break-even. The two sub-reports keep
// their own provenance self-labels; this envelope only carries the headline and the
// honesty fence that the $ NET is an OBSERVED projection.
type TwoTrackReport struct {
	Schema      string `json:"schema"`
	GeneratedAt string `json:"generated_at"`
	Since       string `json:"since,omitempty"` // the --since floor applied, if any

	Track1 Report          `json:"track1_witnessed_kernel"`
	Track2 []SavingsBucket `json:"track2_observed_usd"`

	// LatestNetUSD / CumulativeNetUSD are the most-recent period's net and the
	// running total through it — the P&L headline. BrokeEven is whether the running
	// total has crossed zero.
	LatestNetUSD     float64 `json:"latest_net_usd"`
	CumulativeNetUSD float64 `json:"cumulative_net_usd"`
	BrokeEven        bool    `json:"broke_even"`

	// ProjectionFence is the constant honesty self-label: the $ NET is a COST
	// PROJECTION over OBSERVED quantities, never a fak-WITNESSED claim, and tracks
	// stay side by side, never blended.
	ProjectionFence string `json:"projection_fence"`

	OK         bool   `json:"ok"`
	Verdict    string `json:"verdict"` // MEASURED | INSUFFICIENT
	Finding    string `json:"finding"`
	NextAction string `json:"next_action"`
}

// projectionFence is the #1301 / #1304 honesty fence string.
const projectionFence = "OBSERVED cost projection (provider-relayed cache_read/cache_creation priced from a supplied base $/MTok), never a fak-WITNESSED claim; Track 1 (WITNESSED) and Track 2 (OBSERVED $) stay side by side, never blended"

// FoldTwoTrack is the #1304 two-track P&L fold: it folds the WITNESSED kernel rows
// (Track 1) and the OBSERVED-$ rows (Track 2) into one report that shows both
// tracks side by side with an explicit NET line per period and the running total
// crossing break-even. It is PURE — the only time input is `now` (used to stamp
// GeneratedAt and to delegate the Track-1 fold). The two folds run independently;
// nothing is averaged or summed across the provenance boundary.
func FoldTwoTrack(track1 []cachevalueledger.Row, track2 []SavingsRow, now time.Time) TwoTrackReport {
	t1 := Fold(track1, now)
	t2 := foldSavings(track2)

	rep := TwoTrackReport{
		Schema:          Schema,
		GeneratedAt:     now.UTC().Format(time.RFC3339),
		Track1:          t1,
		Track2:          t2,
		ProjectionFence: projectionFence,
		OK:              true,
		Verdict:         "INSUFFICIENT",
	}
	if n := len(t2); n > 0 {
		last := t2[n-1]
		rep.LatestNetUSD = last.NetUSD
		rep.CumulativeNetUSD = last.CumulativeNetUSD
		rep.BrokeEven = last.BrokeEven
	}

	t1Measured := t1.Verdict == "MEASURED"
	t2Measured := len(t2) > 0
	switch {
	case t1Measured && t2Measured:
		rep.Verdict = "MEASURED"
		rep.Finding = fmt.Sprintf("Track 1 realized reuse %.3f (%s); Track 2 net $%.4f this period, cumulative $%.4f (%s)",
			t1.LatestReuseRatio, t1.LatestTrend, rep.LatestNetUSD, rep.CumulativeNetUSD, breakEvenLabel(rep.BrokeEven))
		rep.NextAction = "post the two-track P&L on a cadence (epic #1301 rung E/F)"
	case t1Measured:
		rep.Verdict = "MEASURED"
		rep.Finding = fmt.Sprintf("Track 1 realized reuse %.3f (%s); Track 2 has no OBSERVED-$ rows yet (rung B not appending)",
			t1.LatestReuseRatio, t1.LatestTrend)
		rep.NextAction = "wire the OBSERVED-$ per-session append (epic #1301 rung B, #1303) to populate Track 2"
	case t2Measured:
		rep.Verdict = "MEASURED"
		rep.Finding = fmt.Sprintf("Track 2 net $%.4f this period, cumulative $%.4f (%s); Track 1 has no multi-turn reuse to trend yet",
			rep.LatestNetUSD, rep.CumulativeNetUSD, breakEvenLabel(rep.BrokeEven))
		rep.NextAction = "accumulate multi-turn guard/serve sessions to populate Track 1"
	default:
		rep.Finding = "no WITNESSED reuse and no OBSERVED-$ rows yet; nothing to fold"
		rep.NextAction = "accumulate guard/serve/run sessions into both ledgers, then re-roll"
	}
	return rep
}

func breakEvenLabel(broke bool) string {
	if broke {
		return "broke even"
	}
	return "below break-even"
}

// RenderTwoTrack produces a compact, deterministic terminal P&L: Track 1 on top
// (delegating to the Track-1 Render), then Track 2's per-period NET table with the
// running cumulative and the explicit break-even crossing. The two tracks are
// printed under separate, provenance-labelled headers so a reader can never mistake
// the OBSERVED $ projection for the WITNESSED reuse.
func RenderTwoTrack(r TwoTrackReport) string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "cache-value P&L (two tracks, side by side) — %s\n", r.Verdict)
	if r.Finding != "" {
		fmt.Fprintf(&sb, "  %s\n", r.Finding)
	}
	fmt.Fprintf(&sb, "  fence: %s\n\n", r.ProjectionFence)

	sb.WriteString(Render(r.Track1))

	fmt.Fprintf(&sb, "\nTrack 2 (OBSERVED $, provider-relayed cost projection)\n")
	if len(r.Track2) == 0 {
		fmt.Fprintf(&sb, "  no OBSERVED-$ rows yet (Track-2 ledger empty)\n")
		return sb.String()
	}
	fmt.Fprintf(&sb, "  %-9s  %5s  %10s  %10s  %10s  %10s  %12s  %s\n",
		"week", "sess", "rebate$", "compact$", "writeprem$", "spend$", "net$", "cumulative$ (break-even)")
	for _, b := range r.Track2 {
		be := ""
		if b.BrokeEven {
			be = "  >= break-even"
		}
		fmt.Fprintf(&sb, "  %-9s  %5d  %10.4f  %10.4f  %10.4f  %10.4f  %12.4f  %12.4f%s\n",
			b.Period, b.Sessions, b.RebateUSD, b.CompactionSavedUSD, b.WritePremiumUSD, b.SpendUSD,
			b.NetUSD, b.CumulativeNetUSD, be)
	}
	return sb.String()
}
