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
// the row, reader, and append helper; the CLI front doors wire live session summaries
// into those rows.

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
	"github.com/anthony-chaudhary/fak/internal/vcachegov"
)

// SavingsLedgerSchema tags each durable OBSERVED-$ row (rung B, #1303). It is
// distinct from the kernel ledger schema (Track 1) so a reader can never confuse
// a WITNESSED row for an OBSERVED one.
const SavingsLedgerSchema = "fak-cache-savings-ledger/1"

// DefaultSavingsLedgerRel is the sibling ledger that carries Track 2 (the OBSERVED
// $ economics), written next to the kernel ledger (docs/nightrun/cache-value.jsonl)
// per the #1303 "sibling file" option, so the two provenances never share a row.
const DefaultSavingsLedgerRel = "docs/nightrun/cache-savings.jsonl"

const providerCacheReadMultiplier = 0.1

const (
	// SavingsDollarStatusBlind marks rows whose token axes are observed but whose
	// dollar axes are intentionally unpriced because no trusted base price was configured.
	SavingsDollarStatusBlind = "dollar_blind"
	savingsDollarStatusMixed = "mixed"
)

// SavingsPricing is the caller-supplied base price for the model in play. DollarBlind
// keeps zero-dollar rows explicit when the live session has provider counters but no
// trusted price table, so $0 cannot be read as a priced no-savings result.
type SavingsPricing struct {
	InputPerMTokUSD  float64
	OutputPerMTokUSD float64
	Source           string
	DollarBlind      bool
}

// SavingsObservation is the live-session input that becomes one or more durable
// Track-2 rows. The provider prompt-cache and fak compaction mechanisms are emitted
// as separate rows so owner/mechanism roll-ups never blend them.
type SavingsObservation struct {
	SessionType string
	Provider    string
	Context     string

	InputTokens          uint64
	CacheReadTokens      uint64
	CacheCreationTokens  uint64
	OutputTokens         uint64
	CompactionShedTokens uint64

	Pricing SavingsPricing
}

// SavingsRow is one durable, append-only cache-economics row — Track 2 of epic #1301,
// the per-session shape rung B (#1303) persists. Provider prompt-cache token axes are
// OBSERVED/provider-relayed; fak compaction rows carry a WITNESSED fak-authored shed-token
// axis whose dollar value is still only a projection from supplied base rates.
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
	Provider    string `json:"provider,omitempty"`
	Mechanism   string `json:"mechanism,omitempty"`
	Context     string `json:"context,omitempty"`
	GeneratedAt string `json:"generated_at,omitempty"`

	// Token axes (OBSERVED, provider-reported).
	InputTokens         uint64 `json:"input_tokens"`
	CacheReadTokens     uint64 `json:"cache_read_tokens"`
	CacheCreationTokens uint64 `json:"cache_creation_tokens"`
	OutputTokens        uint64 `json:"output_tokens"`

	// CompactionShedTokens is the input tokens `--compact-history-budget` (#745)
	// dropped before the turn was sent — a fak-authored token saving on the spend side.
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
	PricingSource    string  `json:"pricing_source,omitempty"`
	DollarStatus     string  `json:"dollar_status,omitempty"`
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
	Period    string `json:"period"` // ISO week, e.g. "2026-W26"
	Start     string `json:"start"`  // earliest row date in the bucket (YYYY-MM-DD)
	Provider  string `json:"provider"`
	Mechanism string `json:"mechanism"`
	Sessions  int    `json:"sessions"`

	InputTokens          uint64 `json:"input_tokens"`
	CacheReadTokens      uint64 `json:"cache_read_tokens"`
	CacheCreationTokens  uint64 `json:"cache_creation_tokens"`
	OutputTokens         uint64 `json:"output_tokens"`
	CompactionShedTokens uint64 `json:"compaction_shed_tokens"`

	SavedTokenEquiv    float64 `json:"saved_token_equiv"`
	NetSavedTokenEquiv float64 `json:"net_saved_token_equiv"`

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

	DollarStatus        string `json:"dollar_status,omitempty"`
	DollarBlindSessions int    `json:"dollar_blind_sessions,omitempty"`
}

// OwnerAttributionBucket is the report-level owner split, in token-equivalent
// units. Provider prompt-cache savings stay separate from fak-authored savings;
// vDSO is currently counted as avoided calls because no token axis exists for it.
type OwnerAttributionBucket struct {
	Period                        string  `json:"period"`
	ProviderPromptCacheTokenEquiv float64 `json:"provider_prompt_cache_token_equiv"`
	FakAuthoredTokenEquiv         float64 `json:"fak_authored_token_equiv"`
	FakKVPrefixReusedTokens       uint64  `json:"fak_kv_prefix_reused_tokens"`
	FakCompactionShedTokens       uint64  `json:"fak_compaction_shed_tokens"`
	FakVDSOAvoidedCalls           uint64  `json:"fak_vdso_avoided_calls"`
}

// NewSavingsRows converts one live session observation into the durable Track-2 row shape.
// Provider prompt-cache and fak-authored compaction are split into separate mechanisms while
// preserving the same session/date/context labels. The dollar fields are projections from the
// caller-supplied base $/MTok; with zero pricing they remain zero rather than fabricating a
// provider-specific price.
func NewSavingsRows(obs SavingsObservation, now time.Time) []SavingsRow {
	now = now.UTC()
	base := SavingsRow{
		Schema:        SavingsLedgerSchema,
		Date:          now.Format("2006-01-02"),
		SessionType:   obs.SessionType,
		Context:       obs.Context,
		GeneratedAt:   now.Format(time.RFC3339),
		PricingSource: strings.TrimSpace(obs.Pricing.Source),
		DollarStatus:  savingsDollarStatus(obs.Pricing),
	}
	var rows []SavingsRow
	if obs.CacheReadTokens > 0 || obs.CacheCreationTokens > 0 {
		row := base
		row.Provider = strings.TrimSpace(obs.Provider)
		row.Mechanism = "provider_prompt_cache"
		row.InputTokens = obs.InputTokens
		row.CacheReadTokens = obs.CacheReadTokens
		row.CacheCreationTokens = obs.CacheCreationTokens
		row.OutputTokens = obs.OutputTokens
		row.SavedTokenEquiv = providerSavedTokenEquiv(obs.CacheReadTokens, obs.CacheCreationTokens)
		row.NetSavedTokenEquiv = row.SavedTokenEquiv
		row.InputPerMTokUSD = obs.Pricing.InputPerMTokUSD
		row.OutputPerMTokUSD = obs.Pricing.OutputPerMTokUSD
		row.RebateUSD = perMTok(obs.Pricing.InputPerMTokUSD, float64(obs.CacheReadTokens)*(1-providerCacheReadMultiplier))
		row.WritePremiumUSD = perMTok(obs.Pricing.InputPerMTokUSD, float64(obs.CacheCreationTokens)*(vcachegov.WriteMult5Minutes-1))
		row.SpendUSD = providerSpendUSD(obs)
		row.NetUSD = row.NetUSDComputed()
		normalizeSavingsDimensions(&row)
		rows = append(rows, row)
	}
	if obs.CompactionShedTokens > 0 {
		row := base
		row.Provider = "fak"
		row.Mechanism = "compaction_shed"
		row.CompactionShedTokens = obs.CompactionShedTokens
		row.SavedTokenEquiv = float64(obs.CompactionShedTokens)
		row.NetSavedTokenEquiv = row.SavedTokenEquiv
		row.InputPerMTokUSD = obs.Pricing.InputPerMTokUSD
		row.OutputPerMTokUSD = obs.Pricing.OutputPerMTokUSD
		row.CompactionSavedUSD = perMTok(obs.Pricing.InputPerMTokUSD, float64(obs.CompactionShedTokens))
		row.NetUSD = row.NetUSDComputed()
		normalizeSavingsDimensions(&row)
		rows = append(rows, row)
	}
	return rows
}

func providerSavedTokenEquiv(read, creation uint64) float64 {
	return float64(read)*(1-providerCacheReadMultiplier) + float64(creation)*(1-vcachegov.WriteMult5Minutes)
}

func providerSpendUSD(obs SavingsObservation) float64 {
	inTok := float64(obs.InputTokens) +
		float64(obs.CacheReadTokens)*providerCacheReadMultiplier +
		float64(obs.CacheCreationTokens)*vcachegov.WriteMult5Minutes
	outTok := float64(obs.OutputTokens)
	return perMTok(obs.Pricing.InputPerMTokUSD, inTok) + perMTok(obs.Pricing.OutputPerMTokUSD, outTok)
}

func perMTok(price, tokens float64) float64 {
	return price * tokens / 1_000_000
}

func savingsDollarStatus(p SavingsPricing) string {
	if p.DollarBlind || (p.InputPerMTokUSD == 0 && p.OutputPerMTokUSD == 0) {
		return SavingsDollarStatusBlind
	}
	return ""
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
		normalizeSavingsDimensions(&row)
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
	normalizeSavingsDimensions(&row)
	b, err := json.Marshal(row)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// AppendSavings appends one durable Track-2 row to the configured JSONL ledger.
func AppendSavings(ledgerPath string, row SavingsRow) error {
	line, err := AppendSavingsLine(row)
	if err != nil {
		return err
	}
	f, err := os.OpenFile(ledgerPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return err
	}
	defer f.Close()
	if _, err := f.WriteString(line + "\n"); err != nil {
		return err
	}
	return nil
}

func normalizeSavingsDimensions(row *SavingsRow) {
	if strings.TrimSpace(row.Provider) == "" {
		row.Provider = "unknown_provider"
	}
	if strings.TrimSpace(row.Mechanism) == "" {
		switch {
		case row.CacheReadTokens > 0 || row.CacheCreationTokens > 0:
			row.Mechanism = "provider_prompt_cache"
		case row.CompactionShedTokens > 0:
			row.Mechanism = "compaction_shed"
		default:
			row.Mechanism = "unknown_mechanism"
		}
	}
	if strings.TrimSpace(row.DollarStatus) == "" &&
		row.InputPerMTokUSD == 0 &&
		row.OutputPerMTokUSD == 0 &&
		row.RebateUSD == 0 &&
		row.WritePremiumUSD == 0 &&
		row.SpendUSD == 0 &&
		row.CompactionSavedUSD == 0 &&
		(row.CacheReadTokens > 0 || row.CacheCreationTokens > 0 || row.CompactionShedTokens > 0) {
		row.DollarStatus = SavingsDollarStatusBlind
	}
}

// foldSavings buckets Track-2 rows by ISO week, provider, and mechanism, then computes each
// bucket's NET plus the running cumulative that crosses break-even. It is pure (no clock, no
// I/O): bucketing comes from each row's own Date. Rows with an unparseable date are skipped,
// mirroring the Track-1 fold.
func foldSavings(rows []SavingsRow) []SavingsBucket {
	type agg struct {
		b     SavingsBucket
		start time.Time
	}
	byPeriod := map[string]*agg{}
	for _, row := range rows {
		normalizeSavingsDimensions(&row)
		d, err := time.Parse("2006-01-02", row.Date)
		if err != nil {
			continue
		}
		period := isoWeek(d)
		key := period + "\x00" + row.Provider + "\x00" + row.Mechanism
		a := byPeriod[key]
		if a == nil {
			a = &agg{b: SavingsBucket{Period: period, Provider: row.Provider, Mechanism: row.Mechanism}, start: d}
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
		b.SavedTokenEquiv += row.SavedTokenEquiv
		b.NetSavedTokenEquiv += row.NetSavedTokenEquiv
		b.RebateUSD += row.RebateUSD
		b.WritePremiumUSD += row.WritePremiumUSD
		b.SpendUSD += row.SpendUSD
		b.CompactionSavedUSD += row.CompactionSavedUSD
		if row.DollarStatus == SavingsDollarStatusBlind {
			b.DollarBlindSessions++
		}
	}

	keys := sortedPeriodKeys(byPeriod)

	buckets := make([]SavingsBucket, 0, len(keys))
	var cumulative float64
	brokeEven := false
	for _, k := range keys {
		a := byPeriod[k]
		b := a.b
		b.Start = a.start.Format("2006-01-02")
		if b.DollarBlindSessions > 0 {
			if b.DollarBlindSessions == b.Sessions {
				b.DollarStatus = SavingsDollarStatusBlind
			} else {
				b.DollarStatus = savingsDollarStatusMixed
			}
		}
		b.NetUSD = b.RebateUSD + b.CompactionSavedUSD - b.WritePremiumUSD - b.SpendUSD
		if b.DollarStatus != SavingsDollarStatusBlind {
			cumulative += b.NetUSD
		}
		b.CumulativeNetUSD = cumulative
		if b.DollarStatus != SavingsDollarStatusBlind && !brokeEven && cumulative >= 0 {
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

	OwnerAttribution []OwnerAttributionBucket `json:"owner_attribution"`

	// LatestNetUSD / CumulativeNetUSD are the most-recent period's net and the
	// running total through it — the P&L headline. BrokeEven is whether the running
	// total has crossed zero.
	LatestNetUSD     float64 `json:"latest_net_usd"`
	CumulativeNetUSD float64 `json:"cumulative_net_usd"`
	BrokeEven        bool    `json:"broke_even"`
	DollarBlindRows  int     `json:"dollar_blind_rows,omitempty"`

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
const projectionFence = "cost projection over labelled sources (OBSERVED provider cache_read/cache_creation plus WITNESSED fak-authored token axes priced from supplied base $/MTok), never blended into one cache claim; Track 1 (WITNESSED) and Track 2 (OBSERVED/projected $) stay side by side"

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
		Schema:           Schema,
		GeneratedAt:      now.UTC().Format(time.RFC3339),
		Track1:           t1,
		Track2:           t2,
		OwnerAttribution: foldOwnerAttribution(t1.Buckets, t2),
		ProjectionFence:  projectionFence,
		OK:               true,
		Verdict:          "INSUFFICIENT",
	}
	if n := len(t2); n > 0 {
		last := t2[n-1]
		rep.LatestNetUSD = last.NetUSD
		rep.CumulativeNetUSD = last.CumulativeNetUSD
		rep.BrokeEven = last.BrokeEven
	}
	for _, b := range t2 {
		rep.DollarBlindRows += b.DollarBlindSessions
	}

	t1Measured := t1.Verdict == "MEASURED"
	t2Measured := len(t2) > 0
	t2AllDollarBlind := t2Measured && allSavingsBucketsDollarBlind(t2)
	switch {
	case t1Measured && t2AllDollarBlind:
		rep.Verdict = "MEASURED"
		rep.Finding = fmt.Sprintf("Track 1 realized reuse %.3f (%s); Track 2 has token evidence but is dollar-blind (no price configured)",
			t1.LatestReuseRatio, t1.LatestTrend)
		rep.NextAction = "set FAK_CACHEVALUE_INPUT_PER_MTOK_USD / FAK_CACHEVALUE_OUTPUT_PER_MTOK_USD or use a known provider/context default before treating Track 2 as dollars"
	case t2AllDollarBlind:
		rep.Verdict = "MEASURED"
		rep.Finding = "Track 2 has token evidence but is dollar-blind (no price configured); Track 1 has no multi-turn reuse to trend yet"
		rep.NextAction = "set FAK_CACHEVALUE_INPUT_PER_MTOK_USD / FAK_CACHEVALUE_OUTPUT_PER_MTOK_USD or use a known provider/context default before treating Track 2 as dollars"
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

func allSavingsBucketsDollarBlind(buckets []SavingsBucket) bool {
	if len(buckets) == 0 {
		return false
	}
	for _, b := range buckets {
		if b.DollarStatus != SavingsDollarStatusBlind {
			return false
		}
	}
	return true
}

func foldOwnerAttribution(track1 []Bucket, track2 []SavingsBucket) []OwnerAttributionBucket {
	byPeriod := map[string]*OwnerAttributionBucket{}
	ensure := func(period string) *OwnerAttributionBucket {
		b := byPeriod[period]
		if b == nil {
			b = &OwnerAttributionBucket{Period: period}
			byPeriod[period] = b
		}
		return b
	}
	for _, b := range track1 {
		if b.Period == "" {
			continue
		}
		dst := ensure(b.Period)
		dst.FakKVPrefixReusedTokens += b.ReusedTokens
		dst.FakAuthoredTokenEquiv += float64(b.ReusedTokens)
	}
	for _, b := range track2 {
		if b.Period == "" {
			continue
		}
		dst := ensure(b.Period)
		switch {
		case b.Provider == "fak" || strings.HasPrefix(b.Mechanism, "compaction"):
			dst.FakCompactionShedTokens += b.CompactionShedTokens
			if b.NetSavedTokenEquiv != 0 {
				dst.FakAuthoredTokenEquiv += b.NetSavedTokenEquiv
			} else {
				dst.FakAuthoredTokenEquiv += float64(b.CompactionShedTokens)
			}
		case b.Mechanism == "provider_prompt_cache":
			if b.NetSavedTokenEquiv != 0 {
				dst.ProviderPromptCacheTokenEquiv += b.NetSavedTokenEquiv
			} else {
				dst.ProviderPromptCacheTokenEquiv += b.SavedTokenEquiv
			}
		}
	}
	keys := sortedPeriodKeys(byPeriod)
	out := make([]OwnerAttributionBucket, 0, len(keys))
	for _, k := range keys {
		out = append(out, *byPeriod[k])
	}
	return out
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
	if r.DollarBlindRows > 0 {
		fmt.Fprintf(&sb, "  pricing: %d row(s) dollar-blind (no price configured); zero dollar fields are placeholders, not priced savings\n",
			r.DollarBlindRows)
	}
	fmt.Fprintf(&sb, "  %-9s  %-16s  %-23s  %5s  %10s  %10s  %10s  %10s  %12s  %-13s  %s\n",
		"week", "provider", "mechanism", "sess", "rebate$", "compact$", "writeprem$", "spend$", "net$", "pricing", "cumulative$ (break-even)")
	for _, b := range r.Track2 {
		be := ""
		if b.BrokeEven {
			be = "  >= break-even"
		}
		pricing := b.DollarStatus
		if pricing == "" {
			pricing = "priced"
		}
		fmt.Fprintf(&sb, "  %-9s  %-16s  %-23s  %5d  %10.4f  %10.4f  %10.4f  %10.4f  %12.4f  %-13s  %12.4f%s\n",
			b.Period, b.Provider, b.Mechanism, b.Sessions, b.RebateUSD, b.CompactionSavedUSD, b.WritePremiumUSD, b.SpendUSD,
			b.NetUSD, pricing, b.CumulativeNetUSD, be)
	}
	fmt.Fprintf(&sb, "\nOwner attribution (token-equiv; provider prompt-cache vs fak-authored)\n")
	if len(r.OwnerAttribution) == 0 {
		fmt.Fprintf(&sb, "  no owner-attribution rows yet\n")
		return sb.String()
	}
	fmt.Fprintf(&sb, "  %-9s  %13s  %10s  %10s  %11s  %s\n",
		"week", "provider_teq", "fak_teq", "kv_tok", "compact_tok", "vdso_calls")
	for _, b := range r.OwnerAttribution {
		fmt.Fprintf(&sb, "  %-9s  %13.0f  %10.0f  %10d  %11d  %d\n",
			b.Period, b.ProviderPromptCacheTokenEquiv, b.FakAuthoredTokenEquiv,
			b.FakKVPrefixReusedTokens, b.FakCompactionShedTokens, b.FakVDSOAvoidedCalls)
	}
	return sb.String()
}
