// Package perfledger is the durable per-request serving-performance record:
// one row per served model turn (TTFT, prefill/decode rates, e2e, cache share),
// the bounded summary fold over the last N rows, the one-line compact renderer,
// a bounded tail reader, and an asynchronous JSONL writer that never blocks the
// request path. The gateway produces rows; `fak perf` reads them back with the
// server up or down.
package perfledger

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cacheobs"
	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
)

const (
	Schema       = "fak.gateway.perf-record.v1"
	ReportSchema = "fak.gateway.perf-report.v1"

	// DefaultLedgerRel sits beside gatewayusageledger.DefaultLedgerRel so both
	// gateway ledgers resolve through the same nightrun root.
	DefaultLedgerRel = ".fak/nightrun/gateway-perf.jsonl"

	RingCap         = 1024
	DefaultRecent   = 50
	DefaultMaxBytes = int64(16 << 20)

	// tailReadBytes bounds the startup seed / CLI read: RingCap rows at a
	// generous ~1 KiB each, so a huge active file never costs more than this.
	tailReadBytes = int64(RingCap) << 10
)

// Locality vocabulary mirrors the gateway's servingLocality; "" means the
// serving side was not resolved and the field is omitted.
const (
	LocalitySelfHosted = "self_hosted"
	LocalityVendor     = "vendor"
	LocalityUnknown    = "unknown"
)

// Record is one served turn. PromptTokens is the UNCACHED prompt (the tokens
// actually prefilled), disjoint from CachedTokens. TTFTMS==0 means the first-token
// boundary was not observed; PrefillTPS/DecodeTPS are present only when it was.
type Record struct {
	Schema           string  `json:"schema"`
	UnixMS           int64   `json:"unix_ms"`
	FinishReason     string  `json:"finish_reason,omitempty"`
	Locality         string  `json:"locality,omitempty"`
	PromptTokens     int     `json:"prompt_tokens"`
	CompletionTokens int     `json:"completion_tokens"`
	CachedTokens     int     `json:"cached_tokens"`
	CacheRegime      string  `json:"cache_regime,omitempty"`
	E2EMS            float64 `json:"e2e_ms"`
	TTFTMS           float64 `json:"ttft_ms,omitempty"`
	PrefillTPS       float64 `json:"prefill_tps,omitempty"`
	DecodeTPS        float64 `json:"decode_tps,omitempty"`
}

// NewRecord builds a row from the raw observation. ttft is clamped into
// (0, dur] the same way the gateway's prefill/decode split clamps it.
func NewRecord(now time.Time, finishReason, locality string, promptTok, complTok, cachedTok int, dur, ttft time.Duration) Record {
	rec := Record{
		Schema:           Schema,
		UnixMS:           now.UnixMilli(),
		FinishReason:     finishReason,
		Locality:         locality,
		PromptTokens:     nonNeg(promptTok),
		CompletionTokens: nonNeg(complTok),
		CachedTokens:     nonNeg(cachedTok),
	}
	rec.CacheRegime = cacheobs.RegimeForTokens(rec.CachedTokens, rec.PromptTokens)
	if dur > 0 {
		rec.E2EMS = roundTo(float64(dur)/float64(time.Millisecond), 1000)
	}
	if ttft <= 0 || dur <= 0 {
		return rec
	}
	if ttft > dur {
		ttft = dur
	}
	rec.TTFTMS = roundTo(float64(ttft)/float64(time.Millisecond), 1000)
	if rec.PromptTokens > 0 {
		rec.PrefillTPS = roundTo(float64(rec.PromptTokens)/ttft.Seconds(), 100)
	}
	if decode := dur - ttft; decode > 0 && rec.CompletionTokens > 0 {
		rec.DecodeTPS = roundTo(float64(rec.CompletionTokens)/decode.Seconds(), 100)
	}
	return rec
}

type Window struct {
	Requests      int    `json:"requests"`
	RetainedCap   int    `json:"retained_cap"`
	Capped        bool   `json:"capped"`
	DroppedWrites uint64 `json:"dropped_writes"`
}

// Summary quantiles are nearest-rank over the rows that measured the axis; an
// axis with no measured row is omitted rather than reported as 0.
type Summary struct {
	Count         int     `json:"count"`
	TTFTMeasured  int     `json:"ttft_measured"`
	TTFTP50MS     float64 `json:"ttft_p50_ms,omitempty"`
	TTFTP99MS     float64 `json:"ttft_p99_ms,omitempty"`
	PrefillTPSP50 float64 `json:"prefill_tps_p50,omitempty"`
	DecodeTPSP50  float64 `json:"decode_tps_p50,omitempty"`
	E2EP50MS      float64 `json:"e2e_p50_ms,omitempty"`
	E2EP99MS      float64 `json:"e2e_p99_ms,omitempty"`
	CacheHitShare float64 `json:"cache_hit_share"`
	// ByRegime splits latency by prompt-cache regime (#5630) so warm vs cold TTFT is
	// one read. Keys are cacheobs.Regimes; a regime with no rows is absent.
	ByRegime map[string]RegimeSummary `json:"by_regime,omitempty"`
}

type RegimeSummary struct {
	Count        int     `json:"count"`
	TTFTMeasured int     `json:"ttft_measured"`
	TTFTP50MS    float64 `json:"ttft_p50_ms,omitempty"`
	TTFTP99MS    float64 `json:"ttft_p99_ms,omitempty"`
	E2EP50MS     float64 `json:"e2e_p50_ms,omitempty"`
}

type Report struct {
	Schema  string   `json:"schema"`
	Source  string   `json:"source,omitempty"`
	Window  Window   `json:"window"`
	Summary Summary  `json:"summary"`
	Records []Record `json:"records"`
}

// BuildReport folds the last n of recs (oldest first). n<=0 takes DefaultRecent.
func BuildReport(recs []Record, n int, capped bool, droppedWrites uint64) Report {
	if n <= 0 {
		n = DefaultRecent
	}
	if n > RingCap {
		n = RingCap
	}
	if len(recs) > n {
		recs = recs[len(recs)-n:]
	}
	out := make([]Record, len(recs))
	copy(out, recs)
	return Report{
		Schema:  ReportSchema,
		Window:  Window{Requests: len(out), RetainedCap: RingCap, Capped: capped, DroppedWrites: droppedWrites},
		Summary: Summarize(out),
		Records: out,
	}
}

// Summarize folds rows into the quantile summary. cache_hit_share is
// cached / (uncached + cached) prompt tokens.
func Summarize(recs []Record) Summary {
	s := Summary{Count: len(recs)}
	var ttft, prefill, decode, e2e []float64
	var prompt, cached int64
	regimeTTFT := map[string][]float64{}
	regimeE2E := map[string][]float64{}
	regimeCount := map[string]int{}
	for _, r := range recs {
		regime := r.regime()
		regimeCount[regime]++
		if r.TTFTMS > 0 {
			regimeTTFT[regime] = append(regimeTTFT[regime], r.TTFTMS)
		}
		if r.E2EMS > 0 {
			regimeE2E[regime] = append(regimeE2E[regime], r.E2EMS)
		}
		if r.TTFTMS > 0 {
			ttft = append(ttft, r.TTFTMS)
		}
		if r.PrefillTPS > 0 {
			prefill = append(prefill, r.PrefillTPS)
		}
		if r.DecodeTPS > 0 {
			decode = append(decode, r.DecodeTPS)
		}
		if r.E2EMS > 0 {
			e2e = append(e2e, r.E2EMS)
		}
		prompt += int64(r.PromptTokens)
		cached += int64(r.CachedTokens)
	}
	s.TTFTMeasured = len(ttft)
	s.TTFTP50MS = quantile(ttft, 0.50)
	s.TTFTP99MS = quantile(ttft, 0.99)
	s.PrefillTPSP50 = quantile(prefill, 0.50)
	s.DecodeTPSP50 = quantile(decode, 0.50)
	s.E2EP50MS = quantile(e2e, 0.50)
	s.E2EP99MS = quantile(e2e, 0.99)
	if total := prompt + cached; total > 0 {
		s.CacheHitShare = roundTo(float64(cached)/float64(total), 10000)
	}
	for regime, n := range regimeCount {
		if s.ByRegime == nil {
			s.ByRegime = make(map[string]RegimeSummary, len(regimeCount))
		}
		s.ByRegime[regime] = RegimeSummary{
			Count:        n,
			TTFTMeasured: len(regimeTTFT[regime]),
			TTFTP50MS:    quantile(regimeTTFT[regime], 0.50),
			TTFTP99MS:    quantile(regimeTTFT[regime], 0.99),
			E2EP50MS:     quantile(regimeE2E[regime], 0.50),
		}
	}
	return s
}

// regime reads the stored regime, deriving it for rows written before the field existed.
func (r Record) regime() string {
	if r.CacheRegime != "" {
		return r.CacheRegime
	}
	return cacheobs.RegimeForTokens(r.CachedTokens, r.PromptTokens)
}

// RenderCompact is the one-line agent read of a report.
func RenderCompact(rep Report) string {
	s := rep.Summary
	var b strings.Builder
	fmt.Fprintf(&b, "PERF n=%d", s.Count)
	if s.TTFTMeasured > 0 {
		fmt.Fprintf(&b, " ttft p50=%s p99=%s (measured %d/%d)", fmtMS(s.TTFTP50MS), fmtMS(s.TTFTP99MS), s.TTFTMeasured, s.Count)
	} else {
		b.WriteString(" ttft n/a")
	}
	fmt.Fprintf(&b, " | prefill p50=%s | decode p50=%s", fmtTPS(s.PrefillTPSP50), fmtTPS(s.DecodeTPSP50))
	fmt.Fprintf(&b, " | e2e p50=%s p99=%s", fmtMS(s.E2EP50MS), fmtMS(s.E2EP99MS))
	fmt.Fprintf(&b, " | cache=%.1f%%", s.CacheHitShare*100)
	if len(s.ByRegime) > 0 {
		b.WriteString(" | ttft p50 by regime:")
		for _, regime := range cacheobs.Regimes {
			if rs, ok := s.ByRegime[regime]; ok {
				fmt.Fprintf(&b, " %s=%s(n=%d)", regime, fmtMS(rs.TTFTP50MS), rs.Count)
			}
		}
	}
	if rep.Window.Capped {
		b.WriteString(" | capped")
	}
	if rep.Window.DroppedWrites > 0 {
		fmt.Fprintf(&b, " | dropped_writes=%d", rep.Window.DroppedWrites)
	}
	return b.String()
}

// ReadTail returns at most RingCap valid rows from the end of the ledger at
// path (oldest first) and whether older rows exist beyond the bounded window.
// A missing file returns nil, false; malformed lines are skipped.
func ReadTail(path string) ([]Record, bool) {
	data := jsonlledger.ReadTail(path, tailReadBytes)
	if len(data) == 0 {
		return nil, false
	}
	var out []Record
	sc := bufio.NewScanner(bytes.NewReader(data))
	sc.Buffer(make([]byte, 0, 4096), 1<<20)
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var r Record
		if json.Unmarshal(line, &r) != nil || r.Schema != Schema {
			continue
		}
		out = append(out, r)
	}
	truncated := int64(len(data)) >= tailReadBytes
	if len(out) > RingCap {
		out = out[len(out)-RingCap:]
		truncated = true
	}
	return out, truncated
}

func quantile(vals []float64, q float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	sorted := append([]float64(nil), vals...)
	sort.Float64s(sorted)
	idx := int(math.Ceil(q*float64(len(sorted)))) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

func fmtMS(v float64) string {
	if v <= 0 {
		return "n/a"
	}
	if v >= 1000 {
		return fmt.Sprintf("%.2fs", v/1000)
	}
	return fmt.Sprintf("%.0fms", v)
}

func fmtTPS(v float64) string {
	if v <= 0 {
		return "n/a"
	}
	return fmt.Sprintf("%.1f tok/s", v)
}

func roundTo(v, scale float64) float64 {
	return math.Round(v*scale) / scale
}

func nonNeg(v int) int {
	if v < 0 {
		return 0
	}
	return v
}
