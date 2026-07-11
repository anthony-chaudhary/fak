// Package cachevalue folds the persisted cache-savings ledger
// (docs/nightrun/cache-savings.jsonl) into per-session cache-efficiency
// metrics and flags regressions (#1992).
//
// The ledger already records the raw axes per guard session — input_tokens,
// cache_read_tokens, cache_creation_tokens — but nothing derives the two
// signals that make a churny session legible:
//
//   - hit rate = cache_read / (cache_read + cache_creation + input) — the
//     fraction of prompt-side tokens served from the provider prompt cache.
//     Healthy long guard sessions sit near 0.99; a session whose prefix keeps
//     mutating (system prompt / tool set churn) falls toward 0.80.
//   - write amplification = cache_creation / cache_read — cache writes per
//     cached read. Writes bill ~1.25x base while reads bill ~0.1x, so every
//     write-amp point is real cost; the observed spread on one night was 23x
//     (0.011 best, 0.256 worst) with no signal firing.
//
// The fold is I/O-free (parse from an io.Reader); FoldFile is the thin
// path-opening helper. FlagRegressions is the closed-verdict rung: a flagged
// session carries stable reason tokens plus the metrics it was judged on, so
// a reader never has to re-derive why it fired.
package cachevalue

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// Default thresholds for FlagRegressions, derived from the observed 2026-07-01
// spread (#1992): the healthy guard sessions cluster at ~0.99 hit / ~0.01
// write-amp, the churn-spike session (T15:07) lands at 0.913 / 0.086, and the
// regressed ones fall to 0.836 / 0.168 and below. A 0.90 floor / 0.10 ceiling
// admits the spike as borderline-healthy and flags the genuine regressions.
const (
	// DefaultHitRateFloor is the hit-rate below which a session is flagged.
	DefaultHitRateFloor = 0.90
	// DefaultWriteAmpCeil is the write-amplification above which a session
	// is flagged.
	DefaultWriteAmpCeil = 0.10
)

// Stable reason tokens carried by a Flag. Fixed strings so a report renderer
// or guard-exit WARN line can match on them rather than parse prose.
const (
	// ReasonHitFloor fires when a session's hit rate fell below the floor.
	ReasonHitFloor = "hit_floor"
	// ReasonWriteAmpCeil fires when a session's write-amp exceeded the ceiling.
	ReasonWriteAmpCeil = "write_amp_ceil"
	// ReasonWriteAmpUnbounded fires when a session wrote cache-creation tokens
	// but read zero cached tokens — write-amp is unbounded (division by zero),
	// the degenerate worst case, and is flagged fail-closed rather than skipped.
	ReasonWriteAmpUnbounded = "write_amp_unbounded"
)

// Row is one parsed cache-savings ledger line (schema
// "fak-cache-savings-ledger/1"). Field names mirror the on-disk JSON exactly;
// fields the fold does not consume (the saved/usd rollups) are omitted —
// encoding/json ignores them on decode, so the fold stays tolerant of
// schema-additive ledger growth.
type Row struct {
	Schema      string `json:"schema"`
	Date        string `json:"date"`
	SessionType string `json:"session_type"`
	Provider    string `json:"provider"`
	Mechanism   string `json:"mechanism"`
	Context     string `json:"context"`
	// GeneratedAt is the session key: one ledger row is one guard session,
	// stamped at write time.
	GeneratedAt         string `json:"generated_at"`
	InputTokens         int64  `json:"input_tokens"`
	CacheReadTokens     int64  `json:"cache_read_tokens"`
	CacheCreationTokens int64  `json:"cache_creation_tokens"`
	OutputTokens        int64  `json:"output_tokens"`
}

// Metrics is the per-session derived reading. The Known bits make the
// divide-by-zero cases explicit instead of smuggling a phantom 0 or an
// unmarshalable +Inf through the value fields.
type Metrics struct {
	// GeneratedAt is the session key, copied from the row.
	GeneratedAt string
	// Row is the ledger row the metrics were derived from, kept whole so a
	// flag can always show its evidence.
	Row Row
	// HitRate is cache_read / (cache_read + cache_creation + input): the
	// prompt-side fraction served from the provider cache. 0 when HitRateKnown
	// is false (an all-zero row never reports a phantom rate).
	HitRate float64
	// HitRateKnown is false when the denominator is zero — a row with no
	// prompt-side tokens at all has no hit rate, not a 0% one.
	HitRateKnown bool
	// WriteAmp is cache_creation / cache_read: cache writes per cached read.
	// 0 when the session did no cache writes; 0 and NOT known when it wrote
	// with zero reads (unbounded — see WriteAmpKnown).
	WriteAmp float64
	// WriteAmpKnown is false only when cache_creation > 0 with
	// cache_read == 0: the ratio is unbounded, and FlagRegressions treats
	// that as the worst case rather than dividing by zero.
	WriteAmpKnown bool
}

// metrics derives the per-session reading from one row, guarding both
// zero denominators.
func metrics(row Row) Metrics {
	m := Metrics{GeneratedAt: row.GeneratedAt, Row: row}
	if denom := row.CacheReadTokens + row.CacheCreationTokens + row.InputTokens; denom > 0 {
		m.HitRate = float64(row.CacheReadTokens) / float64(denom)
		m.HitRateKnown = true
	}
	switch {
	case row.CacheReadTokens > 0:
		m.WriteAmp = float64(row.CacheCreationTokens) / float64(row.CacheReadTokens)
		m.WriteAmpKnown = true
	case row.CacheCreationTokens <= 0:
		// No writes and no reads: zero amplification is an honest reading.
		m.WriteAmpKnown = true
	}
	return m
}

// Fold parses a cache-savings JSONL stream and returns per-session metrics in
// ledger order. Blank lines are skipped; a malformed line is an error naming
// its 1-based line number (a silently dropped row would hide exactly the
// churny session the fold exists to surface). The core is I/O-free beyond the
// reader: no paths, no clock, no globals.
func Fold(r io.Reader) ([]Metrics, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	var out []Metrics
	line := 0
	for sc.Scan() {
		line++
		raw := sc.Bytes()
		if len(trimSpace(raw)) == 0 {
			continue
		}
		var row Row
		if err := json.Unmarshal(raw, &row); err != nil {
			return nil, fmt.Errorf("cachevalue: ledger line %d: %w", line, err)
		}
		out = append(out, metrics(row))
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("cachevalue: reading ledger: %w", err)
	}
	return out, nil
}

// trimSpace trims ASCII whitespace without allocating (bytes.TrimSpace on the
// scanner's buffer would be fine too; this keeps the dependency set minimal).
func trimSpace(b []byte) []byte {
	for len(b) > 0 && isSpace(b[0]) {
		b = b[1:]
	}
	for len(b) > 0 && isSpace(b[len(b)-1]) {
		b = b[:len(b)-1]
	}
	return b
}

func isSpace(c byte) bool { return c == ' ' || c == '\t' || c == '\r' || c == '\n' }

// FoldFile is the thin path-opening helper over Fold for callers holding a
// ledger path (e.g. `fak cachevalue report` over docs/nightrun/cache-savings.jsonl).
func FoldFile(path string) ([]Metrics, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("cachevalue: open ledger: %w", err)
	}
	defer f.Close()
	return Fold(f)
}

// Thresholds are the regression gates FlagRegressions judges against. The
// zero value means "use the documented defaults": a non-positive floor or
// ceiling is normalized to DefaultHitRateFloor / DefaultWriteAmpCeil, so a
// forgotten field can never silently disable a rung.
type Thresholds struct {
	// HitRateFloor flags sessions with HitRate strictly below it.
	HitRateFloor float64
	// WriteAmpCeil flags sessions with WriteAmp strictly above it.
	WriteAmpCeil float64
}

// DefaultThresholds returns the documented default gates (#1992).
func DefaultThresholds() Thresholds {
	return Thresholds{HitRateFloor: DefaultHitRateFloor, WriteAmpCeil: DefaultWriteAmpCeil}
}

// normalized returns the thresholds with non-positive fields replaced by the
// defaults, so the zero value is safe rather than a no-op gate.
func (t Thresholds) normalized() Thresholds {
	if t.HitRateFloor <= 0 {
		t.HitRateFloor = DefaultHitRateFloor
	}
	if t.WriteAmpCeil <= 0 {
		t.WriteAmpCeil = DefaultWriteAmpCeil
	}
	return t
}

// Flag is one flagged session: the stable reason tokens that fired plus the
// metrics they were judged on, so the verdict carries its own evidence.
type Flag struct {
	// GeneratedAt is the flagged session's key.
	GeneratedAt string
	// Metrics is the reading the flag was drawn from.
	Metrics Metrics
	// Reasons are the reason tokens that fired, in a fixed order
	// (hit rung first), each one of the Reason* constants.
	Reasons []string
}

// FlagRegressions judges each session against the thresholds and returns the
// flagged ones, in input order. A session is flagged when its known hit rate
// fell below the floor, its known write-amp exceeded the ceiling, or its
// write-amp is unbounded (writes with zero reads — fail-closed as the worst
// case). An unknown hit rate (an all-zero row) does not fire the hit rung: an
// idle session never reports a phantom regression.
func FlagRegressions(ms []Metrics, th Thresholds) []Flag {
	th = th.normalized()
	var out []Flag
	for _, m := range ms {
		var reasons []string
		if m.HitRateKnown && m.HitRate < th.HitRateFloor {
			reasons = append(reasons, ReasonHitFloor)
		}
		switch {
		case !m.WriteAmpKnown:
			reasons = append(reasons, ReasonWriteAmpUnbounded)
		case m.WriteAmp > th.WriteAmpCeil:
			reasons = append(reasons, ReasonWriteAmpCeil)
		}
		if len(reasons) > 0 {
			out = append(out, Flag{GeneratedAt: m.GeneratedAt, Metrics: m, Reasons: reasons})
		}
	}
	return out
}
