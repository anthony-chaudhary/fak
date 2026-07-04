package compute

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math/rand"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
)

const KVReplayTraceSchema = "fak.kvbm.trace/v1"

// KVReplayTrace is a host-free replay corpus for comparing KV eviction policies.
type KVReplayTrace struct {
	Schema       string               `json:"schema"`
	Name         string               `json:"name"`
	Source       string               `json:"source"`
	BudgetTokens int                  `json:"budget_tokens"`
	Events       []KVReplayTraceEvent `json:"events"`
}

// KVReplayTraceEvent is one prefix-span touch in a replay corpus.
type KVReplayTraceEvent struct {
	Session string `json:"session,omitempty"`
	SpanID  string `json:"span_id"`
	Tokens  int    `json:"tokens"`
}

// KVReplayTraceReport is the policy-vs-oracle rollup for one corpus.
type KVReplayTraceReport struct {
	Name         string                           `json:"name"`
	Source       string                           `json:"source"`
	BudgetTokens int                              `json:"budget_tokens"`
	Policies     map[KVEvictPolicy]KVReplayResult `json:"policies"`
	Oracle       KVReplayOracleResult             `json:"oracle"`
}

// KVReplayGatewayTraceOptions controls reduction of durable usage-ledger rows into
// prefix-touch replay events.
type KVReplayGatewayTraceOptions struct {
	Name          string
	BudgetTokens  int
	MaxSpanTokens int
}

// KVReplaySyntheticOptions controls the deterministic Zipf-ish, bimodal-gap corpus
// generator used when a real gateway ledger is too thin for a stable CI witness.
type KVReplaySyntheticOptions struct {
	Name         string
	Seed         int64
	Events       int
	HotSpans     int
	SpanTokens   int
	BudgetTokens int
}

// ParseKVReplayTrace parses and validates a committed replay trace corpus.
func ParseKVReplayTrace(data []byte) (KVReplayTrace, error) {
	var trace KVReplayTrace
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&trace); err != nil {
		return KVReplayTrace{}, err
	}
	if err := trace.validate(); err != nil {
		return KVReplayTrace{}, err
	}
	return trace, nil
}

// ReplayKVTrace replays every requested policy and scores each policy against the
// finite-trace offline oracle.
func ReplayKVTrace(trace KVReplayTrace, policies ...KVEvictPolicy) (KVReplayTraceReport, error) {
	if err := trace.validate(); err != nil {
		return KVReplayTraceReport{}, err
	}
	events := trace.ReplayEvents()
	return KVReplayTraceReport{
		Name:         trace.Name,
		Source:       trace.Source,
		BudgetTokens: trace.BudgetTokens,
		Policies:     ReplayKVCacheMulti(events, trace.BudgetTokens, policies...),
		Oracle:       BeladyKVReplayOracle(events, trace.BudgetTokens),
	}, nil
}

// ReplayEvents maps stable string span ids to the integer ids the low-level simulator uses.
func (t KVReplayTrace) ReplayEvents() []KVReplayEvent {
	spanIndexes := map[string]int{}
	out := make([]KVReplayEvent, 0, len(t.Events))
	for _, ev := range t.Events {
		idx, ok := spanIndexes[ev.SpanID]
		if !ok {
			idx = len(spanIndexes) + 1
			spanIndexes[ev.SpanID] = idx
		}
		out = append(out, KVReplayEvent{SpanID: idx, Tokens: ev.Tokens})
	}
	return out
}

// GatewayUsageRowsToKVReplayTrace reduces durable gateway-usage rows into replayable
// prefix touches. It never reads prompt bytes; it uses only the token counters already
// admitted to the public usage ledger.
func GatewayUsageRowsToKVReplayTrace(rows []gatewayusageledger.Row, opts KVReplayGatewayTraceOptions) KVReplayTrace {
	if opts.Name == "" {
		opts.Name = "gateway-usage-kv-prefix"
	}
	if opts.BudgetTokens <= 0 {
		opts.BudgetTokens = 1024
	}
	if opts.MaxSpanTokens <= 0 {
		opts.MaxSpanTokens = 512
	}

	sorted := make([]gatewayusageledger.Row, len(rows))
	copy(sorted, rows)
	sort.SliceStable(sorted, func(i, j int) bool {
		if sorted[i].UnixMillis != sorted[j].UnixMillis {
			return sorted[i].UnixMillis < sorted[j].UnixMillis
		}
		return sorted[i].PID < sorted[j].PID
	})

	events := make([]KVReplayTraceEvent, 0, len(sorted)*3)
	for i, row := range sorted {
		c := row.Counters
		prefix := maxCounter(c.KVPrefixPromptTokens, c.CachedPromptTokens, c.CompactionCacheReadTokens)
		if prefix > 0 {
			tokens := clampReplayTokens(prefix, opts.MaxSpanTokens)
			spanID := fmt.Sprintf("gateway:%s:%s:shared-prefix", emptyDefault(row.SessionType, "unknown"), emptyDefault(row.Context, "default"))
			events = append(events, KVReplayTraceEvent{
				Session: gatewayReplaySession(row, i),
				SpanID:  spanID,
				Tokens:  tokens,
			})
			if c.KVPrefixReusedTokens > 0 || c.CachedPromptTokens > 0 || c.CachedTurns > 0 {
				events = append(events, KVReplayTraceEvent{
					Session: gatewayReplaySession(row, i) + "-reuse",
					SpanID:  spanID,
					Tokens:  tokens,
				})
			}
		}

		if c.InputTokens > prefix {
			events = append(events, KVReplayTraceEvent{
				Session: gatewayReplaySession(row, i),
				SpanID:  fmt.Sprintf("gateway:%s:tail", gatewayReplaySession(row, i)),
				Tokens:  clampReplayTokens(c.InputTokens-prefix, opts.MaxSpanTokens),
			})
		}
	}

	return KVReplayTrace{
		Schema:       KVReplayTraceSchema,
		Name:         opts.Name,
		Source:       "gateway-usage-ledger",
		BudgetTokens: opts.BudgetTokens,
		Events:       events,
	}
}

// GenerateKVReplaySyntheticTrace builds a deterministic skewed-popularity corpus with
// short hot-span reuse and longer cold bursts. It is deliberately simple: its job is to
// keep CI's replay witness nonempty without shipping private traces.
func GenerateKVReplaySyntheticTrace(opts KVReplaySyntheticOptions) KVReplayTrace {
	if opts.Name == "" {
		opts.Name = "synthetic-zipf-bimodal-kv-prefix"
	}
	if opts.Events <= 0 {
		opts.Events = 30
	}
	if opts.HotSpans <= 0 {
		opts.HotSpans = 4
	}
	if opts.SpanTokens <= 0 {
		opts.SpanTokens = 50
	}
	if opts.BudgetTokens <= 0 {
		opts.BudgetTokens = opts.SpanTokens * 2
	}

	rng := rand.New(rand.NewSource(opts.Seed))
	var zipf *rand.Zipf
	if opts.HotSpans > 1 {
		zipf = rand.NewZipf(rng, 1.25, 1, uint64(opts.HotSpans-1))
	}
	pickHot := func(forceDominant bool) int {
		if forceDominant || zipf == nil {
			return 0
		}
		return int(zipf.Uint64())
	}

	events := make([]KVReplayTraceEvent, 0, opts.Events)
	cold := 0
	for i := 0; i < opts.Events; i++ {
		phase := i % 5
		session := fmt.Sprintf("synthetic-%02d", i/5)
		if phase == 3 || phase == 4 {
			cold++
			events = append(events, KVReplayTraceEvent{
				Session: session,
				SpanID:  fmt.Sprintf("cold-%03d", cold),
				Tokens:  opts.SpanTokens,
			})
			continue
		}
		hot := pickHot(phase == 0 || phase == 2)
		events = append(events, KVReplayTraceEvent{
			Session: session,
			SpanID:  fmt.Sprintf("hot-%02d", hot),
			Tokens:  opts.SpanTokens,
		})
	}

	return KVReplayTrace{
		Schema:       KVReplayTraceSchema,
		Name:         opts.Name,
		Source:       fmt.Sprintf("synthetic-zipf-bimodal seed=%d", opts.Seed),
		BudgetTokens: opts.BudgetTokens,
		Events:       events,
	}
}

func (t KVReplayTrace) validate() error {
	if t.Schema != KVReplayTraceSchema {
		return fmt.Errorf("compute: replay trace schema %q, want %q", t.Schema, KVReplayTraceSchema)
	}
	if t.Name == "" {
		return errors.New("compute: replay trace name is required")
	}
	if t.BudgetTokens < 0 {
		return errors.New("compute: replay trace budget_tokens must be non-negative")
	}
	if len(t.Events) == 0 {
		return errors.New("compute: replay trace must contain at least one event")
	}
	for i, ev := range t.Events {
		if ev.SpanID == "" {
			return fmt.Errorf("compute: replay trace event %d missing span_id", i)
		}
		if ev.Tokens <= 0 {
			return fmt.Errorf("compute: replay trace event %d has non-positive tokens %d", i, ev.Tokens)
		}
	}
	return nil
}

func gatewayReplaySession(row gatewayusageledger.Row, idx int) string {
	if row.SessionID != "" {
		return row.SessionID
	}
	return fmt.Sprintf("pid%d-%d", row.PID, idx)
}

func maxCounter(values ...uint64) uint64 {
	var max uint64
	for _, v := range values {
		if v > max {
			max = v
		}
	}
	return max
}

func clampReplayTokens(v uint64, max int) int {
	if v == 0 {
		return 0
	}
	if max > 0 && v > uint64(max) {
		return max
	}
	maxInt := int(^uint(0) >> 1)
	if v > uint64(maxInt) {
		return maxInt
	}
	return int(v)
}

func emptyDefault(s, fallback string) string {
	if s == "" {
		return fallback
	}
	return s
}
