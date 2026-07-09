package gatewayusageledger

import (
	"encoding/json"
	"os"
	"sort"
	"time"

	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
)

const (
	// Schema versions the row shape so a future field addition can be detected by a
	// reader without guessing.
	Schema = "fak-gateway-usage-ledger/1"
	// DefaultLedgerRel is the sibling path to cachevalueledger.DefaultLedgerRel
	// (docs/nightrun/cache-value.jsonl) — same directory, same append-only JSONL
	// convention, distinct file because this ledger carries the FULL served-turn
	// counter family rather than only the cache-value axis.
	DefaultLedgerRel = "docs/nightrun/gateway-usage.jsonl"
)

// Counters is the OBSERVED served-turn counter family this ledger snapshots. Every
// field here is a plain count or token/timing total — never a prompt, tool-arg, or
// secret byte — matching the #1610 honesty fence. Callers fill this from the live
// gateway's exported accessors (Server.KernelCounters + Server.AdjudicationSummary);
// this package intentionally does not import internal/gateway or internal/kernel so
// it stays a leaf the gateway (and any future caller, e.g. cmd/fak/guard.go) can
// depend on without risking an import cycle.
type Counters struct {
	// Kernel submission counters (kernel.Counters mirror): the adjudication-boundary
	// view of every fak_syscall this session processed.
	Submits      int64 `json:"submits"`
	VDSOHits     int64 `json:"vdso_hits"`
	EngineCalls  int64 `json:"engine_calls"`
	Denies       int64 `json:"denies"`
	Transforms   int64 `json:"transforms"`
	Quarantines  int64 `json:"quarantines"`
	ResultDenies int64 `json:"result_denies"`
	Admitted     int64 `json:"admitted"`

	// Adjudication roll-up (gateway.AdjudicationSummary mirror, the subset that is a
	// pure count/token total rather than a per-reason map — the map is carried
	// separately in ByReason so a reader does not need the gateway package's type).
	Total       uint64 `json:"total"`
	Allowed     uint64 `json:"allowed"`
	Denied      uint64 `json:"denied"`
	Transformed uint64 `json:"transformed"`
	Quarantined uint64 `json:"quarantined"`
	Deferred    uint64 `json:"deferred"`
	Escalated   uint64 `json:"escalated"`
	Errored     uint64 `json:"errored"`

	// ObservedTurns is the count of served passthrough turns this session folded through
	// the harness coordinator (gateway.HarnessCoherenceSummary.ObservedTurns) — the REAL
	// per-session turn count. It exists because Submits is 0 on the guard proxy path (the
	// kernel submit boundary the *_syscall counters watch is not on the pass-through wire),
	// so the DefaultAssumedSessionTurns calibration in internal/gateway had to lean on
	// CachedTurns (turns that got a provider cache hit) as a session-length PROXY. Recording
	// the observed served-turn count directly makes the turn-distribution claim recomputable
	// from a dedicated field instead of a proxy: a corpus reader can now percentile
	// ObservedTurns straight, and a row with ObservedTurns>0 but CachedTurns==0 (a cold
	// session that never got a cache read) is no longer invisible to the length distribution.
	ObservedTurns uint64 `json:"observed_turns"`

	// Token / cache economy — OBSERVED (provider-relayed) except KVPrefix* which is
	// WITNESSED (fak-authored in-kernel reuse).
	InputTokens          uint64 `json:"input_tokens"`
	OutputTokens         uint64 `json:"output_tokens"`
	CachedPromptTokens   uint64 `json:"cached_prompt_tokens"`
	CachedTurns          uint64 `json:"cached_turns"`
	CacheCreationTokens  uint64 `json:"cache_creation_tokens"`
	KVPrefixPromptTokens uint64 `json:"kv_prefix_prompt_tokens"`
	KVPrefixReusedTokens uint64 `json:"kv_prefix_reused_tokens"`

	// Compaction — WITNESSED attempt counters + OBSERVED post-fire cache read.
	CompactionFired           uint64 `json:"compaction_fired"`
	CompactionBailed          uint64 `json:"compaction_bailed"`
	CompactionOff             uint64 `json:"compaction_off"`
	CompactionDroppedTurns    uint64 `json:"compaction_dropped_turns"`
	CompactionShedTokens      uint64 `json:"compaction_shed_tokens"`
	CompactionCacheReadTokens uint64 `json:"compaction_cache_read_tokens"`
	// Per-reason breakdown of CompactionBailed (the closed agent.CompactReason*
	// vocabulary) plus the anchor-starved subset of under_budget — the per-session
	// WHY behind a zero fak shed slice (#1407/#1408). Durable here so a fleet reader
	// can still tell burst_unprofitable (warm session, no horizon) from a genuinely
	// small session after the process is gone; before this, the breakdown lived only
	// on the in-process /metrics and the console exit summary.
	CompactionBailReasons   map[string]uint64 `json:"compaction_bail_reasons,omitempty"`
	CompactionAnchorStarved uint64            `json:"compaction_anchor_starved,omitempty"`

	// Tool-definition prune (WITNESSED).
	ToolPruneTurns uint64 `json:"tool_prune_turns"`
	ToolPruneCount uint64 `json:"tool_prune_count"`

	// Deny-all stops (WITNESSED) — turns where every proposed tool call was refused.
	DenyAllStops uint64 `json:"deny_all_stops"`

	// Managed-cache 1h TTL upgrade (WITNESSED, epic #1844 C6): outcomes of fak's own
	// stable-prefix cache_control TTL splice on the outbound Anthropic wire, mirrored
	// from the in-process fak_gateway_cache_ttl_upgrade_total family so a managed-cache
	// session leaves DURABLE evidence instead of a witness that dies with the process.
	// Upgraded counts actual 1h-tier upgrades; the reasons map is the closed
	// agent.TTLUpgradeReason* refusal vocabulary. Both are zero/absent while the lever
	// (--managed-cache / CacheTTL1H) is off; a zero upgraded count WITH reason rows is
	// an ON-but-ineligible session (every head refused) — signal, not a bug.
	CacheTTLUpgradesUpgraded uint64            `json:"cache_ttl_upgrades_upgraded"`
	CacheTTLUpgradeReasons   map[string]uint64 `json:"cache_ttl_upgrade_reasons,omitempty"`

	// ByReason is the deny/quarantine reason breakdown (gateway.AdjudicationSummary.ByReason).
	ByReason map[string]uint64 `json:"by_reason,omitempty"`
}

// Provenance stamps the calibration-relevant config knobs (and build revision) that were
// LIVE when a row was written. It exists so the corpus is self-describing: the constants in
// internal/gateway (DefaultAssumedSessionTurns, the compaction budget) are calibrated FROM
// this ledger's distribution, so a reader recomputing that calibration must be able to tell
// which rows were produced under the standard configuration and which under an override
// (e.g. --assume-session-turns 0, which disables the session-length prior entirely and would
// skew a naive percentile). Every field is a plain scalar the CALLER fills from the live
// Server's accessors + binstamp — this package stays a leaf that imports neither. The Row
// holds it by pointer, so a caller that supplies no provenance (nil — older rows, or a test)
// omits the whole object and stays byte-compatible with the pre-provenance schema, while a
// caller that DOES supply it can still carry a meaningful zero (AssumeSessionTurns:0 = prior
// disabled) distinct from "absent" — the reason the field is a pointer rather than a value.
type Provenance struct {
	// AssumeSessionTurns is the resolved session-length prior the head-anchored burst gate
	// used (gateway.Config.AssumeSessionTurns; DefaultAssumedSessionTurns=50 by default,
	// 0 = prior disabled). A recalibrating reader keys the turn-distribution fit on rows
	// whose value matches the default, so an override session cannot silently move the p90.
	AssumeSessionTurns int `json:"assume_session_turns,omitempty"`
	// CompactHistoryBudget is the resident-token budget compaction fired against
	// (gateway.Config.CompactHistoryBudget; 0 = compaction OFF). The volume-aware horizon's
	// heavy/thin split (headHorizonHeavyResidentFloor) is only meaningful when compaction
	// was armed, so this lets a reader exclude compaction-off rows from that analysis.
	CompactHistoryBudget int `json:"compact_history_budget,omitempty"`
	// BuildRevision is the VCS revision of the fak binary that produced the row (binstamp),
	// suffixed "-dirty" for an uncommitted build. It ties a distribution shift to the code
	// that produced it, so a recalibration can scope to rows from a known-good build.
	BuildRevision string `json:"build_revision,omitempty"`
}

// Row is one end-of-session (or periodic, with --metrics-snapshot) counter snapshot.
// Schema/SessionID/PID/UnixMillis identify WHEN and WHICH process/session produced
// it; Kind distinguishes an "exit" row (the final snapshot at session close) from a
// "periodic" row (an interim snapshot from a still-running `fak serve`), so a reader
// folding rows into a trend can choose to fold only exit rows, or watch periodic
// rows for a crash-before-exit trail.
type Row struct {
	Schema      string      `json:"schema"`
	Kind        string      `json:"kind"`              // "exit" | "periodic"
	SessionType string      `json:"session_type"`      // "serve" | "guard"
	Context     string      `json:"context,omitempty"` // free-form label, e.g. transport (http/stdio)
	SessionID   string      `json:"session_id,omitempty"`
	PID         int         `json:"pid"`
	UnixMillis  int64       `json:"unix_millis"`
	UptimeSecs  float64     `json:"uptime_seconds,omitempty"`
	Provenance  *Provenance `json:"provenance,omitempty"`
	Counters    Counters    `json:"counters"`
	GeneratedAt string      `json:"generated_at"`
}

// NewRow builds a Row from a live counter snapshot at now. kind should be "exit" or
// "periodic"; sessionType is the caller's session kind ("serve"/"guard"); context is
// an optional free-form label (e.g. "http"/"stdio"); sessionID identifies the served
// session/trace when the caller has one (empty is fine — PID + UnixMillis are always
// enough to distinguish rows across restarts, matching the acceptance criteria). prov
// stamps the calibration-relevant config that was live (nil = none supplied; see
// Provenance) — the production caller passes it from the Server's accessors, a test may
// pass nil to stay on the pre-provenance byte shape.
func NewRow(kind, sessionType, context, sessionID string, uptime time.Duration, prov *Provenance, c Counters, now time.Time) Row {
	if c.ByReason == nil {
		c.ByReason = map[string]uint64{}
	}
	return Row{
		Schema:      Schema,
		Kind:        kind,
		SessionType: sessionType,
		Context:     context,
		SessionID:   sessionID,
		PID:         os.Getpid(),
		UnixMillis:  now.UnixMilli(),
		UptimeSecs:  uptime.Seconds(),
		Provenance:  prov,
		Counters:    c,
		GeneratedAt: now.UTC().Format(time.RFC3339),
	}
}

// Append serializes row and appends it (plus a trailing newline) to path, creating
// the file and any nothing-else parent behavior identical to os.OpenFile's own
// semantics if it does not exist. Opened O_APPEND on every call (no held handle)
// so concurrent writers from independent processes each get an atomically appended
// line on POSIX; on Windows (this repo's dev box) small single-write appends are
// likewise not interleaved in practice, matching the existing cachevalueledger
// writer this package mirrors.
func Append(path string, row Row) error {
	b, err := json.Marshal(row)
	if err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return err
	}
	defer f.Close()
	if _, err := f.Write(append(b, '\n')); err != nil {
		return err
	}
	return nil
}

// ParseLedger parses the JSONL ledger content into rows, skipping blank lines and any
// line that fails to decode or is missing its Schema (so a foreign/corrupt line never
// aborts the whole read).
func ParseLedger(content string) []Row {
	return jsonlledger.Parse(content, func(r Row) bool { return r.Schema != "" })
}

// ReadLedgerFile reads and parses the ledger at path. A missing/unreadable file
// returns nil (no rows yet), matching cachevalueledger.ReadLedgerFile's fall-open
// posture — an absent ledger is a clean first-run state, not an error.
func ReadLedgerFile(path string) []Row {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	return ParseLedger(string(b))
}

// Trend is a simple fold of >=2 ledger rows into an oldest-vs-newest comparison, the
// minimal reader the #1610 acceptance criteria asks for ("a reader function can fold
// >=2 rows into a trend"). It orders rows by UnixMillis, and reports the first and
// last row plus the delta on the handful of headline counters an operator would look
// at first (tokens served, vDSO hit ratio, denies). A full `fak audit usage` CLI
// surface is a follow-on, not required here.
type Trend struct {
	Sessions int `json:"sessions"`
	First    Row `json:"first"`
	Last     Row `json:"last"`

	DeltaInputTokens        int64 `json:"delta_input_tokens"`
	DeltaOutputTokens       int64 `json:"delta_output_tokens"`
	DeltaCachedPromptTokens int64 `json:"delta_cached_prompt_tokens"`
	DeltaSubmits            int64 `json:"delta_submits"`
	DeltaVDSOHits           int64 `json:"delta_vdso_hits"`
	DeltaDenies             int64 `json:"delta_denies"`
}

// FoldTrend folds rows (already read, e.g. via ReadLedgerFile) into a Trend. It is
// pure and deterministic. ok is false when fewer than 2 rows are given — a single
// row (or none) has nothing to trend against, matching the acceptance criteria's
// ">=2 rows" framing.
func FoldTrend(rows []Row) (Trend, bool) {
	if len(rows) < 2 {
		return Trend{}, false
	}
	sorted := make([]Row, len(rows))
	copy(sorted, rows)
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].UnixMillis < sorted[j].UnixMillis })
	first, last := sorted[0], sorted[len(sorted)-1]
	t := Trend{
		Sessions:                len(sorted),
		First:                   first,
		Last:                    last,
		DeltaInputTokens:        int64(last.Counters.InputTokens) - int64(first.Counters.InputTokens),
		DeltaOutputTokens:       int64(last.Counters.OutputTokens) - int64(first.Counters.OutputTokens),
		DeltaCachedPromptTokens: int64(last.Counters.CachedPromptTokens) - int64(first.Counters.CachedPromptTokens),
		DeltaSubmits:            last.Counters.Submits - first.Counters.Submits,
		DeltaVDSOHits:           last.Counters.VDSOHits - first.Counters.VDSOHits,
		DeltaDenies:             last.Counters.Denies - first.Counters.Denies,
	}
	return t, true
}

// ScoreTrend reads the ledger file at path and folds it into a Trend — the file-based
// convenience wrapper around FoldTrend, mirroring cachevalueledger.ScoreTrendGate's
// read+fold split.
func ScoreTrend(path string) (Trend, bool) {
	return FoldTrend(ReadLedgerFile(path))
}
