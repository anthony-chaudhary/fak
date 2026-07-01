package gatewayusageledger

import (
	"bufio"
	"encoding/json"
	"os"
	"sort"
	"strings"
	"time"
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

	// Tool-definition prune (WITNESSED).
	ToolPruneTurns uint64 `json:"tool_prune_turns"`
	ToolPruneCount uint64 `json:"tool_prune_count"`

	// Deny-all stops (WITNESSED) — turns where every proposed tool call was refused.
	DenyAllStops uint64 `json:"deny_all_stops"`

	// ByReason is the deny/quarantine reason breakdown (gateway.AdjudicationSummary.ByReason).
	ByReason map[string]uint64 `json:"by_reason,omitempty"`
}

// Row is one end-of-session (or periodic, with --metrics-snapshot) counter snapshot.
// Schema/SessionID/PID/UnixMillis identify WHEN and WHICH process/session produced
// it; Kind distinguishes an "exit" row (the final snapshot at session close) from a
// "periodic" row (an interim snapshot from a still-running `fak serve`), so a reader
// folding rows into a trend can choose to fold only exit rows, or watch periodic
// rows for a crash-before-exit trail.
type Row struct {
	Schema      string   `json:"schema"`
	Kind        string   `json:"kind"`              // "exit" | "periodic"
	SessionType string   `json:"session_type"`      // "serve" | "guard"
	Context     string   `json:"context,omitempty"` // free-form label, e.g. transport (http/stdio)
	SessionID   string   `json:"session_id,omitempty"`
	PID         int      `json:"pid"`
	UnixMillis  int64    `json:"unix_millis"`
	UptimeSecs  float64  `json:"uptime_seconds,omitempty"`
	Counters    Counters `json:"counters"`
	GeneratedAt string   `json:"generated_at"`
}

// NewRow builds a Row from a live counter snapshot at now. kind should be "exit" or
// "periodic"; sessionType is the caller's session kind ("serve"/"guard"); context is
// an optional free-form label (e.g. "http"/"stdio"); sessionID identifies the served
// session/trace when the caller has one (empty is fine — PID + UnixMillis are always
// enough to distinguish rows across restarts, matching the acceptance criteria).
func NewRow(kind, sessionType, context, sessionID string, uptime time.Duration, c Counters, now time.Time) Row {
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
	var rows []Row
	sc := bufio.NewScanner(strings.NewReader(content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var row Row
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			continue
		}
		if row.Schema == "" {
			continue
		}
		rows = append(rows, row)
	}
	return rows
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
