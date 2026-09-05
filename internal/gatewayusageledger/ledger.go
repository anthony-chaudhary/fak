package gatewayusageledger

import (
	"encoding/json"
	"fmt"
	"hash/fnv"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
)

const (
	// Schema versions the row shape so a future field addition can be detected by a
	// reader without guessing.
	Schema = "fak-gateway-usage-ledger/1"
	// DefaultLedgerRel is the live, gitignored runtime path. Keeping background
	// guard/serve writes under .fak prevents every session exit from dirtying the
	// shared tree. The tracked docs sibling is a historical publication snapshot.
	DefaultLedgerRel = ".fak/nightrun/gateway-usage.jsonl"
	// PublishedLedgerRel is the TRACKED publication snapshot of the same schema —
	// the committed corpus every checkout carries. It is the READ path for anything
	// whose verdict has to be reproducible by a second person: a reader pointed at
	// DefaultLedgerRel sees only whatever sessions happened to run on THIS box (a
	// gitignored file that is empty in CI and on a fresh clone), so a claim computed
	// from it can never be shown to anyone else, while a reader pointed here sees the
	// same bytes everywhere. Writers must still use DefaultLedgerRel — nothing should
	// append to the tracked snapshot from a session exit; it is refreshed as a
	// deliberate publication step. #5406 is the concrete cost of confusing the two: a
	// calibration drift guard resolved DefaultLedgerRel, so it silently skipped in CI
	// ("thin corpus: only 0 nonzero-length sessions") and red only on fleet boxes,
	// against a per-box population its constants were never sized on.
	PublishedLedgerRel = "docs/nightrun/gateway-usage.jsonl"
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

	// WHO ACTUALLY SERVED THE TURN — the self-hosted split of the volume already
	// totalled in InputTokens/OutputTokens above. fak could not previously answer
	// "what fraction of our tokens did we serve ourselves?" about its own serving
	// path: the side was decided per request (a model id that resolves in-kernel
	// decodes locally, everything else proxies upstream) and then thrown away one
	// line later, so the durable row recorded the volume and lost the attribution.
	//
	// The two groups are DISJOINT SUBSETS of the unsplit totals, accumulated in the
	// same observation, so SelfHostedOutputTokens+VendorOutputTokens <= OutputTokens
	// always — the remainder is the volume whose side could not be resolved, and it
	// is what makes the classified fraction a real coverage number rather than an
	// assumption. FoldSelfHostedShare is the reader.
	//
	// EVERY FIELD IS `omitempty`, WHICH IS THE WHOLE POINT. A row written before
	// this split existed omits all six, and an absent field must read NOT
	// INSTRUMENTED — never "zero self-hosted". Those are opposite claims about a
	// fleet: the first says nobody measured, the second says everyone paid a vendor.
	// A row that DID measure and served nothing locally still carries VendorTurns>0,
	// so the earned zero is distinguishable from the unmeasured one by presence, not
	// by value. Keeping them omitempty also leaves pre-split rows byte-identical and
	// their RowKey unchanged (the key hashes this struct's JSON).
	SelfHostedTurns        uint64 `json:"self_hosted_turns,omitempty"`
	SelfHostedInputTokens  uint64 `json:"self_hosted_input_tokens,omitempty"`
	SelfHostedOutputTokens uint64 `json:"self_hosted_output_tokens,omitempty"`
	VendorTurns            uint64 `json:"vendor_turns,omitempty"`
	VendorInputTokens      uint64 `json:"vendor_input_tokens,omitempty"`
	VendorOutputTokens     uint64 `json:"vendor_output_tokens,omitempty"`

	// Compaction — WITNESSED attempt counters + OBSERVED post-fire cache read.
	CompactionFired           uint64 `json:"compaction_fired"`
	CompactionBailed          uint64 `json:"compaction_bailed"`
	CompactionOff             uint64 `json:"compaction_off"`
	CompactionDroppedTurns    uint64 `json:"compaction_dropped_turns"`
	CompactionRestoredTurns   uint64 `json:"compaction_restored_turns,omitempty"`
	CompactionShedTokens      uint64 `json:"compaction_shed_tokens"`
	CompactionCacheReadTokens uint64 `json:"compaction_cache_read_tokens"`
	// CompactionInducedCreationTokens is the suffix-burst base (#2785): provider
	// cache_creation the fires INDUCED by shifting bytes after the drop point, which
	// invalidates the downstream recent-breakpoint suffix and forces a one-time cold
	// re-write. It is the debit side of the compaction net (CompactionEconomics). No
	// caller populates it yet — the per-fire attribution is #2785 — so it is 0 on
	// today's rows and `omitempty` keeps those rows byte-identical to the pre-field
	// schema (and their RowKey unchanged, since the key hashes this struct's JSON).
	// A 0 here is UNWITNESSED, not measured-zero; CompactionEconomics.NetIsUpperBound
	// is what carries that distinction to a reader.
	CompactionInducedCreationTokens uint64 `json:"compaction_induced_creation_tokens,omitempty"`
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

	// UpstreamErrorKinds is the per-KIND breakdown of upstream/planner turn failures
	// (#5487), keyed by the closed vocabulary the gateway's single classifier
	// (gateway.upstreamErrorKind) already returns: "stalled", "oom", "unreachable",
	// "rate_limited", "auth", "forbidden", "overloaded", "status_4xx", "status_5xx",
	// "transport", "other". It is OBSERVED at the upstream boundary and relayed — a
	// nonzero count here is not a fak fault.
	//
	// It exists because the kind was previously process-local ONLY: it fed the in-memory
	// /metrics counter and the `fak-turn … FAILED` stderr line, both of which die with
	// the process. Under `fak guard` the gateway is a per-invocation process, so once the
	// wrapped command exited a stall left NO trace anywhere — the one thing the
	// idle-deadline detector exists to tell you.
	//
	// Note the row did not merely COARSEN an upstream failure, it dropped it. Errored
	// above is a different population: it counts kernel ADJUDICATION error verdicts (see
	// gateway.AdjudicationSummary.Errored), not upstream turn failures, so it was never
	// even a lossy proxy for this. Nothing in a pre-#5487 row moves when a turn fails
	// upstream.
	//
	// Carrying the classifier's own string-keyed map (rather than one scalar per kind,
	// the way DenyAllStops is shaped) means a future kind needs no new field and no
	// schema bump.
	//
	// omitempty, and that is load-bearing for the same reason it is on the self-hosted
	// split above: a row written before this field existed omits it, and ABSENT must
	// read NOT INSTRUMENTED, never "zero upstream failures". A session that DID measure
	// and hit no upstream error also writes nothing here, so this field alone cannot
	// separate the two — that distinction is a follow-on, not something a reader should
	// assume today. Keeping it omitempty also leaves every pre-field row byte-identical
	// and its RowKey unchanged (the key hashes this struct's JSON).
	UpstreamErrorKinds map[string]uint64 `json:"upstream_error_kinds,omitempty"`

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

	// The two remaining managed-cache levers' INTENT (and the defer lever's EFFECT),
	// #4349. The configured-but-inert diff loop (cachevaluereport.FoldConfiguredButInert,
	// #3649) names three levers, but on a real fleet row it could only ever witness the
	// TTL-upgrade pair above: the other two levers' intent was not in this struct at all,
	// so a session that armed them and did nothing read exactly like a session that never
	// armed them.
	//
	// ManagedCacheActive is the resolved managed-cache lever state (--managed-cache /
	// gateway Config.CacheTTL1H), recorded independently of whether any head was
	// eligible. It has to be its own field because the alternative — INFERRING intent
	// from the WITNESSED KVPrefix* reuse above — is wrong in a specific, silent way: a
	// provider-prompt-cache-only session legitimately reuses zero KV-prefix tokens, so
	// that inference reads "lever off" and "lever on, paying off provider-side" as the
	// same flat zero.
	//
	// DeferColdToolsArmed / DeferColdCount are the same intent/effect pair for the
	// cold-tool-deferral lever (--defer-cold-tools, #3232): whether the lever was armed,
	// and how many cold tool definitions were actually deferred — mirrored from
	// gateway.AdjudicationSummary.DeferColdCount, i.e. the same count the in-process
	// fak_gateway_tool_defer_cold_total exports. That whole counter family lived only on
	// /metrics (#3233/#3536) and died with the process, so under `fak guard` (a
	// per-invocation gateway) an armed-and-inert defer session previously left no durable
	// trace anywhere.
	//
	// All three are omitempty, and as with the self-hosted split above that is
	// load-bearing: a row written before these fields existed omits them, ABSENT must
	// read NOT INSTRUMENTED rather than "the lever was off", and every pre-field row
	// stays byte-identical with its RowKey unchanged (the key hashes this struct's JSON).
	// The cost of that choice, stated plainly: a measured lever-OFF session serializes
	// identically to an unmeasured row, so these fields witness the ARMED case only —
	// separating measured-off from never-measured is a follow-on, not something a reader
	// may assume today.
	ManagedCacheActive  bool   `json:"managed_cache_active,omitempty"`
	DeferColdToolsArmed bool   `json:"defer_cold_tools_armed,omitempty"`
	DeferColdCount      uint64 `json:"defer_cold_count,omitempty"`

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
	// ExposeProfile is the resolved tool-surface profile that produced the row. Empty
	// means the full interactive surface; callers normalize that to "interactive" so
	// headless and interactive cohorts remain distinguishable even at equal budgets.
	ExposeProfile string `json:"expose_profile,omitempty"`
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
	Schema string `json:"schema"`
	// RowKey is a deterministic idempotency key over the row's IDENTITY + PAYLOAD
	// (schema, session_id, pid, unix_millis, and the counter snapshot) — see
	// computeRowKey. It deliberately excludes the write-time labels (Kind, Context,
	// uptime, generated_at) so a retried exit flush, or a periodic and an exit flush
	// of ONE snapshot landing in the same millisecond, collapse to one row at fold,
	// while two genuinely distinct snapshots (any counter differs) both survive.
	// Stamped by NewRow; empty on legacy pre-key rows (which fold as-is, never treated
	// as duplicates) and on synthetic carryforward rows. omitempty keeps a keyless row
	// byte-identical to the pre-key schema.
	RowKey      string      `json:"row_key,omitempty"`
	Kind        string      `json:"kind"`              // "exit" | "periodic" | KindCarryforward
	SessionType string      `json:"session_type"`      // "serve" | "guard"
	Context     string      `json:"context,omitempty"` // free-form label, e.g. transport (http/stdio)
	SessionID   string      `json:"session_id,omitempty"`
	PID         int         `json:"pid"`
	UnixMillis  int64       `json:"unix_millis"`
	UptimeSecs  float64     `json:"uptime_seconds,omitempty"`
	Provenance  *Provenance `json:"provenance,omitempty"`
	Counters    Counters    `json:"counters"`
	// CompactionEconomics is the compaction-economics TRAILER (#2792): the per-session
	// net-of-both WHY — fires, shed, observed cache_read, induced creation, and the
	// signed net they imply — so the economics survive process exit where operators
	// already look, instead of having to be re-joined from the per-fire ledger and the
	// provider rows by hand. It is a PURE projection of Counters (CompactionEconomicsOf),
	// stamped by NewRow, so it adds no observation the row did not already carry and a
	// distrustful reader can recompute it from the same line. Pointer + omitempty: a
	// session that never compacted carries no trailer and stays byte-identical to the
	// pre-trailer schema.
	CompactionEconomics *CompactionEconomics `json:"compaction_economics,omitempty"`
	GeneratedAt         string               `json:"generated_at"`
	// Carryforward is set only on Kind==KindCarryforward rows written by Cut —
	// the fold witness for the pre-cut rows this row's Counters sum. Pointer +
	// omitempty keeps every real session row byte-identical to the pre-cut schema.
	Carryforward *Carryforward `json:"carryforward,omitempty"`
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
	pid := os.Getpid()
	unixMillis := now.UnixMilli()
	return Row{
		Schema:      Schema,
		RowKey:      computeRowKey(Schema, sessionID, pid, unixMillis, c),
		Kind:        kind,
		SessionType: sessionType,
		Context:     context,
		SessionID:   sessionID,
		PID:         pid,
		UnixMillis:  unixMillis,
		UptimeSecs:  uptime.Seconds(),
		Provenance:  prov,
		Counters:    c,
		// Stamped for EVERY writer (guard teardown, serve stdio/http exit, the periodic
		// snapshot) because it is derived from c alone — there is no way for a caller to
		// forget it, and no second source that could disagree with the counters beside it.
		// Cut's synthetic carryforward rows deliberately do NOT get one: they are era-sums,
		// not sessions, and a per-session WHY summed across sessions is not a WHY.
		CompactionEconomics: CompactionEconomicsOf(c),
		GeneratedAt:         now.UTC().Format(time.RFC3339),
	}
}

// computeRowKey derives the deterministic idempotency key stamped into RowKey. It
// hashes the row's IDENTITY (schema, session_id, pid, unix_millis) together with its
// PAYLOAD — the counter snapshot, JSON-serialized (json.Marshal orders struct fields
// deterministically and sorts map keys, so the bytes are stable). Covering the payload
// is load-bearing: two genuinely distinct snapshots in the same millisecond differ in
// at least one counter, so they hash differently and both survive the fold; only a true
// re-emission of the same snapshot (identical counters) collapses. Write-time-only
// labels (Kind, Context, uptime, generated_at) are excluded so a periodic-vs-exit or a
// retried flush of ONE snapshot yields the SAME key. A payload that cannot be marshaled
// (never expected for Counters) yields "" — the row stays keyless and folds as legacy
// rather than carrying a bogus key that could collide with a real one.
func computeRowKey(schema, sessionID string, pid int, unixMillis int64, c Counters) string {
	payload, err := json.Marshal(c)
	if err != nil {
		return ""
	}
	h := fnv.New64a()
	// Length-prefix-free but NUL-delimited: the identity fields never contain a NUL,
	// and the payload is appended last, so no two distinct tuples share a serialization.
	_, _ = h.Write([]byte(schema))
	_, _ = h.Write([]byte{0})
	_, _ = h.Write([]byte(sessionID))
	_, _ = h.Write([]byte{0})
	_, _ = h.Write([]byte(strconv.Itoa(pid)))
	_, _ = h.Write([]byte{0})
	_, _ = h.Write([]byte(strconv.FormatInt(unixMillis, 10)))
	_, _ = h.Write([]byte{0})
	_, _ = h.Write(payload)
	return strconv.FormatUint(h.Sum64(), 16)
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
	return jsonlledger.AppendBounded(path, b, jsonlledger.DefaultActiveBytes)
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
	return ParseLedger(string(jsonlledger.ReadTail(path, jsonlledger.DefaultActiveBytes)))
}

// Trend is a simple fold of >=2 ledger rows into an oldest-vs-newest comparison, the
// minimal reader the #1610 acceptance criteria asks for ("a reader function can fold
// >=2 rows into a trend"). It orders rows by UnixMillis, and reports the first and
// last row plus the delta on the handful of headline counters an operator would look
// at first (tokens served, vDSO hit ratio, denies). A full `fak audit usage` CLI
// surface is a follow-on, not required here.
type Trend struct {
	Sessions int `json:"sessions"`
	// RowsDedupedAtFold counts the rows collapsed by RowKey before this fold — the
	// retried-append / same-millisecond double-flush rows the fold refused to
	// double-count. Zero on a legacy keyless corpus (keyless rows never dedupe). It
	// is the rows-deduped-at-fold signal the dedup census (#2503) reads, so a
	// double-counted row can never inflate a savings claim.
	RowsDedupedAtFold int `json:"rows_deduped_at_fold"`
	First             Row `json:"first"`
	Last              Row `json:"last"`

	DeltaInputTokens        int64 `json:"delta_input_tokens"`
	DeltaOutputTokens       int64 `json:"delta_output_tokens"`
	DeltaCachedPromptTokens int64 `json:"delta_cached_prompt_tokens"`
	DeltaSubmits            int64 `json:"delta_submits"`
	DeltaVDSOHits           int64 `json:"delta_vdso_hits"`
	DeltaDenies             int64 `json:"delta_denies"`
}

// DedupeByKey collapses rows that repeat an earlier row's non-empty RowKey,
// preserving input order and keeping the FIRST occurrence. Rows with an empty RowKey
// — legacy pre-key files and synthetic carryforwards — are NEVER treated as
// duplicates (two keyless rows cannot be proven to describe one snapshot), so they
// always pass through, which is what keeps a legacy keyless file folding unchanged.
// It returns the deduped rows and the count dropped — the rows-deduped-at-fold signal
// the dedup census (#2503) reads.
func DedupeByKey(rows []Row) (deduped []Row, dropped int) {
	seen := make(map[string]struct{}, len(rows))
	out := make([]Row, 0, len(rows))
	for _, r := range rows {
		if r.RowKey != "" {
			if _, ok := seen[r.RowKey]; ok {
				dropped++
				continue
			}
			seen[r.RowKey] = struct{}{}
		}
		out = append(out, r)
	}
	return out, dropped
}

// FoldTrend folds rows (already read, e.g. via ReadLedgerFile) into a Trend. It is
// pure and deterministic. ok is false when fewer than 2 rows are given — a single
// row (or none) has nothing to trend against, matching the acceptance criteria's
// ">=2 rows" framing.
//
// Rows are first collapsed by RowKey (DedupeByKey): a retried or same-millisecond
// double-flush of ONE snapshot folds to a single row so the trend never
// double-counts it, while legacy keyless rows pass through unchanged. The number
// collapsed is reported as Trend.RowsDedupedAtFold.
func FoldTrend(rows []Row) (Trend, bool) {
	deduped, dropped := DedupeByKey(rows)
	// Carryforward rows are synthetic sums, not session snapshots — trending
	// oldest-vs-newest against one would compare a whole folded era's total to a
	// single session. Skip them; the trend stays over real rows only.
	real := make([]Row, 0, len(deduped))
	for _, r := range deduped {
		if r.Kind == KindCarryforward {
			continue
		}
		real = append(real, r)
	}
	if len(real) < 2 {
		return Trend{}, false
	}
	sorted := real
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].UnixMillis < sorted[j].UnixMillis })
	first, last := sorted[0], sorted[len(sorted)-1]
	t := Trend{
		Sessions:                len(sorted),
		RowsDedupedAtFold:       dropped,
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

// DashboardAdoption is the privacy-safe seven-day fold of dashboard event rows.
type DashboardAdoption struct {
	SinceUnixMillis int64             `json:"since_unix_millis"`
	Counts          map[string]uint64 `json:"counts"`
}

// DashboardEventRow creates one bounded dashboard adoption event. event must be
// one of lightweight_open, rich_ready, or rich_unavailable. No request URL,
// dashboard UID, client identity, hostname, or filesystem path is retained.
func DashboardEventRow(event string, now time.Time) (Row, error) {
	switch event {
	case "lightweight_open", "rich_ready", "rich_unavailable":
	default:
		return Row{}, fmt.Errorf("unsupported dashboard event %q", event)
	}
	return NewRow("dashboard_"+event, "serve", "dashboard", "", 0, nil, Counters{}, now), nil
}

// FoldDashboardAdoption counts bounded dashboard events in the inclusive window.
func FoldDashboardAdoption(rows []Row, since time.Time) DashboardAdoption {
	out := DashboardAdoption{SinceUnixMillis: since.UnixMilli(), Counts: map[string]uint64{}}
	for _, row := range rows {
		if row.UnixMillis < out.SinceUnixMillis || !strings.HasPrefix(row.Kind, "dashboard_") {
			continue
		}
		event := strings.TrimPrefix(row.Kind, "dashboard_")
		switch event {
		case "lightweight_open", "rich_ready", "rich_unavailable":
			out.Counts[event]++
		}
	}
	return out
}
