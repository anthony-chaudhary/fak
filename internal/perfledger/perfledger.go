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

	// MinPrefillRateTokens is the smallest uncached prompt that yields a prefill
	// rate. Below it TTFT is fixed overhead, not prefill: a cache hit re-feeding 4
	// tokens in 150ms would read as a "27 tok/s" prefill and drag the median.
	MinPrefillRateTokens = 128
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
// PrefillTPS needs at least MinPrefillRateTokens uncached tokens. DecodeTPS is the
// inter-token rate after the first token: (completion-1) / (e2e-ttft).
type Record struct {
	RequestID string `json:"request_id,omitempty"`
	TraceID   string `json:"trace_id,omitempty"`
	// InboundRequestID is the caller's own id for this attempt (X-Request-ID, else
	// the traceparent trace id): the key a router journal row and usage receipt
	// carry, so a slow turn joins to them. RequestID stays fak-minted and unique.
	InboundRequestID     string         `json:"inbound_request_id,omitempty"`
	WorkloadSHA256       string         `json:"workload_sha256,omitempty"`
	WorkloadIdentityMode string         `json:"workload_identity_mode,omitempty"`
	ConfigSHA256         string         `json:"config_sha256,omitempty"`
	ConfigIdentityMode   string         `json:"config_identity_mode,omitempty"`
	LoadLifecycle        *LoadLifecycle `json:"load_lifecycle,omitempty"`
	Schema               string         `json:"schema"`
	UnixMS               int64          `json:"unix_ms"`
	FinishReason         string         `json:"finish_reason,omitempty"`
	Locality             string         `json:"locality,omitempty"`
	PromptTokens         int            `json:"prompt_tokens"`
	CompletionTokens     int            `json:"completion_tokens"`
	CachedTokens         int            `json:"cached_tokens"`
	CacheRegime          string         `json:"cache_regime,omitempty"`
	E2EMS                float64        `json:"e2e_ms"`
	TTFTMS               float64        `json:"ttft_ms,omitempty"`
	PrefillTPS           float64        `json:"prefill_tps,omitempty"`
	DecodeTPS            float64        `json:"decode_tps,omitempty"`
	NativeTiming         *NativeTiming  `json:"native_timing,omitempty"`
	// PrefillRoute preserves observed and unobserved executed native prefill
	// calls. Absent means no aggregate with executed calls was supplied.
	PrefillRoute *PrefillRoute `json:"prefill_route,omitempty"`

	PartialNativePhases *PartialNativePhases `json:"partial_native_phases,omitempty"`
	// Error is the failure class of a turn that did not complete (one of the Error*
	// constants); empty on a served turn. Status is the HTTP status the client got:
	// 200 when the failure came after the stream was committed, 499 for a client cancel,
	// absent (0) when the client got no status at all (client_write_timeout).
	Error  string `json:"error,omitempty"`
	Status int    `json:"status,omitempty"`
	// Model is the model that served the turn, when the planner reported one.
	Model string `json:"model,omitempty"`
	// Engine is the native engine's decode anatomy for this one request; absent
	// on proxied and mock turns, which have no engine steps of their own.
	Engine *Engine `json:"engine,omitempty"`
	// UpstreamSpec* are a proxied upstream's own speculative-decoding counts
	// (llama.cpp timings draft_n / draft_n_accepted); absent when it drafted nothing.
	UpstreamSpecDraftTokens    int `json:"upstream_spec_draft_tokens,omitempty"`
	UpstreamSpecAcceptedTokens int `json:"upstream_spec_accepted_tokens,omitempty"`
	// Client is the calling harness class (see Clients), never a raw User-Agent;
	// Synthetic marks a readiness canary or health probe the sender declared.
	// Both are absent on rows written before they existed.
	Client    string `json:"client,omitempty"`
	Synthetic bool   `json:"synthetic,omitempty"`
	// UsageUnreported marks a content-bearing turn whose upstream reported no token
	// counts at all: PromptTokens/CompletionTokens/CachedTokens are "unobserved", not
	// zero-cost. Absent on every row where the counts were reported (#14110).
	UsageUnreported bool `json:"usage_unreported,omitempty"`
	// GPUBusyPercent is the device busy share when the row was committed; absent
	// (never 0) when no busy counter was readable.
	GPUBusyPercent *int `json:"gpu_busy_percent,omitempty"`
	Identity

	BenchEvidence *BenchEvidence `json:"bench_evidence,omitempty"`
}

// LoadLifecycle binds a request admission to one installed model profile.
// InitialAfterLoad names the first admission, including probes and failures;
// it does not assert cache warmth or include load cost in request latency.
type LoadLifecycle struct {
	LoadID           string  `json:"load_id"`
	RecordedUnixMS   int64   `json:"recorded_unix_ms"`
	LoadMS           float64 `json:"load_ms,omitempty"`
	RequestOrdinal   uint64  `json:"request_ordinal"`
	InitialAfterLoad bool    `json:"initial_after_load"`
}

// Identity names the server that wrote a row, so a window spanning a planner
// swap, a backend change, a host move or a fak upgrade is visible instead of
// folded into one quantile. Empty fields are omitted; rows written before
// Identity existed decode with every field empty.
type Identity struct {
	Planner string `json:"planner,omitempty"`
	Backend string `json:"backend,omitempty"`
	Host    string `json:"host,omitempty"`
	Version string `json:"fak_version,omitempty"`
}

// ServedBy is the identity of one row including its model; it is what the
// summary counts as distinct.
type ServedBy struct {
	Model string `json:"model,omitempty"`
	Identity
}

func (r Record) servedBy() ServedBy { return ServedBy{Model: r.Model, Identity: r.Identity} }

// Native decode paths (enginestep's closed path vocabulary).
const (
	PathSerial      = "serial"
	PathBatched     = "batched"
	PathSpeculative = "speculative"
)

// Paths is the closed decode-path vocabulary in render order.
var Paths = []string{PathSerial, PathBatched, PathSpeculative}

// Engine is one request's native decode anatomy: which decode path ran, the
// coalesced cohort it rode, and its own speculative draft-verify rounds. It joins
// the process-wide fak_engine_* families and /v1/fak/observation/engine to a
// single served turn.
type Engine struct {
	Path               string                 `json:"path"`
	CohortSize         int                    `json:"cohort_size,omitempty"`
	SpecRounds         int                    `json:"spec_rounds,omitempty"`
	SpecDraftTokens    int                    `json:"spec_draft_tokens,omitempty"`
	SpecAcceptedTokens int                    `json:"spec_accepted_tokens,omitempty"`
	SpeculativeCost    *SpeculativeDecodeCost `json:"speculative_cost,omitempty"`
}

// SpeculativeAttemptCost retains one outcome's request-owned wall times.
// InclusiveElapsed contains DraftElapsed; these durations must not be summed.
type SpeculativeAttemptCost struct {
	Attempts           int           `json:"attempts"`
	EmittedTokens      int           `json:"emitted_tokens"`
	MeasuredAttempts   int           `json:"measured_attempts"`
	UnmeasuredAttempts int           `json:"unmeasured_attempts"`
	DraftElapsed       time.Duration `json:"draft_elapsed_ns"`
	InclusiveElapsed   time.Duration `json:"inclusive_elapsed_ns"`
}

// SpeculativeDecodeCost is absent when no proposal attempt was observed.
// It excludes prefill and the unassisted serial remainder.
type SpeculativeDecodeCost struct {
	Scope             string                 `json:"scope"`
	Attempts          int                    `json:"attempts"`
	Completed         SpeculativeAttemptCost `json:"completed"`
	ProposalError     SpeculativeAttemptCost `json:"proposal_error"`
	EmptyProposal     SpeculativeAttemptCost `json:"empty_proposal"`
	VerificationError SpeculativeAttemptCost `json:"verification_error"`
	Cancelled         SpeculativeAttemptCost `json:"cancelled"`
	Incomplete        SpeculativeAttemptCost `json:"incomplete"`
}

// PartialNativePhases preserves request-owned phase observations even when
// generation fails. Accumulated request-owned elapsed measurements may include
// CPU work, waits and incomplete phases; they are not GPU compute time, complete
// phase times or throughput measurements. A nil duration is unobserved;
// a nonnil zero is an observed zero-duration phase.
type PartialNativePhases struct {
	Scope               string   `json:"scope"`
	PrefillMS           *float64 `json:"prefill_ms,omitempty"`
	DecodeMS            *float64 `json:"decode_ms,omitempty"`
	PrefillObservations uint64   `json:"prefill_observations,omitempty"`
	DecodeObservations  uint64   `json:"decode_observations,omitempty"`
}

// PrefillRoute preserves the request aggregate's call counts, completeness and
// route qualification. Paths, decline reason and fallback flag describe only
// its representative observed status, not every call in a mixed request.
// An executed but wholly unobserved aggregate has no representative strings.
// Qualification summarizes route evidence, not numerical or hardware parity.
// Historical rows without counts decode with zero calls: their completeness is
// unmeasured, and a historical qualification flag is not upgraded by this schema.
type PrefillRoute struct {
	PrefillCalls                int    `json:"prefill_calls"`
	ObservedCalls               int    `json:"observed_calls"`
	Complete                    bool   `json:"complete"`
	RequestedPath               string `json:"requested_path,omitempty"`
	EffectivePath               string `json:"effective_path,omitempty"`
	DeclineReason               string `json:"decline_reason,omitempty"`
	FallbackActive              bool   `json:"fallback_active,omitempty"`
	NativePerformanceQualifying bool   `json:"native_performance_qualifying"`
}

// NativeTiming is the engine-measured phase split for a native request. It is
// independent of client TTFT and request wall time, which can include agent-loop
// tool turns. A zero phase is unknown and omitted.
type NativeTiming struct {
	PrefillMS     float64 `json:"prefill_ms,omitempty"`
	DecodeMS      float64 `json:"decode_ms,omitempty"`
	PrefillTokens int     `json:"prefill_tokens,omitempty"`
	DecodeTokens  int     `json:"decode_tokens,omitempty"`
	PrefillTPS    float64 `json:"prefill_tps,omitempty"`
	DecodeTPS     float64 `json:"decode_tps,omitempty"`
	// CPUMSPerToken is process CPU milliseconds per decode step and HostShare is
	// process CPU over decode wall in cores (can exceed 1), both sampled around
	// the decode loop. Nil is unmeasured; a nonnil zero is a measured zero.
	CPUMSPerToken *float64 `json:"cpu_ms_per_token,omitempty"`
	HostShare     *float64 `json:"host_share,omitempty"`
	// Kernels is the decode window's per-kernel GPU time; absent unless the
	// backend's timestamp profile was on and every decode window measured it.
	Kernels *NativeKernels `json:"kernels,omitempty"`
	// GPUIdleMSPerToken is a lower bound on device-idle host wall per decode
	// step: fence wake and submit latency count as busy. It is absent when any
	// decode window ran a pipelined batch, whose completion the host does not
	// observe until it drains. SyncsPerToken is host waits on device completion
	// per step, and SyncSites splits those waits by waiting entry point and the
	// kernel waited on. Each is absent unless the backend observed every submit
	// and wait in every decode window.
	GPUIdleMSPerToken *float64         `json:"gpu_idle_ms_per_token,omitempty"`
	SyncsPerToken     *float64         `json:"syncs_per_token,omitempty"`
	SyncSites         []NativeSyncSite `json:"sync_sites,omitempty"`
}

// NativeSyncSite is one host-sync (site, stage) cell per decode step.
type NativeSyncSite struct {
	Site             string  `json:"site"`
	Stage            string  `json:"stage"`
	SyncsPerToken    float64 `json:"syncs_per_token"`
	WaitMSPerToken   float64 `json:"wait_ms_per_token"`
	D2HBytesPerToken float64 `json:"d2h_bytes_per_token,omitempty"`
}

// WithDeviceSync returns a copy carrying the decode-window idle and sync rates.
// Values that are negative, NaN or infinite stay absent, and a site table is
// kept only beside a measured syncs/token.
func (n *NativeTiming) WithDeviceSync(idleMSPerToken float64, idleOK bool, syncsPerToken float64, syncsOK bool, sites []NativeSyncSite) *NativeTiming {
	if n == nil {
		return nil
	}
	out := *n
	if v, ok := finiteNonnegativeHostWork(idleMSPerToken, idleOK); ok {
		out.GPUIdleMSPerToken = &v
	}
	if v, ok := finiteNonnegativeHostWork(syncsPerToken, syncsOK); ok {
		out.SyncsPerToken = &v
		out.SyncSites = make([]NativeSyncSite, 0, len(sites))
		for _, s := range sites {
			syncs, okS := finiteNonnegativeHostWork(s.SyncsPerToken, true)
			wait, okW := finiteNonnegativeHostWork(s.WaitMSPerToken, true)
			d2h, okD := finiteNonnegativeHostWork(s.D2HBytesPerToken, true)
			if !okS || !okW || !okD {
				out.SyncsPerToken, out.SyncSites = nil, nil
				break
			}
			out.SyncSites = append(out.SyncSites, NativeSyncSite{Site: s.Site, Stage: s.Stage, SyncsPerToken: syncs, WaitMSPerToken: wait, D2HBytesPerToken: d2h})
		}
	}
	return &out
}

// WithHostWork returns a copy carrying the decode-window CPU split. Values that
// are negative, NaN or infinite stay absent rather than becoming measurements.
func (n *NativeTiming) WithHostWork(cpuMSPerToken float64, cpuOK bool, hostShare float64, shareOK bool) *NativeTiming {
	if n == nil {
		return nil
	}
	out := *n
	if v, ok := finiteNonnegativeHostWork(cpuMSPerToken, cpuOK); ok {
		out.CPUMSPerToken = &v
	}
	if v, ok := finiteNonnegativeHostWork(hostShare, shareOK); ok {
		out.HostShare = &v
	}
	return &out
}

func finiteNonnegativeHostWork(v float64, ok bool) (float64, bool) {
	if !ok || v < 0 || math.IsNaN(v) || math.IsInf(v, 0) {
		return 0, false
	}
	return roundTo(v, 1e6), true
}

// NewNativeTiming keeps only finite, positive engine-authored phases. Missing
// phases remain unknown rather than becoming zero-duration measurements. Measured
// phases retain raw counts even when too short to report a rate.
func NewNativeTiming(prefillTokens, decodeTokens int, prefillMS, decodeMS float64) *NativeTiming {
	n := &NativeTiming{}
	prefillRateTokens := prefillTokens
	if prefillRateTokens < MinPrefillRateTokens {
		prefillRateTokens = 0
	}
	if ms, rate, ok := measuredNativePhase(prefillTokens, prefillRateTokens, prefillMS); ok {
		n.PrefillMS = ms
		n.PrefillTokens = prefillTokens
		n.PrefillTPS = rate
	}
	if ms, rate, ok := measuredNativePhase(decodeTokens, max(decodeTokens, 1)-1, decodeMS); ok {
		n.DecodeMS = ms
		n.DecodeTokens = decodeTokens
		n.DecodeTPS = rate
	}
	if n.PrefillMS == 0 && n.DecodeMS == 0 {
		return nil
	}
	return n
}

// rateTokens is separate from the raw count: prefill needs the token floor,
// and decode excludes the first token emitted by prefill.
func measuredNativePhase(tokens, rateTokens int, ms float64) (float64, float64, bool) {
	// Native durations originate as time.Duration; reject values outside that
	// representable domain when reading persisted rows back from JSONL.
	const maxNativePhaseMS = 9.223372036854775e12
	if tokens < 0 || ms <= 0 || ms > maxNativePhaseMS || math.IsNaN(ms) || math.IsInf(ms, 0) {
		return 0, 0, false
	}
	ms = roundTo(ms, 1000)
	if ms <= 0 || math.IsNaN(ms) || math.IsInf(ms, 0) {
		return 0, 0, false
	}
	if rateTokens == 0 {
		return ms, 0, true
	}
	rate := roundTo(float64(rateTokens)/(ms/1000), 100)
	if rate <= 0 || math.IsNaN(rate) || math.IsInf(rate, 0) {
		return ms, 0, true
	}
	return ms, rate, true
}

// Failure classes for Record.Error.
const (
	ErrorClientCanceled      = "client_canceled"
	ErrorClientWrite         = "client_write"
	ErrorFirstTokenTimeout   = "first_token_timeout"
	ErrorStall               = "stall"
	ErrorDeadline            = "deadline"
	ErrorUpstreamUnreachable = "upstream_unreachable"
	ErrorUpstreamStatus      = "upstream_status"
	ErrorUpstream            = "upstream_error"
	// ErrorClientWriteTimeout marks a buffered turn that outlived the http.Server
	// WriteTimeout: the gateway finished (or failed) it, but the connection was already
	// past its write deadline, so the client received no status and no body.
	ErrorClientWriteTimeout = "client_write_timeout"

	FinishReasonError = "error"
)

// ErrorClasses is the closed Record.Error vocabulary in render order.
var ErrorClasses = []string{
	ErrorClientCanceled, ErrorClientWrite, ErrorFirstTokenTimeout, ErrorStall, ErrorDeadline,
	ErrorUpstreamUnreachable, ErrorUpstreamStatus, ErrorUpstream, ErrorClientWriteTimeout,
}

// NewFailureRecord builds the row for a turn that failed before completing. It
// carries no token counts (the upstream reported none), so its cache regime is
// unknown and it never contributes a rate. ttft>0 means the stream had already
// produced its first token when it failed.
func NewFailureRecord(now time.Time, locality, errClass string, status int, dur, ttft time.Duration) Record {
	rec := Record{
		Schema:       Schema,
		UnixMS:       now.UnixMilli(),
		FinishReason: FinishReasonError,
		Locality:     locality,
		CacheRegime:  cacheobs.RegimeUnknown,
		Error:        errClass,
		Status:       status,
	}
	if dur > 0 {
		rec.E2EMS = roundTo(float64(dur)/float64(time.Millisecond), 1000)
		if ttft > 0 {
			if ttft > dur {
				ttft = dur
			}
			rec.TTFTMS = roundTo(float64(ttft)/float64(time.Millisecond), 1000)
		}
	}
	return rec
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
	if rec.PromptTokens >= MinPrefillRateTokens {
		rec.PrefillTPS = roundTo(float64(rec.PromptTokens)/ttft.Seconds(), 100)
	}
	if decode := dur - ttft; decode > 0 && rec.CompletionTokens > 1 {
		rec.DecodeTPS = roundTo(float64(rec.CompletionTokens-1)/decode.Seconds(), 100)
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
	// PrefillTPSWeighted is total uncached prompt tokens over total TTFT across the
	// rows that carry a prefill rate, so long prompts weigh by their tokens.
	PrefillTPSWeighted float64              `json:"prefill_tps_weighted,omitempty"`
	DecodeTPSP50       float64              `json:"decode_tps_p50,omitempty"`
	E2EP50MS           float64              `json:"e2e_p50_ms,omitempty"`
	E2EP99MS           float64              `json:"e2e_p99_ms,omitempty"`
	CacheHitShare      float64              `json:"cache_hit_share"`
	NativeTiming       *NativeTimingSummary `json:"native_timing,omitempty"`
	// Errors counts failed turns (Record.Error set); ByError splits them by class.
	Errors  int            `json:"errors"`
	ByError map[string]int `json:"by_error,omitempty"`
	// ByRegime splits latency by prompt-cache regime (#5630) so warm vs cold TTFT is
	// one read. Keys are cacheobs.Regimes; a regime with no rows is absent.
	ByRegime map[string]RegimeSummary `json:"by_regime,omitempty"`
	// Native counts the rows the native engine served (rows carrying Engine);
	// ByPath splits them by decode path. Both are absent on a proxy-only window.
	Native      int                    `json:"native,omitempty"`
	ByPath      map[string]int         `json:"by_path,omitempty"`
	ByPathStats map[string]PathSummary `json:"by_path_stats,omitempty"`
	// Spec* sum the window's speculative rounds (native engine rounds plus a
	// proxied upstream's draft counts, which carry no round count); SpecAcceptRate is accepted/draft
	// tokens (vLLM's draft acceptance rate) and is absent when nothing was drafted.
	SpecRounds         int     `json:"spec_rounds,omitempty"`
	SpecDraftTokens    int     `json:"spec_draft_tokens,omitempty"`
	SpecAcceptedTokens int     `json:"spec_accepted_tokens,omitempty"`
	SpecAcceptRate     float64 `json:"spec_accept_rate,omitempty"`
	// Probes counts liveness/probe turns (see Record.IsProbe) left out of every
	// quantile, share, and regime above; Count is the served turns only.
	Probes int `json:"probes,omitempty"`
	// ByClient splits the served turns by Record.Client; absent unless some row
	// in the window carries a client.
	ByClient map[string]ClientSummary `json:"by_client,omitempty"`
	// ServedBy is the newest row's model and server identity; Identities counts
	// the distinct ones in the window, so >1 marks quantiles that mix models,
	// planners, backends, hosts or builds.
	ServedBy   *ServedBy `json:"served_by,omitempty"`
	Identities int       `json:"identities,omitempty"`
	// GPUBusy* fold Record.GPUBusyPercent over the rows that carry one.
	GPUBusyMeasured int     `json:"gpu_busy_measured,omitempty"`
	GPUBusyP50      float64 `json:"gpu_busy_p50,omitempty"`
	// Slowest names the window's slowest served turn so it can be joined to its
	// router attempt and usage receipt.
	Slowest *SlowestTurn `json:"slowest,omitempty"`
}

type SlowestTurn struct {
	RequestID        string  `json:"request_id,omitempty"`
	InboundRequestID string  `json:"inbound_request_id,omitempty"`
	E2EMS            float64 `json:"e2e_ms"`
}

type PathSummary struct {
	Count              int     `json:"count"`
	PrefillMeasured    int     `json:"prefill_measured"`
	DecodeMeasured     int     `json:"decode_measured"`
	PrefillTPSP50      float64 `json:"prefill_tps_p50,omitempty"`
	PrefillTPSWeighted float64 `json:"prefill_tps_weighted,omitempty"`
	DecodeTPSP50       float64 `json:"decode_tps_p50,omitempty"`
	DecodeTPSWeighted  float64 `json:"decode_tps_weighted,omitempty"`
	CacheHitShare      float64 `json:"cache_hit_share"`
}

type NativeTimingSummary struct {
	Count              int     `json:"count"`
	PrefillMeasured    int     `json:"prefill_measured"`
	DecodeMeasured     int     `json:"decode_measured"`
	PrefillP50MS       float64 `json:"prefill_p50_ms,omitempty"`
	DecodeP50MS        float64 `json:"decode_p50_ms,omitempty"`
	PrefillTPSP50      float64 `json:"prefill_tps_p50,omitempty"`
	DecodeTPSP50       float64 `json:"decode_tps_p50,omitempty"`
	PrefillTPSWeighted float64 `json:"prefill_tps_weighted,omitempty"`
	DecodeTPSWeighted  float64 `json:"decode_tps_weighted,omitempty"`
	HostWorkMeasured   int     `json:"host_work_measured,omitempty"`
	CPUMSPerTokenP50   float64 `json:"cpu_ms_per_token_p50,omitempty"`
	HostShareP50       float64 `json:"host_share_p50,omitempty"`
	// Kernels* fold NativeTiming.Kernels over the rows that carry one.
	// TopKernels is the three kernels with the largest mean GPU-time share.
	KernelsMeasured  int           `json:"kernels_measured,omitempty"`
	KernelsComplete  int           `json:"kernels_complete,omitempty"`
	GPUMSPerTokenP50 float64       `json:"gpu_ms_per_token_p50,omitempty"`
	CoverageP50      float64       `json:"kernel_coverage_p50,omitempty"`
	DeviceBusyP50    float64       `json:"device_busy_p50,omitempty"`
	TopKernels       []KernelShare `json:"top_kernels,omitempty"`
}

// ClientSummary is one client's served turns: its prompt-cache hit share over
// cached / (uncached + cached) prompt tokens and its turn count per cache regime.
type ClientSummary struct {
	Count         int            `json:"count"`
	CacheHitShare float64        `json:"cache_hit_share"`
	ByRegime      map[string]int `json:"by_regime,omitempty"`
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
	f := newSummaryFold()
	for _, r := range recs {
		f.foldRecord(r)
	}
	return f.finish(recs)
}

type clientFold struct {
	count          int
	prompt, cached int64
	byRegime       map[string]int
}

func (f *clientFold) add(r Record, regime string) *clientFold {
	if f == nil {
		f = &clientFold{byRegime: map[string]int{}}
	}
	f.count++
	f.prompt += int64(r.PromptTokens)
	f.cached += int64(r.CachedTokens)
	f.byRegime[regime]++
	return f
}

func (f *clientFold) summary() ClientSummary {
	cs := ClientSummary{Count: f.count, ByRegime: f.byRegime}
	if total := f.prompt + f.cached; total > 0 {
		cs.CacheHitShare = roundTo(float64(f.cached)/float64(total), 10000)
	}
	return cs
}

// clientKey folds a row with no recorded client under ClientUnknown.
func (r Record) clientKey() string {
	if r.Client == "" {
		return ClientUnknown
	}
	return r.Client
}

// orderedClients is the window's clients by descending turn count, ties in
// vocabulary order, so the compact line leads with the dominant harness.
func orderedClients(by map[string]ClientSummary) []string {
	rank := make(map[string]int, len(Clients))
	for i, c := range Clients {
		rank[c] = i
	}
	out := make([]string, 0, len(by))
	for c := range by {
		out = append(out, c)
	}
	sort.Slice(out, func(i, j int) bool {
		a, b := by[out[i]].Count, by[out[j]].Count
		if a != b {
			return a > b
		}
		ri, iok := rank[out[i]]
		rj, jok := rank[out[j]]
		if iok != jok {
			return iok
		}
		if ri != rj {
			return ri < rj
		}
		return out[i] < out[j]
	})
	return out
}

func summarizeNativePaths(recs []Record) map[string]PathSummary {
	groups := make(map[string][]Record, len(Paths))
	for _, r := range recs {
		if r.IsProbe() || r.Engine == nil {
			continue
		}
		for _, p := range Paths {
			if r.Engine.Path == p {
				r.Engine = nil
				groups[p] = append(groups[p], r)
				break
			}
		}
	}
	if len(groups) == 0 {
		return nil
	}
	out := make(map[string]PathSummary, len(groups))
	for p, rows := range groups {
		// Clearing Engine above reuses the phase fold without recursively grouping it.
		s := Summarize(rows)
		row := PathSummary{Count: s.Count}
		if n := s.NativeTiming; n != nil {
			row.PrefillMeasured = n.PrefillMeasured
			row.DecodeMeasured = n.DecodeMeasured
			row.PrefillTPSP50 = n.PrefillTPSP50
			row.PrefillTPSWeighted = n.PrefillTPSWeighted
			row.DecodeTPSP50 = n.DecodeTPSP50
			row.DecodeTPSWeighted = n.DecodeTPSWeighted
		}
		var prompt, cached int64
		valid := true
		for _, r := range rows {
			var ok bool
			prompt, ok = addNativeTokens(prompt, r.PromptTokens)
			valid = valid && ok
			cached, ok = addNativeTokens(cached, r.CachedTokens)
			valid = valid && ok
		}
		if valid && (prompt > 0 || cached > 0) {
			row.CacheHitShare = roundTo(float64(cached)/(float64(prompt)+float64(cached)), 10000)
		}
		out[p] = row
	}
	return out
}

func addNativeTokens(total int64, value int) (int64, bool) {
	const maxInt64 = int64(^uint64(0) >> 1)
	if value < 0 || int64(value) > maxInt64-total {
		return total, false
	}
	return total + int64(value), true
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
	fmt.Fprintf(&b, " | prefill p50=%s weighted=%s | decode p50=%s", fmtTPS(s.PrefillTPSP50), fmtTPS(s.PrefillTPSWeighted), fmtTPS(s.DecodeTPSP50))
	fmt.Fprintf(&b, " | e2e p50=%s p99=%s", fmtMS(s.E2EP50MS), fmtMS(s.E2EP99MS))
	fmt.Fprintf(&b, " | cache=%.1f%%", s.CacheHitShare*100)
	if s.GPUBusyMeasured > 0 {
		fmt.Fprintf(&b, " | gpu busy p50=%.0f%% (measured %d/%d)", s.GPUBusyP50, s.GPUBusyMeasured, s.Count)
	}
	if n := s.NativeTiming; n != nil {
		fmt.Fprintf(&b, " | native phases=%d prefill p50=%s/%s weighted=%s (measured %d/%d) decode p50=%s/%s weighted=%s (measured %d/%d)",
			n.Count, fmtMS(n.PrefillP50MS), fmtTPS(n.PrefillTPSP50), fmtTPS(n.PrefillTPSWeighted), n.PrefillMeasured, n.Count,
			fmtMS(n.DecodeP50MS), fmtTPS(n.DecodeTPSP50), fmtTPS(n.DecodeTPSWeighted), n.DecodeMeasured, n.Count)
		if n.HostWorkMeasured > 0 {
			fmt.Fprintf(&b, " host share p50=%.3f cpu p50=%.2fms/tok (measured %d/%d)", n.HostShareP50, n.CPUMSPerTokenP50, n.HostWorkMeasured, n.Count)
		}
		if n.KernelsMeasured > 0 {
			fmt.Fprintf(&b, " kern gpu p50=%.2fms/tok cov=%.0f%% dev_busy=%.0f%%", n.GPUMSPerTokenP50, n.CoverageP50*100, n.DeviceBusyP50*100)
			for i, k := range n.TopKernels {
				sep := ","
				if i == 0 {
					sep = " top="
				}
				fmt.Fprintf(&b, "%s%s:%.0f%%", sep, k.Kernel, k.Share*100)
			}
			fmt.Fprintf(&b, " (measured %d/%d complete %d)", n.KernelsMeasured, n.Count, n.KernelsComplete)
		}
	}
	if s.Errors > 0 {
		classes := make([]string, 0, len(s.ByError))
		for c := range s.ByError {
			classes = append(classes, c)
		}
		sort.Strings(classes)
		fmt.Fprintf(&b, " | errors=%d", s.Errors)
		for _, c := range classes {
			fmt.Fprintf(&b, " %s=%d", c, s.ByError[c])
		}
	}
	if len(s.ByRegime) > 0 {
		b.WriteString(" | ttft p50 by regime:")
		for _, regime := range cacheobs.Regimes {
			if rs, ok := s.ByRegime[regime]; ok {
				fmt.Fprintf(&b, " %s=%s(n=%d)", regime, fmtMS(rs.TTFTP50MS), rs.Count)
			}
		}
	}
	if s.Probes > 0 {
		fmt.Fprintf(&b, " | probes=%d excluded", s.Probes)
	}
	if len(s.ByClient) > 0 {
		b.WriteString(" | by client:")
		for _, c := range orderedClients(s.ByClient) {
			cs := s.ByClient[c]
			fmt.Fprintf(&b, " %s=%d H=%.0f%%", c, cs.Count, cs.CacheHitShare*100)
			var regimes []string
			for _, regime := range cacheobs.Regimes {
				if n := cs.ByRegime[regime]; n > 0 {
					regimes = append(regimes, fmt.Sprintf("%s%d", regime[:1], n))
				}
			}
			if len(regimes) > 0 {
				b.WriteString("(" + strings.Join(regimes, "/") + ")")
			}
		}
	}
	if s.Native > 0 {
		b.WriteString(" | path:")
		for _, p := range Paths {
			if n := s.ByPath[p]; n > 0 {
				fmt.Fprintf(&b, " %s=%d", p, n)
			}
		}
		for _, p := range Paths {
			if row, ok := s.ByPathStats[p]; ok {
				fmt.Fprintf(&b, " | %s native n=%d prefill p50=%s weighted=%s (measured %d/%d) decode p50=%s weighted=%s (measured %d/%d) cache=%.1f%%", p, row.Count, fmtTPS(row.PrefillTPSP50), fmtTPS(row.PrefillTPSWeighted), row.PrefillMeasured, row.Count, fmtTPS(row.DecodeTPSP50), fmtTPS(row.DecodeTPSWeighted), row.DecodeMeasured, row.Count, row.CacheHitShare*100)
			}
		}
	}
	if s.SpecDraftTokens > 0 {
		fmt.Fprintf(&b, " | spec accept=%.1f%% (%d/%d", s.SpecAcceptRate*100, s.SpecAcceptedTokens, s.SpecDraftTokens)
		if s.SpecRounds > 0 {
			fmt.Fprintf(&b, " over %d rounds", s.SpecRounds)
		}
		b.WriteString(")")
	}
	if sb := s.ServedBy; sb != nil {
		b.WriteString(" | served by")
		writeField(&b, "model", sb.Model)
		planner := sb.Planner
		if sb.Backend != "" {
			planner = strings.TrimPrefix(planner+"/"+sb.Backend, "/")
		}
		writeField(&b, "planner", planner)
		writeField(&b, "host", sb.Host)
		writeField(&b, "fak", sb.Version)
		if s.Identities > 1 {
			fmt.Fprintf(&b, " (window mixes %d)", s.Identities)
		}
	}
	if sl := s.Slowest; sl != nil && (sl.RequestID != "" || sl.InboundRequestID != "") {
		fmt.Fprintf(&b, " | slowest e2e=%s", fmtMS(sl.E2EMS))
		writeField(&b, "request_id", sl.RequestID)
		writeField(&b, "inbound_request_id", sl.InboundRequestID)
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

func writeField(b *strings.Builder, key, val string) {
	if val != "" {
		fmt.Fprintf(b, " %s=%s", key, val)
	}
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
