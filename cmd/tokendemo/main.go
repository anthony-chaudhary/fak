// Command tokendemo is the self-contained demo of two CLEAR WINS the kernel's
// tool-call understanding delivers — counted call by call, each grounded in a LIVE
// kernel verdict (the kernel decides; this demo only counts). It is the token-side
// companion to cmd/guarddemo (safety), cmd/turntaxdemo (turns), and cmd/ctxdemo
// (prefix reuse): same "replay a frozen, class-labeled trace through the REAL kernel"
// discipline, two honest token meters.
//
// The two wins live on DIFFERENT layers, and the demo keeps them separate on purpose:
//
//  1. PREFILTER on a mutating /bad call — a MODEL-CONTEXT win.
//     An agent proposes a write_file / delete_path / run_shell the floor does not
//     sanction. WITHOUT the kernel the call EXECUTES and its result — a confirmation,
//     or (more often) a permission-denied stack trace — lands in the MODEL's context
//     (R tokens it must then read and react to). WITH the kernel the call is refused
//     BEFORE it runs: the big result is NEVER PRODUCED, and only a bounded
//     deny-as-value verdict (~a few dozen tokens) enters context. Those (R − verdict)
//     tokens are genuinely kept out of the model. This is the headline token win.
//
//  2. READING THE SAME FILE — a TOOL-SIDE win.
//     An agent re-reads config.yaml on turn 3 and again on turn 5. WITHOUT tool-call
//     understanding the tool RE-EXECUTES every read (re-fetch from disk / DB / API).
//     WITH it, the kernel knows the re-read is the same idempotent call and serves it
//     1-shot from the content cache (vDSO tier-2) — the tool runs ONCE, not N times.
//     HONEST BOUND: the cached content is still RETURNED to the model (gateway
//     resolveBytes re-materializes it), so this is NOT a model-context cut — it saves
//     the tool round-trip / re-execution (latency, compute, $). The model-side
//     prefill/KV reuse that would ALSO cut the re-read's tokens is a separate axis
//     (cmd/ctxdemo); the live agent loop's KV-eviction half is mechanism-proven, not
//     yet production-served (see docs/FAQ.md). So this demo counts the tool-side win
//     here and does not double-count it as model context.
//
// HONEST SCOPE. The result-token sizes are an explicit, documented per-call knob
// (`result_tokens` in the trace meta) — the same kind turnbench's CostModel and
// ctxdemo's tool sizes are; the magnitudes are illustrative, not a measured
// production bill. The DENY / DEDUP classification underneath is the kernel's own
// LIVE verdict. The SAFETY value of refusing the mutating call (the destructive op
// never runs) is cmd/guarddemo's separate axis, the moat; this counts only the token
// consequence. A clean trace (no bad calls, no re-reads) saves ZERO on both meters —
// the anti-inflation control proves the demo cannot cry wolf.
//
// The world here is a CODING-AGENT FILE WORLD, not the airline world the turntax /
// guard demos use: read_file / list_dir / search_code are allow-listed and cacheable;
// write_file / delete_path / run_shell / apply_patch fall to the structural
// DEFAULT_DENY floor (the capability the agent was never granted). It is installed via
// turnbench.RunWithWorld, so the replay, the live-verdict classification, and the
// consistency check are the exact same grounded machinery the other demos use.
//
// Headless — no model, no GPU, no browser, no network. Deterministic:
//
//	go run ./cmd/tokendemo -print
//	# the 30-second point: render the WITHOUT-kernel vs WITH-kernel ledger as a
//	# colored two-column diff in the terminal. -suite picks the trace; honors NO_COLOR.
//
//	go run ./cmd/tokendemo -print -suite reread-same-file
//	go run ./cmd/tokendemo -json            # the exact per-call ledger as JSON (all suites)
//	go run ./cmd/tokendemo -selfcheck       # browserless: replay each suite through the
//	#   kernel, assert the documented ledger invariants, exit non-zero on drift.
package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/ifc"
	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/turnbench"
	"github.com/anthony-chaudhary/fak/internal/vdso"

	// Blank-import the built-in driver list so the full ABI (resolver, vDSO,
	// adjudicator, ctx-MMU, engines) is wired before turnbench.RunWithWorld runs.
	_ "github.com/anthony-chaudhary/fak/internal/registrations"
)

const version = "fak-tokendemo-v1"

// denyVerdictTokens is the bounded size of a deny-as-value verdict that enters the
// model's context in place of an executed bad call's result: the tool name, the
// closed reason code (e.g. DEFAULT_DENY), a one-line human message, and the JSON
// envelope. It is a fixed, small constant BY CONSTRUCTION — the refusal vocabulary is
// closed (see docs/mcp-tool-result.md), so a deny can never balloon the way a real
// tool result can. Conservative on the high side.
const denyVerdictTokens = 32

// defaultResultTokens is the per-call result size assumed when a trace call carries
// no explicit `result_tokens` annotation — a modest read.
const defaultResultTokens = 200

var gomax = runtime.GOMAXPROCS(0)

// knownSuites are the fixtures shipped under testdata/tokendemo. Each isolates ONE
// story so the win is unambiguous; clean-control proves a benign session saves zero.
var knownSuites = []struct{ ID, Label string }{
	{"prefilter-bad-calls", "prefilter: mutating /bad calls refused before they run (win 1 — model-context tokens)"},
	{"reread-same-file", "reread: the same file served from cache, the tool not re-run (win 2 — tool round-trips)"},
	{"clean-control", "clean path (no bad calls, no re-reads — the anti-inflation control, 0)"},
}

// ---------------------------------------------------------------------------
// the coding-agent FILE WORLD — installed via turnbench.RunWithWorld.
// ---------------------------------------------------------------------------

// fileEngine is the dispatch target for the ALLOWED read tools (denied tools never
// reach it — they are refused pre-dispatch at the capability floor). It returns a
// small StatusOK result so the vDSO tier-2 cache fills on the first read of a file
// and the second identical read is a real content-cache hit. The payload bytes are
// not what this demo counts (the ledger uses the trace's `result_tokens` annotation);
// the engine exists only so the dedup path is GROUNDED in a real completion.
type fileEngine struct{}

var (
	fileEngineDelayNs    atomic.Int64
	fileEngineCallSeq    atomic.Int64
	fileEngineByResource sync.Map // resource string -> *atomic.Int64
)

func setFileEngineDelay(d time.Duration) time.Duration {
	prev := time.Duration(fileEngineDelayNs.Swap(int64(d)))
	return prev
}

func resetFileEngineStats() {
	fileEngineCallSeq.Store(0)
	fileEngineByResource.Range(func(k, _ any) bool {
		fileEngineByResource.Delete(k)
		return true
	})
}

func fileEngineCalls() int64 {
	return fileEngineCallSeq.Load()
}

func fileEngineResourceCalls() map[string]int64 {
	out := map[string]int64{}
	fileEngineByResource.Range(func(k, v any) bool {
		name, ok := k.(string)
		ctr, ok2 := v.(*atomic.Int64)
		if ok && ok2 {
			out[name] = ctr.Load()
		}
		return true
	})
	return out
}

// Caps reports the engine's capabilities; this demo file engine declares none.
func (fileEngine) Caps() []abi.Capability { return nil }

func (fileEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	t0 := time.Now()
	if d := time.Duration(fileEngineDelayNs.Load()); d > 0 {
		timer := time.NewTimer(d)
		select {
		case <-timer.C:
		case <-ctx.Done():
			if !timer.Stop() {
				select {
				case <-timer.C:
				default:
				}
			}
			return nil, ctx.Err()
		}
	}
	seq := fileEngineCallSeq.Add(1)
	resource := resourceLabelBytes(toolCallArgsBytes(ctx, c))
	if resource == "" {
		resource = c.Tool
	}
	ctrAny, _ := fileEngineByResource.LoadOrStore(resource, &atomic.Int64{})
	if ctr, ok := ctrAny.(*atomic.Int64); ok {
		ctr.Add(1)
	}
	out := []byte(`{"tool":"` + c.Tool + `","ok":true}`)
	var ref abi.Ref
	if res := abi.ActiveResolver(); res != nil {
		if r, err := res.Put(ctx, out); err == nil {
			ref = r
		}
	}
	if ref.Kind == 0 && ref.Len == 0 {
		ref = abi.Ref{Kind: abi.RefInline, Inline: out, Len: int64(len(out))}
	}
	return &abi.Result{Call: c, Payload: ref, Status: abi.StatusOK, Meta: map[string]string{
		"engine":            "localtools",
		"engine_call":       strconv.FormatInt(seq, 10),
		"engine_elapsed_ns": strconv.FormatInt(time.Since(t0).Nanoseconds(), 10),
	}}, nil
}

// configureFileWorld installs the coding-agent file world: the read family is
// affirmatively allowed (and the read calls in the traces carry the read-only +
// idempotent hints that make them vDSO-cacheable); everything write-shaped is left
// OFF the allow-list, so write_file / delete_path / run_shell / apply_patch fall to
// the structural fail-closed DEFAULT_DENY floor (the strongest refusal — the
// capability was never wired up, not a pattern that could be evaded). It overwrites
// the process-global drivers, replacing whatever world was previously installed; the
// engine name "localtools" matches the dispatch target turnbench's replay builds.
func configureFileWorld() {
	abi.RegisterEngine("localtools", fileEngine{})
	adjudicator.Default.SetPolicy(adjudicator.Policy{
		// The read-only tool family a coding agent inspects a repo with. AllowPrefix
		// covers read_file / list_dir / search_code by name shape; none is write-shaped,
		// so each is vDSO fast-path eligible under its read-only+idempotent hints.
		AllowPrefix: []string{"read_", "list_", "search_", "get_", "find_"},
		// Nothing else is allowed: write_file / delete_path / run_shell / apply_patch
		// are unsanctioned AND write-shaped, so they hit the fail-closed DEFAULT_DENY
		// floor and are counted destructive (the baseline would have executed them).
	})
}

// ---------------------------------------------------------------------------
// the two-meter ledger.
//
//	model-context meter: tokens the MODEL must ingest. The prefilter win lives here
//	  (a denied bad call's result is never produced; only the deny verdict enters).
//	  A dedup'd re-read does NOT save here — the cached content is still returned to
//	  the model — so its model-context columns are equal on both arms (honest).
//	tool-side meter: tool round-trips / re-executions. The re-read win lives here
//	  (the tool runs once, not N times); the bytes are served from cache, not re-fetched.
// ---------------------------------------------------------------------------

// callLedger is one replayed call's contribution, on both meters.
type callLedger struct {
	Index        int    `json:"index"`
	Tool         string `json:"tool"`
	Class        string `json:"class"`         // the kernel's live verdict class
	Axis         string `json:"axis"`          // turn-tax | safety-floor | control
	ResultTokens int    `json:"result_tokens"` // R — the result this call carries
	// model-context meter
	CtxWithout int `json:"ctx_without"` // model context tokens, raw loop
	CtxWith    int `json:"ctx_with"`    // model context tokens, behind fak
	CtxSaved   int `json:"ctx_saved"`
	// tool-side meter
	ToolRanWithout int    `json:"tool_ran_without"` // tool executions, raw loop (0/1)
	ToolRanWith    int    `json:"tool_ran_with"`    // tool executions, behind fak (0 on a cache hit)
	Why            string `json:"why"`
}

// ledger is the rolled-up per-suite accounting on both meters.
type ledger struct {
	Suite string       `json:"suite"`
	Calls []callLedger `json:"calls"`
	// model-context meter (the prefilter win lives here)
	CtxWithout        int `json:"ctx_without_total"`
	CtxWith           int `json:"ctx_with_total"`
	ContextTokensKept int `json:"context_tokens_kept_out"` // headline win 1: sum of CtxSaved (denied bad calls)
	Denies            int `json:"denies"`
	// tool-side meter (the re-read win lives here)
	RoundtripsCollapsed int `json:"roundtrips_collapsed"`   // win 2: re-reads served from cache (tool not re-run)
	ToolTokensFromCache int `json:"tool_tokens_from_cache"` // tool-result tokens served from cache instead of re-fetched
	ToolRunsWithout     int `json:"tool_runs_without"`      // tool executions in the raw loop
	ToolRunsWith        int `json:"tool_runs_with"`         // tool executions behind fak (cache hits + denied calls do not run)
	Dedups              int `json:"dedups"`
	Passes              int `json:"passes"`
	DenyVerdictTokens   int `json:"deny_verdict_tokens"`
}

type timingCall struct {
	Index           int    `json:"index"`
	Tool            string `json:"tool"`
	Resource        string `json:"resource,omitempty"`
	ArgsHash        string `json:"args_hash"`
	Class           string `json:"class"`
	ResultTokens    int    `json:"result_tokens"`
	RawToolTimeNs   int64  `json:"raw_tool_time_ns"`
	FakToolTimeNs   int64  `json:"fak_tool_time_ns"`
	FakSource       string `json:"fak_source"`
	FakTier         string `json:"fak_tier,omitempty"`
	EngineRanRaw    bool   `json:"engine_ran_raw"`
	EngineRanFak    bool   `json:"engine_ran_fak"`
	FakEngineDelta  int64  `json:"fak_engine_call_delta"`
	KernelVDSODelta int64  `json:"kernel_vdso_hit_delta"`
}

type timingProof struct {
	Suite                       string       `json:"suite"`
	Path                        string       `json:"path"`
	Prewarmed                   bool         `json:"prewarmed"`
	EngineDelayMs               int          `json:"engine_delay_ms"`
	Calls                       []timingCall `json:"calls"`
	RawTotalNs                  int64        `json:"raw_total_ns"`
	FakTotalNs                  int64        `json:"fak_total_ns"`
	TimeSavedNs                 int64        `json:"time_saved_ns"`
	RawEngineCalls              int64        `json:"raw_engine_calls"`
	FakEngineCalls              int64        `json:"fak_engine_calls"`
	VDSOHits                    int64        `json:"vdso_hits"`
	RoundtripsCollapsed         int          `json:"roundtrips_collapsed"`
	ToolTokensFromCache         int          `json:"tool_tokens_from_cache"`
	ToolTokensFromCacheMeaning  string       `json:"tool_tokens_from_cache_meaning"`
	ContextTokensKeptOut        int          `json:"context_tokens_kept_out"`
	ContextTokensKeptOutMeaning string       `json:"context_tokens_kept_out_meaning"`
	DenyVerdictTokens           int          `json:"deny_verdict_tokens"`
	ClaimBoundary               []string     `json:"claim_boundary"`
}

type parallelResourceProof struct {
	Resource              string  `json:"resource"`
	ResultTokens          int     `json:"result_tokens"`
	RawEngineCalls        int64   `json:"raw_engine_calls"`
	FakWarmupEngineCalls  int64   `json:"fak_warmup_engine_calls"`
	FakHotEngineCalls     int64   `json:"fak_hot_engine_calls"`
	VDSOHits              int64   `json:"vdso_hits"`
	ToolTokensFromCache   int     `json:"tool_tokens_from_cache"`
	RawP50Ns              int64   `json:"raw_p50_ns"`
	RawP95Ns              int64   `json:"raw_p95_ns"`
	FakP50Ns              int64   `json:"fak_p50_ns"`
	FakP95Ns              int64   `json:"fak_p95_ns"`
	EngineCallsAvoided    int64   `json:"engine_calls_avoided"`
	EngineCallAvoidedRate float64 `json:"engine_call_avoided_rate"`
}

type parallelProof struct {
	Schema                      string                  `json:"schema"`
	Path                        string                  `json:"path"`
	Workers                     int                     `json:"workers"`
	Calls                       int                     `json:"calls"`
	HotFiles                    int                     `json:"hot_files"`
	EngineDelayMs               int                     `json:"engine_delay_ms"`
	Prewarmed                   bool                    `json:"prewarmed"`
	RawWallNs                   int64                   `json:"raw_wall_ns"`
	FakWarmupWallNs             int64                   `json:"fak_warmup_wall_ns"`
	FakHotWallNs                int64                   `json:"fak_hot_wall_ns"`
	RawTotalNs                  int64                   `json:"raw_total_ns"`
	FakHotTotalNs               int64                   `json:"fak_hot_total_ns"`
	TimeSavedNs                 int64                   `json:"time_saved_ns"`
	RawP50Ns                    int64                   `json:"raw_p50_ns"`
	RawP95Ns                    int64                   `json:"raw_p95_ns"`
	FakP50Ns                    int64                   `json:"fak_p50_ns"`
	FakP95Ns                    int64                   `json:"fak_p95_ns"`
	RawEngineCalls              int64                   `json:"raw_engine_calls"`
	FakWarmupEngineCalls        int64                   `json:"fak_warmup_engine_calls"`
	FakHotEngineCalls           int64                   `json:"fak_hot_engine_calls"`
	VDSOHits                    int64                   `json:"vdso_hits"`
	EngineCallsAvoided          int64                   `json:"engine_calls_avoided"`
	EngineCallAvoidedRate       float64                 `json:"engine_call_avoided_rate"`
	ToolTokensFromCache         int                     `json:"tool_tokens_from_cache"`
	ToolTokensFromCacheMeaning  string                  `json:"tool_tokens_from_cache_meaning"`
	ContextTokensKeptOut        int                     `json:"context_tokens_kept_out"`
	ContextTokensKeptOutMeaning string                  `json:"context_tokens_kept_out_meaning"`
	ClaimBoundary               []string                `json:"claim_boundary"`
	PerResource                 []parallelResourceProof `json:"per_resource"`
}

const (
	cacheProofPathKernelSyscall    = "kernel_syscall"
	toolTokensFromCacheMeaning     = "tool-result payload size served from cache instead of refetched; not prompt tokens removed"
	contextTokensKeptOutMeaning    = "model-context tokens omitted only by denied bad-call results; cached read content is still returned"
	cacheProofPositiveClaim        = "proves repeated read-only/idempotent calls routed through kernel.Syscall can be served from vDSO tier-2 after fill"
	cacheProofNotGuardNativeRead   = "does not prove native Claude Code Read calls through fak guard are served from vDSO"
	cacheProofNotModelTokenSaving  = "does not claim model-context token savings for cached rereads"
	cacheProofNotColdSingleflight  = "does not prove cold concurrent singleflight; the parallel proof is warmed hot-cache sharing"
	cacheProofDelayIsObservability = "synthetic engine delay is an observability aid, not a production latency claim"
)

func cacheClaimBoundary(path string, prewarmed bool) []string {
	warmth := "after normal fill"
	if prewarmed {
		warmth = "after explicit warmup"
	}
	return []string{
		cacheProofPositiveClaim + " (" + path + ", " + warmth + ")",
		cacheProofNotGuardNativeRead,
		cacheProofNotModelTokenSaving,
		cacheProofNotColdSingleflight,
		cacheProofDelayIsObservability,
	}
}

func printCacheClaimBoundary(path string, prewarmed bool) {
	lines := cacheClaimBoundary(path, prewarmed)
	fmt.Printf("  scope: %s\n", lines[0])
	fmt.Printf("  non-claims: not native guard Read via vDSO; not model-context token savings for rereads; not cold singleflight; not production latency.\n")
	fmt.Printf("  token terms: tool_tokens_from_cache = %s; context_tokens_kept_out = %s.\n\n",
		toolTokensFromCacheMeaning, contextTokensKeptOutMeaning)
}

type servedCall struct {
	Index             int               `json:"index"`
	Surface           string            `json:"surface"`
	Tool              string            `json:"tool"`
	Resource          string            `json:"resource"`
	ArgsHash          string            `json:"args_hash"`
	RawToolTimeNs     int64             `json:"raw_tool_time_ns"`
	ServedTimeNs      int64             `json:"served_time_ns"`
	Verdict           string            `json:"verdict"`
	ServedBy          string            `json:"served_by"`
	Tier              string            `json:"tier,omitempty"`
	EngineRanRaw      bool              `json:"engine_ran_raw"`
	EngineRanServed   bool              `json:"engine_ran_served"`
	ServedEngineDelta int64             `json:"served_engine_call_delta"`
	ResponseMeta      map[string]string `json:"response_meta,omitempty"`
}

type servedMetricsSnapshot struct {
	VDSOLookups         int64 `json:"vdso_lookups"`
	VDSOHits            int64 `json:"vdso_hits"`
	VDSOFills           int64 `json:"vdso_cache_fills"`
	GatewaySyscalls     int64 `json:"gateway_syscalls"`
	HTTPSyscallRequests int64 `json:"http_syscall_requests"`
	MCPRequests         int64 `json:"mcp_requests"`
	TurnsSavedVDSO      int64 `json:"turns_saved_vdso_dedup"`
}

type servedMetricsEvidence struct {
	Before servedMetricsSnapshot `json:"before"`
	After  servedMetricsSnapshot `json:"after"`
	Delta  servedMetricsSnapshot `json:"delta"`
}

type servedProof struct {
	Schema           string                `json:"schema"`
	Tool             string                `json:"tool"`
	Resource         string                `json:"resource"`
	Surfaces         []string              `json:"surfaces"`
	CallsPerSurface  int                   `json:"calls_per_surface"`
	TotalServedCalls int                   `json:"total_served_calls"`
	DistinctKeys     int                   `json:"distinct_keys"`
	EngineDelayMs    int                   `json:"engine_delay_ms"`
	RawTotalNs       int64                 `json:"raw_total_ns"`
	ServedTotalNs    int64                 `json:"served_total_ns"`
	TimeSavedNs      int64                 `json:"time_saved_ns"`
	RawP50Ns         int64                 `json:"raw_p50_ns"`
	RawP95Ns         int64                 `json:"raw_p95_ns"`
	ServedP50Ns      int64                 `json:"served_p50_ns"`
	ServedP95Ns      int64                 `json:"served_p95_ns"`
	RawEngineCalls   int64                 `json:"raw_engine_calls"`
	FakEngineCalls   int64                 `json:"fak_engine_calls"`
	VDSOHits         int64                 `json:"vdso_hits"`
	GatewayMetrics   servedMetricsEvidence `json:"gateway_metrics"`
	Calls            []servedCall          `json:"calls"`
}

// resultTokens reads the explicit per-call `result_tokens` annotation (the modeled,
// documented knob), falling back to defaultResultTokens when absent or malformed.
func resultTokens(c turnbench.Call) int {
	if c.Meta != nil {
		if s, ok := c.Meta["result_tokens"]; ok {
			if n, err := strconv.Atoi(strings.TrimSpace(s)); err == nil && n >= 0 {
				return n
			}
		}
	}
	return defaultResultTokens
}

// buildLedger replays the suite through the REAL kernel under the file world and
// scores both meters from the LIVE dispositions zipped with the trace's result-size
// annotations. The classification (deny / dedup / pass) is the kernel's; only the
// token arithmetic is this demo's.
func buildLedger(ctx context.Context, suite string) (ledger, error) {
	t, err := turnbench.LoadTrace(suitePath(suite))
	if err != nil {
		return ledger{}, err
	}
	_, disp, err := turnbench.RunWithWorld(ctx, t, turnbench.DefaultCostModel(), configureFileWorld)
	if err != nil {
		return ledger{}, err
	}
	l := ledger{Suite: suite, DenyVerdictTokens: denyVerdictTokens}
	for i, d := range disp {
		R := defaultResultTokens
		if i < len(t.Calls) {
			R = resultTokens(t.Calls[i])
		}
		row := callLedger{Index: d.Index, Tool: d.Tool, Class: d.Class, Axis: d.Axis, ResultTokens: R}
		switch d.Class {
		case "deny":
			// Win 1 (model-context): refused before it runs. The raw loop executes it →
			// R result tokens enter the MODEL's context; behind fak the result is never
			// produced and only the bounded deny verdict enters. The tool also never runs.
			row.CtxWithout, row.CtxWith = R, denyVerdictTokens
			row.CtxSaved = R - denyVerdictTokens
			row.ToolRanWithout, row.ToolRanWith = 1, 0
			row.Why = "prefilter — refused pre-execution; only a bounded deny verdict enters the model, not the op's " + itoa(R) + "-tok result"
			if row.CtxSaved > 0 {
				l.ContextTokensKept += row.CtxSaved
			}
			l.Denies++
		case "vdso_dedup":
			// Win 2 (tool-side): the same idempotent read, served 1-shot from the content
			// cache — the tool does NOT re-execute. HONEST: the cached content is still
			// RETURNED to the model, so the model-context columns are EQUAL on both arms
			// (no model-context cut here); the win is the eliminated tool round-trip.
			row.CtxWithout, row.CtxWith = R, R
			row.CtxSaved = 0
			row.ToolRanWithout, row.ToolRanWith = 1, 0
			row.Why = "dedup — same file already read; the tool is served from cache (not re-run). The content is still returned to the model (a tool-side win, not a model-context cut)."
			l.RoundtripsCollapsed++
			l.ToolTokensFromCache += R
			l.Dedups++
		default:
			// Control: a first read / legitimate call BOTH arms pay identically, and the
			// tool runs once on both (fak is not free on real work — that honesty is the point).
			row.CtxWithout, row.CtxWith = R, R
			row.CtxSaved = 0
			row.ToolRanWithout, row.ToolRanWith = 1, 1
			row.Why = "control — legitimate work; both arms ingest it and the tool runs once on both"
			l.Passes++
		}
		l.CtxWithout += row.CtxWithout
		l.CtxWith += row.CtxWith
		l.ToolRunsWithout += row.ToolRanWithout
		l.ToolRunsWith += row.ToolRanWith
		l.Calls = append(l.Calls, row)
	}
	return l, nil
}

func buildTimingProof(ctx context.Context, suite string, engineDelay time.Duration) (timingProof, error) {
	t, err := turnbench.LoadTrace(suitePath(suite))
	if err != nil {
		return timingProof{}, err
	}
	configureFileWorld()
	res := abi.ActiveResolver()
	if res == nil {
		return timingProof{}, fmt.Errorf("no active Ref resolver registered")
	}

	prevDelay := setFileEngineDelay(engineDelay)
	defer setFileEngineDelay(prevDelay)

	rawTimes := make([]int64, 0, len(t.Calls))
	resetFileEngineStats()
	rawEngine := fileEngine{}
	for _, c := range t.Calls {
		tc, err := traceToolCall(ctx, res, c)
		if err != nil {
			return timingProof{}, err
		}
		start := time.Now()
		if _, err := rawEngine.Complete(ctx, tc); err != nil {
			return timingProof{}, err
		}
		rawTimes = append(rawTimes, elapsedNs(start))
	}
	rawEngineCalls := fileEngineCalls()

	resetFileEngineStats()
	vdso.Default.BumpWorld()
	ifc.Default.Reset("")
	k := kernel.New("localtools")
	k.SetVDSO(true)

	out := timingProof{
		Suite:                       suite,
		Path:                        cacheProofPathKernelSyscall,
		Prewarmed:                   false,
		EngineDelayMs:               int(engineDelay / time.Millisecond),
		ToolTokensFromCacheMeaning:  toolTokensFromCacheMeaning,
		ContextTokensKeptOutMeaning: contextTokensKeptOutMeaning,
		DenyVerdictTokens:           denyVerdictTokens,
		ClaimBoundary:               cacheClaimBoundary(cacheProofPathKernelSyscall, false),
		Calls:                       make([]timingCall, 0, len(t.Calls)),
		RawEngineCalls:              rawEngineCalls,
	}
	for idx, c := range t.Calls {
		tc, err := traceToolCall(ctx, res, c)
		if err != nil {
			return timingProof{}, err
		}
		beforeEngine := fileEngineCalls()
		beforeCounters := k.Counters()
		start := time.Now()
		r, v := k.Syscall(ctx, tc)
		fakElapsed := elapsedNs(start)
		afterCounters := k.Counters()
		afterEngine := fileEngineCalls()

		class, source, tier := timingClass(r, v)
		R := resultTokens(c)
		row := timingCall{
			Index:           idx,
			Tool:            c.Tool,
			Resource:        resourceLabel(c.Args),
			ArgsHash:        argsHash(c.Args),
			Class:           class,
			ResultTokens:    R,
			RawToolTimeNs:   rawTimes[idx],
			FakToolTimeNs:   fakElapsed,
			FakSource:       source,
			FakTier:         tier,
			EngineRanRaw:    true,
			EngineRanFak:    afterEngine > beforeEngine,
			FakEngineDelta:  afterEngine - beforeEngine,
			KernelVDSODelta: afterCounters.VDSOHits - beforeCounters.VDSOHits,
		}
		out.RawTotalNs += row.RawToolTimeNs
		out.FakTotalNs += row.FakToolTimeNs
		switch class {
		case "deny":
			saved := R - denyVerdictTokens
			if saved > 0 {
				out.ContextTokensKeptOut += saved
			}
		case "vdso_dedup":
			out.RoundtripsCollapsed++
			out.ToolTokensFromCache += R
		}
		out.Calls = append(out.Calls, row)
	}
	finalCounters := k.Counters()
	out.FakEngineCalls = fileEngineCalls()
	out.VDSOHits = finalCounters.VDSOHits
	out.TimeSavedNs = out.RawTotalNs - out.FakTotalNs
	return out, nil
}

var parallelHotFileCatalog = []struct {
	path   string
	tokens int
}{
	{"config.yaml", 180},
	{"src/main.go", 540},
	{"README.md", 700},
	{"go.mod", 120},
	{"docs/repro-packet.md", 620},
	{"cmd/tokendemo/main.go", 900},
	{"internal/vdso/vdso.go", 1000},
	{"examples/customer-support-readonly-policy.json", 260},
}

func buildParallelProof(ctx context.Context, workers, calls, hotFiles int, engineDelay time.Duration) (parallelProof, error) {
	if workers <= 0 {
		return parallelProof{}, fmt.Errorf("workers must be positive")
	}
	if calls <= 0 {
		return parallelProof{}, fmt.Errorf("calls must be positive")
	}
	if hotFiles <= 0 {
		return parallelProof{}, fmt.Errorf("hot files must be positive")
	}
	if hotFiles > len(parallelHotFileCatalog) {
		hotFiles = len(parallelHotFileCatalog)
	}
	if hotFiles > calls {
		hotFiles = calls
	}
	configureFileWorld()
	res := abi.ActiveResolver()
	if res == nil {
		return parallelProof{}, fmt.Errorf("no active Ref resolver registered")
	}
	prevDelay := setFileEngineDelay(engineDelay)
	defer setFileEngineDelay(prevDelay)

	hot, workload := parallelWorkload(calls, hotFiles)
	rawEngine := fileEngine{}

	resetFileEngineStats()
	rawRun, err := runParallelToolCalls(ctx, workers, workload, func(ctx context.Context, c turnbench.Call) (string, error) {
		tc, err := traceToolCall(ctx, res, c)
		if err != nil {
			return "", err
		}
		if _, err := rawEngine.Complete(ctx, tc); err != nil {
			return "", err
		}
		return "engine", nil
	})
	if err != nil {
		return parallelProof{}, err
	}
	rawByResource := fileEngineResourceCalls()
	rawEngineCalls := fileEngineCalls()

	resetFileEngineStats()
	vdso.Default.BumpWorld()
	ifc.Default.Reset("")
	k := kernel.New("localtools")
	k.SetVDSO(true)

	warmStart := time.Now()
	for _, c := range hot {
		tc, err := traceToolCall(ctx, res, c)
		if err != nil {
			return parallelProof{}, err
		}
		k.Syscall(ctx, tc)
	}
	warmWall := elapsedNs(warmStart)
	warmByResource := fileEngineResourceCalls()
	warmEngineCalls := fileEngineCalls()

	beforeHotEngine := fileEngineCalls()
	beforeHotByResource := fileEngineResourceCalls()
	fakRun, err := runParallelToolCalls(ctx, workers, workload, func(ctx context.Context, c turnbench.Call) (string, error) {
		tc, err := traceToolCall(ctx, res, c)
		if err != nil {
			return "", err
		}
		r, v := k.Syscall(ctx, tc)
		_, source, _ := timingClass(r, v)
		return source, nil
	})
	if err != nil {
		return parallelProof{}, err
	}
	afterHotByResource := fileEngineResourceCalls()
	fakHotByResource := resourceDelta(afterHotByResource, beforeHotByResource)
	fakHotEngineCalls := fileEngineCalls() - beforeHotEngine

	out := parallelProof{
		Schema:                      "fak.tokendemo.parallel.v1",
		Path:                        cacheProofPathKernelSyscall,
		Workers:                     workers,
		Calls:                       calls,
		HotFiles:                    hotFiles,
		EngineDelayMs:               int(engineDelay / time.Millisecond),
		Prewarmed:                   true,
		RawWallNs:                   rawRun.wallNs,
		FakWarmupWallNs:             warmWall,
		FakHotWallNs:                fakRun.wallNs,
		RawTotalNs:                  sumNs(rawRun.durations),
		FakHotTotalNs:               sumNs(fakRun.durations),
		RawP50Ns:                    percentileNs(rawRun.durations, 50),
		RawP95Ns:                    percentileNs(rawRun.durations, 95),
		FakP50Ns:                    percentileNs(fakRun.durations, 50),
		FakP95Ns:                    percentileNs(fakRun.durations, 95),
		RawEngineCalls:              rawEngineCalls,
		FakWarmupEngineCalls:        warmEngineCalls,
		FakHotEngineCalls:           fakHotEngineCalls,
		VDSOHits:                    int64(countSource(fakRun.sources, "vdso_tier2")),
		ToolTokensFromCacheMeaning:  toolTokensFromCacheMeaning,
		ContextTokensKeptOutMeaning: contextTokensKeptOutMeaning,
		ClaimBoundary:               cacheClaimBoundary(cacheProofPathKernelSyscall, true),
	}
	out.TimeSavedNs = out.RawTotalNs - out.FakHotTotalNs
	out.EngineCallsAvoided = out.RawEngineCalls - out.FakWarmupEngineCalls - out.FakHotEngineCalls
	out.EngineCallAvoidedRate = ratio(out.EngineCallsAvoided, out.RawEngineCalls)
	out.PerResource = parallelResourceProofs(workload, rawRun, fakRun, rawByResource, warmByResource, fakHotByResource)
	for _, r := range out.PerResource {
		out.ToolTokensFromCache += r.ToolTokensFromCache
	}
	return out, nil
}

type parallelRun struct {
	durations []int64
	sources   []string
	wallNs    int64
}

func runParallelToolCalls(ctx context.Context, workers int, calls []turnbench.Call, fn func(context.Context, turnbench.Call) (string, error)) (parallelRun, error) {
	if workers > len(calls) {
		workers = len(calls)
	}
	out := parallelRun{
		durations: make([]int64, len(calls)),
		sources:   make([]string, len(calls)),
	}
	jobs := make(chan int)
	errs := make(chan error, len(calls))
	var wg sync.WaitGroup
	startWall := time.Now()
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for idx := range jobs {
				start := time.Now()
				source, err := fn(ctx, calls[idx])
				out.durations[idx] = elapsedNs(start)
				out.sources[idx] = source
				if err != nil {
					errs <- err
				}
			}
		}()
	}
	for i := range calls {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	out.wallNs = elapsedNs(startWall)
	close(errs)
	for err := range errs {
		if err != nil {
			return out, err
		}
	}
	return out, nil
}

func parallelWorkload(calls, hotFiles int) ([]turnbench.Call, []turnbench.Call) {
	hot := make([]turnbench.Call, 0, hotFiles)
	for i := 0; i < hotFiles; i++ {
		src := parallelHotFileCatalog[i]
		hot = append(hot, turnbench.Call{
			Tool: "read_file",
			Args: json.RawMessage(`{"path":"` + src.path + `"}`),
			Meta: map[string]string{
				"readOnlyHint":   "true",
				"idempotentHint": "true",
				"result_tokens":  strconv.Itoa(src.tokens),
			},
		})
	}
	work := make([]turnbench.Call, 0, calls)
	for i := 0; i < calls; i++ {
		work = append(work, hot[i%len(hot)])
	}
	return hot, work
}

func parallelResourceProofs(workload []turnbench.Call, rawRun, fakRun parallelRun, rawByResource, warmByResource, fakHotByResource map[string]int64) []parallelResourceProof {
	type agg struct {
		tokens       int
		rawDurations []int64
		fakDurations []int64
		vdsoHits     int64
		cacheTokens  int
	}
	aggs := map[string]*agg{}
	for i, c := range workload {
		resource := resourceLabel(c.Args)
		if resource == "" {
			resource = c.Tool
		}
		a := aggs[resource]
		if a == nil {
			a = &agg{tokens: resultTokens(c)}
			aggs[resource] = a
		}
		a.rawDurations = append(a.rawDurations, rawRun.durations[i])
		a.fakDurations = append(a.fakDurations, fakRun.durations[i])
		if fakRun.sources[i] == "vdso_tier2" {
			a.vdsoHits++
			a.cacheTokens += resultTokens(c)
		}
	}
	keys := make([]string, 0, len(aggs))
	for k := range aggs {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := make([]parallelResourceProof, 0, len(keys))
	for _, resource := range keys {
		a := aggs[resource]
		rawCalls := rawByResource[resource]
		warmCalls := warmByResource[resource]
		hotCalls := fakHotByResource[resource]
		avoided := rawCalls - warmCalls - hotCalls
		out = append(out, parallelResourceProof{
			Resource:              resource,
			ResultTokens:          a.tokens,
			RawEngineCalls:        rawCalls,
			FakWarmupEngineCalls:  warmCalls,
			FakHotEngineCalls:     hotCalls,
			VDSOHits:              a.vdsoHits,
			ToolTokensFromCache:   a.cacheTokens,
			RawP50Ns:              percentileNs(a.rawDurations, 50),
			RawP95Ns:              percentileNs(a.rawDurations, 95),
			FakP50Ns:              percentileNs(a.fakDurations, 50),
			FakP95Ns:              percentileNs(a.fakDurations, 95),
			EngineCallsAvoided:    avoided,
			EngineCallAvoidedRate: ratio(avoided, rawCalls),
		})
	}
	return out
}

func resourceDelta(after, before map[string]int64) map[string]int64 {
	out := map[string]int64{}
	for k, v := range after {
		out[k] = v - before[k]
	}
	for k, v := range before {
		if _, ok := out[k]; !ok && v != 0 {
			out[k] = -v
		}
	}
	return out
}

func countSource(sources []string, want string) int {
	n := 0
	for _, s := range sources {
		if s == want {
			n++
		}
	}
	return n
}

func traceToolCall(ctx context.Context, res abi.Resolver, c turnbench.Call) (*abi.ToolCall, error) {
	args := []byte(c.Args)
	if len(args) == 0 {
		args = []byte("{}")
	}
	ref, err := res.Put(ctx, args)
	if err != nil {
		return nil, err
	}
	return &abi.ToolCall{Tool: c.Tool, Args: ref, Meta: c.Meta}, nil
}

func timingClass(r *abi.Result, v abi.Verdict) (class, source, tier string) {
	if v.Kind == abi.VerdictDeny {
		return "deny", "fak_deny", ""
	}
	if v.By == "vdso" {
		if r != nil && r.Meta != nil {
			tier = r.Meta["tier"]
		}
		switch tier {
		case "2":
			return "vdso_dedup", "vdso_tier2", tier
		case "3":
			return "vdso_static", "vdso_tier3", tier
		default:
			if tier == "" {
				tier = "1"
			}
			return "vdso_pure", "vdso_tier" + tier, tier
		}
	}
	if r != nil && r.Meta != nil && r.Meta["engine"] != "" {
		return "pass", "engine", ""
	}
	return "pass", "unknown", ""
}

func resourceLabel(raw json.RawMessage) string {
	return resourceLabelBytes([]byte(raw))
}

func resourceLabelBytes(raw []byte) string {
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		return ""
	}
	for _, k := range []string{"path", "file_path", "filePath", "query", "q", "id"} {
		if v, ok := m[k]; ok {
			switch x := v.(type) {
			case string:
				return x
			case float64:
				return strconv.FormatFloat(x, 'f', -1, 64)
			}
		}
	}
	return ""
}

func toolCallArgsBytes(ctx context.Context, c *abi.ToolCall) []byte {
	if c == nil {
		return nil
	}
	if c.Args.Kind == abi.RefInline {
		return c.Args.Inline
	}
	if res := abi.ActiveResolver(); res != nil {
		if b, err := res.Resolve(ctx, c.Args); err == nil {
			return b
		}
	}
	return nil
}

func argsHash(raw json.RawMessage) string {
	b := []byte(raw)
	var v any
	if err := json.Unmarshal(raw, &v); err == nil {
		if canon, err := json.Marshal(v); err == nil {
			b = canon
		}
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])[:12]
}

func elapsedNs(start time.Time) int64 {
	if ns := time.Since(start).Nanoseconds(); ns > 0 {
		return ns
	}
	return 1
}

func sumNs(vals []int64) int64 {
	var total int64
	for _, v := range vals {
		total += v
	}
	return total
}

func percentileNs(vals []int64, pct int) int64 {
	if len(vals) == 0 {
		return 0
	}
	cp := append([]int64(nil), vals...)
	sort.Slice(cp, func(i, j int) bool { return cp[i] < cp[j] })
	if pct <= 0 {
		return cp[0]
	}
	if pct >= 100 {
		return cp[len(cp)-1]
	}
	idx := (len(cp) - 1) * pct / 100
	return cp[idx]
}

func ratio(num, den int64) float64 {
	if den <= 0 {
		return 0
	}
	return float64(num) / float64(den)
}

func buildServedProof(ctx context.Context, callsPerSurface int, engineDelay time.Duration) (servedProof, error) {
	if callsPerSurface <= 0 {
		return servedProof{}, fmt.Errorf("served calls must be positive")
	}
	const (
		tool     = "read_file"
		resource = "config.yaml"
	)
	args := json.RawMessage(`{"path":"` + resource + `"}`)
	surfaces := []string{"http", "mcp"}
	totalCalls := callsPerSurface * len(surfaces)

	configureFileWorld()
	res := abi.ActiveResolver()
	if res == nil {
		return servedProof{}, fmt.Errorf("no active Ref resolver registered")
	}
	prevDelay := setFileEngineDelay(engineDelay)
	defer setFileEngineDelay(prevDelay)

	rawTimes := make([]int64, 0, totalCalls)
	resetFileEngineStats()
	rawEngine := fileEngine{}
	rawCall := turnbench.Call{Tool: tool, Args: args}
	for i := 0; i < totalCalls; i++ {
		tc, err := traceToolCall(ctx, res, rawCall)
		if err != nil {
			return servedProof{}, err
		}
		start := time.Now()
		if _, err := rawEngine.Complete(ctx, tc); err != nil {
			return servedProof{}, err
		}
		rawTimes = append(rawTimes, elapsedNs(start))
	}
	rawEngineCalls := fileEngineCalls()

	resetFileEngineStats()
	vdso.Default.BumpWorld()
	ifc.Default.Reset("")
	srv, err := gateway.New(gateway.Config{
		EngineID: "localtools",
		Model:    "tokendemo-served",
		VDSO:     true,
		Logf:     func(string, ...any) {},
	})
	if err != nil {
		return servedProof{}, err
	}
	defer srv.Close()
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	client := &http.Client{Timeout: 5 * time.Second}
	before, err := servedMetrics(client, ts.URL)
	if err != nil {
		return servedProof{}, err
	}

	out := servedProof{
		Schema:           "fak.tokendemo.served.v1",
		Tool:             tool,
		Resource:         resource,
		Surfaces:         surfaces,
		CallsPerSurface:  callsPerSurface,
		TotalServedCalls: totalCalls,
		DistinctKeys:     1,
		EngineDelayMs:    int(engineDelay / time.Millisecond),
		RawEngineCalls:   rawEngineCalls,
		Calls:            make([]servedCall, 0, totalCalls),
	}

	idx := 0
	for _, surface := range surfaces {
		for i := 0; i < callsPerSurface; i++ {
			beforeEngine := fileEngineCalls()
			start := time.Now()
			resp, err := servedSyscall(ctx, client, ts.URL, surface, idx+1, gateway.SyscallRequest{
				Tool:      tool,
				Arguments: args,
				ReadOnly:  true,
				TraceID:   "tokendemo-served-" + surface,
			})
			servedElapsed := elapsedNs(start)
			if err != nil {
				return servedProof{}, err
			}
			afterEngine := fileEngineCalls()
			meta := cloneMeta(resp.Result)
			servedBy, tier := servedMetaSource(meta)
			row := servedCall{
				Index:             idx,
				Surface:           surface,
				Tool:              tool,
				Resource:          resource,
				ArgsHash:          argsHash(args),
				RawToolTimeNs:     rawTimes[idx],
				ServedTimeNs:      servedElapsed,
				Verdict:           resp.Verdict.Kind,
				ServedBy:          servedBy,
				Tier:              tier,
				EngineRanRaw:      true,
				EngineRanServed:   afterEngine > beforeEngine,
				ServedEngineDelta: afterEngine - beforeEngine,
				ResponseMeta:      meta,
			}
			if row.ServedBy == "vdso" && row.Tier == "2" {
				out.VDSOHits++
			}
			out.RawTotalNs += row.RawToolTimeNs
			out.ServedTotalNs += row.ServedTimeNs
			out.Calls = append(out.Calls, row)
			idx++
		}
	}

	after, err := servedMetrics(client, ts.URL)
	if err != nil {
		return servedProof{}, err
	}
	out.FakEngineCalls = fileEngineCalls()
	out.TimeSavedNs = out.RawTotalNs - out.ServedTotalNs
	out.RawP50Ns = percentileNs(rawTimes, 50)
	out.RawP95Ns = percentileNs(rawTimes, 95)
	servedTimes := make([]int64, 0, len(out.Calls))
	for _, c := range out.Calls {
		servedTimes = append(servedTimes, c.ServedTimeNs)
	}
	out.ServedP50Ns = percentileNs(servedTimes, 50)
	out.ServedP95Ns = percentileNs(servedTimes, 95)
	out.GatewayMetrics = servedMetricsEvidence{
		Before: before,
		After:  after,
		Delta:  after.sub(before),
	}
	if err := validateServedProof(out); err != nil {
		return out, err
	}
	return out, nil
}

func validateServedProof(p servedProof) error {
	if p.TotalServedCalls <= 1 {
		return fmt.Errorf("served proof needs at least two served calls, got %d", p.TotalServedCalls)
	}
	if p.RawEngineCalls != int64(p.TotalServedCalls) {
		return fmt.Errorf("raw engine calls = %d, want %d", p.RawEngineCalls, p.TotalServedCalls)
	}
	if p.FakEngineCalls != int64(p.DistinctKeys) {
		return fmt.Errorf("fak engine calls = %d, want %d distinct key(s)", p.FakEngineCalls, p.DistinctKeys)
	}
	wantHits := int64(p.TotalServedCalls - p.DistinctKeys)
	if p.VDSOHits != wantHits {
		return fmt.Errorf("response tier-2 vDSO hits = %d, want %d", p.VDSOHits, wantHits)
	}
	if p.GatewayMetrics.Delta.VDSOHits != wantHits {
		return fmt.Errorf("/metrics fak_vdso_hits_total delta = %d, want %d", p.GatewayMetrics.Delta.VDSOHits, wantHits)
	}
	if p.GatewayMetrics.Delta.VDSOLookups != int64(p.TotalServedCalls) {
		return fmt.Errorf("/metrics fak_vdso_lookups_total delta = %d, want %d", p.GatewayMetrics.Delta.VDSOLookups, p.TotalServedCalls)
	}
	if p.GatewayMetrics.Delta.VDSOFills != int64(p.DistinctKeys) {
		return fmt.Errorf("/metrics fak_vdso_cache_fills_total delta = %d, want %d", p.GatewayMetrics.Delta.VDSOFills, p.DistinctKeys)
	}
	if p.GatewayMetrics.Delta.TurnsSavedVDSO != wantHits {
		return fmt.Errorf("/metrics fak_gateway_turns_saved_total{mechanism=vdso_dedup} delta = %d, want %d", p.GatewayMetrics.Delta.TurnsSavedVDSO, wantHits)
	}
	if p.GatewayMetrics.Delta.GatewaySyscalls != int64(p.TotalServedCalls) {
		return fmt.Errorf("/metrics syscall operation delta = %d, want %d", p.GatewayMetrics.Delta.GatewaySyscalls, p.TotalServedCalls)
	}
	if p.GatewayMetrics.Delta.HTTPSyscallRequests != int64(p.CallsPerSurface) {
		return fmt.Errorf("/metrics HTTP syscall request delta = %d, want %d", p.GatewayMetrics.Delta.HTTPSyscallRequests, p.CallsPerSurface)
	}
	if p.GatewayMetrics.Delta.MCPRequests != int64(p.CallsPerSurface) {
		return fmt.Errorf("/metrics MCP request delta = %d, want %d", p.GatewayMetrics.Delta.MCPRequests, p.CallsPerSurface)
	}
	for i, c := range p.Calls {
		if c.Verdict != "ALLOW" {
			return fmt.Errorf("served call %d verdict = %q, want ALLOW", i, c.Verdict)
		}
		if i == 0 {
			if !c.EngineRanServed || c.ServedBy == "vdso" {
				return fmt.Errorf("first served call should run the engine, got served_by=%q engine_delta=%d", c.ServedBy, c.ServedEngineDelta)
			}
			continue
		}
		if c.EngineRanServed || c.ServedEngineDelta != 0 || c.ServedBy != "vdso" || c.Tier != "2" {
			return fmt.Errorf("served call %d should be a tier-2 vDSO hit with no engine run, got served_by=%q tier=%q engine_delta=%d", i, c.ServedBy, c.Tier, c.ServedEngineDelta)
		}
	}
	return nil
}

func servedSyscall(ctx context.Context, client *http.Client, base, surface string, id int, req gateway.SyscallRequest) (gateway.SyscallResponse, error) {
	switch surface {
	case "http":
		return servedHTTPSyscall(ctx, client, base, req)
	case "mcp":
		return servedMCPSyscall(ctx, client, base, id, req)
	default:
		return gateway.SyscallResponse{}, fmt.Errorf("unknown served surface %q", surface)
	}
}

func servedHTTPSyscall(ctx context.Context, client *http.Client, base string, req gateway.SyscallRequest) (gateway.SyscallResponse, error) {
	var out gateway.SyscallResponse
	body, _ := json.Marshal(req)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/v1/fak/syscall", bytes.NewReader(body))
	if err != nil {
		return out, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(httpReq)
	if err != nil {
		return out, err
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return out, fmt.Errorf("POST /v1/fak/syscall = %d: %s", resp.StatusCode, string(b))
	}
	if err := json.Unmarshal(b, &out); err != nil {
		return out, err
	}
	return out, nil
}

func servedMCPSyscall(ctx context.Context, client *http.Client, base string, id int, req gateway.SyscallRequest) (gateway.SyscallResponse, error) {
	var out gateway.SyscallResponse
	frame := map[string]any{
		"jsonrpc": "2.0",
		"id":      id,
		"method":  "tools/call",
		"params": map[string]any{
			"name":      "fak_syscall",
			"arguments": req,
		},
	}
	body, _ := json.Marshal(frame)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/mcp", bytes.NewReader(body))
	if err != nil {
		return out, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(httpReq)
	if err != nil {
		return out, err
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return out, fmt.Errorf("POST /mcp = %d: %s", resp.StatusCode, string(b))
	}
	var rpc struct {
		Result struct {
			Content []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
			IsError bool `json:"isError"`
		} `json:"result"`
		Error *struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
		} `json:"error,omitempty"`
	}
	if err := json.Unmarshal(b, &rpc); err != nil {
		return out, err
	}
	if rpc.Error != nil {
		return out, fmt.Errorf("MCP error %d: %s", rpc.Error.Code, rpc.Error.Message)
	}
	if rpc.Result.IsError {
		return out, fmt.Errorf("MCP tool result isError=true")
	}
	if len(rpc.Result.Content) == 0 {
		return out, fmt.Errorf("MCP tool result had no content")
	}
	if err := json.Unmarshal([]byte(rpc.Result.Content[0].Text), &out); err != nil {
		return out, err
	}
	return out, nil
}

func cloneMeta(env *gateway.ResultEnvelope) map[string]string {
	if env == nil || len(env.Meta) == 0 {
		return nil
	}
	out := make(map[string]string, len(env.Meta))
	for k, v := range env.Meta {
		out[k] = v
	}
	return out
}

func servedMetaSource(meta map[string]string) (servedBy, tier string) {
	if meta == nil {
		return "missing", ""
	}
	if by := meta["served_by"]; by != "" {
		return by, meta["tier"]
	}
	if meta["engine"] != "" {
		return "engine", ""
	}
	return "unknown", meta["tier"]
}

func servedMetrics(client *http.Client, base string) (servedMetricsSnapshot, error) {
	resp, err := client.Get(base + "/metrics")
	if err != nil {
		return servedMetricsSnapshot{}, err
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return servedMetricsSnapshot{}, fmt.Errorf("GET /metrics = %d: %s", resp.StatusCode, string(b))
	}
	text := string(b)
	return servedMetricsSnapshot{
		VDSOLookups:         sumMetricInt(text, "fak_vdso_lookups_total"),
		VDSOHits:            sumMetricInt(text, "fak_vdso_hits_total"),
		VDSOFills:           sumMetricInt(text, "fak_vdso_cache_fills_total"),
		GatewaySyscalls:     sumMetricInt(text, "fak_gateway_operations_total", `operation="syscall"`),
		HTTPSyscallRequests: sumMetricInt(text, "fak_gateway_http_requests_total", `route="/v1/fak/syscall"`, `method="POST"`, `status="200"`),
		MCPRequests:         sumMetricInt(text, "fak_gateway_http_requests_total", `route="/mcp"`, `method="POST"`, `status="200"`),
		TurnsSavedVDSO:      sumMetricInt(text, "fak_gateway_turns_saved_total", `mechanism="vdso_dedup"`),
	}, nil
}

func (s servedMetricsSnapshot) sub(prev servedMetricsSnapshot) servedMetricsSnapshot {
	return servedMetricsSnapshot{
		VDSOLookups:         s.VDSOLookups - prev.VDSOLookups,
		VDSOHits:            s.VDSOHits - prev.VDSOHits,
		VDSOFills:           s.VDSOFills - prev.VDSOFills,
		GatewaySyscalls:     s.GatewaySyscalls - prev.GatewaySyscalls,
		HTTPSyscallRequests: s.HTTPSyscallRequests - prev.HTTPSyscallRequests,
		MCPRequests:         s.MCPRequests - prev.MCPRequests,
		TurnsSavedVDSO:      s.TurnsSavedVDSO - prev.TurnsSavedVDSO,
	}
}

func sumMetricInt(text, name string, labelNeedles ...string) int64 {
	var total int64
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if len(labelNeedles) == 0 {
			if !strings.HasPrefix(line, name+" ") {
				continue
			}
		} else if !strings.HasPrefix(line, name+"{") {
			continue
		}
		matches := true
		for _, needle := range labelNeedles {
			if !strings.Contains(line, needle) {
				matches = false
				break
			}
		}
		if !matches {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) == 0 {
			continue
		}
		v, err := strconv.ParseFloat(fields[len(fields)-1], 64)
		if err == nil {
			total += int64(v)
		}
	}
	return total
}

// ---------------------------------------------------------------------------
// fixture resolution (mirrors cmd/turntaxdemo's turnTaxDir climb).
// ---------------------------------------------------------------------------

func tokenDir() string {
	cands := []string{
		filepath.Join("testdata", "tokendemo"),
		filepath.Join("..", "..", "testdata", "tokendemo"),
	}
	if exe, err := os.Executable(); err == nil {
		cands = append(cands, filepath.Join(filepath.Dir(exe), "testdata", "tokendemo"))
	}
	for _, d := range cands {
		if _, err := os.Stat(d); err == nil {
			return d
		}
	}
	if wd, err := os.Getwd(); err == nil {
		for d := wd; ; {
			cand := filepath.Join(d, "testdata", "tokendemo")
			if _, err := os.Stat(cand); err == nil {
				return cand
			}
			parent := filepath.Dir(d)
			if parent == d {
				break
			}
			d = parent
		}
	}
	return cands[0]
}

func suitePath(suite string) string { return filepath.Join(tokenDir(), suite+".json") }

func itoa(n int) string { return strconv.Itoa(n) }
