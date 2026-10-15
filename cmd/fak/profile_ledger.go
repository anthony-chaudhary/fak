package main

// The ledger consumes saved observations; it neither starts a workload nor enables
// instrumentation. Its output vocabulary is closed so captured content cannot leak
// through model names, paths, labels, tokens, errors, or response text.
import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
)

const profileLedgerMaxBytes = 32 << 20

type profileLedgerBundle struct {
	Schema              string                 `json:"schema"`
	PinsSHA256          string                 `json:"pins_sha256"`
	ProcessBeforeSHA256 string                 `json:"process_before_sha256"`
	ProcessAfterSHA256  string                 `json:"process_after_sha256"`
	Isolated            bool                   `json:"isolated"`
	ClientWallNS        *int64                 `json:"client_wall_ns"`
	ExpertTrace         bool                   `json:"expert_trace"`
	GPUProfile          bool                   `json:"gpu_profile"`
	CPUProfile          bool                   `json:"cpu_profile"`
	Response            *profileLedgerResponse `json:"response"`
	EngineBefore        *profileLedgerEngine   `json:"engine_before"`
	EngineAfter         *profileLedgerEngine   `json:"engine_after"`
	DebugBefore         json.RawMessage        `json:"debug_before"`
	DebugAfter          json.RawMessage        `json:"debug_after"`
	Modelbench          *struct {
		Schema string `json:"schema"`
		Cold   *struct {
			Tokens   *int     `json:"smoke_prefill_tokens"`
			MS       *float64 `json:"smoke_prefill_ms"`
			Restored *int     `json:"smoke_prefill_kv_reused_tokens"`
			Backend  *struct {
				Status   string            `json:"status"`
				Counters map[string]*int64 `json:"counters"`
			} `json:"smoke_prefill_backend_execution"`
		} `json:"cold_prefill_report"`
	} `json:"historical_modelbench"`
}

type profileLedgerResponse struct {
	Usage struct {
		PromptTokens     *int `json:"prompt_tokens"`
		CompletionTokens *int `json:"completion_tokens"`
	} `json:"usage"`
	Fak struct {
		Receipt *struct {
			PrefillSeconds   *float64 `json:"prefill_seconds"`
			DecodeSeconds    *float64 `json:"decode_seconds"`
			TTFTSeconds      *float64 `json:"ttft_seconds"`
			PrefixAccounting *struct {
				Total    *int `json:"total_prompt_tokens"`
				Restored *int `json:"restored_tokens"`
				Computed *int `json:"computed_tokens"`
			} `json:"native_cache_accounting"`
		} `json:"native_inference_receipt"`
		Trace *struct {
			Schema string `json:"schema"`
			Engine string `json:"engine"`
			Events []struct {
				TokenIndex int    `json:"token_index"`
				ElapsedNS  *int64 `json:"elapsed_ns"`
				Forward    *struct {
					Kind        string `json:"kind"`
					DurationNS  *int64 `json:"duration_ns"`
					ActiveLanes int    `json:"active_lanes"`
				} `json:"forward"`
			} `json:"events"`
		} `json:"decode_trace"`
	} `json:"fak"`
}

type profileLedgerEngine struct {
	Schema string `json:"schema"`
	Native bool   `json:"native"`
	Engine struct {
		Schema         string `json:"schema"`
		NowUnixNano    *int64 `json:"now_unix_nano"`
		RequestsActive *int64 `json:"requests_active"`
		Phases         map[string]struct {
			Count      *uint64  `json:"count"`
			SumSeconds *float64 `json:"sum_seconds"`
		} `json:"phases"`
		Recent []struct {
			Seq        *uint64 `json:"seq"`
			AtUnixNano *int64  `json:"at_unix_nano"`
			Kind       string  `json:"kind"`
			Tokens     *int    `json:"tokens"`
			Lanes      *int    `json:"lanes"`
			DurationNS *int64  `json:"duration_ns"`
		} `json:"recent"`
	} `json:"engine"`
}

type profileLedgerDecode struct {
	Status      string   `json:"status"`
	Intervals   int      `json:"committed_intervals"`
	WallNS      *int64   `json:"committed_span_ns"`
	ForwardNS   *int64   `json:"forward_path_wall_ns"`
	RemainderNS *int64   `json:"nonforward_remainder_ns"`
	MeanMS      *float64 `json:"ms_per_committed_interval"`
}

type profileLedgerCounter struct {
	Phase  string `json:"phase"`
	Metric string `json:"metric"`
	Unit   string `json:"unit"`
	Delta  *int64 `json:"delta"`
}

type profileLedgerChunk struct {
	Tokens     int   `json:"tokens"`
	Lanes      *int  `json:"lanes"`
	DurationNS int64 `json:"duration_ns"`
}

type profileLedgerReport struct {
	Schema             string                 `json:"schema"`
	Status             string                 `json:"status"`
	BundleSHA256       string                 `json:"bundle_sha256"`
	PinsSHA256         string                 `json:"pins_sha256,omitempty"`
	Issues             []string               `json:"issues"`
	Provenance         string                 `json:"provenance"`
	ClientWallNS       *int64                 `json:"client_wall_ns"`
	UnattributedWallNS *int64                 `json:"unattributed_request_wall_ns"`
	PrefillMS          *float64               `json:"prefill_ms"`
	PrefillMSPerToken  *float64               `json:"prefill_ms_per_input_token"`
	PrefixState        string                 `json:"prefix_state"`
	EvidenceKind       string                 `json:"evidence_kind"`
	DecodePhaseMS      *float64               `json:"decode_phase_ms"`
	TTFTMS             *float64               `json:"engine_ttft_ms"`
	Decode             profileLedgerDecode    `json:"decode"`
	EngineRingCoverage string                 `json:"engine_ring_coverage"`
	PrefillChunks      []profileLedgerChunk   `json:"prefill_chunks"`
	Counters           []profileLedgerCounter `json:"overlapping_counter_deltas"`
	Missing            []string               `json:"missing_attribution"`
}

func runProfileLedger(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("fak profile ledger", flag.ContinueOnError)
	// flag errors can quote an argument containing captured content.
	fs.SetOutput(io.Discard)
	path := fs.String("bundle", "", "saved JSON bundle")
	if err := fs.Parse(args); err != nil {
		if err == flag.ErrHelp {
			fmt.Fprintln(stdout, "fak profile ledger --bundle FILE\nRead-only, JSON output. Bundle schema: fak-profile-ledger-bundle/1. Embed response, engine_before/after and debug_before/after JSON; pins_sha256 and process_before/after_sha256 bind the caller's capture manifest. Optional isolated, client_wall_ns, expert_trace, cpu_profile and gpu_profile. No workload is launched. Output remains PARTIAL when cross-clock or resource attribution is absent.")
			return 0
		}
		fmt.Fprintln(stderr, "profile ledger: invalid arguments")
		return 2
	}
	if *path == "" || fs.NArg() != 0 {
		fmt.Fprintln(stderr, "profile ledger: --bundle FILE required")
		return 2
	}
	info, err := os.Stat(*path)
	if err != nil || !info.Mode().IsRegular() || info.Size() > profileLedgerMaxBytes {
		fmt.Fprintln(stderr, "profile ledger: bundle must be a regular file of at most 32 MiB")
		return 2
	}
	f, err := os.Open(*path)
	if err != nil {
		fmt.Fprintln(stderr, "profile ledger: cannot read bundle")
		return 2
	}
	defer f.Close()
	raw, err := io.ReadAll(io.LimitReader(f, profileLedgerMaxBytes+1))
	if err != nil || len(raw) > profileLedgerMaxBytes {
		fmt.Fprintln(stderr, "profile ledger: cannot read bounded bundle")
		return 2
	}
	report := buildProfileLedger(raw)
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if enc.Encode(report) != nil {
		return 2
	}
	if report.Status == "INCONCLUSIVE" {
		return 1
	}
	return 0
}

func buildProfileLedger(raw []byte) (r profileLedgerReport) {
	digest := sha256.Sum256(raw)
	r = profileLedgerReport{
		Schema: "fak-profile-ledger/1", Status: "PARTIAL", BundleSHA256: hex.EncodeToString(digest[:]),
		Provenance: "caller_asserted_capture_identity; no cross-clock synchronization", PrefixState: "unknown", EvidenceKind: "saved_native_chat",
		Issues: []string{}, Decode: profileLedgerDecode{Status: "PARTIAL"}, EngineRingCoverage: "unknown",
		PrefillChunks: []profileLedgerChunk{}, Counters: []profileLedgerCounter{},
		Missing: []string{"exclusive_request_boundaries", "cpu_on_off_cpu_intervals", "gpu_stage_intervals", "copy_wait_intervals", "unique_experts_on_unchanged_path", "request_memory_peaks", "profiling_control_delta", "os_cache_and_expert_ring_state"},
	}
	issue := func(code string, invalid bool) {
		r.Issues = append(r.Issues, code)
		if invalid {
			r.Status = "INCONCLUSIVE"
		}
	}
	defer func() {
		if r.Status == "INCONCLUSIVE" {
			// Applies to every return, including historical and malformed arms.
			r.PrefillMSPerToken = nil
			r.Decode = profileLedgerDecode{Status: "INCONCLUSIVE"}
		}
	}()
	var b profileLedgerBundle
	if len(raw) > profileLedgerMaxBytes || !profileLedgerUniqueJSON(raw) || json.Unmarshal(raw, &b) != nil || b.Schema != "fak-profile-ledger-bundle/1" {
		issue("invalid_bundle", true)
		return r
	}
	validDigest := func(s string) bool { v, e := hex.DecodeString(s); return e == nil && len(v) == 32 }
	if validDigest(b.PinsSHA256) {
		r.PinsSHA256 = b.PinsSHA256
	} else {
		issue("missing_pins_digest", false)
	}
	bound := validDigest(b.ProcessBeforeSHA256) && b.ProcessBeforeSHA256 == b.ProcessAfterSHA256 && b.Isolated
	if !bound {
		issue("unbound_or_nonisolated_snapshots", false)
	}
	if b.ProcessBeforeSHA256 != "" && b.ProcessAfterSHA256 != "" && b.ProcessBeforeSHA256 != b.ProcessAfterSHA256 {
		issue("process_identity_mismatch", true)
	}
	if b.ClientWallNS != nil {
		if *b.ClientWallNS < 0 {
			issue("negative_client_wall", true)
		} else {
			r.ClientWallNS = b.ClientWallNS
		}
	}
	if b.ExpertTrace {
		issue("expert_trace_changes_cold_router_path", false)
	}
	if b.GPUProfile {
		issue("gpu_timestamp_profile_perturbs_dispatches", false)
	}
	if b.CPUProfile {
		issue("cpu_profile_overhead_unmeasured", false)
	}
	if b.Modelbench != nil {
		r.EvidenceKind = "historical_modelbench"
		issue("historical_modelbench_not_current_native_serve", false)
		if b.Response != nil || b.EngineBefore != nil || b.EngineAfter != nil {
			issue("mixed_measurement_envelopes", true)
			return r
		}
		m := b.Modelbench
		if m.Schema != "fak.strix-fullmodel-witness-receipt/1" || m.Cold == nil {
			issue("invalid_modelbench_envelope", true)
			return r
		}
		c := m.Cold
		if c.MS == nil || c.Tokens == nil || *c.MS <= 0 || *c.MS > float64(math.MaxInt64)/1e6 || *c.Tokens <= 0 {
			issue("invalid_modelbench_prefill", true)
			return r
		}
		r.PrefillMS = c.MS
		mean := *c.MS / float64(*c.Tokens)
		r.PrefillMSPerToken = &mean
		if c.Restored != nil && *c.Restored == 0 {
			r.PrefixState = "prefix_cold_os_cache_unknown"
		}
		if c.Restored != nil && (*c.Restored < 0 || *c.Restored > *c.Tokens) {
			issue("invalid_modelbench_cache_count", true)
			return r
		}
		if c.Backend != nil && c.Backend.Status == "observed" {
			for _, metric := range []string{"compute_dispatches", "dispatch_submits", "h2d_bytes", "d2h_bytes", "d2d_bytes"} {
				value := c.Backend.Counters[metric]
				if value != nil && *value < 0 {
					issue("invalid_modelbench_counter", true)
					r.PrefillMSPerToken = nil
					return r
				}
				unit := "count"
				if len(metric) >= 5 && metric[len(metric)-5:] == "bytes" {
					unit = "bytes"
				}
				r.Counters = append(r.Counters, profileLedgerCounter{"prefill", metric, unit, value})
			}
		}
		issue("modelbench_counter_scope_requires_pinned_producer", false)
		issue("modelbench_smoke_is_not_steady_decode", false)
		return r
	}
	if b.Response == nil {
		issue("missing_response", false)
	} else {
		p := b.Response
		if p.Usage.PromptTokens == nil {
			issue("missing_prompt_count", false)
		} else if *p.Usage.PromptTokens < 0 {
			issue("negative_prompt_count", true)
		}
		if p.Fak.Receipt != nil {
			issue("full_receipt_adds_vocabulary_scan", false)
			ms := func(s *float64) *float64 {
				if s == nil {
					issue("missing_receipt_timing", false)
					return nil
				}
				if *s < 0 || math.IsNaN(*s) || math.IsInf(*s, 0) || *s > float64(math.MaxInt64)/1e9 {
					issue("invalid_receipt_timing", true)
					return nil
				}
				v := *s * 1000
				return &v
			}
			r.PrefillMS, r.DecodePhaseMS, r.TTFTMS = ms(p.Fak.Receipt.PrefillSeconds), ms(p.Fak.Receipt.DecodeSeconds), ms(p.Fak.Receipt.TTFTSeconds)
			if r.PrefillMS != nil && r.TTFTMS != nil && *r.TTFTMS < *r.PrefillMS {
				issue("ttft_precedes_prefill", true)
			}
			if r.PrefillMS != nil && r.DecodePhaseMS != nil && r.ClientWallNS != nil && *r.PrefillMS+*r.DecodePhaseMS > float64(*r.ClientWallNS)/1e6+0.001 {
				issue("native_phases_exceed_client_wall", true)
			}
			if r.TTFTMS != nil && r.ClientWallNS != nil && *r.TTFTMS > float64(*r.ClientWallNS)/1e6+0.001 {
				issue("ttft_exceeds_client_wall", true)
			}
			if r.PrefillMS != nil && p.Usage.PromptTokens != nil && *p.Usage.PromptTokens > 0 {
				v := *r.PrefillMS / float64(*p.Usage.PromptTokens)
				r.PrefillMSPerToken = &v
			}
			c := p.Fak.Receipt.PrefixAccounting
			if c == nil || c.Total == nil || c.Restored == nil || c.Computed == nil || p.Usage.PromptTokens == nil {
				issue("missing_prefix_cache_accounting", false)
			} else if *c.Total < 0 || *c.Restored < 0 || *c.Computed < 0 || *c.Restored > *c.Total || *c.Computed != *c.Total-*c.Restored || *c.Total != *p.Usage.PromptTokens {
				issue("invalid_prefix_cache_accounting", true)
			} else if *c.Restored == 0 {
				r.PrefixState = "prefix_cold_os_cache_unknown"
			} else {
				r.PrefixState = "prefix_reused_os_cache_unknown"
			}
		} else {
			issue("missing_native_receipt", false)
		}
		profileLedgerTrace(p, &r, issue)
	}
	profileLedgerRing(b.EngineBefore, b.EngineAfter, bound, &r, issue)
	if bound {
		profileLedgerCounters(b.DebugBefore, b.DebugAfter, &r, issue)
	}
	return r
}

func profileLedgerTrace(p *profileLedgerResponse, r *profileLedgerReport, issue func(string, bool)) {
	t := p.Fak.Trace
	if t == nil {
		issue("missing_decode_trace", false)
		return
	}
	issue("decode_trace_clock_overhead_unmeasured", false)
	if t.Schema != "fak.native-decode-trace/1" || t.Engine != "fak-native" || len(t.Events) > 32768 {
		issue("invalid_trace_envelope", true)
		return
	}
	if p.Usage.CompletionTokens == nil {
		issue("missing_completion_count", false)
		return
	}
	if len(t.Events) != *p.Usage.CompletionTokens {
		issue("trace_count_mismatch", true)
		return
	}
	complete := true
	var total int64
	for i, e := range t.Events {
		if e.TokenIndex != i+1 || e.ElapsedNS == nil || *e.ElapsedNS < 0 || (i > 0 && (t.Events[i-1].ElapsedNS == nil || *e.ElapsedNS <= *t.Events[i-1].ElapsedNS)) {
			issue("invalid_trace_sequence", true)
			return
		}
		if e.Forward != nil && (e.Forward.DurationNS == nil || *e.Forward.DurationNS <= 0 || e.Forward.ActiveLanes <= 0) {
			issue("invalid_forward_timing", true)
			return
		}
		if i+1 == len(t.Events) {
			if e.Forward != nil {
				issue("final_forward_outside_committed_span", false)
			}
			continue
		} // Final forward, if any, is outside committed span.
		next := t.Events[i+1].ElapsedNS
		if next == nil || *next <= *e.ElapsedNS {
			issue("invalid_trace_sequence", true)
			return
		}
		if e.Forward == nil {
			complete = false
			continue
		}
		f := e.Forward
		if f.DurationNS == nil || *f.DurationNS <= 0 || *f.DurationNS > *next-*e.ElapsedNS {
			issue("forward_outside_commit_interval", true)
			return
		}
		if f.Kind != "session_step" || f.ActiveLanes != 1 {
			complete = false
			continue
		}
		// Each duration is bounded by its disjoint interval; the total cannot overflow.
		total += *f.DurationNS
	}
	if len(t.Events) < 2 {
		issue("insufficient_committed_intervals", false)
		return
	}
	span := *t.Events[len(t.Events)-1].ElapsedNS - *t.Events[0].ElapsedNS
	if r.DecodePhaseMS != nil && float64(span)/1e6 > *r.DecodePhaseMS+0.000001 {
		issue("commit_span_exceeds_decode_phase", true)
		return
	}
	if r.ClientWallNS != nil && span > *r.ClientWallNS {
		issue("commit_span_exceeds_client_wall", true)
		return
	}
	r.Decode.Intervals, r.Decode.WallNS = len(t.Events)-1, &span
	mean := float64(span) / 1e6 / float64(r.Decode.Intervals)
	r.Decode.MeanMS = &mean
	if !complete {
		issue("forward_coverage_not_single_lane", false)
		return
	}
	rem := span - total
	r.Decode.Status, r.Decode.ForwardNS, r.Decode.RemainderNS = "COMPLETE_COMMITTED_SPAN", &total, &rem
}

func profileLedgerRing(before, after *profileLedgerEngine, bound bool, r *profileLedgerReport, issue func(string, bool)) {
	if before == nil || after == nil {
		issue("missing_engine_snapshots", false)
		return
	}
	for _, s := range []*profileLedgerEngine{before, after} {
		if s.Schema != "fak-observation-engine/1" || !s.Native || s.Engine.Schema != "fak-engine-steps/1" || len(s.Engine.Recent) > 512 {
			issue("invalid_engine_snapshot", true)
			return
		}
		if s.Engine.NowUnixNano == nil || s.Engine.RequestsActive == nil {
			issue("missing_engine_window_fields", false)
			return
		}
		var prev uint64
		for i, e := range s.Engine.Recent {
			if e.Seq == nil || e.DurationNS == nil || e.AtUnixNano == nil {
				issue("missing_engine_record_fields", false)
				return
			}
			if *e.Seq == 0 || (i > 0 && *e.Seq <= prev) || *e.DurationNS < 0 || *e.AtUnixNano <= 0 || (e.Tokens != nil && *e.Tokens < 0) || (e.Lanes != nil && *e.Lanes < 0) {
				issue("invalid_engine_record", true)
				return
			}
			prev = *e.Seq
		}
	}
	if *after.Engine.NowUnixNano <= *before.Engine.NowUnixNano {
		issue("engine_snapshot_clock_reversed", true)
		return
	}
	if len(before.Engine.Recent) == 0 {
		issue("unknown_engine_start_sequence", false)
		return
	}
	last := *before.Engine.Recent[len(before.Engine.Recent)-1].Seq
	newRecords := 0
	gap := false
	for _, e := range after.Engine.Recent {
		if *e.Seq <= last {
			continue
		}
		if *e.Seq-last != 1 {
			gap = true
		}
		last = *e.Seq
		newRecords++
		if e.Kind == "prefill_chunk" {
			if e.Tokens == nil {
				issue("missing_prefill_chunk_shape", false)
				continue
			}
			// Current ObservePrefillChunk emits tokens but no lanes. Preserve null;
			// a requested batch ceiling is not observed lane-count evidence.
			r.PrefillChunks = append(r.PrefillChunks, profileLedgerChunk{*e.Tokens, e.Lanes, *e.DurationNS})
		}
	}
	if gap || newRecords == 0 {
		issue("engine_ring_gap_or_no_new_records", false)
		return
	}
	if !bound || *before.Engine.RequestsActive != 0 || *after.Engine.RequestsActive != 0 {
		issue("engine_window_not_isolated", false)
		return
	}
	requestBefore, ok1 := before.Engine.Phases["request"]
	requestAfter, ok2 := after.Engine.Phases["request"]
	if !ok1 || !ok2 || requestBefore.Count == nil || requestAfter.Count == nil || *requestAfter.Count < *requestBefore.Count || *requestAfter.Count-*requestBefore.Count != 1 {
		issue("engine_request_count_not_one", false)
		return
	}
	r.EngineRingCoverage = "contiguous_under_declared_isolation; publication_clock_only"
	// These per-phase totals overlap. No interval union or wall residual is formed.
	for _, phase := range []string{"request", "prefill", "decode", "sample", "admission_wait", "device_wait", "cohort_wait", "prefix_lookup", "prefix_admit"} {
		a, okA := before.Engine.Phases[phase]
		z, okZ := after.Engine.Phases[phase]
		if !okA || !okZ || a.Count == nil || z.Count == nil || a.SumSeconds == nil || z.SumSeconds == nil {
			issue("missing_engine_phase_counter", false)
			continue
		}
		if *z.Count < *a.Count || *z.SumSeconds < *a.SumSeconds || *a.SumSeconds < 0 || *z.SumSeconds >= float64(math.MaxInt64)/1e9 {
			issue("engine_phase_counter_invalid", true)
			return
		}
		ns := int64((*z.SumSeconds - *a.SumSeconds) * 1e9)
		r.Counters = append(r.Counters, profileLedgerCounter{phase, "engine_phase_wall", "ns", &ns})
	}
}

func profileLedgerCounters(before, after json.RawMessage, r *profileLedgerReport, issue func(string, bool)) {
	if len(before) == 0 || len(after) == 0 {
		issue("missing_debug_counter_snapshots", false)
		return
	}
	read := func(raw json.RawMessage, keys ...string) *int64 {
		for _, key := range keys {
			var obj map[string]json.RawMessage
			if len(raw) == 0 || bytes.Equal(raw, []byte("null")) {
				return nil
			}
			if json.Unmarshal(raw, &obj) != nil {
				issue("invalid_debug_counter_object", true)
				return nil
			}
			raw = obj[key]
		}
		var n int64
		if len(raw) == 0 || bytes.Equal(raw, []byte("null")) {
			return nil
		}
		if json.Unmarshal(raw, &n) != nil {
			issue("invalid_debug_counter_number", true)
			return nil
		}
		return &n
	}
	add := func(phase, metric, unit string, path ...string) {
		a, z := read(before, path...), read(after, path...)
		var delta *int64
		if a != nil && z != nil {
			if *a < 0 || *z < *a {
				issue("debug_counter_reset_or_invalid", true)
			} else {
				n := *z - *a
				delta = &n
			}
		}
		r.Counters = append(r.Counters, profileLedgerCounter{phase, metric, unit, delta})
	}
	for _, metric := range []string{"page_in_bytes", "page_ins", "hits"} {
		unit := "count"
		if metric == "page_in_bytes" {
			unit = "bytes"
		}
		add("request_window", metric, unit, "moe_residency", metric)
	}
	for _, phase := range []string{"prefill", "decode"} {
		for _, metric := range []string{"tokens", "faulted_bytes", "fault_nanos", "dequant_bytes", "dequant_nanos", "contraction_nanos", "dense_projection_nanos", "dense_projection_activation_upload_bytes", "dense_projection_readback_bytes", "expert_activation_nanos", "expert_activation_readback_bytes"} {
			unit := "ns"
			if metric == "tokens" {
				unit = "tokens"
			} else if len(metric) >= 5 && metric[len(metric)-5:] == "bytes" {
				unit = "bytes"
			}
			add(phase, metric, unit, "moe_residency", "v41_phases", phase, metric)
		}
	}
}

// Reject duplicate object keys everywhere, including ignored captured response
// content. Encoding/json otherwise accepts ambiguous last-writer-wins evidence.
func profileLedgerUniqueJSON(raw []byte) bool {
	d := json.NewDecoder(bytes.NewReader(raw))
	d.UseNumber()
	var value func(int) bool
	value = func(depth int) bool {
		if depth > 64 {
			return false
		}
		t, e := d.Token()
		if e != nil {
			return false
		}
		delim, ok := t.(json.Delim)
		if !ok {
			return true
		}
		if delim != '{' && delim != '[' {
			return false
		}
		seen := map[string]bool{}
		for d.More() {
			if delim == '{' {
				k, e := d.Token()
				if e != nil {
					return false
				}
				s, ok := k.(string)
				if !ok || seen[s] {
					return false
				}
				seen[s] = true
			}
			if !value(depth + 1) {
				return false
			}
		}
		end, e := d.Token()
		return e == nil && ((delim == '{' && end == json.Delim('}')) || (delim == '[' && end == json.Delim(']')))
	}
	if !value(0) {
		return false
	}
	_, err := d.Token()
	return err == io.EOF
}
