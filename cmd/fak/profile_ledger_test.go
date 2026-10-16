package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A single contract witness covers the accounting seam, adversarial evidence,
// bounded read-only CLI and content-free output. It launches no workload.
// fak-test:runtime fast est=50ms lane=default
func TestProfileLedgerContract(t *testing.T) {
	const fixture = `{
	 "schema":"fak-profile-ledger-bundle/1",
	 "pins_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
	 "process_before_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
	 "process_after_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
	 "isolated":true,"client_wall_ns":120000000,
	 "response":{"choices":[{"message":{"content":"PRIVATE_CONTENT"}}],
	  "usage":{"prompt_tokens":10,"completion_tokens":3},
	  "fak":{"native_inference_receipt":{"prefill_seconds":0.04,"decode_seconds":0.02,"ttft_seconds":0.045,
	   "native_cache_accounting":{"total_prompt_tokens":10,"restored_tokens":0,"computed_tokens":10}},
	   "decode_trace":{"schema":"fak.native-decode-trace/1","engine":"fak-native","events":[
	    {"token_index":1,"elapsed_ns":1000000,"forward":{"kind":"session_step","duration_ns":3000000,"active_lanes":1}},
	    {"token_index":2,"elapsed_ns":5000000,"forward":{"kind":"session_step","duration_ns":3000000,"active_lanes":1}},
	    {"token_index":3,"elapsed_ns":9000000}]}}},
	 "engine_before":{"schema":"fak-observation-engine/1","native":true,"engine":{
	  "schema":"fak-engine-steps/1","now_unix_nano":1000000000,"requests_active":0,
	  "phases":{"request":{"count":1,"sum_seconds":0.1},"prefill":{"count":1,"sum_seconds":0.04},"decode":{"count":1,"sum_seconds":0.02}},
	  "recent":[{"seq":1,"kind":"phase","phase":"request","at_unix_nano":900000000,"duration_ns":100000000}]}},
	 "engine_after":{"schema":"fak-observation-engine/1","native":true,"engine":{
	  "schema":"fak-engine-steps/1","now_unix_nano":1200000000,"requests_active":0,
	  "phases":{"request":{"count":2,"sum_seconds":0.2},"prefill":{"count":2,"sum_seconds":0.08},"decode":{"count":2,"sum_seconds":0.04}},
	  "recent":[{"seq":2,"kind":"prefill_chunk","at_unix_nano":1050000000,"duration_ns":40000000,"tokens":10},
	    {"seq":3,"kind":"phase","phase":"request","at_unix_nano":1100000000,"duration_ns":100000000}]}},
	 "debug_before":{"moe_residency":{"page_in_bytes":100}},
	 "debug_after":{"moe_residency":{"page_in_bytes":200}}
	}`
	cases := []struct{ name, from, to, status, code string }{
		{"valid_nested", "", "", "PARTIAL", "full_receipt_adds_vocabulary_scan"},
		{"forward_exceeds_interval", `"duration_ns":3000000`, `"duration_ns":5000000`, "INCONCLUSIVE", "forward_outside_commit_interval"},
		{"duplicate_token", `"token_index":2`, `"token_index":1`, "INCONCLUSIVE", "invalid_trace_sequence"},
		{"missing_duration", `"duration_ns":3000000`, `"ignored_duration_ns":3000000`, "INCONCLUSIVE", "invalid_forward_timing"},
		{"overflow", `"elapsed_ns":9000000`, `"elapsed_ns":9223372036854775808`, "INCONCLUSIVE", "invalid_bundle"},
		{"duplicate_json_key", `"isolated":true`, `"isolated":true,"isolated":false`, "INCONCLUSIVE", "invalid_bundle"},
		{"batched_forward", `"active_lanes":1`, `"active_lanes":2`, "PARTIAL", "forward_coverage_not_single_lane"},
		{"missing_counter", `"page_in_bytes":200`, `"unknown":200`, "PARTIAL", "full_receipt_adds_vocabulary_scan"},
		{"counter_reset", `"page_in_bytes":200`, `"page_in_bytes":50`, "INCONCLUSIVE", "debug_counter_reset_or_invalid"},
		{"counter_overflow", `"page_in_bytes":200`, `"page_in_bytes":9223372036854775808`, "INCONCLUSIVE", "invalid_debug_counter_number"},
		{"phase_float_boundary", `"sum_seconds":0.2`, `"sum_seconds":9223372036.854776`, "INCONCLUSIVE", "engine_phase_counter_invalid"},
		{"ring_gap", `"seq":2`, `"seq":5`, "INCONCLUSIVE", "invalid_engine_record"},
		{"filtered_ring", `"seq":3`, `"seq":4`, "PARTIAL", "engine_ring_gap_or_no_new_records"},
		{"missing_active", `"requests_active":0`, `"unknown_active":0`, "PARTIAL", "missing_engine_window_fields"},
		{"missing_prompt", `"prompt_tokens":10`, `"unknown_prompt_tokens":10`, "PARTIAL", "missing_prompt_count"},
		{"negative_prompt", `"prompt_tokens":10`, `"prompt_tokens":-1`, "INCONCLUSIVE", "negative_prompt_count"},
		{"missing_sum", `"sum_seconds":0.04`, `"unknown_sum":0.04`, "PARTIAL", "missing_engine_phase_counter"},
		{"expert_trace", `"isolated":true`, `"isolated":true,"expert_trace":true`, "PARTIAL", "expert_trace_changes_cold_router_path"},
		{"unbound", `"isolated":true`, `"isolated":false`, "PARTIAL", "unbound_or_nonisolated_snapshots"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			in := fixture
			if tc.from != "" {
				in = strings.Replace(in, tc.from, tc.to, 1)
			}
			r := buildProfileLedger([]byte(in))
			if r.Status != tc.status || !strings.Contains(strings.Join(r.Issues, ","), tc.code) {
				t.Fatalf("status/issues = %s/%v", r.Status, r.Issues)
			}
			out, err := json.Marshal(r)
			if err != nil {
				t.Fatal(err)
			}
			if bytes.Contains(out, []byte("PRIVATE_CONTENT")) || r.UnattributedWallNS != nil {
				t.Fatal("leaked content or invented exclusive wall residual")
			}
			if tc.name == "valid_nested" {
				if r.Decode.Status != "COMPLETE_COMMITTED_SPAN" || *r.Decode.WallNS != 8000000 || *r.Decode.ForwardNS != 6000000 || *r.Decode.RemainderNS != 2000000 || *r.Decode.MeanMS != 4 {
					t.Fatalf("decode = %+v", r.Decode)
				}
				if *r.PrefillMSPerToken != 4 || len(r.PrefillChunks) != 1 || r.PrefillChunks[0].Tokens != 10 || r.PrefillChunks[0].Lanes != nil || !strings.HasPrefix(r.EngineRingCoverage, "contiguous_") {
					t.Fatalf("prefill/ring = %+v", r)
				}
			}
			if tc.status == "INCONCLUSIVE" && (r.Decode.MeanMS != nil || r.PrefillMSPerToken != nil) {
				t.Fatal("contradiction emitted derived rate")
			}
			if tc.name == "batched_forward" && r.Decode.ForwardNS != nil {
				t.Fatal("shared batch time assigned exclusively")
			}
			if tc.name == "missing_counter" {
				for _, c := range r.Counters {
					if c.Metric == "page_in_bytes" && c.Delta != nil {
						t.Fatal("missing counter turned into zero")
					}
				}
			}
			if tc.name == "missing_active" && r.EngineRingCoverage != "unknown" {
				t.Fatal("missing activity asserted isolation")
			}
			if tc.name == "missing_sum" {
				for _, c := range r.Counters {
					if c.Metric == "engine_phase_wall" && c.Phase == "prefill" && c.Delta != nil {
						t.Fatal("missing timer became zero")
					}
				}
			}
		})
	}
	t.Run("historical_partial", func(t *testing.T) {
		in := `{"schema":"fak-profile-ledger-bundle/1","historical_modelbench":{"schema":"fak.strix-fullmodel-witness-receipt/1","cold_prefill_report":{"smoke_prefill_tokens":512,"smoke_prefill_ms":224628.245012,"smoke_prefill_kv_reused_tokens":0,"smoke_decode_ms":1005.655237,"smoke_prefill_backend_execution":{"status":"observed","counters":{"h2d_bytes":139251104832,"dispatch_submits":678264}}}}}`
		r := buildProfileLedger([]byte(in))
		dispatches, h2dBytes := int64(678264), int64(139251104832)
		wantCounters := []profileLedgerCounter{
			{Phase: "prefill", Metric: "compute_dispatches", Unit: "count"},
			{Phase: "prefill", Metric: "dispatch_submits", Unit: "count", Delta: &dispatches},
			{Phase: "prefill", Metric: "h2d_bytes", Unit: "bytes", Delta: &h2dBytes},
			{Phase: "prefill", Metric: "d2h_bytes", Unit: "bytes"},
			{Phase: "prefill", Metric: "d2d_bytes", Unit: "bytes"},
		}
		if r.Status != "PARTIAL" || r.EvidenceKind != "historical_modelbench" || r.Decode.WallNS != nil || len(r.Counters) != len(wantCounters) || *r.PrefillMS != 224628.245012 {
			t.Fatalf("historical report = %+v", r)
		}
		for i, want := range wantCounters {
			got := r.Counters[i]
			if got.Phase != want.Phase || got.Metric != want.Metric || got.Unit != want.Unit || (got.Delta == nil) != (want.Delta == nil) {
				t.Fatalf("historical counter %d = %+v, want %+v", i, got, want)
			}
			if want.Delta != nil && *got.Delta != *want.Delta {
				t.Fatalf("historical counter %q = %d, want %d", want.Metric, *got.Delta, *want.Delta)
			}
		}
		bad := strings.Replace(in, `"schema":"fak-profile-ledger-bundle/1"`, `"schema":"fak-profile-ledger-bundle/1","client_wall_ns":-1`, 1)
		r = buildProfileLedger([]byte(bad))
		if r.Status != "INCONCLUSIVE" || r.PrefillMSPerToken != nil {
			t.Fatal("early historical return retained derived rate after contradiction")
		}
	})
	t.Run("cli", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "bundle.json")
		if err := os.WriteFile(path, []byte(fixture), 0600); err != nil {
			t.Fatal(err)
		}
		var out, stderr bytes.Buffer
		if code := runProfileLedger(&out, &stderr, []string{"--bundle", path}); code != 0 || stderr.Len() != 0 || !json.Valid(out.Bytes()) {
			t.Fatalf("CLI code=%d stderr=%s", code, stderr.String())
		}
		out.Reset()
		stderr.Reset()
		if runProfileLedger(&out, &stderr, []string{"--PRIVATE_CONTENT"}) != 2 || strings.Contains(stderr.String(), "PRIVATE_CONTENT") {
			t.Fatal("argument echo leaked content")
		}
		out.Reset()
		stderr.Reset()
		if runProfileLedger(&out, &stderr, []string{"--help"}) != 0 || !strings.Contains(out.String(), "No workload") {
			t.Fatal("missing bounded CLI help")
		}
	})
}
