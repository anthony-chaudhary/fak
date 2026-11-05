// Fixture tests for the decode-anatomy gate: one fixture per pinned corpus
// incident signature (fak-private selection note
// LAUNCH-OVERHEAD-PREVENTION-SELECTION-2026-11-04 §1), asserting the new gate
// flags its incident where no existing verb does. Assertions are closed
// contract only: exact status enums, exact failure-id sets, sentinel exit
// codes — never error prose or flat %+v.

package devcmd

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// fixtureEligibilityFile is a minimal internal/model/cuda_graph_eligibility.go
// carrying every captureEligibilityPins pattern exactly once, so the
// source-scan half of capture_eligibility_drift runs against a deterministic
// root instead of the live repository.
const fixtureEligibilityFile = `package model

const CUDAGraphNoCaptureBackend = "backend-has-no-graph-capture"

func qwen35HybridGraphEligible(cfg Config, be Backend) (bool, string) {
	return true, ""
}

func cudaGraphCaptureEligible(cfg Config, be Backend) (bool, string) {
	if _, ok := be.(graphBackend); !ok {
		return false, CUDAGraphNoCaptureBackend
	}
	if cfg.IsQwen35Hybrid() {
		return qwen35HybridGraphEligible(cfg, be)
	}
	return true, ""
}

func (s *Session) cudaGraphCaptureReady(isSplit bool) bool {
	if s == nil || s.halFailure != nil || isSplit {
		return false
	}
	if fixed, ok := s.halKV.(interface{ GraphCaptureReady() bool }); ok && !fixed.GraphCaptureReady() {
		return false
	}
	if s.cfg.HasQSASparseAttn() && s.halKV.Len()+1 >= QSADynamicGatingThreshold {
		return false
	}
	return true
}
`

// baseAnatomyRecord is a gateway-style perf record with a complete µs-witnessed
// kernel profile (100 launches/tok, host_share nested under native_timing so
// the hostshare reader reports UNKNOWN). extra injects top-level fields.
func baseAnatomyRecord(extra string) string {
	return `{"schema":"fak.gateway.perf-record.v1","unix_ms":1793703910451,"model":"fixture-model","backend":"vulkan","decode_tps":10.0` + extra +
		`,"native_timing":{"prefill_ms":100,"decode_ms":1000,"prefill_tokens":10,"decode_tokens":100,"decode_tps":10.0,"cpu_ms_per_token":5.0,"host_share":0.05,"kernels":{"source":"structured","backend":"vulkan","decode_steps":100,"gpu_ms_per_token":8.0,"wall_ms_per_token":10.0,"device_busy":0.9,"selected_coverage":0.95,"truncated":0,"untimed_compute":0,"untimed_transfer":0,"barriers_elided":0,"complete":true,"pipelining":"disabled-by-profile","selected_kernels":[{"kernel":"k_gemm","op":"matmul","groups":"1x1","calls_per_token":100.0,"gpu_ms_per_token":8.0,"us_per_call":80.0,"share":1.0}]}}}`
}

// writeAnatomyFixture materializes a receipts dir plus the fixture eligibility
// source tree under a fresh temp root; mutateEligibility may rewrite the
// eligibility file (nil = unmodified). It returns (receiptsDir, root).
func writeAnatomyFixture(t *testing.T, receipts map[string]string, mutateEligibility func(files map[string]string)) (string, string) {
	t.Helper()
	root := t.TempDir()
	files := map[string]string{"internal/model/cuda_graph_eligibility.go": fixtureEligibilityFile}
	if mutateEligibility != nil {
		mutateEligibility(files)
	}
	for rel, body := range files {
		full := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatalf("mkdir %s: %v", full, err)
		}
		if err := os.WriteFile(full, []byte(body), 0o644); err != nil {
			t.Fatalf("write %s: %v", full, err)
		}
	}
	recDir := filepath.Join(root, "receipts")
	if err := os.MkdirAll(recDir, 0o755); err != nil {
		t.Fatalf("mkdir receipts: %v", err)
	}
	for name, body := range receipts {
		if err := os.WriteFile(filepath.Join(recDir, name), []byte(body), 0o644); err != nil {
			t.Fatalf("write receipt %s: %v", name, err)
		}
	}
	return recDir, root
}

// runDecodeAnatomyJSON runs the verb with --json and decodes the pinned
// document from stdout.
func runDecodeAnatomyJSON(t *testing.T, argv ...string) (int, DecodeAnatomyResult, string) {
	t.Helper()
	var out, errb bytes.Buffer
	code := RunDecodeAnatomy(&out, &errb, append([]string{"--json"}, argv...))
	var res DecodeAnatomyResult
	if err := json.Unmarshal(out.Bytes(), &res); err != nil {
		t.Fatalf("unmarshal --json output: %v\nstdout=%s stderr=%s", err, out.String(), errb.String())
	}
	return code, res, errb.String()
}

func signalByID(res DecodeAnatomyResult, id string) (AnatomySignal, bool) {
	for _, s := range res.Signals {
		if s.ID == id {
			return s, true
		}
	}
	return AnatomySignal{}, false
}

// TestDecodeAnatomyBarrierStormElidedZeroWhileArmed proves corpus incident #1
// (TICKET-05t host-overhead-unattributed: barrier storm with inert elision).
// No existing verb catches it: hostshare reads host_share 0.04–0.08, below its
// 0.30 line; regime classifies UNKNOWN without gap/busy evidence; graph-audit
// and launch-census are static source scans that never see the runtime
// elision counter. The armed-but-zero-elided guard-drift assert is new here.
func TestDecodeAnatomyBarrierStormElidedZeroWhileArmed(t *testing.T) {
	// fak-test:runtime fast est=50ms lane=default
	recDir, root := writeAnatomyFixture(t, map[string]string{
		"storm.perf.jsonl": baseAnatomyRecord(`,"barriers_per_token":48.0,"hazards_armed":true`) + "\n",
	}, nil)
	code, res, errOut := runDecodeAnatomyJSON(t, "--receipts", recDir, "--root", root)
	sig, ok := signalByID(res, SignalBarriersElidedWhileArmed)
	if !ok {
		t.Fatalf("signal %s absent from document", SignalBarriersElidedWhileArmed)
	}
	if sig.Status != AnatomyFail {
		t.Fatalf("status = %s, want fail; detail=%s", sig.Status, sig.Detail)
	}
	if res.Verdict != "fail" {
		t.Fatalf("verdict = %s, want fail", res.Verdict)
	}
	if len(res.Failures) != 1 || res.Failures[0] != SignalBarriersElidedWhileArmed {
		t.Fatalf("failures = %v, want [%s]", res.Failures, SignalBarriersElidedWhileArmed)
	}
	if code != 1 {
		t.Fatalf("exit = %d, want 1; stderr=%s", code, errOut)
	}
}

// TestDecodeAnatomyHostShareBreach proves corpus incident #2 (PHL-096
// per-byte host conversion loop, host_share ≈0.5, ~2x the 0.30 line). hostshare
// catches this one today (exit 3); the fixture proves the fold into the
// unified gate while the other five signals stay pass/unknown.
func TestDecodeAnatomyHostShareBreach(t *testing.T) {
	// fak-test:runtime fast est=50ms lane=default
	recDir, root := writeAnatomyFixture(t, map[string]string{
		"phl096.perf.jsonl": baseAnatomyRecord(`,"host_share":0.5`) + "\n",
	}, nil)
	code, res, errOut := runDecodeAnatomyJSON(t, "--receipts", recDir, "--root", root)
	sig, ok := signalByID(res, SignalHostShare)
	if !ok {
		t.Fatalf("signal %s absent from document", SignalHostShare)
	}
	if sig.Status != AnatomyFail {
		t.Fatalf("status = %s, want fail; detail=%s", sig.Status, sig.Detail)
	}
	if sig.Value == nil || *sig.Value != 0.5 {
		t.Fatalf("value = %v, want 0.5", sig.Value)
	}
	if sig.Cap == nil || *sig.Cap != defaultHostShareLine {
		t.Fatalf("cap = %v, want %.2f", sig.Cap, defaultHostShareLine)
	}
	if res.Verdict != "fail" || len(res.Failures) != 1 || res.Failures[0] != SignalHostShare {
		t.Fatalf("verdict/failures = %s/%v, want fail/[host_share]", res.Verdict, res.Failures)
	}
	if code != 1 {
		t.Fatalf("exit = %d, want 1; stderr=%s", code, errOut)
	}
}

// TestDecodeAnatomyCaptureEligibilityFlipAndOOM proves corpus incident #3
// (PHL-161 CUDA-graph Instantiate OOM at VRAM fit). graph-audit is
// presence-only: it stays CAPTURED while the eligibility predicate silently
// flips and the fit plan leaves the prealloc uncharged; neither the
// QSA-threshold/KV-grow flip nor the Instantiate-OOM signature is visible to
// any existing verb.
func TestDecodeAnatomyCaptureEligibilityFlipAndOOM(t *testing.T) {
	// fak-test:runtime fast est=50ms lane=default
	recDir, root := writeAnatomyFixture(t, map[string]string{
		"phl161.perf.jsonl": baseAnatomyRecord("") + "\n" +
			`{"schema":"fak.serve.log-line.v1","message":"graph capture failed: Instantiate: out of memory"}` + "\n",
	}, func(files map[string]string) {
		// Flip the QSA-threshold guard out of the ready predicate.
		body := files["internal/model/cuda_graph_eligibility.go"]
		const old = "if s.cfg.HasQSASparseAttn() && s.halKV.Len()+1 >= QSADynamicGatingThreshold {"
		const neu = "if s.cfg.HasQSASparseAttn() && s.halKV.Len()+1 >= SomeOtherConstant {"
		if !containsOnce(body, old) {
			t.Fatalf("fixture eligibility file lost the QSA guard pin")
		}
		files["internal/model/cuda_graph_eligibility.go"] = replaceOnce(body, old, neu)
	})
	code, res, errOut := runDecodeAnatomyJSON(t, "--receipts", recDir, "--root", root)
	sig, ok := signalByID(res, SignalCaptureEligibilityDrift)
	if !ok {
		t.Fatalf("signal %s absent from document", SignalCaptureEligibilityDrift)
	}
	if sig.Status != AnatomyFail {
		t.Fatalf("status = %s, want fail; detail=%s", sig.Status, sig.Detail)
	}
	// One source-pin drift (the flipped QSA guard) + one OOM signature hit.
	if sig.Value == nil || *sig.Value != 2 {
		t.Fatalf("value = %v, want 2 findings", sig.Value)
	}
	if res.Verdict != "fail" || len(res.Failures) != 1 || res.Failures[0] != SignalCaptureEligibilityDrift {
		t.Fatalf("verdict/failures = %s/%v, want fail/[capture_eligibility_drift]", res.Verdict, res.Failures)
	}
	if code != 1 {
		t.Fatalf("exit = %d, want 1; stderr=%s", code, errOut)
	}
}

// containsOnce / replaceOnce are tiny closed helpers for the fixture mutation.
func containsOnce(s, sub string) bool { return strings.Count(s, sub) == 1 }

func replaceOnce(s, old, neu string) string {
	i := strings.Index(s, old)
	return s[:i] + neu + s[i+len(old):]
}

// TestDecodeAnatomyClaimRowMissingUsWitness proves corpus incident #4 (WSL
// launch-bound ~42 ms/tok floor, public issue #115). No receipt was pinned,
// so no regime classification was possible; the ESTIMATED band ("no nsys µs
// witness yet") was ungated by every existing verb. The pinned receipt
// presents a kernel-profile section whose µs fields never landed
// (Complete=false, fields empty — the spec's threshold shape), so a claim
// row with promised-but-incomplete profile evidence must fail
// profile_receipt_missing.
func TestDecodeAnatomyClaimRowMissingUsWitness(t *testing.T) {
	// fak-test:runtime fast est=50ms lane=default
	recDir, root := writeAnatomyFixture(t, map[string]string{
		"wsl-floor.perf.jsonl": `{"schema":"fak.profile-note.v1","model":"fixture-model","backend":"cuda","decode_tps":24.0,"ms_per_token":42.0,"kernels":{"source":"nsys-pending","complete":false}}` + "\n",
	}, nil)
	code, res, errOut := runDecodeAnatomyJSON(t, "--receipts", recDir, "--root", root)
	sig, ok := signalByID(res, SignalProfileReceiptMissing)
	if !ok {
		t.Fatalf("signal %s absent from document", SignalProfileReceiptMissing)
	}
	if sig.Status != AnatomyFail {
		t.Fatalf("status = %s, want fail; detail=%s", sig.Status, sig.Detail)
	}
	if sig.Value == nil || *sig.Value != 1 {
		t.Fatalf("value = %v, want 1 unwitnessed claim row", sig.Value)
	}
	if sig.Cap == nil || *sig.Cap != 0 {
		t.Fatalf("cap = %v, want 0", sig.Cap)
	}
	if res.Verdict != "fail" || len(res.Failures) != 1 || res.Failures[0] != SignalProfileReceiptMissing {
		t.Fatalf("verdict/failures = %s/%v, want fail/[profile_receipt_missing]", res.Verdict, res.Failures)
	}
	if code != 1 {
		t.Fatalf("exit = %d, want 1; stderr=%s", code, errOut)
	}
}

// TestDecodeAnatomyLaunchSyncStormEstimates proves corpus incident #5
// (ATTRIBUTION D2: ~570 kernel launches + 48 stream-sync fences per token,
// ESTIMATED band). launch-census is static (inventory x shape, no runtime
// counters); hostshare sits at 0.05–0.11, below the line; the 48 fences/tok
// and the missing µs witness were ungated. The v3 receipt form carries the
// explicit kernel_launch_count / sync_roundtrips witnesses.
func TestDecodeAnatomyLaunchSyncStormEstimates(t *testing.T) {
	// fak-test:runtime fast est=50ms lane=default
	recDir, root := writeAnatomyFixture(t, map[string]string{
		"d2.perf.jsonl": `{"schema":"fak-native-performance-receipt/v3","envelope_id":"fixture-d2","controls":{"prompt_tokens":10,"decode_tokens":100,"batch":1,"context_tokens":2048,"temperature":0,"sampling":"greedy","cache_state":"cold","warmups":1,"repetitions":1},"repetitions":[{"end_to_end_milliseconds":57000,"tokens_per_second":1.75,"decode_milliseconds":56000,"kernel_launch_count":57000,"sync_roundtrips":4800}]}` + "\n",
	}, nil)
	code, res, errOut := runDecodeAnatomyJSON(t, "--receipts", recDir, "--root", root)
	syncs, ok := signalByID(res, SignalSyncsPerToken)
	if !ok {
		t.Fatalf("signal %s absent from document", SignalSyncsPerToken)
	}
	if syncs.Status != AnatomyFail {
		t.Fatalf("syncs status = %s, want fail; detail=%s", syncs.Status, syncs.Detail)
	}
	if syncs.Value == nil || *syncs.Value != 48 {
		t.Fatalf("syncs value = %v, want 48", syncs.Value)
	}
	launches, ok := signalByID(res, SignalLaunchesPerToken)
	if !ok {
		t.Fatalf("signal %s absent from document", SignalLaunchesPerToken)
	}
	if launches.Value == nil || *launches.Value != 570 {
		t.Fatalf("launches value = %v, want 570", launches.Value)
	}
	if launches.Status != AnatomyPass {
		t.Fatalf("launches status = %s, want pass (570 is under the census-calibrated cap)", launches.Status)
	}
	prof, ok := signalByID(res, SignalProfileReceiptMissing)
	if !ok {
		t.Fatalf("signal %s absent from document", SignalProfileReceiptMissing)
	}
	if prof.Status != AnatomyFail || prof.Value == nil || *prof.Value != 1 {
		t.Fatalf("profile signal = %+v, want fail with 1 unwitnessed claim row", prof)
	}
	if res.Verdict != "fail" {
		t.Fatalf("verdict = %s, want fail", res.Verdict)
	}
	want := []string{SignalSyncsPerToken, SignalProfileReceiptMissing}
	if len(res.Failures) != len(want) {
		t.Fatalf("failures = %v, want %v", res.Failures, want)
	}
	for i, id := range want {
		if res.Failures[i] != id {
			t.Fatalf("failures = %v, want %v (signal order)", res.Failures, want)
		}
	}
	if code != 1 {
		t.Fatalf("exit = %d, want 1; stderr=%s", code, errOut)
	}
}

// TestDecodeAnatomyGreenBaseline passes the gate on an aegis-shaped committed
// receipt: complete µs-witnessed kernel profile, ~2 syncs/tok, host_share
// nested under native_timing (UNKNOWN to the hostshare reader, never a
// fabricated zero). The second row is an ABBA control arm: a decode claim
// with NO profile section at all — it must not be flagged as ESTIMATED,
// which is what keeps committed rollups (whose -off arms carry no structured
// profile) green. Mirrors the green run on real committed receipts.
func TestDecodeAnatomyGreenBaseline(t *testing.T) {
	// fak-test:runtime fast est=50ms lane=default
	recDir, root := writeAnatomyFixture(t, map[string]string{
		"aegis-shape.perf.jsonl": `{"request_id":"r1","schema":"fak.gateway.perf-record.v1","unix_ms":1793703910451,"finish_reason":"stop","prompt_tokens":15,"completion_tokens":33,"cached_tokens":0,"cache_regime":"cold","e2e_ms":425.6,"ttft_ms":117.2,"decode_tps":105.78,"model":"Qwen2.5-Coder-3B-Instruct-Q4_K_M","backend":"cuda","gpu_busy_percent":0,"native_timing":{"prefill_ms":116.8,"decode_ms":302.507,"prefill_tokens":15,"decode_tokens":33,"decode_tps":105.78,"cpu_ms_per_token":9.49575,"host_share":1.004486,"kernels":{"source":"structured","backend":"cuda","decode_steps":32,"gpu_ms_per_token":8.54915,"wall_ms_per_token":9.453344,"device_busy":0.904352,"selected_coverage":0.954,"truncated":0,"untimed_compute":0,"untimed_transfer":66,"barriers_elided":0,"complete":true,"pipelining":"disabled-by-profile","selected_kernels":[{"kernel":"k_q4k_mmvq_gemv","op":"matvec","groups":"11008x1","calls_per_token":74.25,"gpu_ms_per_token":1.32,"us_per_call":17.8,"share":0.15},{"kernel":"k_flash_attention","op":"attention","groups":"16x1","calls_per_token":37.125,"gpu_ms_per_token":1.28,"us_per_call":34.5,"share":0.15},{"kernel":"other","op":"other","groups":"","calls_per_token":92.8125,"gpu_ms_per_token":0.39,"us_per_call":4.2,"share":0.045}]},"gpu_idle_ms_per_token":0.13,"syncs_per_token":2.0625,"sync_sites":[{"site":"readback","stage":"k_argmax","syncs_per_token":1.03125,"wait_ms_per_token":3.47,"d2h_bytes_per_token":4.125},{"site":"upload","stage":"none","syncs_per_token":1.03125,"wait_ms_per_token":0.002}]}}` + "\n" +
			`{"request_id":"r2","schema":"fak.gateway.perf-record.v1","unix_ms":1793703910452,"finish_reason":"stop","prompt_tokens":15,"completion_tokens":30,"decode_tps":105.0,"model":"Qwen2.5-Coder-3B-Instruct-Q4_K_M","backend":"cuda","native_timing":{"decode_ms":300.0,"decode_tokens":30,"decode_tps":105.0,"host_share":1.0}}` + "\n",
	}, nil)
	code, res, errOut := runDecodeAnatomyJSON(t, "--receipts", recDir, "--root", root)
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (green baseline); stderr=%s", code, errOut)
	}
	if res.Verdict != "pass" || len(res.Failures) != 0 {
		t.Fatalf("verdict/failures = %s/%v, want pass/[]", res.Verdict, res.Failures)
	}
	launches, _ := signalByID(res, SignalLaunchesPerToken)
	// 74.25 + 37.125 + 92.8125 = 204.1875 kernel invocations/tok.
	if launches.Value == nil || *launches.Value != 204.1875 {
		t.Fatalf("launches value = %v, want 204.1875", launches.Value)
	}
	if launches.Status != AnatomyPass {
		t.Fatalf("launches status = %s, want pass", launches.Status)
	}
	syncs, _ := signalByID(res, SignalSyncsPerToken)
	if syncs.Value == nil || *syncs.Value != 2.0625 || syncs.Status != AnatomyPass {
		t.Fatalf("syncs = value %v status %s, want 2.0625/pass", syncs.Value, syncs.Status)
	}
	hs, _ := signalByID(res, SignalHostShare)
	if hs.Status != AnatomyUnknown || hs.Value != nil {
		t.Fatalf("host_share = status %s value %v, want unknown/nil (nested host_share is not top-level evidence)", hs.Status, hs.Value)
	}
	prof, _ := signalByID(res, SignalProfileReceiptMissing)
	if prof.Status != AnatomyPass || prof.Value == nil || *prof.Value != 0 {
		t.Fatalf("profile = status %s value %v, want pass/0", prof.Status, prof.Value)
	}
	capSig, _ := signalByID(res, SignalCaptureEligibilityDrift)
	if capSig.Status != AnatomyPass {
		t.Fatalf("capture status = %s, want pass on the clean fixture root; detail=%s", capSig.Status, capSig.Detail)
	}
	bar, _ := signalByID(res, SignalBarriersElidedWhileArmed)
	if bar.Status != AnatomyUnknown {
		t.Fatalf("barriers status = %s, want unknown (arming not declared)", bar.Status)
	}
}

// TestDecodeAnatomyPinnedSchemaShape pins the exact output contract the
// private gate consumes: schema id, RFC3339 observed_at, the receipts array,
// the six signal ids in order, the verdict enum, and the failures slice.
func TestDecodeAnatomyPinnedSchemaShape(t *testing.T) {
	// fak-test:runtime fast est=50ms lane=default
	recDir, root := writeAnatomyFixture(t, map[string]string{
		"green.perf.jsonl": baseAnatomyRecord("") + "\n",
	}, nil)
	_, res, _ := runDecodeAnatomyJSON(t, "--receipts", recDir, "--root", root)
	if res.Schema != DecodeAnatomySchema {
		t.Fatalf("schema = %q, want %q", res.Schema, DecodeAnatomySchema)
	}
	if _, err := time.Parse(time.RFC3339, res.ObservedAt); err != nil {
		t.Fatalf("observed_at %q is not RFC3339: %v", res.ObservedAt, err)
	}
	if len(res.Receipts) != 1 || res.Receipts[0].Model != "fixture-model" {
		t.Fatalf("receipts = %+v, want one fixture-model row", res.Receipts)
	}
	if len(res.Signals) != len(decodeAnatomySignalIDs) {
		t.Fatalf("signals = %d rows, want %d", len(res.Signals), len(decodeAnatomySignalIDs))
	}
	for i, id := range decodeAnatomySignalIDs {
		if res.Signals[i].ID != id {
			t.Fatalf("signals[%d].id = %q, want %q (closed vocabulary, fixed order)", i, res.Signals[i].ID, id)
		}
		switch res.Signals[i].Status {
		case AnatomyPass, AnatomyFail, AnatomyWarn, AnatomyUnknown:
		default:
			t.Fatalf("signals[%d].status = %q, outside the closed enum", i, res.Signals[i].Status)
		}
	}
	if res.Verdict != "pass" {
		t.Fatalf("verdict = %q, want \"pass\"", res.Verdict)
	}
	if res.Failures != nil && len(res.Failures) != 0 {
		t.Fatalf("failures = %v, want empty on a passing run", res.Failures)
	}
}

// TestDecodeAnatomySnapshotDrift proves the trend gate: a second run against
// the same --snapshot path fails when launches/tok drifts above the prior
// snapshot's baseline beyond the tolerance.
func TestDecodeAnatomySnapshotDrift(t *testing.T) {
	// fak-test:runtime fast est=50ms lane=default
	snap := filepath.Join(t.TempDir(), "snap", "decode-anatomy.json")
	recDirA, rootA := writeAnatomyFixture(t, map[string]string{
		"a.perf.jsonl": baseAnatomyRecord("") + "\n", // 100 launches/tok
	}, nil)
	code, res, errOut := runDecodeAnatomyJSON(t, "--receipts", recDirA, "--root", rootA, "--snapshot", snap)
	if code != 0 {
		t.Fatalf("run A exit = %d, want 0; stderr=%s", code, errOut)
	}
	if res.Signals[0].Value == nil || *res.Signals[0].Value != 100 {
		t.Fatalf("run A launches = %v, want 100", res.Signals[0].Value)
	}
	body, err := os.ReadFile(snap)
	if err != nil {
		t.Fatalf("read snapshot: %v", err)
	}
	var prior DecodeAnatomyResult
	if err := json.Unmarshal(body, &prior); err != nil {
		t.Fatalf("snapshot did not round-trip through the result type: %v", err)
	}
	if prior.Schema != DecodeAnatomySchema {
		t.Fatalf("snapshot schema = %q, want %q", prior.Schema, DecodeAnatomySchema)
	}

	// Run B: +30%% launches/tok vs the 100 baseline (> 20%% tolerance).
	recDirB, rootB := writeAnatomyFixture(t, map[string]string{
		"b.perf.jsonl": strings.Replace(baseAnatomyRecord(""), `"calls_per_token":100.0`, `"calls_per_token":130.0`, 1) + "\n",
	}, nil)
	code, res, errOut = runDecodeAnatomyJSON(t, "--receipts", recDirB, "--root", rootB, "--snapshot", snap)
	sig, _ := signalByID(res, SignalLaunchesPerToken)
	if sig.Status != AnatomyFail {
		t.Fatalf("run B launches status = %s, want fail (drift); detail=%s", sig.Status, sig.Detail)
	}
	if res.Verdict != "fail" || len(res.Failures) != 1 || res.Failures[0] != SignalLaunchesPerToken {
		t.Fatalf("run B verdict/failures = %s/%v, want fail/[launches_per_token]", res.Verdict, res.Failures)
	}
	if code != 1 {
		t.Fatalf("run B exit = %d, want 1; stderr=%s", code, errOut)
	}
}

// TestDecodeAnatomyUsageErrors pins the closed-vocabulary refusals: an unknown
// --fail-on id and out-of-range caps exit 2 without touching the tree.
func TestDecodeAnatomyUsageErrors(t *testing.T) {
	// fak-test:runtime fast est=20ms lane=default
	recDir, root := writeAnatomyFixture(t, map[string]string{
		"x.perf.jsonl": baseAnatomyRecord("") + "\n",
	}, nil)
	for _, argv := range [][]string{
		{"--receipts", recDir, "--root", root, "--fail-on", "bogus"},
		{"--receipts", recDir, "--root", root, "--syncs-cap", "0"},
		{"--receipts", recDir, "--root", root, "--threshold", "1.5"},
		{"--receipts", recDir, "--root", root, "--launches-cap", "-3"},
	} {
		var out, errb bytes.Buffer
		if code := RunDecodeAnatomy(&out, &errb, argv); code != 2 {
			t.Fatalf("argv %v: exit = %d, want 2; stderr=%s", argv, code, errb.String())
		}
	}
	// A valid --fail-on subset still gates only the named signal: the
	// host_share breach fails the run when host_share is active but not when
	// only launches_per_token is.
	breachDir, breachRoot := writeAnatomyFixture(t, map[string]string{
		"breach.perf.jsonl": baseAnatomyRecord(`,"host_share":0.5`) + "\n",
	}, nil)
	var out, errb bytes.Buffer
	if code := RunDecodeAnatomy(&out, &errb, []string{"--receipts", breachDir, "--root", breachRoot, "--fail-on", SignalLaunchesPerToken}); code != 0 {
		t.Fatalf("--fail-on launches_per_token over a host_share breach: exit = %d, want 0 (inactive signal cannot gate); stderr=%s", code, errb.String())
	}
	var out2, errb2 bytes.Buffer
	if code := RunDecodeAnatomy(&out2, &errb2, []string{"--receipts", breachDir, "--root", breachRoot, "--fail-on", SignalHostShare}); code != 1 {
		t.Fatalf("--fail-on host_share over a host_share breach: exit = %d, want 1; stderr=%s", code, errb2.String())
	}
}
