package metalgemm

import (
	"os"
	"strings"
	"testing"
)

// This source witness keeps the Darwin-only cgo path compile-safe on hosts that cannot type-check
// Objective-C bridge calls: the repeated-batch helper must receive the owning observation, and its
// only caller must forward the same observation rather than referencing an out-of-scope variable.
func TestQ4KRepeatedBatchForwardsExecutionObservation(t *testing.T) {
	source, err := os.ReadFile("q4k.go")
	if err != nil {
		t.Fatal(err)
	}
	text := string(source)
	for _, want := range []string{
		"func (w *Q4KWeight) gemvBatchRepeatedWithEvents(Xcat []float32, n int, Ycat []float32, observation *ExecutionObservation)",
		"w.gemvBatchRepeatedWithEvents(Xcat, n, Ycat, observation)",
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("q4k repeated-batch observation forwarding missing %q", want)
		}
	}
}

// This source witness runs on non-Darwin hosts and protects the ownership seam that cgo cannot
// type-check there. The Darwin witness in q4k_test.go captures the real Metal lifecycle and parity.
func TestMixedQ4KQ8ObservationSourceIsCallerOwned(t *testing.T) {
	goSource, err := os.ReadFile("q4k.go")
	if err != nil {
		t.Fatal(err)
	}
	nativeSource, err := os.ReadFile("q4k.m")
	if err != nil {
		t.Fatal(err)
	}
	q8Source, err := os.ReadFile("q8.m")
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"GEMVGroupMixedQ4KQ8(", "observation *ExecutionObservation",
		"observation.record(uintptr(event.command_buffer)", "MixedQ4KQ8PostSubmitError",
	} {
		if !strings.Contains(string(goSource), want) {
			t.Fatalf("mixed Go source missing %q", want)
		}
	}
	for _, want := range []string{
		"mg_q4k_q8_gemv_group(", "mg_execution_event* event",
		"event->command_buffer", "event->host_readback = 1",
		"inject_post_submit_failure", "[cb waitUntilCompleted]",
	} {
		if !strings.Contains(string(nativeSource), want) {
			t.Fatalf("mixed native source missing %q", want)
		}
	}
	for _, want := range []string{"mg_q8_prepare_gemv_group", "mg_q8_encode_gemv_group", "mg_q8_read_gemv_group"} {
		if !strings.Contains(string(q8Source), want) {
			t.Fatalf("Q8 caller-owned helper missing %q", want)
		}
	}
	for _, forbidden := range []string{"gQuantCommandBuffers", "mg_quant_event_snapshot", "ResetMixed"} {
		if strings.Contains(string(goSource), forbidden) || strings.Contains(string(nativeSource), forbidden) || strings.Contains(string(q8Source), forbidden) {
			t.Fatalf("mixed implementation reintroduced package-global attribution %q", forbidden)
		}
	}
}

func TestMixedQ4KQ8MetalWitnessReadback(t *testing.T) {
	stdout, err := os.ReadFile("../../docs/_witnesses/issue-8973-mixed-q4k-q8/metal-test.stdout")
	if err != nil {
		t.Fatal(err)
	}
	text := string(stdout)
	for _, want := range []string{
		"control_events=2 candidate_events=1",
		"Operation:mixed-q4_k-q8-qkv",
		"Q4_K parity: outputs=64 cosine=1.000000 max_rel=0.000000",
		"Q8 parity: groups=2 exact=true",
		"typed=*metalgemm.MixedQ4KQ8PostSubmitError committed=true waited=true readback=false encoders=2",
		"PASS",
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("mixed Metal witness missing %q", want)
		}
	}
	for _, forbidden := range []string{"/Users/", "credential", "token="} {
		if strings.Contains(text, forbidden) {
			t.Fatalf("mixed Metal witness contains private marker %q", forbidden)
		}
	}
}
