package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/engine"
)

// TestHealthzEngineFieldNamesDispatchAxis is the #2454 witness: /healthz carries
// two orthogonal axes — the dispatch-capability identity ("engine" = the abi
// engine fak_syscall dispatches to, default "inkernel") and the live serving mode
// ("planner" = the /v1/chat/completions backend). The body used to expose only
// the bare pair, so a live proxy serve reporting engine:"mock" read as "this
// deployment serves scripted text".
//
// This test FAILS on the old body (no "axes"/"serving" keys, and "engine" adjacent
// to a real planner with nothing naming its axis) and PASSES on the new one. It
// asserts, on the SAME document:
//
//  1. "engine" reports the DISPATCH/CAPABILITY value, unchanged (the legacy read
//     contract must not regress).
//  2. The document states, in-band, that "engine" names the dispatch axis and is
//     not a serving mode — so a reader needs no prior knowledge to tell them apart.
//  3. A reader can answer "is real model output being served?" WITHOUT consulting
//     "engine": the answer is carried by the planner-sourced serving verdict.
//  4. "planner" semantics are untouched (same value, still the serving-mode axis).
func TestHealthzEngineFieldNamesDispatchAxis(t *testing.T) {
	// A real proxied backend next to a left-unset (default "inkernel") dispatch
	// engine is exactly the live trap: real serving, non-serving engine identity.
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	// ResetForTest wiped the "mock" engine internal/engine's init registered;
	// restore it so the trap case below can construct a server whose dispatch
	// engine id reads "mock" while a real backend answers chat.
	abi.RegisterEngine("mock", engine.MockEngine)
	abi.RegisterAdjudicator(0, toolAdj{})
	srv, err := New(Config{
		EngineID: "test",
		Model:    "test-model",
		BaseURL:  "http://127.0.0.1:1",
		VDSO:     true,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)

	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/healthz", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("/healthz status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode /healthz: %v (body=%s)", err, rec.Body.String())
	}

	// (1) The dispatch-capability value itself is unchanged — this is the syscall
	// engine identity, NOT the chat backend.
	if body["engine"] != "test" {
		t.Fatalf(`/healthz engine = %v, want the dispatch engine id "test"`, body["engine"])
	}
	// (4) planner semantics unchanged: this config is a live proxy backend.
	if body["planner"] != "proxy" {
		t.Fatalf(`/healthz planner = %v, want "proxy" (serving-mode axis must not regress)`, body["planner"])
	}

	// (2) The axis split is stated in-band on the same document, so "engine"
	// cannot be read as a serving verdict without the reader first overriding a
	// label that says it is not one.
	axes, ok := body["axes"].(map[string]any)
	if !ok {
		t.Fatalf(`/healthz "axes" missing or not an object: %#v — the engine/planner axes are unlabeled`, body["axes"])
	}
	engineAxis, _ := axes["engine"].(string)
	plannerAxis, _ := axes["planner"].(string)
	if !strings.Contains(engineAxis, "dispatch") {
		t.Fatalf(`axes.engine = %q, want it to name the dispatch/capability axis`, engineAxis)
	}
	if !strings.Contains(engineAxis, "not a serving mode") && !strings.Contains(engineAxis, "not a serving") {
		t.Fatalf(`axes.engine = %q, want it to state it is NOT a serving mode`, engineAxis)
	}
	if !strings.Contains(plannerAxis, "serving") {
		t.Fatalf(`axes.planner = %q, want it to name the serving-mode axis`, plannerAxis)
	}
	if engineAxis == plannerAxis {
		t.Fatalf("axes.engine and axes.planner are identical (%q); the axes are still collapsed", engineAxis)
	}

	// (3) The serving question is answerable from a planner-sourced verdict, with
	// no reference to "engine". Here: a live proxy backend => real model output.
	serving, ok := body["serving"].(map[string]any)
	if !ok {
		t.Fatalf(`/healthz "serving" missing or not an object: %#v`, body["serving"])
	}
	if serving["real_model_output"] != true {
		t.Fatalf(`serving.real_model_output = %v, want true for a live "proxy" planner (body=%#v)`, serving["real_model_output"], body)
	}
	if serving["source"] != "planner" {
		t.Fatalf(`serving.source = %v, want "planner" (the serving-mode axis, not the dispatch engine)`, serving["source"])
	}

	// The trap, restated as a negative: the serving verdict must not be derived
	// from the dispatch engine. A deployment whose engine id reads "mock" while a
	// real backend answers must still report real model output.
	mockEngineSrv, err := New(Config{
		EngineID: "mock",
		Model:    "test-model",
		BaseURL:  "http://127.0.0.1:1",
	})
	if err != nil {
		t.Fatalf("New(mock engine id): %v", err)
	}
	t.Cleanup(mockEngineSrv.Close)
	rec2 := httptest.NewRecorder()
	mockEngineSrv.Handler().ServeHTTP(rec2, httptest.NewRequest(http.MethodGet, "/healthz", nil))
	var body2 map[string]any
	if err := json.Unmarshal(rec2.Body.Bytes(), &body2); err != nil {
		t.Fatalf("decode /healthz (mock engine id): %v", err)
	}
	if body2["engine"] != "mock" {
		t.Fatalf(`/healthz engine = %v, want "mock" (dispatch identity passed through)`, body2["engine"])
	}
	serving2, _ := body2["serving"].(map[string]any)
	if serving2 == nil || serving2["real_model_output"] != true {
		t.Fatalf(`engine:"mock" + live proxy must still report real model output; serving = %#v (body=%#v)`,
			body2["serving"], body2)
	}

	// And the mirror on the metrics surface: fak_gateway_build_info must name the
	// dispatch axis apart from the serving axis.
	metrics := srv.renderMetrics()
	line := buildInfoSeriesLine(metrics)
	if line == "" {
		t.Fatalf("fak_gateway_build_info series missing from /metrics")
	}
	if hasPromLabel(line, "engine") {
		t.Fatalf("fak_gateway_build_info still carries an unlabeled engine= label: %s", line)
	}
	if !hasPromLabelValue(line, "dispatch_engine", "test") || !hasPromLabelValue(line, "planner", "proxy") {
		t.Fatalf("fak_gateway_build_info must label both axes (dispatch_engine=\"test\" + planner=\"proxy\"); got %s", line)
	}
}

// buildInfoSeriesLine returns the samples line of fak_gateway_build_info (the
// line starting with the metric name, not its HELP/TYPE preamble), or "" when
// the family is absent.
func buildInfoSeriesLine(metrics string) string {
	for _, line := range strings.Split(metrics, "\n") {
		if strings.HasPrefix(line, "fak_gateway_build_info{") {
			return line
		}
	}
	return ""
}

// hasPromLabel reports whether a Prometheus samples line carries the exact label
// name (not a substring of a longer label such as dispatch_engine for engine).
// The label set is space-separated as `{a="1",b="2"}`, so requiring a leading
// `{` or `,` before the name avoids the substring match.
func hasPromLabel(line, name string) bool {
	return strings.Contains(line, "{"+name+`="`) || strings.Contains(line, ","+name+`="`)
}

// hasPromLabelValue reports whether a samples line carries name="value".
func hasPromLabelValue(line, name, value string) bool {
	return strings.Contains(line, name+`="`+value+`"`)
}

// TestHealthzServingVerdictRefusesUnknownPlanner keeps the serving verdict
// fail-closed: an unrecognized planner must never be reported as real model
// output (the same rule plannerKind already applies by returning "unknown"
// rather than masquerading as a real backend).
func TestHealthzServingVerdictRefusesUnknownPlanner(t *testing.T) {
	for _, planner := range []string{"unknown", "", "weird"} {
		got := servingModeVerdict(planner)
		if got["real_model_output"] != false {
			t.Errorf("servingModeVerdict(%q).real_model_output = %v, want false", planner, got["real_model_output"])
		}
	}
	// A nil/unrecognized planner flows through plannerKind as "unknown".
	if kind := plannerKind(nil); kind != "unknown" {
		t.Fatalf("plannerKind(nil) = %q, want \"unknown\"", kind)
	}
	// And the mock verdict names the scripted fallback explicitly.
	mock := servingModeVerdict("mock")
	if mock["real_model_output"] != false {
		t.Fatalf("servingModeVerdict(\"mock\").real_model_output = %v, want false", mock["real_model_output"])
	}
	real := servingModeVerdict("proxy")
	if real["real_model_output"] != true {
		t.Fatalf("servingModeVerdict(\"proxy\").real_model_output = %v, want true", real["real_model_output"])
	}
}
