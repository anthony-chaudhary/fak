package deepseekbench

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// TestRequiredFields is the FIELD LOCK: every required JSON key from the issue must
// be present on every emitted row, and no others (including the #3020 speculative
// axis: "speculative" + "accepted_token_ratio"). If a field is renamed/dropped
// without updating RequiredFields, this fails.
func TestRequiredFields(t *testing.T) {
	for i, r := range DryRunRows() {
		b, err := json.Marshal(r)
		if err != nil {
			t.Fatalf("row %d marshal: %v", i, err)
		}
		var m map[string]json.RawMessage
		if err := json.Unmarshal(b, &m); err != nil {
			t.Fatalf("row %d not valid JSON: %v\n%s", i, err, b)
		}
		for _, want := range RequiredFields() {
			if _, ok := m[want]; !ok {
				t.Fatalf("row %d missing required field %q:\n%s", i, want, b)
			}
		}
		// No stray fields beyond the locked schema — the row struct IS the schema.
		if len(m) != len(RequiredFields()) {
			t.Fatalf("row %d has %d fields, schema locks %d:\n%s", i, len(m), len(RequiredFields()), b)
		}
	}
}

// TestDryRunHonesty pins the no-key fixture invariants: every DeepSeek row is
// labelled a fixture placeholder (never a measurement), and both V4 Pro and V4
// Flash appear side by side.
func TestDryRunHonesty(t *testing.T) {
	rows := DryRunRows()
	sawPro, sawFlash := false, false
	for _, r := range rows {
		if r.ProviderRoute == "deepseek" {
			if r.Measurement != "dry-run-fixture" {
				t.Fatalf("dry-run row not labelled a fixture: %+v", r)
			}
			if r.SpeedProvenance == "provider-observed" {
				t.Fatalf("dry-run row claims provider-observed speed: %+v", r)
			}
		}
		switch r.ModelID {
		case ModelV4Pro:
			sawPro = true
		case ModelV4Flash:
			sawFlash = true
		}
	}
	if !sawPro || !sawFlash {
		t.Fatalf("scorecard must carry BOTH V4 Pro and V4 Flash (pro=%v flash=%v)", sawPro, sawFlash)
	}
}

// TestAxisCoverage confirms every locked axis value appears in the fixture.
func TestAxisCoverage(t *testing.T) {
	rows := DryRunRows()
	buckets, targets, modes := map[string]bool{}, map[string]bool{}, map[string]bool{}
	for _, r := range rows {
		if r.ProviderRoute != "deepseek" {
			continue
		}
		buckets[r.ContextBucket] = true
		targets[r.OutputTarget] = true
		modes[r.ReasoningMode] = true
	}
	for _, b := range ContextBuckets {
		if !buckets[b] {
			t.Fatalf("context bucket %q never appears", b)
		}
	}
	for _, tg := range OutputTargets {
		if !targets[tg] {
			t.Fatalf("output target %q never appears", tg)
		}
	}
	for _, md := range ReasoningModes {
		if !modes[md] {
			t.Fatalf("reasoning mode %q never appears", md)
		}
	}
}

// TestSpeedupRefusal is the honesty gate: the scorecard must NOT print a speedup for
// a dry-run fixture, a shape mismatch, an unverified parity, or a missing/invalid
// speculative label (#3020) — and MUST print one only when shape + parity + live +
// speculative label all line up, with the acceptance evidence beside any
// speculative-on delta.
func TestSpeedupRefusal(t *testing.T) {
	base := Row{Measurement: "live", QualityParity: "verified", E2EMillis: 100, PromptShapeKey: "4K|short|non-thinking|stream=true", ModelID: "baseline", Speculative: "off", AcceptedTokenRatio: "unknown"}
	subj := Row{Measurement: "live", QualityParity: "verified", E2EMillis: 50, PromptShapeKey: "4K|short|non-thinking|stream=true", ModelID: ModelV4Flash, Speculative: "off", AcceptedTokenRatio: "unknown"}

	// (a) dry-run fixture -> refuse.
	if line, printed := CompareSpeedup(Row{Measurement: "dry-run-fixture"}, base); printed || !strings.Contains(line, "NOT COMPARABLE") {
		t.Fatalf("dry-run must refuse a speedup, got printed=%v line=%q", printed, line)
	}
	// (b) shape mismatch -> refuse.
	mismatch := subj
	mismatch.PromptShapeKey = "1M|8K|max|stream=false"
	if line, printed := CompareSpeedup(mismatch, base); printed || !strings.Contains(line, "prompt shape differs") {
		t.Fatalf("shape mismatch must refuse, got printed=%v line=%q", printed, line)
	}
	// (c) parity not verified -> refuse.
	noparity := subj
	noparity.QualityParity = "unknown"
	if line, printed := CompareSpeedup(noparity, base); printed || !strings.Contains(line, "quality parity") {
		t.Fatalf("unverified parity must refuse, got printed=%v line=%q", printed, line)
	}
	// (d) all aligned -> a labelled OBSERVED delta.
	line, printed := CompareSpeedup(subj, base)
	if !printed || !strings.Contains(line, "OBSERVED provider speed") || !strings.Contains(line, "not a fak-authored saving") {
		t.Fatalf("aligned rows must print an OBSERVED delta, got printed=%v line=%q", printed, line)
	}
	// (e) an mtp subject vs an off baseline WITHOUT verified parity -> refuse: a
	// speculative speedup can never surface without a verified quality parity.
	mtpNoParity := subj
	mtpNoParity.Speculative = "mtp"
	mtpNoParity.QualityParity = "unknown"
	if line, printed := CompareSpeedup(mtpNoParity, base); printed || !strings.Contains(line, "quality parity") {
		t.Fatalf("mtp without verified parity must refuse, got printed=%v line=%q", printed, line)
	}
	// (f) an empty or off-vocabulary speculative label -> refuse.
	unlabelled := subj
	unlabelled.Speculative = ""
	if line, printed := CompareSpeedup(unlabelled, base); printed || !strings.Contains(line, "missing speculative label") {
		t.Fatalf("empty speculative label must refuse, got printed=%v line=%q", printed, line)
	}
	invalid := subj
	invalid.Speculative = "eagle"
	if line, printed := CompareSpeedup(invalid, base); printed || !strings.Contains(line, "missing speculative label") {
		t.Fatalf("off-vocabulary speculative label must refuse, got printed=%v line=%q", printed, line)
	}
	// (g) an mtp subject vs an off baseline WITH verified parity, shared shape, both
	// live, measured E2E -> prints, and the acceptance evidence rides beside it.
	mtp := subj
	mtp.Speculative = "mtp"
	mtp.AcceptedTokenRatio = "0.87"
	line, printed = CompareSpeedup(mtp, base)
	if !printed || !strings.Contains(line, "OBSERVED provider speed") {
		t.Fatalf("aligned mtp rows must print an OBSERVED delta, got printed=%v line=%q", printed, line)
	}
	if !strings.Contains(line, "speculative=mtp") || !strings.Contains(line, "accepted_token_ratio=0.87") {
		t.Fatalf("a speculative speedup line must carry its acceptance evidence, got %q", line)
	}
}

// TestLiveGate confirms the pure gate refuses without a key or without --spend, and
// admits only when both are present.
func TestLiveGate(t *testing.T) {
	if _, ok := LiveGate(false, true); ok {
		t.Fatal("missing key must refuse")
	}
	if _, ok := LiveGate(true, false); ok {
		t.Fatal("missing spend must refuse")
	}
	if msg, ok := LiveGate(true, true); !ok || msg != "" {
		t.Fatalf("key+spend must admit, got ok=%v msg=%q", ok, msg)
	}
}

// TestMeasureStreamedAgainstFakeSSE drives the real MeasureStreamed parsing/timing
// logic against a canned OpenAI-compatible SSE stream — no key, loopback only —
// locking counter extraction, the provider-cache attribution flip, and the
// live-row provenance without a live DeepSeek endpoint.
func TestMeasureStreamedAgainstFakeSSE(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		frames := []string{
			`{"choices":[{"delta":{"content":"one "}}]}`,
			`{"choices":[{"delta":{"content":"two "}}]}`,
			`{"choices":[{"delta":{"content":"three"}}]}`,
			`{"choices":[],"usage":{"prompt_tokens":128,"completion_tokens":40,` +
				`"reasoning_tokens":12,"prompt_cache_hit_tokens":96,"prompt_cache_miss_tokens":32}}`,
		}
		for _, f := range frames {
			w.Write([]byte("data: " + f + "\n\n"))
		}
		w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer srv.Close()

	row, err := MeasureStreamed(srv.Client(), srv.URL, "", ModelV4Flash)
	if err != nil {
		t.Fatalf("MeasureStreamed: %v", err)
	}
	if row.Measurement != "live" || row.SpeedProvenance != "provider-observed" {
		t.Errorf("live row provenance wrong: %+v", row)
	}
	if row.PromptTokens != 128 || row.CompletionTokens != 40 || row.ReasoningTokens != 12 {
		t.Errorf("token counters = p%d/c%d/r%d, want 128/40/12", row.PromptTokens, row.CompletionTokens, row.ReasoningTokens)
	}
	if row.PromptCacheHitTokens != 96 || row.PromptCacheMissTokens != 32 {
		t.Errorf("cache counters = hit%d/miss%d, want 96/32", row.PromptCacheHitTokens, row.PromptCacheMissTokens)
	}
	if row.CacheAttribution != "provider-observed" {
		t.Errorf("cache attribution = %q, want provider-observed (a hit/miss split was present)", row.CacheAttribution)
	}
	if row.E2EMillis < 0 || row.TTFTMillis < 0 || row.E2EMillis < row.TTFTMillis {
		t.Errorf("timings incoherent: ttft=%v e2e=%v", row.TTFTMillis, row.E2EMillis)
	}
	// A live row against a dry-run fixture row must still refuse (baseline unmeasured).
	if _, printed := CompareSpeedup(row, DryRunRows()[0]); printed {
		t.Error("speedup against a dry-run fixture row must refuse")
	}
}
