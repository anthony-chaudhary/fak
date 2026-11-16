package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
)

// toolcall_parse_metrics_test.go — adversarial witness for the
// fak_gateway_toolcall_parse_total{outcome,dialect,reason} family: tool-call
// parse outcomes must be a live scrapeable counter, not only a log line.
//
// These drive the REAL production entrypoints — (*Server).validateChatCompletionConformance,
// (*Server).failClosedOnUnparsedToolCalls, (*Server).withToolCallParseObserver and
// (*Server).renderMetrics — so the assertions are on the served path, not a
// re-implementation of it.

const toolCallParseFamily = "fak_gateway_toolcall_parse_total"

// toolCallParseDefs builds the offered-tool set a text lift is admitted against.
func toolCallParseDefs(names ...string) []agent.ToolDef {
	tools := make([]agent.ToolDef, len(names))
	for i, n := range names {
		tools[i] = agent.ToolDef{Type: "function", Function: agent.ToolDefFunction{Name: n, Parameters: json.RawMessage(`{"type":"object"}`)}}
	}
	return tools
}

// toolCallParseSampleLines returns the family's SAMPLE lines (the ones carrying a
// `{` label set), in render order, so a test can assert both content and order.
func toolCallParseSampleLines(scrape string) []string {
	var out []string
	for _, line := range strings.Split(scrape, "\n") {
		if strings.HasPrefix(line, toolCallParseFamily+"{") {
			out = append(out, line)
		}
	}
	return out
}

// toolCallParseCount reads one (outcome,dialect,reason) series out of a live
// metrics receiver. A missing series reads 0, which is what a scrape would show.
func toolCallParseCount(m *gatewayMetrics, outcome, dialect, reason string) uint64 {
	return m.toolCallParseSnapshot()[toolCallParseKey{outcome: outcome, dialect: dialect, reason: reason}]
}

// renderToolCallParseFamily renders just this family. The full (*Server).renderMetrics
// needs a fully-wired kernel, so the family-level assertions use the named contract
// renderer; the end-to-end wiring through renderMetrics is asserted separately on a
// real server in the zero-event subtest.
func renderToolCallParseFamily(m *gatewayMetrics) string {
	var b strings.Builder
	writeToolCallParseMetrics(&b, m)
	return b.String()
}

// parsePromLabelValues parses the `{k="v",k2="v2"}` label set of one sample line
// and returns the DECODED values. It fails on any raw newline, an unterminated
// quote, or a trailing backslash — i.e. exactly the ways a scrape line can be
// malformed. This is a real parser, not a quote count: an escaped `\"` is a
// legitimate quote character inside a value.
func parsePromLabelValues(t *testing.T, line string) map[string]string {
	t.Helper()
	open := strings.Index(line, "{")
	close := strings.LastIndex(line, "}")
	if open < 0 || close < open {
		t.Fatalf("sample line has no label set: %q", line)
	}
	body := line[open+1 : close]
	values := map[string]string{}
	for i := 0; i < len(body); {
		eq := strings.IndexByte(body[i:], '=')
		if eq < 0 {
			t.Fatalf("label without '=' in %q", body)
		}
		name := body[i : i+eq]
		i += eq + 1
		if i >= len(body) || body[i] != '"' {
			t.Fatalf("label %q value is not quoted in %q", name, body)
		}
		i++ // consume the opening quote
		var sb strings.Builder
		closed := false
		for i < len(body) {
			c := body[i]
			if c == '\\' {
				if i+1 >= len(body) {
					t.Fatalf("label %q ends in a dangling backslash: %q", name, body)
				}
				switch body[i+1] {
				case '\\', '"':
					sb.WriteByte(body[i+1])
				case 'n':
					sb.WriteByte('\n')
				default:
					t.Fatalf("label %q has an invalid escape \\%c: %q", name, body[i+1], body)
				}
				i += 2
				continue
			}
			if c == '"' {
				closed = true
				i++
				break
			}
			if c == '\n' || c == '\r' {
				t.Fatalf("label %q carries a raw newline: %q", name, body)
			}
			sb.WriteByte(c)
			i++
		}
		if !closed {
			t.Fatalf("label %q is unterminated: %q", name, body)
		}
		values[name] = sb.String()
		if i < len(body) {
			if body[i] != ',' {
				t.Fatalf("expected ',' after label %q, got %q", name, body[i:])
			}
			i++
		}
	}
	return values
}

// TestToolCallParseMetrics is the ticket's mandatory witness. It covers the
// fail-closed and structured outcomes through the real conformance gate, the
// nil-metrics regression risk, the zero-event header, nil-receiver safety,
// render determinism, and label escaping.
func TestToolCallParseMetrics(t *testing.T) {
	t.Run("fail_closed_through_conformance_gate", func(t *testing.T) {
		m := newGatewayMetrics(time.Now())
		s := &Server{logf: func(string, ...any) {}, metrics: m}
		comp := &agent.Completion{
			ToolCallsDropped:       true,
			ToolCallsDroppedReason: abi.ReasonMalformed,
			FinishReason:           "length",
			Usage:                  agent.Usage{CompletionTokens: 24},
			Message:                agent.Message{Role: agent.RoleAssistant},
		}
		rec := httptest.NewRecorder()
		if s.validateChatCompletionConformance(rec, nil, comp, comp.Message, false, false, false, nil) {
			t.Fatal("conformance gate admitted a dropped tool call; want fail-closed")
		}
		if rec.Code != http.StatusBadGateway {
			t.Fatalf("status = %d, want 502", rec.Code)
		}
		if got := toolCallParseCount(m, toolCallParseOutcomeFailClosed, "", "MALFORMED"); got != 1 {
			t.Fatalf("fail_closed/MALFORMED = %d, want 1 (snapshot=%v)", got, m.toolCallParseSnapshot())
		}
		// The spec pins dialect empty for fail_closed: no other series may exist.
		if snap := m.toolCallParseSnapshot(); len(snap) != 1 {
			t.Fatalf("fail_closed recorded extra series: %v", snap)
		}
		// The rendered scrape must carry the exact sample line.
		scrape := renderToolCallParseFamily(m)
		want := `fak_gateway_toolcall_parse_total{outcome="fail_closed",dialect="",reason="MALFORMED"} 1`
		if !strings.Contains(scrape, want+"\n") {
			t.Fatalf("scrape missing exact sample line %q\n--- scrape excerpt ---\n%s", want, toolCallParseExcerpt(scrape))
		}
	})

	t.Run("fail_closed_reason_none_is_NONE", func(t *testing.T) {
		// The spec: reason carries the closed abi drop-reason name, "NONE" when
		// unspecified. ReasonNone is the unspecified case.
		m := newGatewayMetrics(time.Now())
		s := &Server{logf: func(string, ...any) {}, metrics: m}
		comp := &agent.Completion{
			ToolCallsDropped:       true,
			ToolCallsDroppedReason: abi.ReasonNone,
			Message:                agent.Message{Role: agent.RoleAssistant},
		}
		rec := httptest.NewRecorder()
		if s.validateChatCompletionConformance(rec, nil, comp, comp.Message, false, false, false, nil) {
			t.Fatal("ReasonNone drop admitted; want fail-closed")
		}
		if got := toolCallParseCount(m, toolCallParseOutcomeFailClosed, "", "NONE"); got != 1 {
			t.Fatalf("fail_closed/NONE = %d, want 1 (snapshot=%v)", got, m.toolCallParseSnapshot())
		}
	})

	t.Run("structured_through_conformance_gate", func(t *testing.T) {
		m := newGatewayMetrics(time.Now())
		s := &Server{logf: func(string, ...any) {}, metrics: m}
		comp := &agent.Completion{
			Message: agent.Message{
				Role: agent.RoleAssistant,
				ToolCalls: []agent.ToolCall{{
					ID:       "call_1",
					Type:     "function",
					Function: agent.Func{Name: "Read", Arguments: `{"file_path":"x"}`},
				}},
			},
		}
		rec := httptest.NewRecorder()
		if !s.validateChatCompletionConformance(rec, nil, comp, comp.Message, false, false, false, nil) {
			t.Fatalf("a turn carrying structured calls must pass; body=%s", rec.Body.String())
		}
		if got := toolCallParseCount(m, toolCallParseOutcomeStructured, "", ""); got != 1 {
			t.Fatalf("structured = %d, want 1 (snapshot=%v)", got, m.toolCallParseSnapshot())
		}
		if got := toolCallParseCount(m, toolCallParseOutcomeFailClosed, "", "NONE"); got != 0 {
			t.Fatalf("structured turn also incremented fail_closed (%d); the outcomes must be disjoint", got)
		}
		scrape := renderToolCallParseFamily(m)
		want := `fak_gateway_toolcall_parse_total{outcome="structured",dialect="",reason=""} 1`
		if !strings.Contains(scrape, want+"\n") {
			t.Fatalf("scrape missing exact sample line %q\n--- scrape excerpt ---\n%s", want, toolCallParseExcerpt(scrape))
		}
	})

	t.Run("dropped_flag_with_calls_present_counts_structured", func(t *testing.T) {
		// Adversarial edge (mirrors floor_2088's dropped-but-calls-present case):
		// ToolCallsDropped=true but calls DID survive. The gate must not refuse,
		// and the outcome must be structured — never fail_closed.
		m := newGatewayMetrics(time.Now())
		s := &Server{logf: func(string, ...any) {}, metrics: m}
		comp := &agent.Completion{
			ToolCallsDropped:       true,
			ToolCallsDroppedReason: abi.ReasonMalformed,
			Message: agent.Message{
				Role:      agent.RoleAssistant,
				ToolCalls: []agent.ToolCall{{ID: "call_1", Type: "function", Function: agent.Func{Name: "Read", Arguments: `{}`}}},
			},
		}
		rec := httptest.NewRecorder()
		if !s.validateChatCompletionConformance(rec, nil, comp, comp.Message, false, false, false, nil) {
			t.Fatalf("dropped flag WITH surviving calls must pass; body=%s", rec.Body.String())
		}
		if got := toolCallParseCount(m, toolCallParseOutcomeStructured, "", ""); got != 1 {
			t.Fatalf("structured = %d, want 1 (snapshot=%v)", got, m.toolCallParseSnapshot())
		}
		if snap := m.toolCallParseSnapshot(); len(snap) != 1 {
			t.Fatalf("a turn with surviving calls recorded a fail_closed series too: %v", snap)
		}
	})

	t.Run("nil_metrics_server_does_not_panic", func(t *testing.T) {
		// Two existing tests (floor_2088_test.go, toolcall_refusal_log_test.go)
		// build a bare &Server{logf: ...} with NO metrics. Driving the same
		// conformance gate through it must not panic.
		s := &Server{logf: func(string, ...any) {}}
		if s.metrics != nil {
			t.Fatal("fixture precondition: metrics must be nil for this probe")
		}
		comp := &agent.Completion{
			ToolCallsDropped:       true,
			ToolCallsDroppedReason: abi.ReasonOversize,
			Message:                agent.Message{Role: agent.RoleAssistant},
		}
		rec := httptest.NewRecorder()
		if s.validateChatCompletionConformance(rec, nil, comp, comp.Message, false, false, false, nil) {
			t.Fatal("nil-metrics server admitted a dropped call; want fail-closed")
		}
		if rec.Code != http.StatusBadGateway {
			t.Fatalf("status = %d, want 502", rec.Code)
		}
		// The streamed sibling shares the same nil-metrics risk.
		rec2 := httptest.NewRecorder()
		if !s.failClosedOnUnparsedToolCalls(rec2, comp, false, "pre", "mid", func() {}) {
			t.Fatal("nil-metrics server did not take the streamed fail-closed turn")
		}
		// And the structured arm of the streamed rule.
		structured := &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant,
			ToolCalls: []agent.ToolCall{{ID: "c", Type: "function", Function: agent.Func{Name: "Read", Arguments: `{}`}}}}}
		if s.failClosedOnUnparsedToolCalls(httptest.NewRecorder(), structured, false, "pre", "mid", func() {}) {
			t.Fatal("nil-metrics server refused a conformant structured turn")
		}
	})

	t.Run("nil_receiver_observer_and_snapshot", func(t *testing.T) {
		var nilMetrics *gatewayMetrics
		// Must not panic.
		nilMetrics.observeToolCallParse(toolCallParseOutcomeFailClosed, "", "MALFORMED")
		nilMetrics.observeToolCallParse(toolCallParseOutcomeLifted, "hermes", "")
		nilMetrics.observeToolCallParse(toolCallParseOutcomeStructured, "", "")
		snap := nilMetrics.toolCallParseSnapshot()
		if snap == nil {
			t.Fatal("nil receiver snapshot returned a nil map; want an empty (non-nil) map")
		}
		if len(snap) != 0 {
			t.Fatalf("nil receiver snapshot = %v, want empty", snap)
		}
		// A nil-metrics Server's observer install must also be inert, not panicking.
		s := &Server{}
		ctx := s.withToolCallParseObserver(context.Background())
		if ctx == nil {
			t.Fatal("withToolCallParseObserver returned a nil context")
		}
	})

	t.Run("header_renders_with_zero_events", func(t *testing.T) {
		m := newGatewayMetrics(time.Now())
		var b strings.Builder
		writeToolCallParseMetrics(&b, m)
		out := b.String()
		if !strings.Contains(out, "# HELP "+toolCallParseFamily) {
			t.Fatalf("zero-event render missing HELP header:\n%s", out)
		}
		if !strings.Contains(out, "# TYPE "+toolCallParseFamily+" counter") {
			t.Fatalf("zero-event render missing TYPE counter header:\n%s", out)
		}
		if lines := toolCallParseSampleLines(out); len(lines) != 0 {
			t.Fatalf("zero-event render emitted sample lines: %v", lines)
		}
		// The same must hold through the full server scrape.
		srv := newTestServer(t)
		scrape := srv.renderMetrics()
		if !strings.Contains(scrape, "# HELP "+toolCallParseFamily) || !strings.Contains(scrape, "# TYPE "+toolCallParseFamily+" counter") {
			t.Fatalf("full scrape missing the family header on a fresh server")
		}
		if lines := toolCallParseSampleLines(scrape); len(lines) != 0 {
			t.Fatalf("fresh server scrape emitted sample lines: %v", lines)
		}
	})

	t.Run("render_is_deterministic_across_insertion_orders", func(t *testing.T) {
		type ev struct{ outcome, dialect, reason string }
		events := []ev{
			{toolCallParseOutcomeStructured, "", ""},
			{toolCallParseOutcomeFailClosed, "", "OVERSIZE"},
			{toolCallParseOutcomeLifted, "hermes", ""},
			{toolCallParseOutcomeFailClosed, "", "MALFORMED"},
			{toolCallParseOutcomeLifted, "bare_json", ""},
			{toolCallParseOutcomeLifted, "hermes", ""},
		}
		render := func(order []ev) string {
			m := newGatewayMetrics(time.Now())
			for _, e := range order {
				m.observeToolCallParse(e.outcome, e.dialect, e.reason)
			}
			var b strings.Builder
			writeToolCallParseMetrics(&b, m)
			return b.String()
		}
		forward := render(events)
		reversed := make([]ev, len(events))
		for i, e := range events {
			reversed[len(events)-1-i] = e
		}
		backward := render(reversed)
		if forward != backward {
			t.Fatalf("render is insertion-order dependent:\n--- forward ---\n%s\n--- reversed ---\n%s", forward, backward)
		}
		// And the order is the documented sorted (outcome, dialect, reason) order.
		lines := toolCallParseSampleLines(forward)
		if !sort.SliceIsSorted(lines, func(i, j int) bool { return lines[i] < lines[j] }) {
			t.Fatalf("sample lines are not in sorted order:\n%s", strings.Join(lines, "\n"))
		}
		// The repeated hermes lift must have accumulated to 2, not been overwritten.
		if !strings.Contains(forward, `fak_gateway_toolcall_parse_total{outcome="lifted",dialect="hermes",reason=""} 2`) {
			t.Fatalf("repeated lift did not accumulate:\n%s", forward)
		}
	})

	t.Run("label_values_are_prometheus_quoted", func(t *testing.T) {
		m := newGatewayMetrics(time.Now())
		// A hostile dialect/reason: a quote, a backslash and a newline. A raw
		// newline would split the scrape into a malformed line.
		m.observeToolCallParse(toolCallParseOutcomeLifted, "evil\"dialect\ninjected", "")
		m.observeToolCallParse(toolCallParseOutcomeFailClosed, "", "REASON\\WITH\"QUOTE\nAND_NEWLINE")
		var b strings.Builder
		writeToolCallParseMetrics(&b, m)
		out := b.String()
		lines := toolCallParseSampleLines(out)
		if len(lines) != 2 {
			t.Fatalf("expected 2 sample lines, got %d:\n%s", len(lines), out)
		}
		seen := map[string]string{}
		for _, line := range lines {
			if strings.ContainsAny(line, "\n\r") {
				t.Fatalf("sample line carries a raw newline: %q", line)
			}
			if !strings.HasPrefix(line, toolCallParseFamily+"{") || !strings.HasSuffix(line, "} 1") {
				t.Fatalf("malformed sample line: %q", line)
			}
			// A real label parser: it rejects a raw newline, an unterminated
			// quote, or a dangling backslash, and decodes the escapes.
			values := parsePromLabelValues(t, line)
			if len(values) != 3 {
				t.Fatalf("label set = %v, want exactly outcome/dialect/reason", values)
			}
			seen[values["outcome"]] = values["dialect"] + "\x00" + values["reason"]
		}
		// The hostile values must round-trip through the escape/parse cycle.
		if got := seen[toolCallParseOutcomeLifted]; got != "evil\"dialect\ninjected\x00" {
			t.Fatalf("dialect did not round-trip: %q", got)
		}
		if got := seen[toolCallParseOutcomeFailClosed]; got != "\x00REASON\\WITH\"QUOTE\nAND_NEWLINE" {
			t.Fatalf("reason did not round-trip: %q", got)
		}
		// And the raw bytes must be the escaped form, not the literal one.
		if !strings.Contains(out, `dialect="evil\"dialect\ninjected"`) {
			t.Fatalf("dialect was not escaped as a valid Prometheus label:\n%s", out)
		}
		if !strings.Contains(out, `reason="REASON\\WITH\"QUOTE\nAND_NEWLINE"`) {
			t.Fatalf("reason was not escaped as a valid Prometheus label:\n%s", out)
		}
	})

	t.Run("lifted_through_real_text_lift_observer", func(t *testing.T) {
		// Drive a REAL text lift end to end: an upstream that emits a Hermes
		// <tool_call> block as CONTENT, through the real HTTPPlanner, with the
		// gateway's own observer installed on the request context. The lift is
		// performed by agent.normalizeCompletionToolCalls, the same code the
		// served chat path runs.
		const content = `before <tool_call>{"name":"Bash","arguments":{"command":"ls"}}</tool_call> after`
		up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			quoted, _ := json.Marshal(content)
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprintf(w, `{"choices":[{"message":{"role":"assistant","content":%s},"finish_reason":"stop"}]}`, quoted)
		}))
		defer up.Close()

		m := newGatewayMetrics(time.Now())
		s := &Server{metrics: m}
		ctx := s.withToolCallParseObserver(context.Background())
		planner := agent.NewHTTPPlanner(up.URL, "model", "")
		comp, err := planner.Complete(ctx, nil, toolCallParseDefs("Bash"))
		if err != nil {
			t.Fatalf("planner.Complete: %v", err)
		}
		if len(comp.Message.ToolCalls) != 1 {
			t.Fatalf("the text lift did not promote the call: %+v", comp.Message)
		}
		if got := toolCallParseCount(m, toolCallParseOutcomeLifted, "hermes", ""); got != 1 {
			t.Fatalf("lifted/hermes = %d, want 1 (snapshot=%v)", got, m.toolCallParseSnapshot())
		}
		scrape := renderToolCallParseFamily(m)
		want := `fak_gateway_toolcall_parse_total{outcome="lifted",dialect="hermes",reason=""} 1`
		if !strings.Contains(scrape, want+"\n") {
			t.Fatalf("scrape missing exact lifted sample line %q\n--- scrape excerpt ---\n%s", want, toolCallParseExcerpt(scrape))
		}
	})

	t.Run("full_server_scrape_carries_the_sample_line", func(t *testing.T) {
		// The end-to-end wiring claim: a REAL server (metrics built by New) that
		// refused a turn must show the sample line in its FULL /metrics scrape,
		// not just in the family renderer. This is the assertion that
		// renderMetrics actually invokes writeToolCallParseMetrics.
		srv := newTestServer(t)
		comp := &agent.Completion{
			ToolCallsDropped:       true,
			ToolCallsDroppedReason: abi.ReasonOversize,
			FinishReason:           "length",
			Usage:                  agent.Usage{CompletionTokens: 24},
			Message:                agent.Message{Role: agent.RoleAssistant},
		}
		rec := httptest.NewRecorder()
		if srv.validateChatCompletionConformance(rec, nil, comp, comp.Message, false, false, false, nil) {
			t.Fatal("real server admitted a dropped call; want fail-closed")
		}
		scrape := srv.renderMetrics()
		want := `fak_gateway_toolcall_parse_total{outcome="fail_closed",dialect="",reason="OVERSIZE"} 1`
		if !strings.Contains(scrape, want+"\n") {
			t.Fatalf("full server scrape missing %q\n--- family lines ---\n%s", want, toolCallParseExcerpt(scrape))
		}
	})

	t.Run("concurrent_observe_and_snapshot", func(t *testing.T) {
		// The spec: the snapshot copies the live map UNDER ITS LOCK. A torn read
		// or a lost increment shows up here (run with -race for the full check).
		m := newGatewayMetrics(time.Now())
		const writers, perWriter = 8, 200
		done := make(chan struct{})
		for w := 0; w < writers; w++ {
			go func(w int) {
				defer func() { done <- struct{}{} }()
				for i := 0; i < perWriter; i++ {
					m.observeToolCallParse(toolCallParseOutcomeLifted, fmt.Sprintf("dialect_%d", w), "")
				}
			}(w)
		}
		// Concurrent readers must never observe a torn map.
		for r := 0; r < 4; r++ {
			go func() {
				defer func() { done <- struct{}{} }()
				for i := 0; i < perWriter; i++ {
					_ = m.toolCallParseSnapshot()
				}
			}()
		}
		for i := 0; i < writers+4; i++ {
			<-done
		}
		var total uint64
		for _, v := range m.toolCallParseSnapshot() {
			total += v
		}
		if want := uint64(writers * perWriter); total != want {
			t.Fatalf("total observed = %d, want %d (lost increments)", total, want)
		}
	})

	t.Run("observer_ignores_zero_call_lifts", func(t *testing.T) {
		// The observer contract: a lift event with zero promoted calls is not an
		// outcome. A rejected/unoffered block must not mint a lifted series.
		m := newGatewayMetrics(time.Now())
		s := &Server{metrics: m}
		ctx := s.withToolCallParseObserver(context.Background())
		// "Other" is not offered, so the block is rejected and nothing lifts.
		content := `<tool_call>{"name":"Other","arguments":{}}</tool_call>`
		up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			quoted, _ := json.Marshal(content)
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprintf(w, `{"choices":[{"message":{"role":"assistant","content":%s},"finish_reason":"stop"}]}`, quoted)
		}))
		defer up.Close()
		planner := agent.NewHTTPPlanner(up.URL, "model", "")
		if _, err := planner.Complete(ctx, nil, toolCallParseDefs("Bash")); err != nil {
			t.Fatalf("planner.Complete: %v", err)
		}
		if snap := m.toolCallParseSnapshot(); len(snap) != 0 {
			t.Fatalf("an unoffered block minted a parse outcome: %v", snap)
		}
	})
}

// toolCallParseExcerpt trims a full scrape down to the family's own lines so a
// failure message stays readable.
func toolCallParseExcerpt(scrape string) string {
	var out []string
	for _, line := range strings.Split(scrape, "\n") {
		if strings.Contains(line, toolCallParseFamily) {
			out = append(out, line)
		}
	}
	if len(out) == 0 {
		return "(family absent from scrape)"
	}
	return strings.Join(out, "\n")
}
