package gateway

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"
)

// legacyGatewayOperationJSON is the map[string]any encoding logGatewayOperation used
// before #4969 replaced it with a struct. It is kept HERE, in the test, as the oracle
// the new encoder must reproduce byte-for-byte: the struct swap is a pure allocation
// optimization on the request path, so any observable change in the emitted log line
// would be a regression for every consumer parsing the JSON stream.
func legacyGatewayOperationJSON(operation, traceID, tool string, v WireVerdict, opErr error, dur time.Duration) []byte {
	verdict := v.Kind
	if opErr != nil && verdict == "" {
		verdict = "ERROR"
	}
	ev := map[string]any{
		"event":       "gateway_operation",
		"operation":   operation,
		"tool":        tool,
		"trace_id":    strings.TrimSpace(traceID),
		"verdict":     verdict,
		"duration_ms": float64(dur.Microseconds()) / 1000.0,
	}
	if v.Reason != "" {
		ev["reason"] = v.Reason
	}
	if v.By != "" {
		ev["by"] = v.By
	}
	if v.Disposition != "" {
		ev["disposition"] = v.Disposition
	}
	if opErr != nil {
		ev["error"] = opErr.Error()
	}
	b, err := json.Marshal(ev)
	if err != nil {
		panic(err)
	}
	return b
}

// TestGatewayOperationLogIsByteIdenticalToLegacyMapForm is the failure-class proof for
// the #4969 request-path allocation cut. logGatewayOperation now marshals a struct
// instead of a map[string]any; encoding/json sorts MAP keys but emits STRUCT fields in
// declaration order, so a field declared out of alphabetical order (or a stray
// omitempty on an always-present field) would silently reorder or drop a key in the
// live log stream. This pins the real captured bytes against the pre-#4969 oracle.
func TestGatewayOperationLogIsByteIdenticalToLegacyMapForm(t *testing.T) {
	cases := []struct {
		name      string
		operation string
		traceID   string
		tool      string
		v         WireVerdict
		err       error
		dur       time.Duration
	}{
		{
			name:      "allow_minimal_only_always_present_keys",
			operation: "adjudicate",
			traceID:   "trace-1",
			tool:      "read_kb",
			v:         WireVerdict{Kind: "ALLOW"},
			dur:       5 * time.Microsecond,
		},
		{
			name:      "deny_all_optional_keys_present",
			operation: "syscall",
			traceID:   "  trace-2  ", // must be TrimSpace'd, as the map form did
			tool:      "Bash",
			v:         WireVerdict{Kind: "DENY", Reason: "POLICY_BLOCK", By: "monitor", Disposition: "TERMINAL"},
			dur:       1234 * time.Microsecond,
		},
		{
			name:      "error_supplies_verdict_when_kind_empty",
			operation: "adjudicate",
			traceID:   "trace-3",
			tool:      "x",
			v:         WireVerdict{},
			err:       errors.New("boom"),
			dur:       0, // duration_ms is always emitted, including the 0 value
		},
		{
			name:      "error_alongside_explicit_verdict",
			operation: "admit",
			traceID:   "trace-4",
			tool:      "y",
			v:         WireVerdict{Kind: "DENY", Reason: "r"},
			err:       errors.New("upstream 500"),
			dur:       999 * time.Microsecond,
		},
		{
			name:      "empty_tool_and_trace_still_emit_their_keys",
			operation: "adjudicate",
			traceID:   "   ",
			tool:      "",
			v:         WireVerdict{Kind: "ALLOW"},
			dur:       time.Microsecond,
		},
		{
			// encoding/json HTML-escapes <, > and & by default. The struct path must not
			// quietly change that for a tool name or reason carrying markup.
			name:      "html_escaping_and_quotes_preserved",
			operation: "adjudicate",
			traceID:   "trace-5",
			tool:      `<script>&"quoted"`,
			v:         WireVerdict{Kind: "DENY", Reason: `a<b&c>d`, By: `he said "hi"`},
			dur:       42 * time.Microsecond,
		},
		{
			// duration_ms is float64(micros)/1000.0 — pin the float formatting, which is
			// where a hand-rolled encoder would most easily drift.
			name:      "sub_millisecond_float_formatting",
			operation: "adjudicate",
			traceID:   "trace-6",
			tool:      "t",
			v:         WireVerdict{Kind: "ALLOW"},
			dur:       1 * time.Nanosecond, // Microseconds() truncates to 0 => "0"
		},
		{
			name:      "multi_second_float_formatting",
			operation: "syscall",
			traceID:   "trace-7",
			tool:      "t",
			v:         WireVerdict{Kind: "ALLOW"},
			dur:       90*time.Second + 500*time.Microsecond,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Capture what the LIVE path actually writes, not a re-derived copy.
			var got string
			srv := &Server{logf: func(format string, args ...any) {
				got = strings.TrimSpace(fmt.Sprintf(format, args...))
			}}
			srv.logGatewayOperation(tc.operation, tc.traceID, tc.tool, tc.v, tc.err, tc.dur)

			want := string(legacyGatewayOperationJSON(tc.operation, tc.traceID, tc.tool, tc.v, tc.err, tc.dur))
			if got != want {
				t.Fatalf("gateway_operation log line drifted from the pre-#4969 map form:\n got: %s\nwant: %s", got, want)
			}
			// Belt and braces: it must still be parseable JSON naming this event.
			var back map[string]any
			if err := json.Unmarshal([]byte(got), &back); err != nil {
				t.Fatalf("emitted log line is not valid JSON: %v (%s)", err, got)
			}
			if back["event"] != "gateway_operation" {
				t.Fatalf("event = %v, want gateway_operation", back["event"])
			}
		})
	}
}

// countingError counts how many times its Error() string is demanded. It is the probe
// that makes the sinkless path observable: logGatewayOperation reads opErr.Error() ONLY
// while BUILDING the event, so the call count separates the two outcomes that produce
// the same (empty) output — "returned before doing the work" and "did the work, then
// threw it away". Nothing else can tell them apart, because with no sink there is by
// definition nothing to inspect.
type countingError struct{ n int }

func (e *countingError) Error() string { e.n++; return "boom" }

// TestGatewayOperationLogSkippedWithoutSink pins the cheap exit: a Server with no log
// sink must not build or marshal the event at all. Every gateway operation calls this
// on the request path, so a guard that ran AFTER the map build + json.Marshal would be
// an allocation on every adjudication of every unsunk server — invisible in output,
// visible only in cost.
func TestGatewayOperationLogSkippedWithoutSink(t *testing.T) {
	// Control first: with a sink the probe MUST be consulted and a line MUST be
	// written. Without this, the zero-count assertions below would also hold for a
	// probe the production path never touches under ANY input — an assertion that
	// cannot fail is the same vacuum in a costume.
	probe := &countingError{}
	lines := 0
	sunk := &Server{logf: func(format string, args ...any) { lines++; _ = fmt.Sprintf(format, args...) }}
	sunk.logGatewayOperation("adjudicate", "t", "x", WireVerdict{Kind: "ALLOW"}, probe, time.Microsecond)
	if lines != 1 {
		t.Fatalf("with a sink: %d log lines, want 1", lines)
	}
	if probe.n == 0 {
		t.Fatal("with a sink: Error() was never called, so the probe witnesses nothing and the skip assertions below are vacuous")
	}

	// The two sinkless shapes. Neither may panic (a nil receiver is reachable: the
	// helper is called on servers that were never given a logger), neither can write
	// anywhere, and neither may BUILD the event first.
	for _, tc := range []struct {
		name string
		srv  *Server
	}{
		{"nil_receiver", nil},
		{"server_without_logf", &Server{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := &countingError{}
			// Every optional field is populated so a build would touch the most work
			// possible — and the error field, which is the one the probe watches.
			tc.srv.logGatewayOperation("adjudicate", "  t  ", "x",
				WireVerdict{Kind: "DENY", Reason: "POLICY_BLOCK", By: "monitor", Disposition: "TERMINAL"},
				p, time.Microsecond)
			if p.n != 0 {
				t.Fatalf("Error() called %d times with no sink: the event was BUILT and then discarded, not skipped", p.n)
			}
		})
	}
}
