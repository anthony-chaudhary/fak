package gateway

// The TOON wire (#3067): flag-gated, fail-safe application of the governed TOON
// auto-fire decision (internal/toon.Decide, #3066) at the ONE place every MCP tool
// result leaves the gateway — mcpToolResult. Default OFF: with FAK_TOON_WIRE unset
// the wire is a byte-identical no-op and every tool result ships exactly the JSON it
// shipped before this file existed. With the flag on, a payload is TOON-encoded ONLY
// when every Decide gate proves a net token win on a reversible encoding; any skip,
// error, or doubt falls back to the canonical JSON (fail-safe identity).
//
// Signal wiring (the part #3066 deliberately left as inputs):
//   - CacheResident <- the span was served by the vDSO fast path (Verdict.By ==
//     "vdso"): its bytes repeat an earlier emission that may already sit inside the
//     client's cached prompt prefix, so re-encoding it turns a cheap cache read into
//     a full-price recompute. CACHE_PREFIX_RESIDENT wins over every shape gate.
//   - Volatile <- agent.HeadValueIsVolatile over the payload's head value — the SAME
//     UUID/sub-day-timestamp evidence anthropic_cachebp.go anchors breakpoints with.
//   - ModelFitnessKnown stays false: the gateway cannot see which model reads an MCP
//     client's tool results, and a fabricated fitness would be dishonest. The
//     MODEL_TOON_UNFIT gate (and its one-time primer) stays inert until a model
//     registry signal reaches this seam.
//   - Tokenizer stays nil (the codec's bytes/4 yardstick): no target-model tokenizer
//     is identifiable here for the same reason.

import (
	"bytes"
	"encoding/json"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/toon"
)

// toonWireEnabled reports whether FAK_TOON_WIRE is on. Default OFF (#3067 ships the
// wire dark). Read per call — not latched at process start — so the ablate sweep's
// subprocess env controls each arm deterministically and tests flip it with t.Setenv.
func toonWireEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("FAK_TOON_WIRE"))) {
	case "1", "on", "true", "yes":
		return true
	}
	return false
}

// toonCacheResident reports whether a tool-result payload is a cache-resident span:
// a syscall response the vDSO fast path answered (Verdict.By == "vdso", set by the
// kernel's fast-path arm and carried to the wire by renderVerdict). A vDSO-served
// result repeats bytes this session already shipped, so it must ship byte-identical
// JSON — re-encoding would invalidate any client-side cached prefix holding the
// first emission.
func toonCacheResident(v any) bool {
	switch r := v.(type) {
	case SyscallResponse:
		return r.Verdict.By == "vdso"
	case *SyscallResponse:
		return r != nil && r.Verdict.By == "vdso"
	}
	return false
}

// toonHeadRaw returns the payload's head value as raw JSON — the first element of an
// array, the first field value of an object (encoding/json emits struct fields in
// declaration order, so for a SyscallResponse this is the verdict, not the trace id),
// or the whole scalar. This mirrors how anthropic_cachebp.go judges prefix stability
// by the span's LEADING value: a volatile tail (e.g. a trailing trace_id) does not
// condemn a stable head.
func toonHeadRaw(jsonBytes []byte) json.RawMessage {
	d := json.NewDecoder(bytes.NewReader(jsonBytes))
	t, err := d.Token()
	if err != nil {
		return nil
	}
	switch t {
	case json.Delim('{'):
		if _, err := d.Token(); err != nil { // first key (or '}' on an empty object)
			return nil
		}
	case json.Delim('['):
	default:
		return jsonBytes // scalar: the whole value is the head
	}
	var raw json.RawMessage
	if err := d.Decode(&raw); err != nil {
		return nil
	}
	return raw
}

// toonWireText runs the governed TOON decision over a tool-result payload and returns
// the TOON text and true ONLY when the flag is on AND every #3066 gate fires. Every
// other path — flag off, cache-resident span, volatile head, shape/size/round-trip/
// net-token skip, malformed payload — returns ok=false, leaving the caller's canonical
// JSON untouched. jsonBytes is the already-marshaled canonical encoding; the payload is
// re-decoded from it into json-native values (map/slice/float64/...) because Decide's
// round-trip witness — Decode(Encode(p)) deep-equals p — is only provable on that
// domain, and the wire's unit of meaning IS those bytes, not the Go struct behind them.
func toonWireText(jsonBytes []byte, cacheResident bool) (string, bool) {
	if !toonWireEnabled() {
		return "", false
	}
	var payload any
	if err := json.Unmarshal(jsonBytes, &payload); err != nil {
		return "", false
	}
	d := toon.Decide(payload, toon.DecideInput{
		CacheResident: cacheResident,
		Volatile:      agent.HeadValueIsVolatile(toonHeadRaw(jsonBytes)),
	})
	if !d.Encode {
		return "", false
	}
	enc, err := toon.Encode(payload, toon.Options{})
	if err != nil {
		return "", false
	}
	return string(enc), true
}
