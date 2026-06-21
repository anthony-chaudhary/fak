package ctxmmu_test

// Deterministic witness tests closing two OPEN math-proof obligations for the
// context-MMU (see fak/docs/proofs). Both are metamorphic/invariant assertions,
// not smoke tests:
//
//	(1) page-out-idempotent  — once Admit has paged-out / quarantined a result
//	    in-place (so r.Payload now points at the {"_quarantined":true,...} or
//	    {"_paged":true,...} stub), a SECOND Admit on that same *Result is a no-op:
//	    it returns VerdictAllow and leaves the paged/quarantine counters AND
//	    r.Payload unchanged. (The stub is a small, clean JSON object, so it can
//	    never re-trip Screen/oversize.)
//
//	(2) benign-byte-identical — for a clean body (no secret, no injection marker,
//	    no degenerate repeat, len <= OversizeBytes) Admit returns VerdictAllow and
//	    leaves r.Payload byte-identical to the input.
//
// These use the same external-test harness as ctxmmu_test.go: the blank blob
// import below registers the "blob" PageOut/Resolver backend so paged stubs
// actually materialize.

import (
	"bytes"
	"context"
	"fmt"
	"math/rand"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	_ "github.com/anthony-chaudhary/fak/internal/blob" // registers the "blob" PageOut/Resolver backend
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// admitResult builds a StatusOK Result with an inline payload body.
func admitResult(c *abi.ToolCall, body []byte) *abi.Result {
	return &abi.Result{
		Call:    c,
		Status:  abi.StatusOK,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: body},
	}
}

func admitCall(tool string) *abi.ToolCall {
	return &abi.ToolCall{
		Tool: tool,
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{}`)},
		Meta: map[string]string{"readOnlyHint": "true"},
	}
}

// materialize resolves a (possibly paged-out) Ref to its bytes.
func materialize(t *testing.T, ctx context.Context, r abi.Ref) []byte {
	t.Helper()
	if r.Kind == abi.RefInline {
		return append([]byte(nil), r.Inline...)
	}
	res := abi.ActiveResolver()
	if res == nil {
		t.Fatalf("no active resolver to materialize ref")
	}
	b, err := res.Resolve(ctx, r)
	if err != nil {
		t.Fatalf("resolve ref: %v", err)
	}
	return b
}

// refEqual reports whether two Refs denote the same in-context payload: same
// kind, same digest, and byte-identical materialized content. (abi.Ref is not
// `==`-comparable because it carries a []byte, so we compare structurally.)
func refEqual(t *testing.T, ctx context.Context, a, b abi.Ref) bool {
	t.Helper()
	if a.Kind != b.Kind || a.Digest != b.Digest || a.Len != b.Len {
		return false
	}
	return bytes.Equal(materialize(t, ctx, a), materialize(t, ctx, b))
}

// --- OPEN (1): page-out-idempotent ------------------------------------------
//
// A second Admit on an already-paged-out / already-quarantined result is a no-op
// (VerdictAllow, counters unchanged, r.Payload unchanged). Witnessed on BOTH
// stub shapes: the quarantine stub ({"_quarantined":...}) and the oversize
// page-out stub ({"_paged":...}).

func TestProofPageOutIdempotent(t *testing.T) {
	ctx := context.Background()

	cases := []struct {
		name      string
		tool      string
		body      []byte
		firstKind abi.Verdict
	}{
		{
			name: "quarantine-secret",
			tool: "read_file",
			// secret-shaped body -> first Admit quarantines in-place.
			body:      []byte("config: api_key=sk-abcdef0123456789abcdef0123 from prod env"),
			firstKind: abi.Verdict{Kind: abi.VerdictQuarantine},
		},
		{
			name:      "oversize-paged",
			tool:      "dump_table",
			body:      distinctBenign(100 * 1024), // > OversizeBytes, non-repeating, clean
			firstKind: abi.Verdict{Kind: abi.VerdictTransform},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := ctxmmu.New()
			c := admitCall(tc.tool)
			r := admitResult(c, append([]byte(nil), tc.body...))

			// First Admit: pages the bytes out and rewrites r.Payload in-place.
			v1 := m.Admit(ctx, c, r)
			if v1.Kind != tc.firstKind.Kind {
				t.Fatalf("first Admit: want %v, got %v", tc.firstKind.Kind, v1.Kind)
			}

			// For the Transform (oversize) case, Admit returns the new pointer in
			// the verdict payload but does NOT itself mutate r.Payload — the
			// caller applies the transform. To witness idempotence on the STUB we
			// install the paged pointer as r.Payload, exactly as the result-admit
			// pipeline does, then re-admit. For the Quarantine case Admit already
			// rewrote r.Payload in-place.
			if v1.Kind == abi.VerdictTransform {
				tp, ok := v1.Payload.(abi.TransformPayload)
				if !ok {
					t.Fatalf("Transform payload is not TransformPayload: %T", v1.Payload)
				}
				r.Payload = tp.NewArgs
			}

			// Snapshot the post-page-out state: the in-context payload bytes and
			// the gate's counters.
			stubBytes := materialize(t, ctx, r.Payload)
			payloadBefore := r.Payload
			qBefore, totalBefore, _ := m.PollutionRate()

			// Sanity: the stub really is a ctxmmu page-out/quarantine stub.
			if !(bytes.Contains(stubBytes, []byte(`"_quarantined"`)) ||
				bytes.Contains(stubBytes, []byte(`"_paged"`))) {
				t.Fatalf("post-page-out payload is not a ctxmmu stub: %q", stubBytes)
			}
			// And it is small + clean, so it cannot re-trip Screen/oversize.
			if reason, screened := ctxmmu.ScreenBytes(stubBytes); screened {
				t.Fatalf("stub unexpectedly screens unsafe (reason %s): %q",
					abi.ReasonName(reason), stubBytes)
			}
			if len(stubBytes) > ctxmmu.OversizeBytes {
				t.Fatalf("stub unexpectedly oversize (%d > %d)", len(stubBytes), ctxmmu.OversizeBytes)
			}

			// Second Admit on the already-paged-out result.
			v2 := m.Admit(ctx, c, r)

			// (a) it allows the stub through as-is.
			if v2.Kind != abi.VerdictAllow {
				t.Fatalf("re-admit of stub: want VerdictAllow, got %v (reason %s)",
					v2.Kind, abi.ReasonName(v2.Reason))
			}
			// (b) it did NOT page anything out again: the quarantine counter is
			//     unchanged. (total advances by one — every Admit counts a call —
			//     but no NEW quarantine/page-out occurred.)
			qAfter, totalAfter, _ := m.PollutionRate()
			if qAfter != qBefore {
				t.Fatalf("re-admit changed quarantine count: before=%d after=%d", qBefore, qAfter)
			}
			if totalAfter != totalBefore+1 {
				t.Fatalf("re-admit total: want before+1 (%d), got %d", totalBefore+1, totalAfter)
			}
			// (c) r.Payload is unchanged by the re-admission (same ref, same bytes).
			if !refEqual(t, ctx, payloadBefore, r.Payload) {
				t.Fatalf("re-admit mutated r.Payload: before=%+v after=%+v", payloadBefore, r.Payload)
			}
			if got := materialize(t, ctx, r.Payload); !bytes.Equal(got, stubBytes) {
				t.Fatalf("re-admit mutated stub bytes:\n before %q\n after  %q", stubBytes, got)
			}
		})
	}
}

// --- OPEN (2): benign-byte-identical ----------------------------------------
//
// For any clean body (no secret, no injection, no degenerate repeat, len <=
// OversizeBytes) Admit returns VerdictAllow and leaves r.Payload byte-identical
// to the input. Driven over many randomized-but-deterministic clean bodies
// (fixed seed) plus targeted edge cases.

func TestProofBenignByteIdentical(t *testing.T) {
	ctx := context.Background()
	rng := rand.New(rand.NewSource(0xC7C7C7C7)) // fixed seed -> deterministic

	// A safe alphabet that contains no secret prefix / injection marker and,
	// because each emitted token carries a fresh counter, cannot form a >50x
	// 16-byte repeat run.
	const alpha = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:-_/(){}[]"

	bodies := make([][]byte, 0, 260)

	// Edge cases first.
	bodies = append(bodies,
		[]byte(``),
		[]byte(`{}`),
		[]byte(`{"reservation_id":"ABC123","status":"confirmed","seat":"14C"}`),
		[]byte("plain log line: request handled in 12ms, status 200"),
		// Exactly OversizeBytes, clean & non-repeating (distinct 16-byte windows):
		// boundary case proving len==OversizeBytes is NOT treated as oversize.
		distinctBenign(ctxmmu.OversizeBytes)[:ctxmmu.OversizeBytes],
	)

	// Randomized clean bodies of varying length (all <= OversizeBytes), each
	// salted with an incrementing per-token counter so the repeat detector
	// (16-byte chunk x >50) never fires.
	for n := 0; n < 256; n++ {
		size := 1 + rng.Intn(ctxmmu.OversizeBytes) // 1 .. OversizeBytes
		var b bytes.Buffer
		i := 0
		for b.Len() < size {
			fmt.Fprintf(&b, "t%08d-", i)
			// add a few random safe chars to vary content
			for k := 0; k < 3 && b.Len() < size; k++ {
				b.WriteByte(alpha[rng.Intn(len(alpha))])
			}
			i++
		}
		body := b.Bytes()[:size]
		bodies = append(bodies, append([]byte(nil), body...))
	}

	for idx, body := range bodies {
		// Precondition guard: the body must genuinely be clean & in-size, else
		// the case is out of the theorem's scope (and we'd be asserting a false
		// hypothesis). Skip any that aren't, loudly.
		if len(body) > ctxmmu.OversizeBytes {
			t.Fatalf("case %d body exceeds OversizeBytes (%d) — test bug", idx, len(body))
		}
		if reason, screened := ctxmmu.ScreenBytes(body); screened {
			t.Fatalf("case %d body unexpectedly screens unsafe (reason %s): %q",
				idx, abi.ReasonName(reason), body)
		}

		m := ctxmmu.New()
		c := admitCall("benign")
		input := append([]byte(nil), body...) // independent copy for comparison
		r := admitResult(c, body)

		v := m.Admit(ctx, c, r)

		// (a) clean body -> VerdictAllow.
		if v.Kind != abi.VerdictAllow {
			t.Fatalf("case %d: clean body should Allow, got %v (reason %s): %q",
				idx, v.Kind, abi.ReasonName(v.Reason), body)
		}
		// (b) not flagged as quarantined.
		if ctxmmu.Quarantined(r) {
			t.Fatalf("case %d: clean body marked Quarantined", idx)
		}
		// (c) payload is byte-identical to the input — no mutation / no page-out.
		if r.Payload.Kind != abi.RefInline {
			t.Fatalf("case %d: clean body payload kind changed to %d (page-out?)", idx, r.Payload.Kind)
		}
		if !bytes.Equal(r.Payload.Inline, input) {
			t.Fatalf("case %d: clean body bytes mutated\n before %q\n after  %q", idx, input, r.Payload.Inline)
		}
		// (d) no quarantine/page-out counted.
		if q, _, _ := m.PollutionRate(); q != 0 {
			t.Fatalf("case %d: clean body bumped quarantine count to %d", idx, q)
		}
	}
}

// distinctBenign builds a > n-byte clean body whose 16-byte windows are all
// distinct (so the repeat detector never fires) and that holds no secret /
// injection marker — the oversize-but-benign shape that pages out.
func distinctBenign(n int) []byte {
	var b bytes.Buffer
	b.WriteString(`{"x":"`)
	i := 0
	for b.Len() < n {
		fmt.Fprintf(&b, "row-%08d-filler;", i)
		i++
	}
	b.WriteString(`"}`)
	return b.Bytes()
}
