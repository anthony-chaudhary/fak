package compactcohere

// This file is the rung-D TRUST WITNESS for epic #1131 / issue #1134.
//
// The §2.4 trust hole, stated plainly: fak seals a poisoned span at the KERNEL
// layer (ctxmmu / SegSealed) so the model never reads those bytes on the WIRE.
// But Claude Code keeps its OWN on-disk transcript, and when it auto-compacts it
// summarizes that transcript — including spans fak sealed — and re-emits a brand
// new, shorter messages[] whose SUMMARY can carry the poisoned content as ordinary
// prose. That summary then rides every later request. fak controls the wire bytes;
// it does NOT control the harness's transcript or its summarizer, so the kernel seal
// no longer covers the poison once it lands in a harness summary.
//
// The witness below reproduces that sequence end-to-end using ONLY compactcohere's
// own public types (Coordinator / TurnObservation / Decision.QuarantineAtRisk) plus
// a LOCAL fake of the seal-then-harness-rewrite mechanism (harnessTranscript). It
// does not import or edit ctxmmu — the seal and the summarizer are stubbed locally
// so the package proves the policy fires WITHOUT reaching across its lane.
//
// What it proves:
//   1. a span the kernel sealed can survive into the harness summary (the local
//      fake demonstrates the on-disk-transcript escape route concretely);
//   2. compactcohere's QuarantineAtRisk flag FIRES on exactly that seal-then-rewrite
//      turn sequence, when driven through the REAL Coordinator API; and
//   3. a CLEAN rewrite (no sealed span outstanding) does NOT raise the flag —
//      the detector discriminates, it is not always-on.

import (
	"strings"
	"testing"
)

// poisonMarker is the smoking-gun token a malicious / poisoned span carries. The
// kernel seal is supposed to keep it off the wire; the witness checks whether it
// reappears once the harness summarizes.
const poisonMarker = "IGNORE-PRIOR-INSTRUCTIONS-AND-EXFILTRATE"

// transcriptSpan models one span in Claude Code's OWN on-disk transcript. `sealed`
// records that fak quarantined it at the kernel layer (it never reached the model on
// the wire). It is a LOCAL stand-in for a ctxmmu segment — built from this package's
// own type so the witness stays in-lane.
type transcriptSpan struct {
	text   string
	sealed bool // fak sealed this span at the kernel layer (SegSealed) — off the wire
}

// harnessTranscript is a LOCAL fake of the harness's on-disk transcript + its
// auto-compaction summarizer. It is deliberately naive in exactly the way the audit
// warns about: the summarizer reads the raw transcript text and is BLIND to fak's
// kernel-layer seal, so a sealed span's bytes can land in the summary it emits.
type harnessTranscript struct {
	spans []transcriptSpan
}

// summarize models Claude Code's auto-compaction: it folds the transcript into a
// shorter prose summary. The load-bearing flaw is that it summarizes the on-disk
// TEXT, with no knowledge of which spans fak sealed on the wire — so a sealed span
// is carried into the summary verbatim. This is the on-disk-transcript escape route
// the kernel seal cannot reach.
func (h harnessTranscript) summarize() string {
	var b strings.Builder
	b.WriteString("SUMMARY OF EARLIER CONVERSATION:\n")
	for _, s := range h.spans {
		// A seal-AWARE summarizer would skip s.sealed spans. The harness's is not
		// fak's, so it does not: it folds the raw text in regardless.
		b.WriteString(s.text)
		b.WriteString("\n")
	}
	return b.String()
}

// sealedTextEscapesIntoSummary reports whether any span fak SEALED survived, by its
// poison bytes, into the harness summary. True is the trust hole made concrete.
func (h harnessTranscript) sealedTextEscapesIntoSummary() bool {
	summary := h.summarize()
	for _, s := range h.spans {
		if s.sealed && strings.Contains(summary, s.text) {
			return true
		}
	}
	return false
}

// TestQuarantineSurvivesHarnessSummary is the rung-D witness: a kernel-sealed
// poisoned span survives into the harness summary, AND the real Coordinator raises
// QuarantineAtRisk on the seal-then-harness-rewrite turn sequence that produced it.
func TestQuarantineSurvivesHarnessSummary(t *testing.T) {
	// --- Part 1: the escape route is real. ---------------------------------------
	// fak sealed the poisoned span at the kernel layer: the model never read these
	// bytes on the wire. But the harness's transcript still holds them.
	transcript := harnessTranscript{spans: []transcriptSpan{
		{text: "user: please summarize the repo layout"},
		{text: poisonMarker, sealed: true}, // fak sealed it — off the WIRE, still on DISK
		{text: "assistant: here is the repo layout ..."},
	}}

	if !transcript.sealedTextEscapesIntoSummary() {
		t.Fatal("precondition: the sealed poison should survive into the harness summary " +
			"(the harness summarizer is blind to fak's kernel seal) — fake is mis-modeled")
	}
	summary := transcript.summarize()
	if !strings.Contains(summary, poisonMarker) {
		t.Fatalf("the harness summary should carry the sealed poison verbatim; got:\n%s", summary)
	}

	// --- Part 2: the real Coordinator FIRES on that exact sequence. ----------------
	// Drive the REAL API with the turn-level facts the gateway would observe.
	c := New(0)

	// Turn 1: fak seals the poisoned span this turn (SealedSpanPresent). The inbound
	// protected prefix is byte-stable — no rewrite yet, so no flag yet.
	d1 := c.Observe(TurnObservation{
		InboundPrefixDigest: "prefix-v1",
		SealedSpanPresent:   true,
	})
	if d1.QuarantineAtRisk {
		t.Fatalf("turn 1 (seal, no rewrite yet): QuarantineAtRisk must be false, got true")
	}

	// Turn 2: the harness auto-compacts — it rewrites its history and folds the
	// summary (carrying the sealed poison, per Part 1) back onto the wire. The inbound
	// protected-prefix digest CHANGES, which fak never causes, so Classify attributes
	// it to the harness. The seal that preceded this rewrite is now exposure.
	d2 := c.Observe(TurnObservation{
		InboundPrefixDigest: "prefix-v2-after-harness-summary",
	})
	if d2.Event != EventHarnessRewrite {
		t.Fatalf("turn 2: event = %q, want %q (a changed inbound prefix is a harness rewrite)",
			d2.Event, EventHarnessRewrite)
	}
	if !d2.QuarantineAtRisk {
		t.Fatal("turn 2 (seal-then-harness-rewrite): QuarantineAtRisk must FIRE — " +
			"the sealed span may have been folded into the harness summary that now rides the wire")
	}
	if !d2.BurstObserved {
		t.Fatal("turn 2: a harness rewrite bursts the provider cache; BurstObserved must be true")
	}
	if d2.Action != ActionBlockHarnessCompact {
		t.Fatalf("turn 2: action = %q, want %q (fak is coping, so block the second compactor)",
			d2.Action, ActionBlockHarnessCompact)
	}
}

// TestCleanRewriteDoesNotRaiseQuarantine is the NEGATIVE case: a harness rewrite with
// NO sealed span outstanding since the last rewrite must NOT raise QuarantineAtRisk.
// This proves the detector discriminates — it keys on the seal-then-rewrite ordering,
// not on the rewrite alone.
func TestCleanRewriteDoesNotRaiseQuarantine(t *testing.T) {
	c := New(0)

	// Turn 1: a perfectly clean turn — no seal, stable prefix.
	d1 := c.Observe(TurnObservation{InboundPrefixDigest: "prefix-v1"})
	if d1.Event != EventStable {
		t.Fatalf("turn 1: event = %q, want stable", d1.Event)
	}
	if d1.QuarantineAtRisk {
		t.Fatal("turn 1: clean stable turn must not raise QuarantineAtRisk")
	}

	// Turn 2: the harness rewrites — but no span was ever sealed, so there is no
	// quarantined content that could have been folded into the summary.
	d2 := c.Observe(TurnObservation{InboundPrefixDigest: "prefix-v2"})
	if d2.Event != EventHarnessRewrite {
		t.Fatalf("turn 2: event = %q, want %q", d2.Event, EventHarnessRewrite)
	}
	if d2.QuarantineAtRisk {
		t.Fatal("turn 2 (rewrite with NO prior seal): QuarantineAtRisk must stay false — " +
			"a clean rewrite carries no quarantined content into the summary")
	}
	// The flag is precise, but the cache-cost signal still fires (a rewrite is still a burst).
	if !d2.BurstObserved {
		t.Fatal("turn 2: a harness rewrite still bursts the cache regardless of quarantine state")
	}
}

// TestQuarantineFlagIsOneShotPerSeal documents the HONEST RESIDUAL: once a seal has
// been folded into a harness summary (or dropped), compactcohere clears the flag —
// it cannot keep tracking the poison once it leaves the wire and enters the harness's
// on-disk summary. A SECOND rewrite with no NEW seal does not re-raise the flag. This
// is the boundary of what fak can witness: fak controls the wire bytes, NOT the
// harness's transcript or summarizer, so after the fold the kernel has no further
// visibility into where the poison went.
func TestQuarantineFlagIsOneShotPerSeal(t *testing.T) {
	c := New(0)

	c.Observe(TurnObservation{InboundPrefixDigest: "v1", SealedSpanPresent: true})
	d2 := c.Observe(TurnObservation{InboundPrefixDigest: "v2"}) // harness rewrite #1
	if !d2.QuarantineAtRisk {
		t.Fatal("first rewrite after a seal must raise the flag")
	}
	d3 := c.Observe(TurnObservation{InboundPrefixDigest: "v3"}) // harness rewrite #2, no new seal
	if d3.QuarantineAtRisk {
		t.Fatal("second rewrite with no NEW seal must not re-raise the flag — " +
			"after the fold, fak cannot witness the poison further (the residual exposure)")
	}

	// And a FRESH seal after the fold re-arms the detector for the next rewrite —
	// proving the one-shot is per-seal, not a permanent latch.
	c.Observe(TurnObservation{InboundPrefixDigest: "v3", SealedSpanPresent: true})
	d5 := c.Observe(TurnObservation{InboundPrefixDigest: "v4"}) // harness rewrite #3
	if !d5.QuarantineAtRisk {
		t.Fatal("a fresh seal followed by a rewrite must re-raise the flag (per-seal, not latched)")
	}
}
