package sessionledger

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
)

// TestSessionKeyIsDeterministicAndSurfaceAgnostic pins the identity half of
// #2885: the key drops the surface and is a PURE function of the normalized
// conversation, so the same conversation on two surfaces collapses to one key
// while two conversations stay distinct.
func TestSessionKeyIsDeterministicAndSurfaceAgnostic(t *testing.T) {
	cli := SessionKey(SurfaceRef{Surface: "cli", Conversation: "c1"})
	relay := SessionKey(SurfaceRef{Surface: "relay", Conversation: "c1"})
	if cli == "" || cli != relay {
		t.Fatalf("same conversation on different surfaces must share one key: cli=%q relay=%q", cli, relay)
	}

	// Purity: repeated derivation must not drift. A key that depended on process
	// state (a counter, an address, a clock) would fail here and, worse, would fail
	// to resolve after a restart — the first confusion risk the issue names.
	if again := SessionKey(SurfaceRef{Surface: "cli", Conversation: "c1"}); again != cli {
		t.Errorf("derivation is not pure: %q then %q", cli, again)
	}

	// Case/whitespace must not fork the key, or a hop that trivially reformats the
	// id would cold-re-send.
	if got := SessionKey(SurfaceRef{Surface: "slack", Conversation: " C1 "}); got != cli {
		t.Errorf("normalized conversation id must share the key: got %q want %q", got, cli)
	}

	// A different conversation is a different identity — not a hop.
	if other := SessionKey(SurfaceRef{Surface: "cli", Conversation: "c2"}); other == cli {
		t.Errorf("distinct conversations must not alias: c1=%q c2=%q", cli, other)
	}

	// No conversation identity -> empty key, never an alias.
	if got := SessionKey(SurfaceRef{Surface: "cli", Conversation: "  "}); got != "" {
		t.Errorf("empty conversation must yield empty key, got %q", got)
	}

	// Shape: a scheme-tagged, fixed-width trace id that reads cleanly in a dump.
	if !strings.HasPrefix(cli, "sess:") || len(cli) != len("sess:")+32 {
		t.Errorf("unexpected key shape %q", cli)
	}
	t.Logf("deterministic session key for conversation c1: %s", cli)
}

// TestCrossSurfaceResumeReusesPrefixAcrossLedgerReopen is the done-condition
// witness for #2885, and the property that separates it from the in-memory
// gateway.SessionPrefixIndex (#2852): a conversation established on one mock
// surface and resumed on a SECOND mock surface reuses the established prefix even
// though the ledger was CLOSED and REOPENED from disk in between.
//
// The reopen is the whole point. An in-memory index would have evaporated and the
// relay hop would cold-re-send; because the identity is derived deterministically
// and the establishment lives on the durable hash chain, the hop still resolves to
// the same trace and serves the prefix warm.
func TestCrossSurfaceResumeReusesPrefixAcrossLedgerReopen(t *testing.T) {
	dir := t.TempDir()
	const conv = "conv-2885"

	// --- Surface 1 (cli): establish the prefix on a durable ledger. ---
	first, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	cold, err := first.ResumeSurface(SurfaceRef{Surface: "cli", Conversation: conv}, 4096, 0)
	if err != nil {
		t.Fatal(err)
	}
	if cold.Reused || cold.Warm {
		t.Fatalf("first sight of a conversation must be cold, got %+v", cold)
	}
	if cold.Event == "" {
		t.Fatal("the establish must be witnessed by a ledger entry")
	}

	// --- Simulate eviction / restart: drop every in-process handle and reopen
	// the ledger purely from disk. Nothing in-memory survives this line. ---
	first = nil
	second, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}

	// --- Surface 2 (relay): resume the SAME conversation on a different surface. ---
	hop, err := second.ResumeSurface(SurfaceRef{Surface: "relay", Conversation: conv}, 4096, 4000)
	if err != nil {
		t.Fatal(err)
	}
	if !hop.Reused {
		t.Fatalf("cross-surface resume must reuse the established prefix after reopen, got %+v", hop)
	}
	if !hop.SurfaceHop || hop.FromSurface != "cli" || hop.ToSurface != "relay" {
		t.Errorf("hop provenance wrong: from=%q to=%q hop=%v", hop.FromSurface, hop.ToSurface, hop.SurfaceHop)
	}
	if want := 4000.0 / 4096.0; math.Abs(hop.CacheReadFraction-want) > 1e-12 {
		t.Errorf("cache-read fraction = %.12f, want %.12f", hop.CacheReadFraction, want)
	}
	if !hop.Warm {
		t.Errorf("a 0.977 cache-read fraction on a reuse must witness warm continuity, got %+v", hop)
	}
	if hop.Key != cold.Key {
		t.Errorf("both surfaces must resolve to one identity: cli=%q relay=%q", cold.Key, hop.Key)
	}

	b, _ := json.MarshalIndent(hop, "", "  ")
	t.Logf("cross-surface warm resume across a ledger reopen:\n%s", b)
}

// TestCrossSurfaceResumeIsWitnessedLedgerEvent pins the second half of the issue:
// every resume is a WITNESSED ledger event — hash-chained, Verify-able, and
// replayable in order — not an in-memory mutation that leaves no trace. It reads
// the history back off a reopened ledger so the witness is proven durable.
func TestCrossSurfaceResumeIsWitnessedLedgerEvent(t *testing.T) {
	dir := t.TempDir()
	ref := func(s string) SurfaceRef { return SurfaceRef{Surface: s, Conversation: "c-witness"} }

	l, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range []string{"cli", "relay", "slack"} {
		if _, err := l.ResumeSurface(ref(s), 4096, 4000); err != nil {
			t.Fatal(err)
		}
	}

	// Read the history back off a ledger reopened from disk.
	reopened, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	chain, err := reopened.History(ref("cli"))
	if err != nil {
		t.Fatalf("history must replay and verify off the durable chain: %v", err)
	}
	if len(chain) != 3 {
		t.Fatalf("every surface turn must be witnessed: got %d entries, want 3", len(chain))
	}

	// The first entry establishes; every later one is a witnessed resume.
	wantKinds := []string{KindEstablish, KindResume, KindResume}
	wantSurfaces := []string{"cli", "relay", "slack"}
	for i, e := range chain {
		if e.Kind != wantKinds[i] {
			t.Errorf("entry %d kind = %q, want %q", i, e.Kind, wantKinds[i])
		}
		var ev surfaceEvent
		if err := json.Unmarshal(e.Content, &ev); err != nil {
			t.Fatalf("entry %d content: %v", i, err)
		}
		if ev.Surface != wantSurfaces[i] {
			t.Errorf("entry %d surface = %q, want %q", i, ev.Surface, wantSurfaces[i])
		}
	}

	// Tamper detection: the chain is hash-linked, so a doctored event is caught
	// rather than silently trusted — that is what "witnessed" buys over a map.
	doctored := append([]Entry(nil), chain...)
	doctored[1].Content = json.RawMessage(`{"key":"forged","surface":"relay"}`)
	if err := Verify(doctored); err == nil {
		t.Error("a doctored resume event must fail chain verification")
	}
}

// TestReusedButColdResumeIsNotWarm pins the honesty rule: the witnessed
// cache-read fraction reflects the ACTUAL resumed turn, so a reused prefix whose
// turn served little from cache is reported as NOT warm. A cold re-send can never
// be mislabeled warm just because the identity matched.
func TestReusedButColdResumeIsNotWarm(t *testing.T) {
	l := Memory()
	ref := func(s string) SurfaceRef { return SurfaceRef{Surface: s, Conversation: "c-cold"} }

	if _, err := l.ResumeSurface(ref("cli"), 4096, 0); err != nil {
		t.Fatal(err)
	}

	// Reused identity, but the provider served only 100 of 4096 tokens from cache
	// (fraction ~0.024, below the 0.5 floor) — a miss, not warm continuity.
	out, err := l.ResumeSurface(ref("relay"), 4096, 100)
	if err != nil {
		t.Fatal(err)
	}
	if !out.Reused {
		t.Fatalf("identity matched, so it is a reuse, got %+v", out)
	}
	if out.Warm {
		t.Errorf("a below-floor cache-read fraction must NOT witness warm, got %+v", out)
	}

	// A reused identity that served nothing from cache at all is likewise cold.
	zero, err := l.ResumeSurface(ref("slack"), 4096, 0)
	if err != nil {
		t.Fatal(err)
	}
	if zero.Warm {
		t.Errorf("a zero-cache-read reused turn must not be warm, got %+v", zero)
	}
}

// TestResumeWithoutConversationIdentityWritesNothing guards the aliasing edge: a
// ref with no conversation identity has no key, so it must never land on the
// ledger where it could alias another conversation's trace.
func TestResumeWithoutConversationIdentityWritesNothing(t *testing.T) {
	l := Memory()
	before := l.NodeCount()

	out, err := l.ResumeSurface(SurfaceRef{Surface: "cli", Conversation: "   "}, 4096, 4000)
	if err != nil {
		t.Fatal(err)
	}
	if out.Key != "" || out.Event != "" || out.Reused {
		t.Errorf("an identity-less ref must not resolve or witness anything, got %+v", out)
	}
	if l.NodeCount() != before {
		t.Errorf("an identity-less ref must not write to the ledger: %d -> %d", before, l.NodeCount())
	}
}
