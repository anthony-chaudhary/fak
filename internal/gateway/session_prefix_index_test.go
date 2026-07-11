package gateway

import (
	"encoding/json"
	"math"
	"testing"
)

// TestSessionPrefixKeyIsChannelAgnostic pins the core identity claim (#2852): the
// platform-agnostic session-prefix key drops the channel and keys on the
// conversation identity alone, so the same conversation on two channels collapses
// to one key while two conversations stay distinct. This is the exact ambiguity the
// issue names — "distinguish same conversation, new channel from new conversation."
func TestSessionPrefixKeyIsChannelAgnostic(t *testing.T) {
	tg := SessionPrefixKey(ConversationRef{Channel: "telegram", Conversation: "c1"})
	cli := SessionPrefixKey(ConversationRef{Channel: "cli", Conversation: "c1"})
	if tg == "" || tg != cli {
		t.Fatalf("same conversation on different channels must share one key: telegram=%q cli=%q", tg, cli)
	}

	// Case/whitespace on the conversation identity must not fork the key, or a hop
	// that trivially reformats the id would cold-re-send.
	if got := SessionPrefixKey(ConversationRef{Channel: "slack", Conversation: " C1 "}); got != tg {
		t.Errorf("normalized conversation id must share the key: got %q want %q", got, tg)
	}

	// A different conversation is a different prefix — a new conversation, not a hop.
	if other := SessionPrefixKey(ConversationRef{Channel: "telegram", Conversation: "c2"}); other == tg {
		t.Errorf("distinct conversations must not alias: c1=%q c2=%q", tg, other)
	}

	// No conversation identity -> empty key, never an alias, never a reuse.
	if got := SessionPrefixKey(ConversationRef{Channel: "cli", Conversation: "  "}); got != "" {
		t.Errorf("empty conversation must yield empty key, got %q", got)
	}
}

// TestResumeReusesPrefixOnChannelHop is the done-condition witness: a conversation
// established on one channel and resumed on another reuses the established prefix
// (not a cold re-send), and the cache-read fraction on the resumed turn is recorded
// and clears the warm floor.
func TestResumeReusesPrefixOnChannelHop(t *testing.T) {
	var idx SessionPrefixIndex
	idx.Establish(ConversationRef{Channel: "telegram", Conversation: "c1"}, 4000)

	out := idx.Resume(ConversationRef{Channel: "cli", Conversation: "c1"}, 4096, 4000)
	if !out.Reused {
		t.Fatalf("channel hop must reuse the established prefix, got %+v", out)
	}
	if !out.ChannelHop {
		t.Errorf("resume from a different channel must be a hop: from=%q to=%q", out.FromChannel, out.ToChannel)
	}
	if out.FromChannel != "telegram" || out.ToChannel != "cli" {
		t.Errorf("hop provenance wrong: from=%q to=%q", out.FromChannel, out.ToChannel)
	}
	if want := 4000.0 / 4096.0; math.Abs(out.CacheReadFraction-want) > 1e-12 {
		t.Errorf("cache-read fraction = %.12f, want %.12f", out.CacheReadFraction, want)
	}
	if !out.Warm {
		t.Errorf("a 0.977 cache-read fraction must witness warm continuity, got %+v", out)
	}
}

// TestResumeColdConversationIsNotReusedThenHops guards the other half: the FIRST
// time a conversation is seen it is cold (nothing established yet, not a reuse), but
// it is recorded so a subsequent channel hop reuses it.
func TestResumeColdConversationIsNotReusedThenHops(t *testing.T) {
	var idx SessionPrefixIndex

	cold := idx.Resume(ConversationRef{Channel: "telegram", Conversation: "c9"}, 4096, 0)
	if cold.Reused || cold.ChannelHop || cold.Warm {
		t.Fatalf("first sight of a conversation must be cold, got %+v", cold)
	}

	hop := idx.Resume(ConversationRef{Channel: "cli", Conversation: "c9"}, 4096, 4000)
	if !hop.Reused || !hop.ChannelHop || !hop.Warm {
		t.Errorf("resume after a cold establish must hop and warm, got %+v", hop)
	}
	if hop.FromChannel != "telegram" {
		t.Errorf("home channel should be the cold establisher telegram, got %q", hop.FromChannel)
	}
}

// TestReusedButColdResumeIsNotWarm pins the second confusion risk verbatim: the
// witnessed cache-read fraction must reflect the ACTUAL resumed turn, so a reused
// prefix whose turn served little from cache is reported honestly as NOT warm — a
// cold baseline can never be mislabeled warm just because a key matched.
func TestReusedButColdResumeIsNotWarm(t *testing.T) {
	var idx SessionPrefixIndex
	idx.Establish(ConversationRef{Channel: "telegram", Conversation: "c1"}, 4000)

	// Reused key, but the provider served only 100 of 4096 tokens from cache
	// (fraction ~0.024, below the 0.5 floor) — a burst/miss, not warm continuity.
	out := idx.Resume(ConversationRef{Channel: "cli", Conversation: "c1"}, 4096, 100)
	if !out.Reused {
		t.Fatalf("key matched, so it is a reuse, got %+v", out)
	}
	if out.Warm {
		t.Errorf("a below-floor cache-read fraction must NOT witness warm, got %+v", out)
	}
	// Same for a reused key that served nothing from cache at all.
	if zero := idx.Resume(ConversationRef{Channel: "slack", Conversation: "c1"}, 4096, 0); zero.Warm {
		t.Errorf("a zero-cache-read reused turn must not be warm, got %+v", zero)
	}
}

// TestCrossPlatformContinuityWarmResume is the committed witness for the
// cross-platform continuity claim (#2852): the SAME channel-hop resume priced as
// the Hermes cold-re-send (no shared prefix identity -> cache_read 0) vs fak's
// platform-agnostic prefix reuse (the hop serves the established prefix from
// cache). It pins the deterministic fractions so the artifact cannot silently
// drift, and asserts fak's warm continuity beats the Hermes baseline it can
// structurally never reach.
func TestCrossPlatformContinuityWarmResume(t *testing.T) {
	res := MeasureCrossPlatformContinuity(DefaultChannelHopScenario)

	const (
		wantFakFraction = 4000.0 / 4096.0 // 0.9765625 — near-whole-prompt reuse
		eps             = 1e-12
	)

	if res.HermesColdReSend.Reused {
		t.Errorf("Hermes has no shared prefix identity; its hop must NOT reuse: %+v", res.HermesColdReSend)
	}
	if res.HermesColdReSend.CacheReadFraction != 0 {
		t.Errorf("Hermes cold re-send must serve nothing from cache, got %.12f", res.HermesColdReSend.CacheReadFraction)
	}
	if !res.FakPrefixReuse.Reused || !res.FakPrefixReuse.ChannelHop {
		t.Errorf("fak arm must reuse the established prefix on the channel hop: %+v", res.FakPrefixReuse)
	}
	if math.Abs(res.FakPrefixReuse.CacheReadFraction-wantFakFraction) > eps {
		t.Errorf("fak cache-read fraction = %.12f, want %.12f", res.FakPrefixReuse.CacheReadFraction, wantFakFraction)
	}
	if !res.FakPrefixReuse.Warm {
		t.Errorf("fak resumed turn must witness warm continuity, got %+v", res.FakPrefixReuse)
	}
	if math.Abs(res.CacheReadFractionLift-wantFakFraction) > eps {
		t.Errorf("cache-read fraction lift = %.12f, want %.12f", res.CacheReadFractionLift, wantFakFraction)
	}
	// The whole point: warm continuity Hermes structurally cannot get.
	if res.FakPrefixReuse.CacheReadFraction <= res.HermesColdReSend.CacheReadFraction {
		t.Errorf("fak warm resume %.4f must beat Hermes cold re-send %.4f",
			res.FakPrefixReuse.CacheReadFraction, res.HermesColdReSend.CacheReadFraction)
	}

	b, _ := json.MarshalIndent(res, "", "  ")
	t.Logf("\n%s", b)
}
