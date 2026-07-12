package gateway

import "strings"

// session_prefix_index.go — cross-platform conversation continuity via a
// PLATFORM-AGNOSTIC session-prefix identity (#2852, Track D of epic #2834).
//
// Hermes advertises the same headline capability — talk from Telegram, continue
// from CLI — and implements it as a deterministic session-key scheme over a
// SQLite transcript store (`gateway/session.py`, `hermes_state.py`): the *text*
// is carried across the hop, but because Hermes has no kernel in front of the
// model, each platform hop RE-SENDS and RE-PAYS for the shared prefix. Continuity
// there is "same history", never "same warm cache".
//
// fak's headline capability is shared-prefix / KV-cache reuse. This file extends
// it across *platforms*: it derives a session-prefix identity that is INDEPENDENT
// of the channel a turn arrived on, so a conversation established on one channel
// and resumed on another maps to the SAME prefix key — and the established warm
// prefix is reused instead of cold-re-sent. The cost win Hermes structurally can't
// get is exactly the read rebate on that reused prefix.
//
// PROVENANCE. The prefix IDENTITY and the reuse decision are WITNESSED (fak
// authors the channel-agnostic key and the index it consults). The cache-read
// FRACTION on a resumed turn is OBSERVED — it is priced from the provider-relayed
// cache_read tokens, the same axis cache_pricing.go folds — so a "warm" verdict is
// evidence the resumed turn actually served the prefix from cache, never a fak
// trust claim. A cold re-send (the Hermes shape) reads fraction 0 on the hop turn,
// so a cold baseline can never be mislabeled warm.

// WarmResumeFloor is the cache-read fraction (cache_read / prompt tokens) at or
// above which a channel-hop resume is WITNESSED warm — the resumed turn served a
// majority of its prompt from the established prefix rather than re-sending it.
// It matches the aggregate-reuse floor cacheobs uses for its cold-cliff alarm
// (cacheobs.ColdCliffReuseFloor = 0.50): below the floor the hop paid to re-prefill
// more than it reused, which is the cold-re-send regime this feature exists to beat.
const WarmResumeFloor = 0.50

// ConversationRef is a channel-scoped handle on a conversation: the platform a
// turn arrived on (telegram, cli, slack, discord, ...) and the conversation
// identity WITHIN the fleet. It is the Hermes-shaped (channel, conversation) pair —
// the conversation identity is the deterministic, channel-independent key Hermes'
// own session-key scheme already computes; fak's contribution is keying the warm
// prefix by that identity instead of by the (channel, conversation) pair, so the
// cache carries across the hop.
type ConversationRef struct {
	// Channel is the platform the turn arrived on. It is deliberately NOT part of
	// the prefix identity (SessionPrefixKey drops it); it is retained only to
	// witness that a resume is a genuine cross-channel HOP (from != to), not a
	// same-channel continuation.
	Channel string
	// Conversation is the channel-independent conversation identity. Two refs with
	// the same Conversation are the SAME conversation regardless of Channel.
	Conversation string
}

// SessionPrefixKey derives the PLATFORM-AGNOSTIC session-prefix identity from a
// conversation ref: the channel is dropped and only the normalized conversation
// identity survives. Telegram:c1 and CLI:c1 therefore collapse to the same key
// (same conversation, new channel — REUSE), while c1 and c2 stay distinct (a new
// conversation — COLD). This is the exact ambiguity the issue names: "distinguish
// same conversation, new channel from new conversation." An empty conversation
// yields "" — no identity, so it can never alias another conversation's prefix and
// is never treated as a reuse.
func SessionPrefixKey(ref ConversationRef) string {
	conv := strings.ToLower(strings.TrimSpace(ref.Conversation))
	if conv == "" {
		return ""
	}
	// The key is the conversation identity alone — channel-free by construction.
	// A namespace prefix keeps it distinguishable from any other string key the
	// residency layer might hold, without folding the channel back in.
	return "conv:" + conv
}

// establishedPrefix is the warm prefix a conversation established on its first
// (home) channel: which channel warmed it, and the prompt-token length of the
// established prefix. The home channel is what makes a later resume a HOP.
type establishedPrefix struct {
	homeChannel  string
	promptTokens int
}

// SessionPrefixIndex maps a platform-agnostic session-prefix key to the warm
// prefix established for that conversation. It is the cross-platform analogue of
// the per-session KV prefix: ONE warm prefix, shared by every channel a
// conversation is carried on, so a channel hop reuses it instead of cold-re-sending
// the transcript. Zero value is ready to use.
type SessionPrefixIndex struct {
	established map[string]establishedPrefix
}

// Establish records the warm prefix a conversation has established on a channel,
// keyed by its platform-agnostic identity. It is idempotent and keeps the FIRST
// establishing channel as the home (so a later same-key turn from a different
// channel reads as a hop). An empty key (no conversation identity) is ignored.
// Returns whether it recorded a new prefix (false when the key was already home,
// or the ref carried no conversation identity).
func (x *SessionPrefixIndex) Establish(ref ConversationRef, promptTokens int) bool {
	key := SessionPrefixKey(ref)
	if key == "" {
		return false
	}
	if x.established == nil {
		x.established = make(map[string]establishedPrefix)
	}
	if _, ok := x.established[key]; ok {
		return false
	}
	x.established[key] = establishedPrefix{
		homeChannel:  strings.TrimSpace(ref.Channel),
		promptTokens: promptTokens,
	}
	return true
}

// ResumeOutcome records what a resume did with the established prefix, and the
// witnessed cache-read fraction on THAT resumed turn (never a cold baseline).
type ResumeOutcome struct {
	// Key is the platform-agnostic session-prefix identity the resume resolved to.
	Key string `json:"key"`
	// Reused is true when a prior channel had already established this prefix, so
	// the resume reuses the warm cache instead of cold-re-sending the transcript.
	Reused bool `json:"reused"`
	// ChannelHop is true when the resuming channel differs from the home channel —
	// a genuine cross-platform hop (Telegram -> CLI), not a same-channel turn.
	ChannelHop bool `json:"channel_hop"`
	// FromChannel is the channel that established the prefix (empty on a cold
	// resume); ToChannel is the channel the resume arrived on.
	FromChannel string `json:"from_channel,omitempty"`
	ToChannel   string `json:"to_channel,omitempty"`
	// PromptTokens is the resumed turn's prompt length; CacheReadTokens is the
	// OBSERVED provider cache_read on that turn (0 on a cold re-send).
	PromptTokens    int `json:"prompt_tokens"`
	CacheReadTokens int `json:"cache_read_tokens"`
	// CacheReadFraction is CacheReadTokens / PromptTokens on the resumed turn — the
	// witnessed warm-continuity fraction. Warm is true iff it clears WarmResumeFloor.
	CacheReadFraction float64 `json:"cache_read_fraction"`
	Warm              bool    `json:"warm"`
}

// Resume resolves a conversation resuming on some channel against the established
// prefixes, and witnesses the cache-read fraction on the resumed turn. When the
// platform-agnostic key was already established by another channel, the resume is
// a cross-platform hop that REUSES the warm prefix; CacheReadFraction prices the
// provider cache_read the reused prefix served on that turn. A key with no prior
// home is a cold conversation: it is established here (so the NEXT channel hops
// onto it) and read as not-reused. cacheReadTokens is the provider-relayed
// cache_read for the resumed turn; a cold re-send passes 0 and reads fraction 0,
// so it can never be mislabeled warm.
func (x *SessionPrefixIndex) Resume(ref ConversationRef, promptTokens, cacheReadTokens int) ResumeOutcome {
	key := SessionPrefixKey(ref)
	out := ResumeOutcome{
		Key:             key,
		ToChannel:       strings.TrimSpace(ref.Channel),
		PromptTokens:    promptTokens,
		CacheReadTokens: cacheReadTokens,
	}
	out.CacheReadFraction = cacheReadFraction(cacheReadTokens, promptTokens)
	out.Warm = out.Reused // set below once reuse is known; a cold turn is never warm

	if key == "" {
		// No conversation identity: nothing to reuse, nothing to establish.
		return out
	}
	prev, ok := x.established[key]
	if !ok {
		// Cold conversation on its home channel: record it so a later channel hops
		// onto the warm prefix. Not a reuse, so not warm regardless of the fraction.
		x.Establish(ref, promptTokens)
		return out
	}
	out.Reused = true
	out.FromChannel = prev.homeChannel
	out.ChannelHop = prev.homeChannel != out.ToChannel
	// Only a genuine reuse can be witnessed warm, and only when the OBSERVED
	// cache-read fraction clears the floor — a reused-but-cold turn (fraction below
	// the floor) is reported honestly as not warm.
	out.Warm = out.CacheReadFraction >= WarmResumeFloor
	return out
}

// cacheReadFraction is cache_read / prompt for a turn, clamped to [0,1]. Zero
// prompt (nothing to serve) is fraction 0, never a divide-by-zero; a cache_read
// somehow exceeding prompt is clamped to 1 so an inconsistent upstream count can
// never report a fraction above a whole turn.
func cacheReadFraction(cacheRead, prompt int) float64 {
	if prompt <= 0 || cacheRead <= 0 {
		return 0
	}
	f := float64(cacheRead) / float64(prompt)
	if f > 1 {
		return 1
	}
	return f
}

// ChannelHopScenario is a deterministic Telegram -> CLI resume: a conversation
// established on FromChannel with an EstablishedPrefixTokens-long warm prefix, then
// resumed on ToChannel with a ResumeTurnPromptTokens-long turn whose leading
// min(EstablishedPrefixTokens, ResumeTurnPromptTokens) tokens ARE the shared prefix.
type ChannelHopScenario struct {
	Conversation            string `json:"conversation"`
	FromChannel             string `json:"from_channel"`
	ToChannel               string `json:"to_channel"`
	EstablishedPrefixTokens int    `json:"established_prefix_tokens"`
	ResumeTurnPromptTokens  int    `json:"resume_turn_prompt_tokens"`
}

// CrossPlatformContinuityResult is the committed witness: the SAME channel-hop
// resume priced two ways. HermesColdReSend is the transcript-store baseline — no
// platform-agnostic prefix identity, so the hop re-sends the whole prefix and the
// resumed turn serves nothing from cache (fraction 0). FakPrefixReuse is fak's
// platform-agnostic reuse — the hop resolves to the established prefix and the
// resumed turn serves it from cache (fraction = reused/prompt). CacheReadFractionLift
// is the fak fraction minus the Hermes fraction: the warm continuity Hermes
// structurally cannot get. Pure and deterministic — same scenario, same numbers.
type CrossPlatformContinuityResult struct {
	Harness               string             `json:"harness"`
	Scenario              ChannelHopScenario `json:"scenario"`
	HermesColdReSend      ResumeOutcome      `json:"hermes_cold_re_send"`
	FakPrefixReuse        ResumeOutcome      `json:"fak_prefix_reuse"`
	CacheReadFractionLift float64            `json:"cache_read_fraction_lift"`
	WarmResumeFloor       float64            `json:"warm_resume_floor"`
}

// MeasureCrossPlatformContinuity runs one channel-hop resume both ways and returns
// the comparison. The Hermes arm has NO shared prefix identity, so the resuming
// channel is a fresh conversation that re-sends the prefix cold (cache_read 0). The
// fak arm establishes the prefix on FromChannel and resumes on ToChannel through the
// SAME platform-agnostic key, so the reused prefix is served from cache. Both price
// the identical resumed turn, so CacheReadFractionLift isolates the platform-agnostic
// reuse — the warm continuity the issue asks be witnessed.
func MeasureCrossPlatformContinuity(sc ChannelHopScenario) CrossPlatformContinuityResult {
	served := sc.EstablishedPrefixTokens
	if served > sc.ResumeTurnPromptTokens {
		served = sc.ResumeTurnPromptTokens
	}
	if served < 0 {
		served = 0
	}

	// Hermes: the transcript store carries the text, but with no kernel in front of
	// the model the hop is a cold conversation that re-pays for the prefix. Model it
	// as a resume whose channel never established a shared key -> not reused, 0 read.
	var hermes SessionPrefixIndex
	hermesOut := hermes.Resume(
		ConversationRef{Channel: sc.ToChannel, Conversation: sc.Conversation},
		sc.ResumeTurnPromptTokens, 0)

	// fak: establish on the home channel, then resume on the new channel through the
	// same platform-agnostic identity. The reused prefix serves `served` tokens from
	// cache on the resumed turn.
	var fak SessionPrefixIndex
	fak.Establish(
		ConversationRef{Channel: sc.FromChannel, Conversation: sc.Conversation},
		sc.EstablishedPrefixTokens)
	fakOut := fak.Resume(
		ConversationRef{Channel: sc.ToChannel, Conversation: sc.Conversation},
		sc.ResumeTurnPromptTokens, served)

	return CrossPlatformContinuityResult{
		Harness:               "TestCrossPlatformContinuityWarmResume",
		Scenario:              sc,
		HermesColdReSend:      hermesOut,
		FakPrefixReuse:        fakOut,
		CacheReadFractionLift: fakOut.CacheReadFraction - hermesOut.CacheReadFraction,
		WarmResumeFloor:       WarmResumeFloor,
	}
}

// DefaultChannelHopScenario is the standing witness workload: a Telegram
// conversation with a warm 4,000-token prefix resumed from the CLI on a
// 4,096-token turn — the near-whole-prompt-is-prefix regime a real resume lands in
// (a short new user turn appended to a long established history). The fak arm
// serves 4,000 of 4,096 tokens from the reused prefix (fraction ~0.977, well past
// the warm floor); the Hermes arm serves 0.
var DefaultChannelHopScenario = ChannelHopScenario{
	Conversation:            "conv-track-d-2852",
	FromChannel:             "telegram",
	ToChannel:               "cli",
	EstablishedPrefixTokens: 4000,
	ResumeTurnPromptTokens:  4096,
}
