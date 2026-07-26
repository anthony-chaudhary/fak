package sessionledger

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"
)

// crosssurface.go — deterministic, witnessed cross-surface session identity
// (#2885, Hermes-inspiration epic #2871). Generation: gen/next.
//
// Hermes keeps one conversation coherent across platforms with a deterministic
// session-key scheme (`gateway/session.py`) over a SQLite transcript store: the
// TEXT survives the hop, but with no kernel in front of the model each hop
// re-sends and re-pays for the shared prefix.
//
// fak already answers the identity half in-process: gateway.SessionPrefixKey
// (#2852) derives a channel-agnostic prefix key and gateway.SessionPrefixIndex
// resolves a hop against it. That index is an in-memory map — exactly the
// "in-memory LRU entry that evaporates on eviction" this issue calls out. Once it
// is evicted or the process restarts, the established prefix is forgotten and the
// next surface cold-re-sends, silently degrading to the Hermes shape.
//
// This file closes that gap by moving the identity and the resume decision ONTO
// the durable session ledger:
//
//   - SessionKey is a PURE function of the normalized conversation identity —
//     no process state, no counter, no address — so it is deterministic across
//     processes AND across restarts, not merely stable per-process.
//   - The derived key IS the ledger trace. Continuity is therefore the ledger's
//     own hash-chained head, which is persisted to disk by Append, so reopening
//     the ledger recovers the established prefix instead of losing it.
//   - Every establish and every resume is appended as a ledger entry, so each
//     resume is a WITNESSED event (hash-chained, Verify-able, replayable via
//     Chain) rather than a mutation of an in-memory map that leaves no trace.
//
// PROVENANCE. The session identity, the reuse decision, and the ledger event are
// WITNESSED (fak authors the key, the trace, and the hash chain). The cache-read
// FRACTION on a resumed turn is OBSERVED — it is priced from the provider-relayed
// cache_read tokens the caller passes, never asserted by fak — so a "warm" verdict
// is evidence the resumed turn actually served the prefix from cache. A cold
// re-send reads fraction 0 and can never be mislabeled warm.

// Ledger event kinds for the cross-surface identity chain. The first entry on a
// session trace is always KindEstablish; every later hop appends a KindResume,
// so Chain(key) replays the full cross-surface history.
const (
	KindEstablish = "session-establish"
	KindResume    = "session-resume"
)

// keyScheme versions the SessionKey derivation. It is mixed into the digest so a
// future scheme change yields different keys rather than silently aliasing
// conversations derived under the old rules onto the new ones.
const keyScheme = "fak-session/1"

// WarmResumeFloor is the cache-read fraction (cache_read / prompt tokens) at or
// above which a cross-surface resume is WITNESSED warm — the resumed turn served
// a majority of its prompt from the established prefix rather than re-sending it.
// It matches gateway.WarmResumeFloor (and cacheobs.ColdCliffReuseFloor); the value
// is restated rather than imported because the ledger sits below the gateway in
// the tier graph and must not depend upward on it.
const WarmResumeFloor = 0.50

// SurfaceRef is a surface-scoped handle on a conversation: the surface a turn
// arrived on (cli, relay, slack, ...) and the conversation identity within the
// fleet. It is the ledger-backed analogue of gateway.ConversationRef.
type SurfaceRef struct {
	// Surface is where the turn arrived. It is deliberately NOT part of the
	// session identity (SessionKey drops it); it is retained only to witness that
	// a resume is a genuine cross-surface HOP (from != to), not a continuation on
	// the same surface.
	Surface string
	// Conversation is the surface-independent conversation identity. Two refs with
	// the same Conversation are the SAME conversation regardless of Surface.
	Conversation string
}

// SessionKey derives the deterministic cross-surface session key for a ref: the
// surface is dropped and the normalized conversation identity is hashed under the
// scheme tag. cli:c1 and relay:c1 therefore collapse to one key (same
// conversation, new surface — REUSE) while c1 and c2 stay distinct (a new
// conversation — COLD).
//
// It is a pure function: the same conversation yields the same key in any process
// and after any restart, which is what makes the identity resumable rather than
// merely stable for the life of one index. An empty conversation yields "" — no
// identity, so it can never alias another conversation's trace.
func SessionKey(ref SurfaceRef) string {
	conv := strings.ToLower(strings.TrimSpace(ref.Conversation))
	if conv == "" {
		return ""
	}
	h := sha256.New()
	h.Write([]byte(keyScheme))
	h.Write([]byte{0})
	h.Write([]byte(conv))
	// 128 bits is ample to keep conversations collision-free while keeping the
	// trace id readable in a ledger dump.
	return "sess:" + hex.EncodeToString(h.Sum(nil)[:16])
}

// surfaceEvent is the JSON content of an establish/resume ledger entry. The
// establish entry carries the home surface and the established prefix length; a
// resume entry additionally carries the OBSERVED cache-read of that turn.
type surfaceEvent struct {
	Key               string  `json:"key"`
	Surface           string  `json:"surface"`
	Conversation      string  `json:"conversation"`
	PromptTokens      int     `json:"prompt_tokens"`
	CacheReadTokens   int     `json:"cache_read_tokens,omitempty"`
	CacheReadFraction float64 `json:"cache_read_fraction,omitempty"`
	Reused            bool    `json:"reused,omitempty"`
	SurfaceHop        bool    `json:"surface_hop,omitempty"`
	FromSurface       string  `json:"from_surface,omitempty"`
}

// SurfaceResume records what a cross-surface resume did with the established
// prefix, plus the ledger entry that witnesses it.
type SurfaceResume struct {
	// Key is the deterministic session identity, which is also the ledger trace.
	Key string `json:"key"`
	// Reused is true when the ledger already held an established prefix for this
	// key, so the resume reuses the warm cache instead of cold-re-sending.
	Reused bool `json:"reused"`
	// SurfaceHop is true when the resuming surface differs from the home surface —
	// a genuine cross-surface hop (cli -> relay), not a same-surface turn.
	SurfaceHop bool `json:"surface_hop"`
	// FromSurface established the prefix (empty on a cold resume); ToSurface is
	// where the resume arrived.
	FromSurface string `json:"from_surface,omitempty"`
	ToSurface   string `json:"to_surface,omitempty"`
	// PromptTokens is the resumed turn's prompt length; CacheReadTokens is the
	// OBSERVED provider cache_read on that turn (0 on a cold re-send).
	PromptTokens    int `json:"prompt_tokens"`
	CacheReadTokens int `json:"cache_read_tokens"`
	// CacheReadFraction is CacheReadTokens / PromptTokens on the resumed turn.
	// Warm is true iff the turn is a reuse AND the fraction clears WarmResumeFloor.
	CacheReadFraction float64 `json:"cache_read_fraction"`
	Warm              bool    `json:"warm"`
	// Event is the hash of the ledger entry that witnesses this resume. It is the
	// whole point: the resume is recoverable from the durable chain, not a mutation
	// of an in-memory map. Empty only when the ref carried no conversation identity.
	Event Hash `json:"event,omitempty"`
}

// ResumeSurface resolves a conversation arriving on some surface against the
// DURABLE ledger and appends the witnessing entry.
//
// When the derived key already has a trace, the ledger is holding an established
// prefix: the resume is a reuse, and cacheReadTokens prices what the reused prefix
// actually served on that turn. A key with no trace is a cold conversation — it is
// established here (so the NEXT surface hops onto it) and read as not-reused,
// never warm. Because the establishing entry lives on disk, this holds across a
// process restart: reopen the ledger and the hop still reuses.
//
// A ref with no conversation identity yields the zero SurfaceResume and writes
// nothing, so it can never alias another conversation's trace.
func (l *Ledger) ResumeSurface(ref SurfaceRef, promptTokens, cacheReadTokens int) (SurfaceResume, error) {
	key := SessionKey(ref)
	out := SurfaceResume{
		Key:             key,
		ToSurface:       strings.TrimSpace(ref.Surface),
		PromptTokens:    promptTokens,
		CacheReadTokens: cacheReadTokens,
	}
	if key == "" {
		return out, nil
	}
	out.CacheReadFraction = cacheReadFraction(cacheReadTokens, promptTokens)

	kind := KindEstablish
	if home, ok := l.establishedSurface(key); ok {
		kind = KindResume
		out.Reused = true
		out.FromSurface = home
		out.SurfaceHop = home != out.ToSurface
		// Only a genuine reuse can be witnessed warm, and only when the OBSERVED
		// fraction clears the floor — a reused-but-cold turn is reported honestly
		// as not warm.
		out.Warm = out.CacheReadFraction >= WarmResumeFloor
	}

	ev := surfaceEvent{
		Key:               key,
		Surface:           out.ToSurface,
		Conversation:      strings.ToLower(strings.TrimSpace(ref.Conversation)),
		PromptTokens:      promptTokens,
		CacheReadTokens:   cacheReadTokens,
		CacheReadFraction: out.CacheReadFraction,
		Reused:            out.Reused,
		SurfaceHop:        out.SurfaceHop,
		FromSurface:       out.FromSurface,
	}
	content, err := json.Marshal(ev)
	if err != nil {
		return out, err
	}
	// The key is the trace: continuity rides the ledger's own hash-chained head,
	// which Append persists to disk.
	entry, err := l.Append(key, kind, content)
	if err != nil {
		return out, err
	}
	out.Event = entry.Hash
	return out, nil
}

// establishedSurface reads the home surface for a session key back off the
// durable chain. The FIRST entry on a session trace is the establish event, so
// the home surface is recovered from the ledger rather than from any in-process
// state — the property that survives a restart.
func (l *Ledger) establishedSurface(key string) (surface string, ok bool) {
	if l.Head(key) == "" {
		return "", false
	}
	chain, err := l.Chain(key)
	if err != nil || len(chain) == 0 {
		return "", false
	}
	var ev surfaceEvent
	if err := json.Unmarshal(chain[0].Content, &ev); err != nil {
		return "", false
	}
	return ev.Surface, true
}

// History replays the full cross-surface history for a conversation off the
// durable ledger and verifies the chain. It is the audit path: every surface the
// conversation was carried on, in order, hash-verified — the witnessed
// alternative to an in-memory index that can only report its current state.
func (l *Ledger) History(ref SurfaceRef) ([]Entry, error) {
	key := SessionKey(ref)
	if key == "" {
		return nil, nil
	}
	chain, err := l.Chain(key)
	if err != nil {
		return nil, err
	}
	if err := Verify(chain); err != nil {
		return nil, err
	}
	return chain, nil
}

// cacheReadFraction is cache_read / prompt for a turn, clamped to [0,1]. Zero
// prompt is fraction 0, never a divide-by-zero; a cache_read somehow exceeding
// prompt is clamped to 1 so an inconsistent upstream count can never report a
// fraction above a whole turn.
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
