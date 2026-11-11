// SPDX-License-Identifier: Apache-2.0
// session_retention.go — cross-turn generation-block reuse for finished decode lanes,
// ADAPTED (clean-room re-expression, no copied code) from NVIDIA/TensorRT-LLM's
// KVCacheManagerV2 retention mechanism:
//   - PR #19710 "reuse generation KV blocks across turns" and
//   - PR #20086 "native retention priority/TTL in KVCM V2",
//
// at tag v1.4.0rc2, commit b96ba434e8a05e2ab52d2082bd8d70b1d844739b
// (https://github.com/NVIDIA/TensorRT-LLM, Apache-2.0).
//
// Upstream mechanism: when a request lane finishes, its generated decode blocks are
// retained (not freed) under a session-scoped retention key with a priority and a TTL;
// the next turn of the same session prefix-extends into them instead of re-prefilling
// the prior assistant output.
//
// Fak adaptation: this file extends the existing retention tiers (the RetentionRequest
// channel stamped through Tree.SetNodeRetention and scored by nodeTierSegAt /
// isNodePinnedOrImmune in radixkv_eviction.go) rather than building a parallel pin
// path. A finished lane's generated suffix is stamped with a clock-governed
// RetentionRequest (priority >= 90 => Tier 0 protection segment while the window is
// live; past the window the descriptor auto-demotes to probationary, evict-first), so
// admission-time prefix lookup on the next turn hits the prior-turn KV. A bounded byte
// budget across all sessions reclaims the weakest entries first, in the same total
// order as ReclaimOrder (expired first, then ascending reclaimKey).
//
// Default-off contract: nothing here changes tree behavior unless a caller constructs
// an enabled SessionRetention. The opt-in switch is FAK_RADIX_SESSION_RETENTION
// (SessionRetentionConfigFromEnv); unset or off reproduces the exact baseline free
// behavior (generated blocks age normally and are evicted like any other leaf).
package radixkv

import (
	"errors"
	"os"
	"sort"
	"strings"
	"sync"
)

const (
	// SessionRetentionEnvVar is the opt-in switch for cross-turn generation-block
	// reuse. Unset, empty, "0", "off", "false" (case-insensitive) leave it disabled;
	// "1", "on", "true" enable it with the defaults below. Any other value fails
	// closed to disabled.
	SessionRetentionEnvVar = "FAK_RADIX_SESSION_RETENTION"

	// DefaultSessionRetentionTTL is the default retention window in LOGICAL ticks
	// (tree clock advances, never wall-clock): long enough to span the gap between a
	// finished turn and the session's next turn, short enough that an abandoned
	// session's blocks return to normal LRU aging without operator intervention.
	DefaultSessionRetentionTTL int64 = 1024

	// DefaultSessionRetentionByteBudget bounds the total retained bytes across ALL
	// sessions (footprint estimated at sessionRetentionBytesPerToken per token, the
	// same estimate evictToBudget uses for demotion decisions). 0 means unbounded.
	DefaultSessionRetentionByteBudget int64 = 4 << 20 // 4 MiB

	// DefaultSessionRetentionPriority is the minimum priority that earns Tier 0
	// protection-segment immunity (nodeTierSeg seg = 3) while the TTL window is live.
	DefaultSessionRetentionPriority = 90

	// sessionRetentionBytesPerToken is the per-token footprint estimate for the byte
	// budget — identical to the estimate evictToBudget applies to a victim leaf, so
	// the registry's accounting and the tree's own pressure math agree.
	sessionRetentionBytesPerToken = 4
)

var (
	// ErrSessionNilTree is returned when a nil tree is passed to NewSessionRetention.
	ErrSessionNilTree = errors.New("radixkv: session retention requires a non-nil tree")
	// ErrSessionConfig is returned when a SessionRetentionConfig field is out of range
	// (priority outside 0..100, negative TTL, negative byte budget) — validation fails
	// CLOSED rather than clamping a caller's intent to a silently different retention.
	ErrSessionConfig = errors.New("radixkv: session retention config out of range")
	// ErrSessionEmpty is returned when Retain is called with an empty session id — the
	// retention key must be session-scoped, so an empty id is refused, not admitted.
	ErrSessionEmpty = errors.New("radixkv: session retention requires a non-empty session id")
	// ErrSessionGenRange is returned when genStart lies outside [0, len(tokens)] — the
	// generated suffix cannot start before the transcript begins or after it ends.
	ErrSessionGenRange = errors.New("radixkv: session retention generated range out of bounds")
	// ErrSessionPathNotCached is returned when the lane's full token path is not fully
	// resident in the tree — retaining a partially cached path would protect only part
	// of the suffix, so Retain fails CLOSED instead of half-retaining.
	ErrSessionPathNotCached = errors.New("radixkv: session retention path not fully cached")
)

// SessionRetentionConfig is the closed configuration for one SessionRetention registry.
// A disabled registry (Enabled=false, otherwise any valid field values) changes nothing
// about tree behavior — that is the default-off guarantee; the disabled default is
// DefaultSessionRetentionConfig() with Enabled=false.
type SessionRetentionConfig struct {
	// Enabled turns the mechanism on. When false, Retain/Release/Reclaim are no-ops and
	// the tree behaves exactly as if no SessionRetention existed.
	Enabled bool
	// Priority is the retention strength stamped on retained suffix nodes (0..100).
	// >= 90 earns Tier 0 protection-segment immunity while the window is live.
	Priority int
	// TTL is the retention window in logical ticks measured from the stamp. It MUST be a
	// positive finite window: cross-turn reuse is inherently windowed (the session's next
	// turn arrives within a bounded gap), and a never-expiring descriptor at Tier-strength
	// priority would take SetNodeRetention's permanent-pin path (pinPath over the whole
	// ancestor chain), pinning far more than the generated suffix. A caller that keeps a
	// session alive longer re-stamps it on each finished lane to refresh the window.
	TTL int64
	// ByteBudget bounds total retained bytes across all sessions; 0 is unbounded.
	ByteBudget int64
}

// Validate reports whether the config is well-formed, failing CLOSED on any out-of-range
// field (mirrors RetentionRequest.Validate's fail-closed posture): priority outside
// 0..100, a non-positive TTL (window must be finite), or a negative byte budget.
func (c SessionRetentionConfig) Validate() error {
	if c.Priority < MinRetentionPriority || c.Priority > MaxRetentionPriority {
		return ErrSessionConfig
	}
	if c.TTL <= 0 {
		return ErrSessionConfig
	}
	if c.ByteBudget < 0 {
		return ErrSessionConfig
	}
	return nil
}

// DefaultSessionRetentionConfig returns the enabled default: Tier-strength priority, a
// 1024-tick window, and a 4 MiB cross-session byte budget.
func DefaultSessionRetentionConfig() SessionRetentionConfig {
	return SessionRetentionConfig{
		Enabled:    true,
		Priority:   DefaultSessionRetentionPriority,
		TTL:        DefaultSessionRetentionTTL,
		ByteBudget: DefaultSessionRetentionByteBudget,
	}
}

// SessionRetentionConfigFromEnv reads SessionRetentionEnvVar. Unset, empty, "0", "off",
// and "false" (case-insensitive) yield the DEFAULTS with Enabled=false (the default-off
// contract); "1", "on", and "true" yield the defaults with Enabled=true. Any other value
// fails closed to disabled — a typo can never silently widen retention behavior.
func SessionRetentionConfigFromEnv() SessionRetentionConfig {
	cfg := DefaultSessionRetentionConfig()
	switch strings.ToLower(strings.TrimSpace(os.Getenv(SessionRetentionEnvVar))) {
	case "1", "on", "true":
		cfg.Enabled = true
	default:
		cfg.Enabled = false
	}
	return cfg
}

// sessionEntry is one session's retained generated suffix: the nodes stamped with the
// shared RetentionRequest plus its byte accounting and reclaim-order fields.
type sessionEntry struct {
	session  string
	ns       string
	nodes    []*node // suffix nodes carrying the retention descriptor
	bytes    int64   // Σ len(edge)·sessionRetentionBytesPerToken over the suffix edges
	admitted uint64  // tree clock tick the entry was stamped (window anchor + tie-break)
	req      RetentionRequest
}

// SessionRetention is the session-scoped retention registry for cross-turn
// generation-block reuse. It owns NO eviction logic of its own: it stamps the tree's
// existing retention channel (SetNodeRetention) and clears descriptors on reclaim, and
// the tree's own tier scoring (nodeTierSegAt / isNodePinnedOrImmune) does the rest.
//
// Concurrency: the mutex guards the registry only. The underlying Tree assumes external
// serialization of mutations (its documented contract), so callers serialize tree
// access the way they already do; the mutex makes the registry itself safe to share.
type SessionRetention struct {
	t   *Tree
	cfg SessionRetentionConfig

	mu         sync.Mutex
	entries    map[string]*sessionEntry
	totalBytes int64
}

// NewSessionRetention builds the registry for tree t with the given config. It fails
// closed on a nil tree or an invalid config and leaves the tree untouched.
func NewSessionRetention(t *Tree, cfg SessionRetentionConfig) (*SessionRetention, error) {
	if t == nil {
		return nil, ErrSessionNilTree
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return &SessionRetention{t: t, cfg: cfg, entries: map[string]*sessionEntry{}}, nil
}

// Enabled reports whether the mechanism is active.
func (s *SessionRetention) Enabled() bool { return s.cfg.Enabled }

// Config returns the registry's configuration.
func (s *SessionRetention) Config() SessionRetentionConfig { return s.cfg }

// Sessions reports how many sessions currently hold a retained suffix.
func (s *SessionRetention) Sessions() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.entries)
}

// RetainedBytes reports the total bytes currently retained across all sessions.
func (s *SessionRetention) RetainedBytes() int64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.totalBytes
}

// Has reports whether session currently holds a retained suffix.
func (s *SessionRetention) Has(session string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	_, ok := s.entries[session]
	return ok
}

// Retain pins the generated suffix of a FINISHED decode lane under session so the
// session's next turn prefix-extends into it instead of re-prefilling.
//
// tokens is the lane's full transcript (prompt + generated) and genStart the index of
// the first GENERATED token: every tree node whose edge lies entirely within
// tokens[genStart:] is stamped with a clock-governed RetentionRequest (cfg.Priority,
// cfg.TTL anchored at the current tree clock) through the existing SetNodeRetention
// seam. While the window is live and the priority reaches Tier 0, those nodes sit in
// the immune protection segment, so admission-time prefix lookups hit the prior-turn KV
// under budget pressure; once the window lapses the descriptor auto-demotes the nodes
// to probationary (evict-first) and the byte budget can reclaim them.
//
// Closed contract:
//   - disabled registry (Enabled=false): no-op, returns nil — the default-off baseline.
//   - empty session: ErrSessionEmpty.
//   - genStart outside [0, len(tokens)]: ErrSessionGenRange.
//   - no generated tokens (len(tokens)==0 or genStart==len(tokens)): no-op, nil.
//   - full path not resident: ErrSessionPathNotCached (fail closed, no half-retention).
//   - success: the suffix is retained; a pre-existing entry for the same session is
//     replaced (its descriptors cleared first), and the byte budget is enforced by
//     reclaiming the weakest OTHER entries first — the just-admitted entry is never
//     self-reclaimed, so an oversized single retain may leave the registry over budget
//     until more sessions exist to displace.
//
// Retain replaces any retention descriptor already present on the covered nodes; the
// previous descriptor is not restored on release (last writer owns the suffix).
func (s *SessionRetention) Retain(session, ns string, tokens []int, genStart int) error {
	if !s.cfg.Enabled {
		return nil // default-off: baseline free behavior, nothing stamped
	}
	if session == "" {
		return ErrSessionEmpty
	}
	if genStart < 0 || genStart > len(tokens) {
		return ErrSessionGenRange
	}
	if len(tokens) == 0 || genStart == len(tokens) {
		return nil // nothing generated, nothing to retain
	}

	root := s.t.rootFor(ns)
	end, matched := s.t.boundaryFor(root, tokens)
	if matched < len(tokens) {
		return ErrSessionPathNotCached
	}
	// Materialize the prompt boundary at exactly genStart (splitting mid-edge when the
	// cache has no node there), then collect the nodes strictly between it and end —
	// every collected edge lies entirely inside the generated range.
	promptBoundary, _ := s.t.boundaryFor(root, tokens[:genStart])
	var suffixNodes []*node
	for n := end; n != nil && n != promptBoundary && n.parent != nil; n = n.parent {
		suffixNodes = append(suffixNodes, n)
	}
	if len(suffixNodes) == 0 {
		return nil // defensive: the range check above guarantees a non-empty suffix
	}

	req := RetentionRequest{Priority: s.cfg.Priority, TTL: s.cfg.TTL, Admitted: int64(s.t.clock)}
	if err := req.Validate(); err != nil {
		return err // validated in NewSessionRetention; defensive against a mutated cfg
	}
	for _, n := range suffixNodes {
		if err := s.t.SetNodeRetention(n, req); err != nil {
			return err
		}
	}

	var bytes int64
	for _, n := range suffixNodes {
		bytes += int64(len(n.key)) * sessionRetentionBytesPerToken
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if old, ok := s.entries[session]; ok {
		s.releaseLocked(old)
	}
	e := &sessionEntry{session: session, ns: ns, nodes: suffixNodes, bytes: bytes, admitted: s.t.clock, req: req}
	s.entries[session] = e
	s.totalBytes += bytes
	// Enforce the byte budget by displacing the weakest OTHER entries first; the
	// just-admitted entry is never self-reclaimed.
	for s.cfg.ByteBudget > 0 && s.totalBytes > s.cfg.ByteBudget {
		next := s.reclaimOrderLocked(s.t.clock)[0]
		if next.session == session {
			break
		}
		s.releaseLocked(next)
	}
	return nil
}

// Release drops session's retained suffix, clearing its retention descriptors so the
// blocks return to normal LRU eligibility. It returns the bytes freed (0 when the
// session holds no entry or the registry is disabled).
func (s *SessionRetention) Release(session string) int64 {
	if !s.cfg.Enabled {
		return 0
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	e, ok := s.entries[session]
	if !ok {
		return 0
	}
	return s.releaseLocked(e)
}

// Reclaim sweeps the registry at the injected logical clock now: every TTL-expired
// entry is reclaimed, and while the byte budget is exceeded the weakest live entry is
// reclaimed next (unlike Retain's enforcement, an explicit Reclaim may reclaim ANY
// entry, including the newest, down to within budget or empty). It returns the total
// bytes freed. No-op when disabled.
func (s *SessionRetention) Reclaim(now uint64) int64 {
	if !s.cfg.Enabled {
		return 0
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	var freed int64
	for {
		order := s.reclaimOrderLocked(now)
		if len(order) == 0 {
			break
		}
		next := order[0]
		expired := next.req.expiredAtClock(now)
		if !expired && (s.cfg.ByteBudget == 0 || s.totalBytes <= s.cfg.ByteBudget) {
			break
		}
		freed += s.releaseLocked(next)
	}
	return freed
}

// releaseLocked clears e's retention descriptors and removes it from the registry.
// Callers hold s.mu.
func (s *SessionRetention) releaseLocked(e *sessionEntry) int64 {
	for _, n := range e.nodes {
		n.retention = nil
	}
	delete(s.entries, e.session)
	s.totalBytes -= e.bytes
	return e.bytes
}

// reclaimOrderLocked returns the registry's entries in RECLAIM ORDER at clock now,
// mirroring ReclaimOrder's closed contract: TTL-expired entries first, then live, each
// batch ordered by ascending reclaimKey (weakest priority first, then older admission,
// then session id). Callers hold s.mu.
func (s *SessionRetention) reclaimOrderLocked(now uint64) []*sessionEntry {
	all := make([]*sessionEntry, 0, len(s.entries))
	for _, e := range s.entries {
		all = append(all, e)
	}
	sort.SliceStable(all, func(i, j int) bool {
		a, b := all[i], all[j]
		ae, be := a.req.expiredAtClock(now), b.req.expiredAtClock(now)
		if ae != be {
			return ae // expired entries reclaim before live ones
		}
		if a.req.Priority != b.req.Priority {
			return a.req.Priority < b.req.Priority
		}
		if a.admitted != b.admitted {
			return a.admitted < b.admitted
		}
		return a.session < b.session
	})
	return all
}
