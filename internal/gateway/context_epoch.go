package gateway

import (
	"context"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// context_epoch.go — the Context Epoch: the span during which one initially rendered
// System Context remains the immutable provider-cache baseline, ending at completed
// compaction, Session movement, or an incompatible context transition that requires a
// fresh baseline. Without it a provider-side cache hit can return state the current
// epoch already invalidated.
//
// Direct port from anomalyco/opencode @ 4eb29a64f0054672950acf789f2b09487ebfbb20
// (MIT, Copyright (c) 2025 opencode):
//
//	packages/core/src/system-context/index.ts -> ContextEpochSource, RenderContextEpochBaseline,
//	                                               ReconcileContextEpoch, ReplaceContextEpoch
//	packages/core/src/session/context-epoch.ts -> ContextEpoch, ContextEpochStore.Prepare/Reset
//
// Upstream keys provider prefix caching on session identity and lets the epoch change
// that prefix by changing the baseline bytes (session/runner/llm.ts:204-217). Fak's radix
// namespace is token-addressed rather than content-addressed, so the epoch needs an
// explicit segment: contextEpochCacheTag folds the session and the baseline sequence into
// the prefix-cache identity, and a superseded epoch's prefixes become unreachable.

// contextEpochPartSeparator joins rendered baseline and update parts. Ported from the
// upstream renderer (system-context/index.ts:278).
const contextEpochPartSeparator = "\n\n"

// contextEpochMaxSessions bounds the durable epoch rows. Eviction only forces the next
// turn to re-initialize, which is the same cost as Session movement.
const contextEpochMaxSessions = 4096

// ContextEpochSource is one independently refreshable system-context source: how to
// observe, compare, and render one value. Ported from SystemContext.Source<A>
// (system-context/index.ts:31-39).
type ContextEpochSource struct {
	// Key is the source's stable namespaced identity.
	Key string
	// Load observes the source now. unavailable means observation failed temporarily;
	// it differs from the source being absent, so a refresh preserves the admitted
	// snapshot instead of dropping the source (system-context/index.ts:10-16).
	Load func(ctx context.Context) (value string, unavailable bool)
	// Baseline renders the source's contribution to a fresh baseline generation.
	Baseline func(value string) string
	// Decode maps a stored snapshot value back to its compared form. It is the Go
	// stand-in for the upstream JSON codec: ok=false means the stored value cannot be
	// compared, which upstream reports as Incompatible and resolves by replacement
	// (system-context/index.ts:216-233). A nil Decode compares the raw string.
	Decode func(stored string) (value string, ok bool)
	// Update renders the mid-conversation system message when the value changed but the
	// epoch survives (context-epoch.ts:70-77). A nil Update means the change has no
	// in-place representation, so the observation turns the epoch over instead.
	Update func(previous, current string) string
	// Removed renders a source that disappeared from the context. A nil Removed means
	// the disappearance cannot be represented and forces replacement
	// (system-context/index.ts:236-259).
	Removed func(previous string) string
}

// ContextEpochSnapshot is the durable comparison state for one admitted source.
// Ported from SystemContext.SourceSnapshot (system-context/index.ts:44-46).
type ContextEpochSnapshot struct {
	Value   string
	Removed string
}

// ContextEpochGeneration is a complete baseline plus the snapshot that produced it.
// Ported from SystemContext.Generation (system-context/index.ts:53-56).
type ContextEpochGeneration struct {
	Baseline string
	Snapshot map[string]ContextEpochSnapshot
}

// ContextEpochOutcome classifies one observation of the composed context.
// Ported from the upstream tagged union ReconcileResult/ReplacementResult
// (system-context/index.ts:58-75).
type ContextEpochOutcome uint8

const (
	ContextEpochUnchanged ContextEpochOutcome = iota
	ContextEpochUpdated
	ContextEpochReplacementReady
	ContextEpochReplacementBlocked
	ContextEpochInitializationBlocked
)

// ContextEpochObservation is the result of observing every source once.
type ContextEpochObservation struct {
	Outcome     ContextEpochOutcome
	Text        string // Outcome == ContextEpochUpdated
	Snapshot    map[string]ContextEpochSnapshot
	Generation  ContextEpochGeneration // Outcome == ContextEpochReplacementReady
	BlockedKeys []string
}

// ContextEpoch is the durable per-session row. Ported from the upstream
// session_context_epoch table (packages/core/src/session/sql.ts:168-176), minus the
// agent column: Fak binds the epoch to the harness session that already scopes the turn.
type ContextEpoch struct {
	SessionID   string
	Baseline    string
	Snapshot    map[string]ContextEpochSnapshot
	BaselineSeq int64
}

// ContextEpochPrepared is one turn's epoch decision.
type ContextEpochPrepared struct {
	Epoch       ContextEpoch
	Outcome     ContextEpochOutcome
	Turnover    bool     // the baseline was replaced this turn
	Update      string   // mid-conversation system message; empty when none
	BlockedKeys []string // sources that withheld a decision; empty when none
}

// ContextEpochStore holds one ContextEpoch per session.
type ContextEpochStore struct {
	mu     sync.Mutex
	epochs map[string]ContextEpoch
}

// NewContextEpochStore returns an empty store.
func NewContextEpochStore() *ContextEpochStore {
	return &ContextEpochStore{epochs: make(map[string]ContextEpoch)}
}

// Prepare drives one provider turn's epoch. It mirrors upstream prepareOnce
// (context-epoch.ts:39-77): mint once, replace on compaction or an incompatible
// transition, advance the snapshot in place otherwise.
func (s *ContextEpochStore) Prepare(ctx context.Context, sessionID string, sources []ContextEpochSource, replacementSeq, latestSeq int64) ContextEpochPrepared {
	if s == nil || strings.TrimSpace(sessionID) == "" {
		return ContextEpochPrepared{Outcome: ContextEpochInitializationBlocked}
	}
	stored, ok := s.Lookup(sessionID)
	if !ok {
		obs := RenderContextEpochBaseline(ctx, sources)
		if obs.Outcome != ContextEpochReplacementReady {
			// An incomplete baseline is never persisted; the next turn retries.
			return ContextEpochPrepared{Outcome: obs.Outcome, BlockedKeys: obs.BlockedKeys}
		}
		epoch := ContextEpoch{
			SessionID:   sessionID,
			Baseline:    obs.Generation.Baseline,
			Snapshot:    obs.Generation.Snapshot,
			BaselineSeq: latestSeq,
		}
		s.store(epoch)
		return ContextEpochPrepared{Epoch: epoch, Outcome: ContextEpochReplacementReady, Turnover: true}
	}

	// Compaction only forces replacement once it has passed the epoch's baseline.
	if replacementSeq > 0 && replacementSeq <= stored.BaselineSeq {
		replacementSeq = 0
	}
	var obs ContextEpochObservation
	if replacementSeq > 0 {
		obs = ReplaceContextEpoch(ctx, sources, stored.Snapshot)
	} else {
		obs = ReconcileContextEpoch(ctx, sources, stored.Snapshot)
	}
	switch obs.Outcome {
	case ContextEpochUnchanged, ContextEpochReplacementBlocked, ContextEpochInitializationBlocked:
		return ContextEpochPrepared{Epoch: stored, Outcome: obs.Outcome, BlockedKeys: obs.BlockedKeys}
	case ContextEpochReplacementReady:
		seq := replacementSeq
		if seq == 0 {
			seq = latestSeq
		}
		epoch := ContextEpoch{
			SessionID:   sessionID,
			Baseline:    obs.Generation.Baseline,
			Snapshot:    obs.Generation.Snapshot,
			BaselineSeq: seq,
		}
		s.store(epoch)
		return ContextEpochPrepared{Epoch: epoch, Outcome: ContextEpochReplacementReady, Turnover: true}
	default: // ContextEpochUpdated
		// advance writes the snapshot alone: the baseline and its sequence — the
		// provider-cache baseline this epoch froze — do not move (context-epoch.ts:159-174).
		stored.Snapshot = obs.Snapshot
		s.store(stored)
		return ContextEpochPrepared{Epoch: stored, Outcome: ContextEpochUpdated, Update: obs.Text}
	}
}

// Reset drops the session's epoch so the destination must re-initialize. Ported from
// upstream reset (context-epoch.ts:111-118), reached on Session movement.
func (s *ContextEpochStore) Reset(sessionID string) {
	if s == nil || strings.TrimSpace(sessionID) == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.epochs, sessionID)
}

// Lookup returns the session's current epoch.
func (s *ContextEpochStore) Lookup(sessionID string) (ContextEpoch, bool) {
	if s == nil {
		return ContextEpoch{}, false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	epoch, ok := s.epochs[sessionID]
	return epoch, ok
}

func (s *ContextEpochStore) store(epoch ContextEpoch) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.epochs == nil {
		s.epochs = make(map[string]ContextEpoch)
	}
	if _, ok := s.epochs[epoch.SessionID]; !ok && len(s.epochs) >= contextEpochMaxSessions {
		evictOne(s.epochs)
	}
	s.epochs[epoch.SessionID] = epoch
}

// RenderContextEpochBaseline creates the immutable baseline and durable snapshot for a
// new generation. Ported from initialize (system-context/index.ts:194-214).
func RenderContextEpochBaseline(ctx context.Context, sources []ContextEpochSource) ContextEpochObservation {
	entries, blocked := observeContextEpochSources(ctx, sources)
	if len(blocked) > 0 {
		return ContextEpochObservation{Outcome: ContextEpochInitializationBlocked, BlockedKeys: blocked}
	}
	if dup, ok := duplicateContextEpochKey(sources); ok {
		return ContextEpochObservation{Outcome: ContextEpochReplacementBlocked, BlockedKeys: []string{dup}}
	}
	return ContextEpochObservation{Outcome: ContextEpochReplacementReady, Generation: initializeContextEpoch(entries)}
}

// ReconcileContextEpoch reconciles current source values with one active generation.
// Ported from reconcile (system-context/index.ts:216-259): a change the snapshot cannot
// represent falls through to replacement rather than silently mutating the baseline.
func ReconcileContextEpoch(ctx context.Context, sources []ContextEpochSource, previous map[string]ContextEpochSnapshot) ContextEpochObservation {
	entries, _ := observeContextEpochSources(ctx, sources)
	obs := reconcileContextEpochObservation(entries, previous)
	if obs.Outcome == ContextEpochUnchanged || obs.Outcome == ContextEpochUpdated {
		return obs
	}
	return replaceContextEpochObservation(entries, previous)
}

// ReplaceContextEpoch creates a complete replacement generation, or blocks while an
// admitted source is unavailable. Ported from replace (system-context/index.ts:261-269).
func ReplaceContextEpoch(ctx context.Context, sources []ContextEpochSource, previous map[string]ContextEpochSnapshot) ContextEpochObservation {
	entries, _ := observeContextEpochSources(ctx, sources)
	return replaceContextEpochObservation(entries, previous)
}

type contextEpochEntry struct {
	source  ContextEpochSource
	value   string
	loaded  bool
	renders string
}

// observeContextEpochSources loads every source once and names the unavailable keys.
func observeContextEpochSources(ctx context.Context, sources []ContextEpochSource) ([]contextEpochEntry, []string) {
	entries := make([]contextEpochEntry, 0, len(sources))
	var blocked []string
	for _, source := range sources {
		entry := contextEpochEntry{source: source}
		if source.Load != nil {
			value, unavailable := source.Load(ctx)
			if unavailable {
				blocked = append(blocked, source.Key)
			} else {
				entry.value, entry.loaded = value, true
			}
		} else {
			blocked = append(blocked, source.Key)
		}
		entries = append(entries, entry)
	}
	sort.Strings(blocked)
	return entries, blocked
}

// initializeContextEpoch renders every loaded entry into a baseline and snapshot.
func initializeContextEpoch(entries []contextEpochEntry) ContextEpochGeneration {
	var parts []string
	snapshot := make(map[string]ContextEpochSnapshot, len(entries))
	for _, entry := range entries {
		if !entry.loaded {
			continue
		}
		rendered, ok := entry.renderBaseline()
		if !ok {
			return ContextEpochGeneration{}
		}
		parts = append(parts, rendered)
		snapshot[entry.source.Key] = entry.snapshotOf(entry.value)
	}
	return ContextEpochGeneration{Baseline: strings.Join(parts, contextEpochPartSeparator), Snapshot: snapshot}
}

// replaceContextEpochObservation blocks replacement while an ADMITTED source is
// unavailable: an incomplete baseline is never silently constructed.
func replaceContextEpochObservation(entries []contextEpochEntry, previous map[string]ContextEpochSnapshot) ContextEpochObservation {
	var blocked []string
	for _, entry := range entries {
		if !entry.loaded {
			if _, admitted := previous[entry.source.Key]; admitted {
				blocked = append(blocked, entry.source.Key)
			}
		}
	}
	if len(blocked) > 0 {
		sort.Strings(blocked)
		return ContextEpochObservation{Outcome: ContextEpochReplacementBlocked, BlockedKeys: blocked}
	}
	generation := initializeContextEpoch(entries)
	if generation.Baseline == "" {
		return ContextEpochObservation{Outcome: ContextEpochReplacementBlocked}
	}
	return ContextEpochObservation{Outcome: ContextEpochReplacementReady, Generation: generation}
}

// reconcileContextEpochObservation is the in-epoch path. It refuses to carry a change it
// cannot represent — an undecodable stored value, an unremovable disappearance, or a
// changed source with no update renderer — and lets the caller replace instead.
func reconcileContextEpochObservation(entries []contextEpochEntry, previous map[string]ContextEpochSnapshot) ContextEpochObservation {
	live := make(map[string]bool, len(entries))
	for _, entry := range entries {
		live[entry.source.Key] = true
		if !entry.loaded {
			continue
		}
		stored, admitted := previous[entry.source.Key]
		if !admitted {
			continue
		}
		if _, ok := entry.compare(stored); !ok {
			return ContextEpochObservation{Outcome: ContextEpochReplacementReady}
		}
	}
	for _, key := range sortedContextEpochKeys(previous) {
		if live[key] {
			continue
		}
		if previous[key].Removed == "" {
			return ContextEpochObservation{Outcome: ContextEpochReplacementReady}
		}
	}

	snapshot := make(map[string]ContextEpochSnapshot, len(entries))
	var updates []string
	for _, entry := range entries {
		stored, admitted := previous[entry.source.Key]
		if !entry.loaded {
			// Unavailable preserves the admitted snapshot and contributes no text
			// (system-context/index.ts:288-291).
			if admitted {
				snapshot[entry.source.Key] = stored
			}
			continue
		}
		if !admitted {
			rendered, ok := entry.renderBaseline()
			if !ok {
				return ContextEpochObservation{Outcome: ContextEpochReplacementBlocked, BlockedKeys: []string{entry.source.Key}}
			}
			updates = append(updates, rendered)
			snapshot[entry.source.Key] = entry.snapshotOf(entry.value)
			continue
		}
		// compare decodes the STORED value into the form compared against the freshly
		// loaded one; a change is then previousValue != entry.value.
		previousValue, ok := entry.compare(stored)
		if !ok {
			return ContextEpochObservation{Outcome: ContextEpochReplacementReady}
		}
		if previousValue == entry.value {
			snapshot[entry.source.Key] = stored
			continue
		}
		if entry.source.Update == nil {
			return ContextEpochObservation{Outcome: ContextEpochReplacementReady}
		}
		text, rendered := entry.renderUpdate(previousValue, entry.value)
		if !rendered {
			return ContextEpochObservation{Outcome: ContextEpochReplacementBlocked, BlockedKeys: []string{entry.source.Key}}
		}
		updates = append(updates, text)
		snapshot[entry.source.Key] = entry.snapshotOf(entry.value)
	}
	for _, key := range sortedContextEpochKeys(previous) {
		if live[key] {
			continue
		}
		updates = append(updates, previous[key].Removed)
	}
	if len(updates) == 0 {
		return ContextEpochObservation{Outcome: ContextEpochUnchanged}
	}
	return ContextEpochObservation{
		Outcome:  ContextEpochUpdated,
		Text:     strings.Join(updates, contextEpochPartSeparator),
		Snapshot: snapshot,
	}
}

// compare maps a stored snapshot value back to the comparable form the freshly loaded
// value is checked against. ok=false means the stored value cannot be decoded, which
// upstream reports as Incompatible and resolves by replacement
// (system-context/index.ts:216-233). A nil Decode compares the raw string.
func (e contextEpochEntry) compare(stored ContextEpochSnapshot) (string, bool) {
	if e.source.Decode != nil {
		return e.source.Decode(stored.Value)
	}
	return stored.Value, true
}

// snapshotOf records the source's admitted value, with the removal text when the source
// can render one (system-context/index.ts:39).
func (e contextEpochEntry) snapshotOf(value string) ContextEpochSnapshot {
	snap := ContextEpochSnapshot{Value: value}
	if e.source.Removed != nil {
		snap.Removed = e.source.Removed(value)
	}
	return snap
}

func (e contextEpochEntry) renderBaseline() (string, bool) {
	if e.source.Baseline == nil {
		return "", false
	}
	return requireContextEpochText(e.source.Key, e.source.Baseline(e.value))
}

func (e contextEpochEntry) renderUpdate(previous, current string) (string, bool) {
	return requireContextEpochText(e.source.Key, e.source.Update(previous, current))
}

// requireContextEpochText mirrors upstream requireText (system-context/index.ts:319-322):
// a source that renders nothing is a defect, not an empty contribution.
func requireContextEpochText(key, text string) (string, bool) {
	if strings.TrimSpace(text) == "" {
		return "", false
	}
	return text, true
}

func duplicateContextEpochKey(sources []ContextEpochSource) (string, bool) {
	seen := make(map[string]bool, len(sources))
	for _, source := range sources {
		if seen[source.Key] {
			return source.Key, true
		}
		seen[source.Key] = true
	}
	return "", false
}

func sortedContextEpochKeys(snapshot map[string]ContextEpochSnapshot) []string {
	keys := make([]string, 0, len(snapshot))
	for key := range snapshot {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

// ContextEpochGate arms the port per Server. It is default-off; FAK_ABLATE_CONTEXT_EPOCH=1
// arms it. Unarmed and nil receivers are inert, so the ordinary served path is unchanged.
type ContextEpochGate struct {
	armed bool
	store *ContextEpochStore
	seq   atomic.Int64
}

func newContextEpochGate(armed bool) *ContextEpochGate {
	if !armed {
		return &ContextEpochGate{}
	}
	return &ContextEpochGate{armed: true, store: NewContextEpochStore()}
}

func (g *ContextEpochGate) isArmed() bool { return g != nil && g.armed }

// bind prepares this turn's epoch and folds its isolation tag into the prefix-cache
// identity already bound for the turn. A turn with no session identity gets no epoch:
// upstream keys prefix caching on session identity, so there is nothing to isolate.
func (g *ContextEpochGate) bind(ctx context.Context, messages []agent.Message) context.Context {
	if !g.isArmed() || ctx == nil {
		return ctx
	}
	session := prefixReuseSessionFromContext(ctx)
	if session == "" {
		return ctx
	}
	prepared := g.store.Prepare(ctx, session, gatewayContextEpochSources(messages), 0, g.seq.Add(1))
	if prepared.Outcome == ContextEpochInitializationBlocked {
		return ctx
	}
	tag := contextEpochCacheTag(session, prepared.Epoch)
	ctx = withContextEpochTag(ctx, tag)
	if owner, ok := agent.PrefixCacheIdentityFromContext(ctx); ok {
		ctx = agent.WithPrefixCacheIdentityEpoch(ctx, owner.Tenant, owner.Agent, tag)
	}
	return ctx
}

// gatewayContextEpochSources describes the gateway's one system-context source: the
// leading contiguous system-message block, which is exactly the model-visible baseline a
// provider prefix cache is keyed on. A turn carrying no system block observes it as
// unavailable rather than as removed, so the admitted baseline survives.
func gatewayContextEpochSources(messages []agent.Message) []ContextEpochSource {
	return []ContextEpochSource{{
		Key: "gateway/system",
		Load: func(context.Context) (string, bool) {
			return leadingSystemContext(messages), leadingSystemContext(messages) == ""
		},
		Baseline: func(value string) string { return value },
		// No in-place renderer: a harness rewrite of the system block cannot be folded
		// into the current baseline, so it turns the epoch over and restamps the
		// isolation tag instead of leaving a stale prefix reachable.
		Removed: func(string) string { return "gateway/system removed" },
	}}
}

func leadingSystemContext(messages []agent.Message) string {
	var parts []string
	for _, message := range messages {
		if message.Role != agent.RoleSystem {
			break
		}
		if strings.TrimSpace(message.Content) == "" {
			continue
		}
		parts = append(parts, message.Content)
	}
	return strings.Join(parts, contextEpochPartSeparator)
}

// contextEpochCacheTag is the isolation token for one epoch: the session identity plus the
// baseline sequence that stamps it. Content-free, like the rest of the gateway's cache
// metadata — an opaque id and an integer, never prompt text.
func contextEpochCacheTag(session string, epoch ContextEpoch) string {
	return session + "@" + strconv.FormatInt(epoch.BaselineSeq, 10)
}

type contextEpochTagKey struct{}

func withContextEpochTag(ctx context.Context, tag string) context.Context {
	if ctx == nil || strings.TrimSpace(tag) == "" {
		return ctx
	}
	return context.WithValue(ctx, contextEpochTagKey{}, tag)
}

func contextEpochTagFromContext(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	tag, _ := ctx.Value(contextEpochTagKey{}).(string)
	return tag
}
