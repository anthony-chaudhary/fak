// Package eveimport is the read-only importer for issue #2606: it folds saved Eve
// observability artifacts — an NDJSON session stream and/or OpenTelemetry spans
// carrying `eve.*` / `$eve.*` attributes — into fak's session-ledger row shape, so an
// Eve run can be debugged with fak's witnessed-status discipline instead of
// dashboard-only inspection.
//
// The trust rule this package enforces: the reconstruction consumes ONLY
// framework-owned workflow tags (session ids, subagent lineage, model id, token and
// cache-read counters, tool names, failure events). Free-form assistant text is never
// an input to any counter or verdict, and message/reasoning BODIES are redacted by
// default — kept only as a sha256 + byte-count witness — unless a fixture/test
// explicitly opts in (Options.IncludeBodies). The ledger row shape (LedgerRow) has no
// field that could carry prose, so "don't trust assistant text" is structural, not a
// convention.
//
// Missing best-effort tags never fabricate success: a session without a model id or a
// turn without usage counters degrades to a MISSING_TAG diagnostic and an Observation
// of "partial"; an input from which no root session can be reconstructed is
// "INDETERMINATE". Both are legible verdicts — not errors, and not fake passes.
//
// Honesty boundary (read before quoting a number): the parsers accept fixture-modeled
// forms of Eve's documented surfaces (the session/turn/subagent/usage workflow tags of
// docs/concepts/sessions-runs-and-streaming.md and the OTel export of
// agent/instrumentation.ts per docs/guides/instrumentation.md). The package is
// READ-ONLY and offline: no live host, no `eve` CLI, no network, no clock — saved
// artifact bytes in, a deterministic reconstruction out. Importing a live host's
// export is the residual named in #2606.
package eveimport

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
)

// Schema is the stable identifier and version tag for the eveimport reconstruction artifact.
const Schema = "fak.eve-import.v1"

// LedgerRowSchema is the stable id of the joined session-ledger row shape. It mirrors
// the fold the session-ledger consumers (#2392 content-addressed ledger, #2565
// status-card) read: ids, lineage, model, counters — never prose.
const LedgerRowSchema = "fak.eve-import.session-ledger-row.v1"

// Issue is the GitHub issue number tracking the read-only Eve observability importer contract.
const Issue = 2606

// Observation status values defining the closed partial-observation vocabulary.
const (
	// ObservationComplete indicates that every framework tag requested for reconstruction was present.
	ObservationComplete = "complete"
	// ObservationPartial indicates that session trees were reconstructed but non-fatal diagnostics fired.
	ObservationPartial = "partial"
	// ObservationIndeterminate indicates that no root session could be determined from the evidence.
	ObservationIndeterminate = "INDETERMINATE"
)

// Diagnostic codes define the closed vocabulary of reasons an import degrades. Every code is a
// partial-observation fact, not a fabricated success and not a hard error.
const (
	// DiagBadInput indicates that the artifact as a whole was structurally unreadable.
	DiagBadInput = "BAD_INPUT"
	// DiagBadLine indicates that one NDJSON line was not valid JSON; the line is skipped.
	DiagBadLine = "BAD_LINE"
	// DiagUnknownEvent indicates an NDJSON event type outside the modeled vocabulary.
	DiagUnknownEvent = "UNKNOWN_EVENT"
	// DiagOrphanEvent indicates a turn-scoped event for a session no session.start declared.
	DiagOrphanEvent = "ORPHAN_EVENT"
	// DiagOrphanSpan indicates an OTel span carrying no eve.session.id — it cannot join.
	DiagOrphanSpan = "ORPHAN_SPAN"
	// DiagMissingTag indicates a best-effort tag (model id, usage counters) was absent.
	DiagMissingTag = "MISSING_TAG"
	// DiagDuplicateSession indicates a second session.start for an already-known id.
	DiagDuplicateSession = "DUPLICATE_SESSION"
	// DiagMultipleRoots indicates more than one parentless session; the first-seen one is root.
	DiagMultipleRoots = "MULTIPLE_ROOTS"
	// DiagNoRoot indicates no session could be reconstructed; the run is INDETERMINATE.
	DiagNoRoot = "NO_ROOT"
)

// Options specifies execution settings and explicit redaction controls during reconstruction.
// The zero value is the DEFAULT and the safe posture: message/reasoning bodies are dropped
// (sha256 + byte count kept as a witness). IncludeBodies exists so a fixture/test can
// explicitly opt in — nothing else should.
type Options struct {
	IncludeBodies bool
}

// Usage records framework-owned token counters, including prompt, completion, and cache reads
// that cost ledgers price separately.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	CacheReadTokens  int `json:"cache_read_tokens"`
}

func (u Usage) add(v Usage) Usage {
	return Usage{
		PromptTokens:     u.PromptTokens + v.PromptTokens,
		CompletionTokens: u.CompletionTokens + v.CompletionTokens,
		CacheReadTokens:  u.CacheReadTokens + v.CacheReadTokens,
	}
}

// Body is one message/reasoning body observation. By default the body TEXT is dropped
// and only a witness survives: Redacted=true, the sha256 of the original bytes, and
// the byte count — enough to prove later that a body existed and what it hashed to,
// without carrying prose into the artifact. Text is populated ONLY under
// Options.IncludeBodies.
type Body struct {
	Kind     string `json:"kind"` // "message" | "reasoning"
	Role     string `json:"role,omitempty"`
	Redacted bool   `json:"redacted"`
	SHA256   string `json:"sha256"`
	Bytes    int    `json:"bytes"`
	Text     string `json:"text,omitempty"` // only when a fixture/test explicitly opts in
}

func newBody(kind, role, text string, opt Options) Body {
	sum := sha256.Sum256([]byte(text))
	b := Body{Kind: kind, Role: role, SHA256: hex.EncodeToString(sum[:]), Bytes: len(text)}
	if opt.IncludeBodies {
		b.Text = text
	} else {
		b.Redacted = true
	}
	return b
}

// Failure is one framework-emitted failure event (an NDJSON failure event or an OTel
// span with status ERROR). Reason is the framework's own message, never assistant text.
type Failure struct {
	Turn   int    `json:"turn,omitempty"`
	Reason string `json:"reason"`
}

// Turn is one reconstructed turn: tool names, token counters, and (redacted) bodies.
// UsageObserved distinguishes a genuine zero-token turn from a turn whose usage tags
// were simply absent — the latter is a MISSING_TAG partial observation.
type Turn struct {
	Index         int      `json:"index"`
	ToolCalls     []string `json:"tool_calls,omitempty"`
	Usage         Usage    `json:"usage"`
	UsageObserved bool     `json:"usage_observed"`
	Bodies        []Body   `json:"bodies,omitempty"`
}

// Session is one reconstructed Eve session: the root run or a subagent child. Usage
// and ToolCalls are this session's OWN totals (its turns); tree rollups live in the
// ledger rows and the operator summary.
type Session struct {
	SessionID string     `json:"session_id"`
	ParentID  string     `json:"parent_session_id,omitempty"`
	ModelID   string     `json:"model_id,omitempty"`
	Turns     []Turn     `json:"turns,omitempty"`
	ToolCalls int        `json:"tool_calls"`
	Usage     Usage      `json:"usage"`
	Failures  []Failure  `json:"failures,omitempty"`
	Children  []*Session `json:"children,omitempty"`
}

// Diagnostic is one degradation fact. The set of codes is closed (Diag* above), so a
// consumer can route on Code without parsing Detail.
type Diagnostic struct {
	Code      string `json:"code"`
	SessionID string `json:"session_id,omitempty"`
	Turn      int    `json:"turn,omitempty"`
	Detail    string `json:"detail"`
}

// Source names the evidence artifact an import came from. Path is recorded verbatim
// (this package never opens it — the caller owns I/O); Kind is the wire it was
// parsed as.
type Source struct {
	Kind string `json:"kind"` // "eve-ndjson" | "eve-otel-spans"
	Path string `json:"path"`
}

// Run is the reconstruction artifact: the deterministic session tree plus the honest
// account of everything the evidence did NOT say (Diagnostics, Observation).
// BodiesIncluded records the redaction posture the artifact was built under, so a
// reader can see at a glance whether prose could possibly be inside.
type Run struct {
	Schema         string       `json:"schema"`
	Issue          int          `json:"issue"`
	Source         Source       `json:"source"`
	Root           *Session     `json:"root,omitempty"`
	Sessions       int          `json:"sessions"`
	Observation    string       `json:"observation"`
	BodiesIncluded bool         `json:"bodies_included"`
	Diagnostics    []Diagnostic `json:"diagnostics,omitempty"`
}

// LedgerRow represents a normalized session row suitable for joining into fak's cost ledger.
// Deliberately, the type has NO free-text field — the only strings are framework-owned ids,
// the model id, the closed observation vocabulary, and the evidence path — so free-form
// assistant text structurally cannot reach the ledger.
//
// Invariant: LedgerRow contains only framework-owned identifiers, model tags, and counters; free-form prose is structurally excluded.
type LedgerRow struct {
	Schema           string `json:"schema"`
	SessionID        string `json:"session_id"`
	RootSessionID    string `json:"root_session_id"`
	ParentSessionID  string `json:"parent_session_id,omitempty"`
	Depth            int    `json:"depth"`
	ModelID          string `json:"model_id,omitempty"`
	Turns            int    `json:"turns"`
	Subagents        int    `json:"subagents"`
	ToolCalls        int    `json:"tool_calls"`
	PromptTokens     int    `json:"prompt_tokens"`
	CompletionTokens int    `json:"completion_tokens"`
	CacheReadTokens  int    `json:"cache_read_tokens"`
	FailedSteps      int    `json:"failed_steps"`
	Observation      string `json:"observation"`
	EvidenceKind     string `json:"evidence_kind"`
	EvidencePath     string `json:"evidence_path"`
}

// JoinLedger projects a reconstructed Run into session-ledger rows using deterministic preorder traversal.
// Projects root first, then children depth-first in first-seen order — a deterministic flattening
// of the tree. An INDETERMINATE run yields no rows: a run the evidence could not establish must
// not mint ledger rows that look like facts.
//
// Precondition: r.Observation must be evaluated prior to projection; INDETERMINATE runs yield a nil slice to prevent unverified ledger entries.
// Postcondition: reconstructed rows are emitted root-first and children depth-first in deterministic first-seen traversal order.
func JoinLedger(r Run) []LedgerRow {
	if r.Root == nil {
		return nil
	}
	var rows []LedgerRow
	var walk func(s *Session, depth int)
	walk = func(s *Session, depth int) {
		rows = append(rows, LedgerRow{
			Schema:           LedgerRowSchema,
			SessionID:        s.SessionID,
			RootSessionID:    r.Root.SessionID,
			ParentSessionID:  s.ParentID,
			Depth:            depth,
			ModelID:          s.ModelID,
			Turns:            len(s.Turns),
			Subagents:        len(s.Children),
			ToolCalls:        s.ToolCalls,
			PromptTokens:     s.Usage.PromptTokens,
			CompletionTokens: s.Usage.CompletionTokens,
			CacheReadTokens:  s.Usage.CacheReadTokens,
			FailedSteps:      len(s.Failures),
			Observation:      r.Observation,
			EvidenceKind:     r.Source.Kind,
			EvidencePath:     r.Source.Path,
		})
		for _, c := range s.Children {
			walk(c, depth+1)
		}
	}
	walk(r.Root, 0)
	return rows
}

// Summary generates a compact operator-facing text overview of reconstructed sessions and token totals.
// Renders root session, turns, subagents, failed steps, token totals (tree-wide rollups), and
// the evidence source. Pure string building — the caller decides where to print it.
//
// Postcondition: returns an operator summary containing non-empty status details matching the run observation state.
func Summary(r Run) string {
	head := fmt.Sprintf("eve-import %s %s", r.Source.Kind, r.Source.Path)
	if r.Root == nil {
		return head + fmt.Sprintf(
			"\n  observation=%s (no root session reconstructed; diagnostics=%d)",
			r.Observation, len(r.Diagnostics))
	}
	var turns, subagents, tools, failed int
	var total Usage
	var walk func(s *Session)
	walk = func(s *Session) {
		turns += len(s.Turns)
		tools += s.ToolCalls
		failed += len(s.Failures)
		total = total.add(s.Usage)
		subagents += len(s.Children)
		for _, c := range s.Children {
			walk(c)
		}
	}
	walk(r.Root)
	return head + fmt.Sprintf(
		"\n  root=%s model=%s observation=%s"+
			"\n  turns=%d subagents=%d tool_calls=%d failed_steps=%d"+
			"\n  tokens prompt=%d completion=%d cache_read=%d",
		r.Root.SessionID, r.Root.ModelID, r.Observation,
		turns, subagents, tools, failed,
		total.PromptTokens, total.CompletionTokens, total.CacheReadTokens)
}

// builder accumulates sessions/turns in first-seen order while a parser walks the
// evidence, then finalize folds them into a Run. Shared by both wire formats so the
// lineage/redaction/partial-observation rules cannot drift between them.
type builder struct {
	opt      Options
	source   Source
	sessions map[string]*Session
	order    []string
	turns    map[string]map[int]*Turn
	diags    []Diagnostic
}

func newBuilder(source Source, opt Options) *builder {
	return &builder{
		opt:      opt,
		source:   source,
		sessions: map[string]*Session{},
		turns:    map[string]map[int]*Turn{},
	}
}

func (b *builder) diag(code, sessionID string, turn int, detail string) {
	b.diags = append(b.diags, Diagnostic{Code: code, SessionID: sessionID, Turn: turn, Detail: detail})
}

// startSession registers a session id with its lineage/model tags. A repeated start
// for a known id is a DUPLICATE_SESSION diagnostic; empty tags on the first sighting
// may be filled by a later fillSession (the OTel path sees tags span-by-span).
func (b *builder) startSession(id, parentID, modelID string) {
	if id == "" {
		return
	}
	if _, ok := b.sessions[id]; ok {
		b.diag(DiagDuplicateSession, id, 0, "session.start repeated for a known session id")
		return
	}
	b.sessions[id] = &Session{SessionID: id, ParentID: parentID, ModelID: modelID}
	b.order = append(b.order, id)
	b.turns[id] = map[int]*Turn{}
}

// fillSession backfills lineage/model tags that arrived on a later record without
// overwriting an already-observed value (first observation wins — deterministic).
func (b *builder) fillSession(id, parentID, modelID string) {
	s, ok := b.sessions[id]
	if !ok {
		return
	}
	if s.ParentID == "" {
		s.ParentID = parentID
	}
	if s.ModelID == "" {
		s.ModelID = modelID
	}
}

// turn returns the (session, index) turn, creating it on first sight. A turn-scoped
// record for an undeclared session cannot join: it degrades to an ORPHAN diagnostic
// (the caller checks hasSession first).
func (b *builder) turn(sessionID string, index int) *Turn {
	ts := b.turns[sessionID]
	if ts == nil {
		ts = map[int]*Turn{}
		b.turns[sessionID] = ts
	}
	if t, ok := ts[index]; ok {
		return t
	}
	t := &Turn{Index: index}
	ts[index] = t
	return t
}

func (b *builder) hasSession(id string) bool {
	_, ok := b.sessions[id]
	return ok
}

func (b *builder) addUsage(sessionID string, index int, u Usage) {
	t := b.turn(sessionID, index)
	t.Usage = t.Usage.add(u)
	t.UsageObserved = true
}

func (b *builder) addToolCall(sessionID string, index int, tool string) {
	t := b.turn(sessionID, index)
	t.ToolCalls = append(t.ToolCalls, tool)
}

func (b *builder) addBody(sessionID string, index int, kind, role, text string) {
	t := b.turn(sessionID, index)
	t.Bodies = append(t.Bodies, newBody(kind, role, text, b.opt))
}

func (b *builder) addFailure(sessionID string, index int, reason string) {
	s, ok := b.sessions[sessionID]
	if !ok {
		return
	}
	s.Failures = append(s.Failures, Failure{Turn: index, Reason: reason})
}

// finalize folds the accumulated state into the Run artifact: turns sorted by index,
// own-usage/tool rollups, MISSING_TAG checks, parent/child attachment in first-seen
// order, root election, and the closed observation verdict.
func (b *builder) finalize() Run {
	for _, id := range b.order {
		s := b.sessions[id]
		idxs := make([]int, 0, len(b.turns[id]))
		for i := range b.turns[id] {
			idxs = append(idxs, i)
		}
		sort.Ints(idxs)
		for _, i := range idxs {
			t := b.turns[id][i]
			s.Turns = append(s.Turns, *t)
			s.ToolCalls += len(t.ToolCalls)
			s.Usage = s.Usage.add(t.Usage)
			if !t.UsageObserved {
				b.diag(DiagMissingTag, id, t.Index,
					fmt.Sprintf("usage counters absent for turn %d (token totals are partial)", t.Index))
			}
		}
		if s.ModelID == "" {
			b.diag(DiagMissingTag, id, 0, "model id absent for session (best-effort tag not emitted)")
		}
	}

	// Lineage: attach children in first-seen order; elect the first parentless (or
	// parent-unknown) session as root.
	var root *Session
	roots := 0
	for _, id := range b.order {
		s := b.sessions[id]
		if p, ok := b.sessions[s.ParentID]; ok && s.ParentID != id {
			p.Children = append(p.Children, s)
			continue
		}
		roots++
		if root == nil {
			root = s
		}
	}
	if roots > 1 {
		b.diag(DiagMultipleRoots, root.SessionID, 0,
			fmt.Sprintf("%d parentless sessions; first-seen elected root, others detached", roots))
	}

	r := Run{
		Schema:         Schema,
		Issue:          Issue,
		Source:         b.source,
		Root:           root,
		Sessions:       len(b.order),
		BodiesIncluded: b.opt.IncludeBodies,
		Diagnostics:    b.diags,
	}
	switch {
	case root == nil:
		b.diag(DiagNoRoot, "", 0, "no session reconstructible from the evidence")
		r.Diagnostics = b.diags
		r.Observation = ObservationIndeterminate
	case len(b.diags) > 0:
		r.Observation = ObservationPartial
	default:
		r.Observation = ObservationComplete
	}
	return r
}
