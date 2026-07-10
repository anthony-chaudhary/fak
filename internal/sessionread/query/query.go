package query

// query.go — the structured session-query grammar of the session read plane (epic #4176,
// child C2 #4193). The goal's literal ask: "query parts of the transcript." The gap it
// closes — the gateway serves no "last N turns / tool failures / files touched" route;
// fak_context_restore returns exactly one span, fak_context_value returns only counts, and
// reading real transcript content otherwise requires filesystem access to
// ~/.claude*/projects/*.jsonl. C2 is the closed query grammar over a session's turns that
// returns a bounded PROJECTION, never the whole JSONL.
//
// Three disciplines, all pure (no I/O, no side effects — a read cannot advance the loop,
// the Temporal-Query rule):
//
//  1. A CLOSED grammar. Five query kinds and nothing else: last-n-turns N, tool-failures,
//     files-touched, decisions-about <term>, spans-matching <term>. ParseQuery accepts
//     exactly these; an unknown kind is a closed parse error, not an open-ended filter.
//
//  2. DISCLOSURE gating (composes C1's read-plane grammar). Each kind declares the
//     projection level it needs — spans-matching returns verbatim span bytes and is
//     DisclosureFull; the rest are bounded descriptors (DisclosureMetadata / redacted text).
//     Answer refuses READ_SCOPE_DENIED when a query needs a higher disclosure than the
//     caller was granted: raw-bytes disclosure is SEPARATELY gated from metadata, never
//     folded into it.
//
//  3. TAINT filtering (composes C1's outbound screen). Every turn a query would surface is
//     run through screen.ScreenOutbound: a sealed (trust-quarantine) or tombstoned
//     (context-control) turn is WITHHELD — it appears in the projection as a bytes-free
//     marker carrying READ_TAINT_WITHHELD, never its content. A query can never return a
//     quarantined span, in ANY kind, at ANY disclosure.
//
// The projection is always BOUNDED (at most maxProjectionItems, and last-n-turns is
// clamped to its N): a scoped query never streams the whole transcript. TurnsFromRecords
// adapts the real transcript parser (internal/resume/transcript) into the Turn projection
// the engine runs over; the engine itself is parser-agnostic so a durable projection (C3)
// can feed it the same shape.

import (
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/sessionread"
	"github.com/anthony-chaudhary/fak/internal/sessionread/screen"
)

// maxProjectionItems bounds every query answer: a scoped read returns at most this many
// projection rows regardless of session size, so the whole transcript is never streamed.
const maxProjectionItems = 256

// Kind is the closed set of query verbs. A value outside this set is not a query.
type Kind string

const (
	// KindLastNTurns — the last N turns of the session (any role), bounded by N.
	KindLastNTurns Kind = "last-n-turns"
	// KindToolFailures — tool-terminal turns whose tool-call carried an error verdict.
	KindToolFailures Kind = "tool-failures"
	// KindFilesTouched — the files a session's tool calls read or wrote (from tool args/results).
	KindFilesTouched Kind = "files-touched"
	// KindDecisionsAbout — assistant decision turns whose text mentions the term.
	KindDecisionsAbout Kind = "decisions-about"
	// KindSpansMatching — turns whose raw bytes match the term; the one raw-bytes projection.
	KindSpansMatching Kind = "spans-matching"
)

// ErrUnknownQueryKind is returned by ParseQuery for a verb outside the closed grammar.
var ErrUnknownQueryKind = errors.New("unknown session-query kind")

// ErrMalformedQuery is returned by ParseQuery when a kind's required argument is missing
// or ill-formed (a non-positive N, an empty term).
var ErrMalformedQuery = errors.New("malformed session query")

// Query is one parsed query: a Kind plus its argument (N for last-n-turns, Term for the
// two match kinds; unused fields are zero).
type Query struct {
	Kind Kind
	N    int
	Term string
}

// Disclosure is the projection level a kind requires. spans-matching returns verbatim span
// bytes (DisclosureFull); last-n-turns and decisions-about return bounded excerpt text
// (DisclosureRedacted); tool-failures and files-touched return pure descriptors
// (DisclosureMetadata). The level is fixed per kind so a caller cannot escalate disclosure
// by choice of argument.
func (k Kind) Disclosure() sessionread.Disclosure {
	switch k {
	case KindSpansMatching:
		return sessionread.DisclosureFull
	case KindLastNTurns, KindDecisionsAbout:
		return sessionread.DisclosureRedacted
	default:
		return sessionread.DisclosureMetadata
	}
}

// disclosureRank orders the projection levels: metadata < redacted < full. A grant covers a
// query iff the query's required rank is <= the granted rank.
func disclosureRank(d sessionread.Disclosure) int {
	switch d {
	case sessionread.DisclosureFull:
		return 2
	case sessionread.DisclosureRedacted:
		return 1
	default:
		return 0
	}
}

// ParseQuery parses one line of the closed grammar. The first whitespace token is the kind;
// last-n-turns takes a positive integer, decisions-about / spans-matching take a non-empty
// term (the rest of the line). Any other verb is ErrUnknownQueryKind.
func ParseQuery(s string) (Query, error) {
	fields := strings.Fields(s)
	if len(fields) == 0 {
		return Query{}, ErrMalformedQuery
	}
	kind := Kind(fields[0])
	switch kind {
	case KindToolFailures, KindFilesTouched:
		if len(fields) != 1 {
			return Query{}, fmt.Errorf("%w: %q takes no argument", ErrMalformedQuery, kind)
		}
		return Query{Kind: kind}, nil
	case KindLastNTurns:
		if len(fields) != 2 {
			return Query{}, fmt.Errorf("%w: %q needs an integer N", ErrMalformedQuery, kind)
		}
		n, err := strconv.Atoi(fields[1])
		if err != nil || n <= 0 {
			return Query{}, fmt.Errorf("%w: %q needs a positive integer N", ErrMalformedQuery, kind)
		}
		return Query{Kind: kind, N: n}, nil
	case KindDecisionsAbout, KindSpansMatching:
		term := strings.TrimSpace(strings.TrimPrefix(s, fields[0]))
		if term == "" {
			return Query{}, fmt.Errorf("%w: %q needs a term", ErrMalformedQuery, kind)
		}
		return Query{Kind: kind, Term: term}, nil
	default:
		return Query{}, fmt.Errorf("%w: %q", ErrUnknownQueryKind, kind)
	}
}

// Turn is the projected view of one transcript turn the engine runs over — the shape a
// durable projection (C3) or the transcript adapter fills. Bytes is the verbatim span
// payload (surfaced only at full disclosure and only for a non-quarantined turn); Sealed /
// Tombstoned are the C1 suppression gates.
type Turn struct {
	Index      int
	Role       string   // "user" | "assistant"
	Tool       string   // tool name for a tool-terminal turn ("" otherwise)
	ToolTerm   bool     // this turn is a tool-terminal (a tool_use paired with its result)
	ToolFailed bool     // the tool-terminal carried an error verdict
	Files      []string // paths the turn's tool call touched
	Text       string   // decision / turn text (a bounded excerpt, taint-screened before use)
	Bytes      []byte   // verbatim span bytes (full disclosure only)
	Sealed     bool     // trust-quarantine gate
	Tombstoned bool     // context-control gate
}

func (t Turn) suppressed() bool { return t.Sealed || t.Tombstoned }

// Item is one bounded projection row. For a suppressed turn only Index + Withheld + Reason
// are set — never any content. Bytes is populated only for a full-disclosure, non-suppressed
// spans-matching hit.
type Item struct {
	Index    int      `json:"index"`
	Role     string   `json:"role,omitempty"`
	Tool     string   `json:"tool,omitempty"`
	Files    []string `json:"files,omitempty"`
	Text     string   `json:"text,omitempty"`
	Bytes    []byte   `json:"bytes,omitempty"`
	Withheld bool     `json:"withheld,omitempty"`
	Reason   string   `json:"reason,omitempty"`
}

// Result is a query answer: the parsed query, the disclosure level it was answered at, the
// bounded items, and whether the projection was truncated by the item cap.
type Result struct {
	Query      Query                  `json:"query"`
	Disclosure sessionread.Disclosure `json:"disclosure"`
	Items      []Item                 `json:"items"`
	Truncated  bool                   `json:"truncated,omitempty"`
}

// Answer runs q over the session turns at the granted disclosure level. It refuses
// READ_SCOPE_DENIED when the query needs a higher disclosure than granted (raw bytes are
// separately gated), and taint-screens every surfaced turn through C1's ScreenOutbound so a
// quarantined span is withheld rather than disclosed. The projection is bounded to
// maxProjectionItems (and to N for last-n-turns): the whole transcript is never returned.
func Answer(q Query, turns []Turn, granted sessionread.Disclosure) (Result, error) {
	need := q.Kind.Disclosure()
	if disclosureRank(need) > disclosureRank(granted) {
		return Result{}, &screen.Refusal{
			Reason: sessionread.ReasonReadScopeDenied,
			Detail: fmt.Sprintf("query %q needs %s disclosure, caller granted %s", q.Kind, need, granted),
		}
	}
	res := Result{Query: q, Disclosure: need}
	add := func(it Item) bool {
		if len(res.Items) >= maxProjectionItems {
			res.Truncated = true
			return false
		}
		res.Items = append(res.Items, it)
		return true
	}

	switch q.Kind {
	case KindLastNTurns:
		n := q.N
		if n > maxProjectionItems {
			n = maxProjectionItems
			res.Truncated = true
		}
		start := len(turns) - n
		if start < 0 {
			start = 0
		}
		for i := start; i < len(turns); i++ {
			if !add(projectTurn(turns[i], need)) {
				break
			}
		}
	case KindToolFailures:
		for _, t := range turns {
			if !t.ToolTerm || !t.ToolFailed {
				continue
			}
			if !add(projectTurn(t, need)) {
				break
			}
		}
	case KindFilesTouched:
		for _, t := range turns {
			if len(t.Files) == 0 {
				continue
			}
			if !add(projectTurn(t, need)) {
				break
			}
		}
	case KindDecisionsAbout:
		term := strings.ToLower(q.Term)
		for _, t := range turns {
			if t.Role != "assistant" {
				continue
			}
			// A content-match query cannot confirm a hidden turn matches without seeing its
			// withheld text, and surfacing a marker would itself leak that a quarantined
			// decision exists — so a suppressed turn is skipped silently, the same treatment
			// spans-matching gives. (Structural queries — last-n-turns / files-touched —
			// instead surface a bytes-free withheld marker, since position/structure is not
			// the quarantined content.)
			if t.suppressed() {
				continue
			}
			if !strings.Contains(strings.ToLower(t.Text), term) {
				continue
			}
			if !add(projectTurn(t, need)) {
				break
			}
		}
	case KindSpansMatching:
		term := strings.ToLower(q.Term)
		for _, t := range turns {
			if t.suppressed() {
				// A matching-or-not suppressed span is never disclosed; skip silently rather
				// than leak the fact of a match on hidden bytes.
				continue
			}
			hay := strings.ToLower(t.Text) + "\x00" + strings.ToLower(string(t.Bytes))
			if !strings.Contains(hay, term) {
				continue
			}
			if !add(projectTurn(t, need)) {
				break
			}
		}
	default:
		return Result{}, fmt.Errorf("%w: %q", ErrUnknownQueryKind, q.Kind)
	}
	return res, nil
}

// projectTurn renders one turn into an Item at the given disclosure, applying the C1 taint
// screen. A suppressed turn returns a bytes-free withheld marker. A non-suppressed turn
// carries content up to the disclosure level: metadata → descriptors only; redacted →
// + excerpt text; full → + verbatim (screened) bytes.
func projectTurn(t Turn, d sessionread.Disclosure) Item {
	it := Item{Index: t.Index, Role: t.Role, Tool: t.Tool}
	// Every surfaced turn passes through the outbound screen. For a suppressed turn this
	// refuses, and we emit only the marker — no Text, no Files, no Bytes.
	body, err := screen.ScreenOutbound(screen.Span{Bytes: t.Bytes, Sealed: t.Sealed, Tombstoned: t.Tombstoned})
	if err != nil {
		return Item{Index: t.Index, Withheld: true, Reason: screen.RefusalReason(err)}
	}
	if len(t.Files) > 0 {
		it.Files = append([]string(nil), t.Files...)
	}
	if disclosureRank(d) >= disclosureRank(sessionread.DisclosureRedacted) {
		it.Text = t.Text
	}
	if disclosureRank(d) >= disclosureRank(sessionread.DisclosureFull) {
		it.Bytes = body
	}
	return it
}
