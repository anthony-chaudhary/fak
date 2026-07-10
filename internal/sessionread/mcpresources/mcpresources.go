// Package mcpresources is the sessionread C6 MCP resource projection (issue #4197,
// child of epic #4176): it exposes a session's queryable transcript/context/decisions
// as standard MCP resources — the exact resources/list + resources/read shape any MCP
// client already understands (see internal/gateway/mcp_resources_prompts.go, read for
// wire-shape reference only, never imported here).
//
// # The URI scheme
//
// One directory root plus four per-session views:
//
//	session://                    the C4 cross-session directory, scoped to the caller
//	session://<trace>/turns       bounded last-n-turns transcript excerpt (redacted)
//	session://<trace>/context     tool-failures + files-touched descriptors (metadata)
//	session://<trace>/decisions   assistant decision turns (redacted)
//	session://<trace>/spans       raw span bytes (full disclosure, separately gated)
//
// A resources/list entry is {uri, name, description, mimeType}; a resources/read
// result is {contents:[{uri, mimeType, text}]} — both mirrored here as the Resource
// and ReadResult Go shapes, matching the gateway's mcpResource / readResource
// convention byte-for-byte in field names, so a future adapter is a type conversion,
// not a redesign.
//
// # Compose, don't duplicate
//
// This package invents NO taint or scope logic of its own. Every view is answered by
// delegating to the already-landed sibling packages:
//
//   - internal/sessionread/screen (C1) supplies the per-principal scope floor
//     (screen.Authorize) this package calls before serving any per-session view, and
//     the outbound taint screen (screen.ScreenOutbound) that C2's query engine already
//     runs over every surfaced turn.
//   - internal/sessionread/query (C2) supplies the closed query grammar and the
//     taint-filtering + disclosure-gating engine (query.Answer): every view here is
//     one (or, for "context", two) query.Answer call(s) over the session's turns. A
//     sealed or tombstoned turn is therefore withheld — or, for decisions/spans,
//     silently absent — by C2's own screening, not by anything reimplemented here.
//   - internal/sessionread/directory (C4) supplies the cross-session directory fold
//     (directory.Directory) the "session://" root projects.
//
// Because query.Answer already screens every turn through screen.ScreenOutbound
// before it ever reaches an Item, and Answer itself refuses a disclosure escalation
// (a full-disclosure view addressed with a lower grant refuses READ_SCOPE_DENIED —
// see query.go's disclosureRank gate), this package's own job is narrow: parse a URI,
// resolve the addressed session, apply the C1 per-principal scope check the sibling
// packages do not themselves apply (they are principal-agnostic), and shape the
// composed answer into the MCP wire envelope.
//
// # Scope
//
// Every per-session view requires screen.Authorize(read-self): the caller principal
// must equal the trace addressed by the URI. A cross-principal ListResources omits
// that session's views entirely (rather than surfacing an empty-but-listed resource);
// a cross-principal ReadResource refuses READ_SCOPE_DENIED. The directory root applies
// the same self-scoping to its own rows: a caller sees only the directory row(s) whose
// TraceID equals its own principal.
//
// No registered sessionread.ReadOp names "an MCP resource read" yet — the S vocabulary
// spine (internal/sessionread/vocab.go) deliberately leaves this seam unregistered
// until its child lands (see that file's header comment), and this package must not
// edit the vocab. Scope checks here reuse the closest already-registered CapReadSelf
// op (OpContextValue for the general session views, OpContextRestore for the
// full-disclosure spans view) purely so screen.Authorize has a valid capability to
// key its self/fleet rule off of; only the Capability field of that op's spec is
// consulted; the Op token itself is not asserted as the "true" identity of an MCP
// resource read. Registering a dedicated op is follow-on, alongside the wiring below.
//
// # Purity and follow-on
//
// This package is pure: stdlib plus the four internal/sessionread* siblings only, no
// I/O, no clock, no internal/gateway or internal/session import. A caller (a test
// fixture today, the live gateway tomorrow) hands it an in-memory Store — sessions as
// []query.Turn (e.g. via query.TurnsFromRecords) and a pre-folded []directory.DirectoryRow
// (e.g. via directory.Directory). Wiring this into the live gateway's s.resources() /
// s.readResource() so a real MCP client can reach a live session is explicitly OUT OF
// SCOPE here and is the follow-on adoption child.
package mcpresources

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/sessionread"
	"github.com/anthony-chaudhary/fak/internal/sessionread/directory"
	"github.com/anthony-chaudhary/fak/internal/sessionread/query"
	"github.com/anthony-chaudhary/fak/internal/sessionread/screen"
)

// DirectoryURI is the directory-root resource: the C4 cross-session fold, scoped to
// the caller principal.
const DirectoryURI = "session://"

// The four per-session view suffixes making up a session://<trace>/<view> URI.
const (
	viewTurns     = "turns"
	viewContext   = "context"
	viewDecisions = "decisions"
	viewSpans     = "spans"
)

const uriScheme = "session://"

// Resource mirrors the MCP resources/list entry shape {uri, name, description,
// mimeType} — the exact field set internal/gateway/mcp_resources_prompts.go's
// resourceDescriptors returns, so json.Marshal(Resource{}) needs no field renaming
// to serve a real MCP resources/list response.
type Resource struct {
	URI         string `json:"uri"`
	Name        string `json:"name"`
	Description string `json:"description"`
	MimeType    string `json:"mimeType"`
}

// Content is one entry of a resources/read result's "contents" array —
// {uri, mimeType, text}, mirroring the gateway's readResource shape.
type Content struct {
	URI      string `json:"uri"`
	MimeType string `json:"mimeType"`
	Text     string `json:"text"`
}

// ReadResult is the resources/read response shape: {contents:[...]}.
type ReadResult struct {
	Contents []Content `json:"contents"`
}

// Session is the pure, in-memory view of one addressable session this package
// projects as MCP resources: its trace id and its turns, in the C2 query.Turn
// projection shape (populate via query.TurnsFromRecords, or directly in a fixture).
// mcpresources performs no I/O to obtain this — the caller supplies it.
type Session struct {
	// Trace is the session's trace id — the addressing key used in
	// session://<trace>/... URIs and the principal a read-self scope check compares
	// the caller against.
	Trace string
	// Turns is the session's turn projection, already adapted from whatever backing
	// store (transcript file, durable C3 span store, ...) the caller reads from.
	Turns []query.Turn
}

// Store is the pure, hermetic index this package serves resources/list and
// resources/read from: the addressable sessions plus the folded cross-session
// directory. It holds no other state and performs no I/O; a caller assembles it once
// per read (or reuses it across reads of the same live snapshot).
type Store struct {
	// Sessions is every addressable session, keyed by trace (must equal each
	// Session.Trace).
	Sessions map[string]Session
	// Directory is the pre-folded cross-session directory (e.g. the output of
	// directory.Directory) the "session://" root projects, scoped per-principal at
	// read time.
	Directory []directory.DirectoryRow
}

// perSessionViews is the static, content-free catalog of the four per-session views.
// Name/Description never derive from turn content — the resources/list payload can
// never leak a byte of any turn, sealed or not, by construction.
var perSessionViews = []struct {
	view, name, desc string
}{
	{viewTurns, "session turns", "bounded, taint-filtered transcript turns (last-n-turns query, redacted disclosure)"},
	{viewContext, "session context report", "tool-failure and files-touched descriptors for the session (metadata disclosure, no raw text or bytes)"},
	{viewDecisions, "session decisions", "assistant decision turns mentioning the query term, taint-filtered (redacted disclosure)"},
	{viewSpans, "session dropped spans", "raw span bytes for matching turns, taint-filtered; a sealed or tombstoned span is never listed (full disclosure, separately gated)"},
}

// sessionURI builds the session://<trace>/<view> URI for one per-session view.
func sessionURI(trace, view string) string {
	return uriScheme + trace + "/" + view
}

// parseSessionURI recognizes a session://<trace>/<view> URI where view is one of the
// four registered per-session views. It does not recognize DirectoryURI itself —
// callers check that separately.
func parseSessionURI(uri string) (trace, view string, ok bool) {
	if uri == DirectoryURI || !strings.HasPrefix(uri, uriScheme) {
		return "", "", false
	}
	rest := strings.TrimPrefix(uri, uriScheme)
	parts := strings.SplitN(rest, "/", 2)
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		return "", "", false
	}
	switch parts[1] {
	case viewTurns, viewContext, viewDecisions, viewSpans:
		return parts[0], parts[1], true
	default:
		return "", "", false
	}
}

// opForView returns the CapReadSelf op used purely to key screen.Authorize's
// self/fleet rule for a per-session view read. See the package doc "Scope" section:
// no dedicated MCP-resource op is registered yet, so this borrows the closest
// existing self-read op's Capability — OpContextRestore for the full-disclosure raw
// span view (the closest semantic match: verbatim bytes, taint-withheld), and
// OpContextValue for the bounded/metadata views.
func opForView(view string) sessionread.ReadOp {
	if view == viewSpans {
		return sessionread.OpContextRestore
	}
	return sessionread.OpContextValue
}

// unknownResource builds the closed-vocabulary refusal for a URI, trace, or view this
// package cannot address — the pure-package analog of the gateway's InvalidParams
// fault for an unknown resources/read uri.
func unknownResource(detail string) error {
	return &screen.Refusal{Reason: sessionread.ReasonReadUnknownTrace, Detail: detail}
}

// ListResources enumerates the resources/list payload visible to principal: the
// directory root, plus the four per-session views for every session principal may
// read. Per C1, a session is enumerated only when principal equals that session's
// trace (the read-self scope floor) — a cross-principal listing yields no entries for
// that session, rather than an error, so a bulk resources/list from an unprivileged
// caller degrades to "just the directory root" instead of faulting. An empty
// principal fails closed with READ_SCOPE_DENIED, matching screen.Authorize's own
// empty-caller rule.
func (s Store) ListResources(principal string) ([]Resource, error) {
	if strings.TrimSpace(principal) == "" {
		return nil, &screen.Refusal{Reason: sessionread.ReasonReadScopeDenied, Detail: "empty caller principal — the scope floor fails closed"}
	}
	out := []Resource{{
		URI:         DirectoryURI,
		Name:        "session directory",
		Description: "cross-session directory (C4 fold), scoped to the caller principal's own row(s)",
		MimeType:    "application/json",
	}}

	traces := make([]string, 0, len(s.Sessions))
	for t := range s.Sessions {
		traces = append(traces, t)
	}
	sort.Strings(traces)
	for _, trace := range traces {
		if trace != principal {
			continue // cross-principal: this session's views are omitted, not listed-then-denied
		}
		for _, v := range perSessionViews {
			out = append(out, Resource{
				URI:         sessionURI(trace, v.view),
				Name:        v.name,
				Description: v.desc,
				MimeType:    "application/json",
			})
		}
	}
	return out, nil
}

// ReadResource resolves uri and returns its taint-filtered projection as the
// {contents:[{uri,mimeType,text}]} shape, scoped to principal at the granted
// disclosure level. An unknown uri (unrecognized scheme, unknown trace, or unknown
// view) is a parameter fault, returned as an error carrying READ_UNKNOWN_TRACE. A
// cross-principal read of a known session, or a disclosure escalation (e.g. the
// full-disclosure spans view addressed with a metadata/redacted grant), refuses
// READ_SCOPE_DENIED. A sealed or tombstoned turn is never present in any returned
// content, at any disclosure level — C2's query.Answer screens it before this
// function ever sees a byte.
func (s Store) ReadResource(uri, principal string, disclosure sessionread.Disclosure) (ReadResult, error) {
	if strings.TrimSpace(principal) == "" {
		return ReadResult{}, &screen.Refusal{Reason: sessionread.ReasonReadScopeDenied, Detail: "empty caller principal — the scope floor fails closed"}
	}

	if uri == DirectoryURI {
		return s.readDirectory(principal)
	}

	trace, view, ok := parseSessionURI(uri)
	if !ok {
		return ReadResult{}, unknownResource(fmt.Sprintf("unrecognized resource uri %q", uri))
	}
	sess, ok := s.Sessions[trace]
	if !ok {
		return ReadResult{}, unknownResource(fmt.Sprintf("unknown session trace %q", trace))
	}
	if err := screen.Authorize(screen.ScopeRequest{Op: opForView(view), Caller: principal, TargetOwner: trace}); err != nil {
		return ReadResult{}, err
	}

	var payload any
	switch view {
	case viewTurns:
		res, err := query.Answer(query.Query{Kind: query.KindLastNTurns, N: len(sess.Turns)}, sess.Turns, disclosure)
		if err != nil {
			return ReadResult{}, err
		}
		payload = res
	case viewContext:
		failures, err := query.Answer(query.Query{Kind: query.KindToolFailures}, sess.Turns, disclosure)
		if err != nil {
			return ReadResult{}, err
		}
		filesTouched, err := query.Answer(query.Query{Kind: query.KindFilesTouched}, sess.Turns, disclosure)
		if err != nil {
			return ReadResult{}, err
		}
		payload = map[string]query.Result{"toolFailures": failures, "filesTouched": filesTouched}
	case viewDecisions:
		res, err := query.Answer(query.Query{Kind: query.KindDecisionsAbout, Term: ""}, sess.Turns, disclosure)
		if err != nil {
			return ReadResult{}, err
		}
		payload = res
	case viewSpans:
		res, err := query.Answer(query.Query{Kind: query.KindSpansMatching, Term: ""}, sess.Turns, disclosure)
		if err != nil {
			return ReadResult{}, err
		}
		payload = res
	default:
		// Unreachable: parseSessionURI only accepts the four cases above.
		return ReadResult{}, unknownResource(fmt.Sprintf("unrecognized view %q", view))
	}

	b, err := json.Marshal(payload)
	if err != nil {
		return ReadResult{}, err
	}
	return ReadResult{Contents: []Content{{URI: uri, MimeType: "application/json", Text: string(b)}}}, nil
}

// readDirectory answers a read of the "session://" root: the C4-folded directory
// rows, filtered to the ones principal itself owns (TraceID == principal) — the
// directory root's self-scoping. The Authorize call is a formality (TargetOwner is
// set to principal itself, so it can only ever pass for a non-empty principal, which
// is already checked by the caller) kept for auditability and consistency with every
// other read path in this package.
func (s Store) readDirectory(principal string) (ReadResult, error) {
	if err := screen.Authorize(screen.ScopeRequest{Op: sessionread.OpContextValue, Caller: principal, TargetOwner: principal}); err != nil {
		return ReadResult{}, err
	}
	rows := make([]directory.DirectoryRow, 0, len(s.Directory))
	for _, r := range s.Directory {
		if r.TraceID == principal {
			rows = append(rows, r)
		}
	}
	b, err := json.Marshal(rows)
	if err != nil {
		return ReadResult{}, err
	}
	return ReadResult{Contents: []Content{{URI: DirectoryURI, MimeType: "application/json", Text: string(b)}}}, nil
}
