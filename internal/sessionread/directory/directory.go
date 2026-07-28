// Package directory is the sessionread C4 read projection (issue #4195, child of
// epic #4176): it exposes, over the session READ plane, the two things an external
// process needs to DISCOVER and ADDRESS a session it did not launch —
//
//  1. the durable UUID<->trace identity join (both directions), and
//  2. a UNIFIED session directory that folds the two disjoint "which sessions
//     exist" stores into one addressing view.
//
// # Why two source-of-truths, folded into one view
//
// A fak session is recorded in TWO stores that key on DISJOINT namespaces:
//
//   - internal/sessionjournal — the lifecycle journal (LIVE/CRASHED/STALE/CLOSED,
//     boot-epoch classified). Its rows answer "did this session survive / crash?"
//   - the gateway session table (drive-state: run-state, priority, parent) — its
//     rows answer "what is this session's live operator drive?".
//
// The join that bridges the two keyspaces — the Claude Code transcript UUID
// (CLAUDE_CODE_SESSION_ID) on one side and the gateway/guard TRACE id on the other
// — lives in a THIRD, GC-immune store (internal/resume/identitymap.go,
// resume_identity.jsonl, append-only). Before this leaf none of that was reachable
// as one read: a consumer had to read the JSONL off disk AND reconcile two session
// tables by hand. Directory folds all three into one row per session, keyed by
// trace, each row tagged with the SOURCE(s) it came from ("journal" | "drive" |
// "both") and carrying the trace/UUID pair a caller uses to address it.
//
// The identity map is LOAD-BEARING in the fold, not decoration: a journal row keyed
// by a transcript UUID and a drive row keyed by a gateway trace merge into ONE
// Source="both" row ONLY because the identity join resolves the UUID to that trace.
// That is precisely the "discover and address over the wire" gap #4195 names.
//
// # The OBSERVED qualifier: live, not attested
//
// Every DirectoryRow carries sessionread.EvidenceObserved. This is deliberate and
// load-bearing: a directory read is a LIVE reading of two mutable stores, true at
// read-time and stale the instant after — NOT a durably attested artifact fak
// authored (that would be EvidenceWitnessed). Tagging the evidence OBSERVED stops a
// consumer from treating "session X existed a moment ago" as a witnessed fact it can
// build a durable decision on. The unknown-lookup miss reuses the read plane's
// closed refusal token sessionread.ReasonReadUnknownTrace ("READ_UNKNOWN_TRACE") so
// a "never had it" miss speaks the same vocabulary as the rest of the read plane.
//
// # Why a LOCAL identity fold instead of importing internal/resume
//
// internal/resume already exposes FoldIdentity / ResolveIdentity over the exact
// IdentityRow shape below, and importing it would be "compose, don't duplicate".
// But `go list -deps ./internal/resume/` pulls internal/sessionaudit (a
// substring-sibling of the volatile internal/session family) into the transitive
// closure. To keep THIS read leaf hermetic — buildable and testable with
// `go test ./internal/sessionread/directory/` alone, uncoupled from any sibling's
// health in the shared multi-writer tree — the identity fold is reimplemented here
// over the same on-disk resume_identity.jsonl row shape. It is a TOTAL function
// (same rows in, same maps out, no clock, no I/O) mirroring resume.FoldIdentity, so
// the projection stays faithful to the one durable join. The only imports are
// stdlib + internal/sessionjournal (proven session-free) + internal/sessionread
// (the read-plane vocab).
package directory

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
	"github.com/anthony-chaudhary/fak/internal/sessionread"
)

// IdentityRow is one append-only line of resume_identity.jsonl reduced to the typed
// facts the fold reads — the SAME shape internal/resume.IdentityRow parses, so this
// leaf reads the identical durable store. UUID is the Claude Code transcript UUID
// (== CLAUDE_CODE_SESSION_ID); Trace is the gateway / guard --session-id trace id.
// TS/Handle/Account/Via are provenance carried for humans and audit — the fold
// orders by FILE order (append-only ⇒ chronological) and joins on UUID<->Trace alone.
type IdentityRow struct {
	// TS is the row's ISO-8601 write time (audit only; the fold ignores it).
	TS string `json:"ts,omitempty"`
	// UUID is the Claude Code transcript UUID — one endpoint of the join.
	UUID string `json:"uuid"`
	// Trace is the gateway / guard trace id — the other endpoint of the join.
	Trace string `json:"trace"`
	// Handle is the optional human-facing session handle recorded with the row.
	Handle string `json:"handle,omitempty"`
	// Account is the optional worker account the session ran under (provenance).
	Account string `json:"account,omitempty"`
	// Via names what wrote the row (e.g. "guard SessionStart"), for provenance.
	Via string `json:"via,omitempty"`
}

// FoldIdentity folds the append-only rows into the two lookup directions: traceByUUID
// answers "which trace is this transcript UUID?" and uuidByTrace the reverse. Last row
// per key wins in each direction (the store is append-only, so slice order is write
// order). A row missing either endpoint is skipped in BOTH directions — a half row is
// not a join and must never clobber a prior valid pairing. Total over any input:
// nil/empty rows yield empty (non-nil) maps. Mirrors resume.FoldIdentity.
func FoldIdentity(rows []IdentityRow) (traceByUUID, uuidByTrace map[string]string) {
	traceByUUID = make(map[string]string, len(rows))
	uuidByTrace = make(map[string]string, len(rows))
	for _, r := range rows {
		uuid := strings.TrimSpace(r.UUID)
		trace := strings.TrimSpace(r.Trace)
		if uuid == "" || trace == "" {
			continue // a join needs both endpoints; a half row maps neither way
		}
		traceByUUID[uuid] = trace
		uuidByTrace[trace] = uuid
	}
	return traceByUUID, uuidByTrace
}

// IdentityMatch is the resolved join a lookup returns: the query id, the id it pairs
// to, the direction that resolved, and the winning row (for handle/account/via
// provenance). OK is false when no row pairs the query; in that case Reason carries
// the read-plane miss token sessionread.ReasonReadUnknownTrace so a caller reports the
// miss in the same closed vocabulary the rest of the plane uses. Direction is
// "uuid->trace" when the query was a transcript UUID (Paired is its trace) or
// "trace->uuid" when it was a gateway trace (Paired is its UUID).
type IdentityMatch struct {
	Query     string
	Paired    string
	Direction string
	Row       IdentityRow
	OK        bool
	Reason    string
}

// ResolveIdentity resolves query against the append-only rows in EITHER direction,
// honoring the same "last row per key wins" rule FoldIdentity applies (the store is
// append-only, so slice order is write order): it scans forward and keeps the newest
// row whose UUID — or, failing that, whose Trace — equals the query. Half rows (missing
// an endpoint) are skipped, exactly as the fold skips them. A blank query, or one no
// row pairs, yields OK=false with Reason=READ_UNKNOWN_TRACE. Pure and total: no clock,
// no I/O, deterministic over any input.
func ResolveIdentity(rows []IdentityRow, query string) IdentityMatch {
	q := strings.TrimSpace(query)
	if q == "" {
		return IdentityMatch{Query: query, Reason: sessionread.ReasonReadUnknownTrace}
	}
	m := IdentityMatch{Query: q}
	for _, r := range rows {
		uuid := strings.TrimSpace(r.UUID)
		trace := strings.TrimSpace(r.Trace)
		if uuid == "" || trace == "" {
			continue // a half row is not a join (same discipline as FoldIdentity)
		}
		switch q {
		case uuid:
			m = IdentityMatch{Query: q, Paired: trace, Direction: "uuid->trace", Row: r, OK: true}
		case trace:
			m = IdentityMatch{Query: q, Paired: uuid, Direction: "trace->uuid", Row: r, OK: true}
		}
	}
	if !m.OK {
		m.Reason = sessionread.ReasonReadUnknownTrace
	}
	return m
}

// IdentityLedgerPath is the durable, GC-immune identity store under regDir — the SAME
// resume_identity.jsonl the resume tick writes, so this reader and the producer always
// agree on the file.
func IdentityLedgerPath(regDir string) string {
	return filepath.Join(regDir, "resume_identity.jsonl")
}

// LoadIdentityRows reads and parses the append-only identity store into its raw rows.
// A missing / unreadable file yields nil, so an absent store simply resolves no join
// (fail-open). Blank lines and '#' comment lines are skipped; an un-decodable line is
// dropped rather than trusted (a forward-extended row still decodes, unknown fields
// ignored).
func LoadIdentityRows(regDir string) []IdentityRow {
	raw, err := os.ReadFile(IdentityLedgerPath(regDir))
	if err != nil {
		return nil
	}
	var out []IdentityRow
	for _, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		var r IdentityRow
		if json.Unmarshal([]byte(line), &r) != nil {
			continue
		}
		out = append(out, r)
	}
	return out
}

// DriveRow is a minimal projection of the gateway session table's drive-state row (the
// gateway.SessionState the host would inject). It is DEFINED here rather than imported
// because internal/gateway is volatile and must never enter this hermetic read leaf's
// closure; the host projects its live table onto this shape before calling Directory.
type DriveRow struct {
	// TraceID is the gateway trace id the drive-state row keys on.
	TraceID string
	// Run is the live run-state (e.g. "RUNNING", "PARKED") — lands on RunState.
	Run string
	// Priority is the session's dispatch priority.
	Priority int
	// ParentTrace is the parent session's trace, if this is a child.
	ParentTrace string
}

// Source names which store(s) a DirectoryRow was folded from.
type Source string

const (
	// SourceJournal — the row exists only in the sessionjournal lifecycle store.
	SourceJournal Source = "journal"
	// SourceDrive — the row exists only in the gateway drive-state table.
	SourceDrive Source = "drive"
	// SourceBoth — the same trace was found in BOTH stores and merged (the join
	// the identity map makes possible).
	SourceBoth Source = "both"
)

// DirectoryRow is one folded, addressable session in the unified directory. TraceID and
// UUID are the two ways to address it (UUID present only when the identity join knows
// it). Lifecycle is the sessionjournal verdict when a journal row contributed it;
// RunState the gateway run-state when a drive row did. Evidence is ALWAYS
// sessionread.EvidenceObserved — a live reading of mutable stores, not an attested
// artifact.
type DirectoryRow struct {
	TraceID     string
	UUID        string
	Source      Source
	Lifecycle   sessionjournal.Status // "" when no journal row contributed
	RunState    string                // "" when no drive row contributed
	Priority    int                   // from the drive row, if any
	ParentTrace string                // from the drive row, if any
	Evidence    sessionread.Evidence  // always EvidenceObserved
}

// addSource folds a contributing source into the row: the first source sets it; a
// second, DIFFERENT source promotes it to "both"; a repeat is a no-op.
func (r *DirectoryRow) addSource(s Source) {
	switch r.Source {
	case "":
		r.Source = s
	case s:
		// already recorded this source
	default:
		r.Source = SourceBoth
	}
}

// resolveTrace maps a session/journal id onto the (trace, uuid) pair using the identity
// join, robustly for either keyspace: if id is a known transcript UUID it returns its
// trace; if id is a known gateway trace it returns its UUID; if the identity map knows
// neither, id is treated as a bare trace (uuid unknown). This is what lets a journal
// row keyed by a UUID and a drive row keyed by a trace collapse to one key.
func resolveTrace(id string, traceByUUID, uuidByTrace map[string]string) (trace, uuid string) {
	id = strings.TrimSpace(id)
	if id == "" {
		return "", ""
	}
	if t := traceByUUID[id]; t != "" {
		return t, id // id was a transcript UUID
	}
	if u := uuidByTrace[id]; u != "" {
		return id, u // id was a gateway trace
	}
	return id, "" // unknown to the identity map: treat as a bare trace
}

// rowKey is the internal accumulation key. Both stores fold onto the resolved TRACE, so
// a journal row and a drive row for the same session share a key and merge. A trace-less
// row (empty id) falls back to its uuid so it is not silently dropped.
func rowKey(trace, uuid string) string {
	if t := strings.TrimSpace(trace); t != "" {
		return "T:" + t
	}
	if u := strings.TrimSpace(uuid); u != "" {
		return "U:" + u
	}
	return "" // both empty — a degenerate row folds under the empty key
}

// Directory folds the three sources — sessionjournal lifecycle rows, gateway drive-state
// rows, and the durable identity join — into one addressing view, one DirectoryRow per
// session keyed by trace. A trace seen in BOTH the journal and the drive table yields a
// single Source="both" row (the identity join is what bridges the journal's UUID key to
// the drive's trace key); traces seen in only one store yield Source="journal" or
// Source="drive" rows. Every row is tagged sessionread.EvidenceObserved: a live reading,
// not an attested artifact. Rows are returned in first-seen order (journal rows, then
// any drive-only rows). Pure and total: no clock, no I/O.
func Directory(journalRows []sessionjournal.Classified, driveRows []DriveRow, identityRows []IdentityRow) []DirectoryRow {
	traceByUUID, uuidByTrace := FoldIdentity(identityRows)

	acc := map[string]*DirectoryRow{}
	order := make([]string, 0, len(journalRows)+len(driveRows))
	get := func(key string) *DirectoryRow {
		if r, ok := acc[key]; ok {
			return r
		}
		r := &DirectoryRow{Evidence: sessionread.EvidenceObserved}
		acc[key] = r
		order = append(order, key)
		return r
	}

	// rowFor resolves the row for one (trace, uuid) identity — creating it on first sight —
	// and fills in whichever half of the identity is still blank. Blank-only filling IS the
	// merge rule: the store that saw an identity first keeps it, which is exactly why the
	// lifecycle store is folded in before the drive store below.
	rowFor := func(trace, uuid string) *DirectoryRow {
		r := get(rowKey(trace, uuid))
		if r.TraceID == "" {
			r.TraceID = trace
		}
		if r.UUID == "" {
			r.UUID = uuid
		}
		return r
	}

	// Lifecycle store first, so a "both" row keeps its journal identity in the output order.
	for _, c := range journalRows {
		r := rowFor(resolveTrace(c.ID, traceByUUID, uuidByTrace))
		r.Lifecycle = c.Status
		r.addSource(SourceJournal)
	}

	// Drive-state store: fold onto the same trace key; a match merges to Source="both".
	for _, d := range driveRows {
		trace := strings.TrimSpace(d.TraceID)
		r := rowFor(trace, uuidByTrace[trace])
		r.RunState = d.Run
		r.Priority = d.Priority
		r.ParentTrace = d.ParentTrace
		r.addSource(SourceDrive)
	}

	out := make([]DirectoryRow, 0, len(order))
	for _, k := range order {
		out = append(out, *acc[k])
	}
	return out
}

// Lookup addresses a session in the directory by its trace id OR its transcript UUID,
// returning the folded row on a hit. A blank id, or one no row carries, is the closed
// "never had it" miss: OK=false with reason sessionread.ReasonReadUnknownTrace, matching
// the read plane's refusal vocabulary rather than inventing a row.
func Lookup(dir []DirectoryRow, id string) (row DirectoryRow, reason string, ok bool) {
	q := strings.TrimSpace(id)
	if q == "" {
		return DirectoryRow{}, sessionread.ReasonReadUnknownTrace, false
	}
	for _, r := range dir {
		if r.TraceID == q || r.UUID == q {
			return r, "", true
		}
	}
	return DirectoryRow{}, sessionread.ReasonReadUnknownTrace, false
}
