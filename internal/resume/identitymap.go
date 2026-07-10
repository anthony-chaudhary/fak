// identitymap.go — the durable, GC-immune join between the two session keyspaces
// the resume watchdog straddles: the Claude Code transcript UUID
// (CLAUDE_CODE_SESSION_ID, the id `claude --resume` takes) and the gateway /
// guard TRACE id the operator control plane keys on. It is the keystone of the
// A-identity-join cluster (#1193): every consumer (the SessionStart producer, the
// recontinue refresh, the watchdog read-side join, the `fak session ls` surface)
// joins the two namespaces through the map this leaf folds.
//
// # Why a second store, disjoint from drivestate
//
// drivestate.go smuggles exactly ONE fact (the operator hold token) across the
// keyspace wall by re-recording it in the UUID keyspace. That is enough to VETO a
// resume but cannot ANSWER "which trace is this crashed UUID?" — the question a
// crash-journal row keyed by UUID must answer to resolve its gateway drive. This
// store carries the join itself: a row per observed (UUID, trace) pairing, folded
// into both lookup directions.
//
// # Pure by construction, durable by discipline
//
// FoldIdentity is a total function — same rows in, same maps out, no clock, no
// I/O — mirroring FoldDriveStates (drivestate.go). The shell reads the append-only
// resume_identity.jsonl and hands the parsed rows to the fold; because the store is
// append-only, slice order IS write order and the fold is "last row per key wins"
// in each direction. The store has NO TTL and is never swept, so a join survives
// the descriptor registry's 30-minute GC — a crashed UUID still resolves its trace
// long after the descriptor that first paired them has evaporated.
package resume

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
)

// IdentityRow is one append-only line of resume_identity.jsonl, reduced to the
// typed facts the fold reads. The shell parses the JSONL (jsonlledger.Parse);
// unknown fields are dropped, not trusted, so a forward-extended row still decodes.
// TS/Handle/Account/Via are carried for humans and audit only — the fold orders by
// FILE order (append-only ⇒ chronological) and joins on UUID<->Trace alone.
type IdentityRow struct {
	// TS is the row's ISO-8601 write time (audit only; the fold ignores it).
	TS string `json:"ts,omitempty"`
	// UUID is the Claude Code transcript UUID (== CLAUDE_CODE_SESSION_ID) — the
	// watchdog plan-row key and one endpoint of the join.
	UUID string `json:"uuid"`
	// Trace is the gateway / guard --session-id trace id — the operator control
	// plane key and the other endpoint of the join.
	Trace string `json:"trace"`
	// Handle is the optional human-facing session handle recorded with the row.
	Handle string `json:"handle,omitempty"`
	// Account is the optional worker account the session ran under (provenance).
	Account string `json:"account,omitempty"`
	// Via names what wrote the row (e.g. "guard SessionStart"), for provenance.
	Via string `json:"via,omitempty"`
}

// FoldIdentity folds the append-only rows into the two lookup directions every
// consumer of the join reads: traceByUUID answers "which trace is this transcript
// UUID?" and uuidByTrace the reverse. Last row per key wins in each direction (the
// store is append-only, so slice order is write order), so a re-paired UUID picks
// up its newest trace. A row missing either endpoint is skipped in BOTH directions
// — a half row is not a join and must never clobber a prior valid pairing (the same
// blank-key discipline FoldDriveStates applies to its Session key). Total over any
// input: nil/empty rows yield empty (non-nil) maps.
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

// IdentityLedgerPath is the durable, GC-immune identity store under the SAME regDir
// the resume tick already resolves, so the producer (guard SessionStart) and every
// reader always agree on the file. It is deliberately NOT the descriptor registry
// (session-registry.json): that store is keyed by the guard trace id — a disjoint
// keyspace from the transcript UUID — AND it is TTL-GC'd, so a join recorded there
// would evaporate. This file is append-only and never swept.
func IdentityLedgerPath(regDir string) string {
	return filepath.Join(regDir, "resume_identity.jsonl")
}

// LoadIdentity reads the append-only identity store and folds it — through the pure
// leaf FoldIdentity — into the two join directions, reusing jsonlledger.Parse exactly
// as the drive-state loader does. A missing / unreadable / empty file yields two
// empty (non-nil) maps, so an absent store leaves any join lookup inert (fail-open):
// it simply answers "no join" rather than stranding a session.
func LoadIdentity(regDir string) (traceByUUID, uuidByTrace map[string]string) {
	raw, _ := os.ReadFile(IdentityLedgerPath(regDir))
	return FoldIdentity(jsonlledger.Parse[IdentityRow](string(raw), nil))
}
