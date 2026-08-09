package gateway

// session_teleport.go — #2419 (part of #2392, harness-native program #2387).
// Generation: gen/next.
//
// The portability layer over the durable session chain (#2416): fork, export and
// import move a served session BETWEEN hosts as a verifiable hash closure instead
// of a file copy.
//
// Claude Code sessions teleport between web and CLI and auto-resume on another
// worker after a crash. The copy is the easy part; the hard part is that the
// receiving host has no reason to believe what it was handed. So the portable unit
// here is not a blob — it is the ledger head plus the closure of entries that reach
// it, and the receiver RE-DERIVES every hash from the content it was given rather
// than trusting the hashes in the file. A bundle whose bytes were altered in flight
// cannot reproduce the head it declares, so import refuses it.
//
// Three verbs, all on the existing /v1/fak/session/{id}/{verb} control plane
// (handleFakSession, http.go):
//
//   - export — walk the trace to its root, verify the chain, and emit the closure
//     plus the re-arm state (budget, taint high-water) as one portable document.
//   - import — re-derive the whole chain on the receiving host, refuse anything
//     that does not reproduce the declared head, then re-arm the trace so the next
//     turn continues where the source host stopped.
//   - fork — mint a NEW trace whose head points at the shared prefix. No entry and
//     no content byte is copied: a fork is a second name for one immutable prefix.
//     Two forks of one session are therefore two sessions to the arbiter and to the
//     budget accountant, which is the property that makes divergent exploration
//     accountable rather than free.
//
// WHAT IS AND IS NOT PROVEN. The chain is WITNESSED: import recomputes
// digest(parent, kind, content) for every entry, so the entry bytes are pinned by
// the head. The re-arm state is sealed into the bundle's Closure digest, so a
// budget or taint edit in flight is caught too. What this does NOT do is
// AUTHENTICATE the exporter: the closure digest is unkeyed, so a party willing to
// recompute the whole document can hand the receiver a self-consistent bundle that
// simply is not the one host A exported. That is the invalidating assumption to
// carry forward — verification here answers "is this closure internally true",
// never "did A author it". Binding a bundle to its exporter needs a signature over
// the closure digest, which is a separate wire (#2214's descriptor is the natural
// place to hang it).

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/sessionledger"
)

// TeleportSchema versions the portable document. It is mixed into the closure
// digest, so a future schema change yields different closures rather than letting
// a v1 reader silently accept a v2 document it would misread.
const TeleportSchema = "fak.session.teleport.v1"

// maxTeleportEntries bounds the closure a single import will replay. The ledger's
// own MaxNodes ceiling already bounds what a chain walk can return, so this is the
// receiving side's independent bound: a hostile or corrupt document must not be
// able to make an import loop for an unbounded time.
const maxTeleportEntries = sessionledger.MaxNodes

// TeleportArm is the state that must be restored alongside the chain for the next
// turn to continue where the source host stopped. The chain says what happened; the
// arm says what the session is still allowed to do.
type TeleportArm struct {
	// Budget is the remaining work allotment. It travels because a teleport must
	// not refund a session's spend: a session that had four turns left on host A
	// has four turns left on host B.
	Budget SessionBudget `json:"budget"`
	// TaintHighWater is the most dangerous provenance label the session has ever
	// carried, as the /v1/fak/trace route spells it. It travels as a HIGH-WATER
	// mark, never as the current reading, so a teleport can only ever preserve or
	// tighten the receiving host's posture — moving hosts must not launder a
	// tainted session clean.
	TaintHighWater string `json:"taint_high_water,omitempty"`
	// Generation counts how many times this lineage has been re-armed (a budget
	// reset carryover, a teleport, a fork). It is provenance for the accountant,
	// not a limit.
	Generation int `json:"generation,omitempty"`
}

// TeleportBundle is the portable unit: one trace's ledger head, the closure of
// entries that reach it (root-first), and the re-arm state.
type TeleportBundle struct {
	Schema  string                `json:"schema"`
	TraceID string                `json:"trace_id"`
	Head    sessionledger.Hash    `json:"head"`
	Entries []sessionledger.Entry `json:"entries"`
	Arm     TeleportArm           `json:"arm"`
	// Closure seals the parts the chain hashes do not reach — the schema, the trace
	// name, the declared head and the re-arm state. Verify recomputes it, so an
	// edited budget or a relabelled taint is refused the same way an edited entry
	// is. It is an integrity seal, not a signature; see the file header.
	Closure string `json:"closure"`
}

// TeleportFork is what a fork verb answers with: the freshly minted trace and the
// prefix the two traces now share.
type TeleportFork struct {
	TraceID      string             `json:"trace_id"`
	ForkTraceID  string             `json:"fork_trace_id"`
	SharedPrefix sessionledger.Hash `json:"shared_prefix"`
}

// errTeleportResident is the refusal for importing onto a trace that already holds
// history on this host. Import re-derives the chain from the root, which is only
// truthful onto an empty trace; splicing a foreign closure under an existing head
// would mint hashes that match neither host.
var errTeleportResident = errors.New("trace already holds history on this host")

// teleportLedger opens the durable session ledger the control-plane verbs act on.
// It is a var so a test can hand the verbs an in-memory ledger instead of the
// process-wide durable one; the core functions below take their ledger explicitly
// and never consult this.
var teleportLedger = func() (*sessionledger.Ledger, error) { return sessionledger.OpenDefault() }

// closureDigest seals the document's non-chain parts. The arm is hashed through its
// canonical JSON encoding (Go emits struct fields in declaration order, so the
// encoding is stable for a given schema), with a length prefix on every field so no
// two different documents can concatenate to the same preimage.
func closureDigest(trace string, head sessionledger.Hash, arm TeleportArm) (string, error) {
	armJSON, err := json.Marshal(arm)
	if err != nil {
		return "", err
	}
	h := sha256.New()
	for _, part := range [][]byte{[]byte(TeleportSchema), []byte(trace), []byte(head), armJSON} {
		fmt.Fprintf(h, "%d:", len(part))
		h.Write(part)
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// ExportTeleport walks a trace to its root, verifies the chain, and returns the
// portable closure.
//
// A chain the ledger can no longer root — because its oldest entries aged out under
// the MaxNodes bound — is REFUSED rather than exported as a suffix. An unrooted
// suffix cannot be re-derived on the receiving host, so exporting one would only
// move the failure to import time, on a different host, after the operator had
// already committed to the move.
func ExportTeleport(l *sessionledger.Ledger, trace string, arm TeleportArm) (TeleportBundle, error) {
	trace = strings.TrimSpace(trace)
	if l == nil || trace == "" {
		return TeleportBundle{}, errors.New("teleport export: a ledger and a trace are required")
	}
	entries, err := l.Chain(trace)
	if err != nil {
		return TeleportBundle{}, fmt.Errorf("teleport export: %w", err)
	}
	if len(entries) == 0 {
		return TeleportBundle{}, fmt.Errorf("teleport export: trace %q has no entries", trace)
	}
	if err := sessionledger.Verify(entries); err != nil {
		return TeleportBundle{}, fmt.Errorf("teleport export: trace %q is not exportable: %w "+
			"(its oldest entries have aged out of the ledger, so the closure cannot be rooted)", trace, err)
	}
	if err := checkCanonicalContent(entries); err != nil {
		return TeleportBundle{}, fmt.Errorf("teleport export: %w", err)
	}
	head := entries[len(entries)-1].Hash
	if live := l.Head(trace); live != head {
		return TeleportBundle{}, fmt.Errorf("teleport export: trace %q moved under the walk (head %s, chain ends %s)", trace, live, head)
	}
	b := TeleportBundle{Schema: TeleportSchema, TraceID: trace, Head: head, Entries: entries, Arm: arm}
	if b.Closure, err = closureDigest(trace, head, arm); err != nil {
		return TeleportBundle{}, fmt.Errorf("teleport export: seal closure: %w", err)
	}
	return b, nil
}

// checkCanonicalContent refuses a chain whose entry content is not already in the
// compact form a JSON encoder emits.
//
// This is the sharp edge of shipping a hash closure over JSON. An entry's hash
// covers its content BYTES, but an encoder is free to re-flow the JSON it embeds —
// json.Marshal compacts a RawMessage, and MarshalIndent re-indents it. So content
// carrying insignificant whitespace hashes one way in the ledger and arrives on the
// receiving host hashing another, and the operator learns about it after the move.
//
// Every content byte the ledger persists has already been through an encoder
// (Append marshals the record it writes, and Elide marshals its stub), so a durable
// chain is canonical by construction and this check costs one pass and never fires.
// It fires for a chain assembled in memory from hand-written JSON — which is exactly
// the case that would otherwise corrupt silently in flight.
func checkCanonicalContent(entries []sessionledger.Entry) error {
	var buf bytes.Buffer
	for i, e := range entries {
		if len(e.Content) == 0 {
			continue
		}
		buf.Reset()
		if err := json.Compact(&buf, e.Content); err != nil {
			return fmt.Errorf("entry %d content is not JSON: %w", i, err)
		}
		if !bytes.Equal(buf.Bytes(), e.Content) {
			return fmt.Errorf("entry %d content is not in canonical compact form, so its hash "+
				"would not survive a JSON hop; re-append it through an encoder before exporting", i)
		}
	}
	return nil
}

// Verify checks that the bundle is internally true: the schema is one this build
// reads, the seal covers the head and the re-arm state, every entry's hash is the
// digest of its own parent/kind/content, and the last entry IS the declared head.
// It writes nothing, so a caller can screen a document before touching a ledger.
func (b TeleportBundle) Verify() error {
	if b.Schema != TeleportSchema {
		return fmt.Errorf("teleport: unknown schema %q (want %q)", b.Schema, TeleportSchema)
	}
	if strings.TrimSpace(b.TraceID) == "" {
		return errors.New("teleport: bundle names no trace")
	}
	if len(b.Entries) == 0 {
		return errors.New("teleport: bundle carries no entries")
	}
	if len(b.Entries) > maxTeleportEntries {
		return fmt.Errorf("teleport: bundle carries %d entries, over the %d ceiling", len(b.Entries), maxTeleportEntries)
	}
	want, err := closureDigest(b.TraceID, b.Head, b.Arm)
	if err != nil {
		return fmt.Errorf("teleport: seal closure: %w", err)
	}
	if b.Closure != want {
		return errors.New("teleport: closure seal does not cover this document (head, trace or re-arm state was altered)")
	}
	if err := sessionledger.Verify(b.Entries); err != nil {
		return fmt.Errorf("teleport: chain verification failed: %w", err)
	}
	if got := b.Entries[len(b.Entries)-1].Hash; got != b.Head {
		return fmt.Errorf("teleport: chain ends at %s but the bundle declares head %s", got, b.Head)
	}
	return nil
}

// ImportTeleport re-arms a session on THIS host from a portable closure.
//
// It refuses before it writes: Verify runs first, so a tampered or truncated
// document never reaches the ledger. Then, rather than trusting the hashes it was
// handed, it REPLAYS the closure — appending each entry's kind and content to the
// named trace, which re-derives that entry's hash from the parent the ledger itself
// is holding. The replay is checked entry by entry, so the import can only complete
// if the source's content reproduces the source's head byte for byte. That is the
// difference between verifying a chain and deserializing one.
//
// The returned bundle is the one this host re-derived, which a caller can compare
// against what it was sent.
func ImportTeleport(l *sessionledger.Ledger, b TeleportBundle) (TeleportBundle, error) {
	if l == nil {
		return TeleportBundle{}, errors.New("teleport import: a ledger is required")
	}
	if err := b.Verify(); err != nil {
		return TeleportBundle{}, err
	}
	trace := strings.TrimSpace(b.TraceID)
	if l.Head(trace) != "" {
		return TeleportBundle{}, fmt.Errorf("teleport import: %w: trace %q is at %s", errTeleportResident, trace, l.Head(trace))
	}
	for i, e := range b.Entries {
		got, err := l.Append(trace, e.Kind, e.Content)
		if err != nil {
			return TeleportBundle{}, fmt.Errorf("teleport import: replay entry %d: %w", i, err)
		}
		if got.Hash != e.Hash {
			return TeleportBundle{}, fmt.Errorf("teleport import: entry %d re-derived as %s, not the %s the bundle claims", i, got.Hash, e.Hash)
		}
	}
	if head := l.Head(trace); head != b.Head {
		return TeleportBundle{}, fmt.Errorf("teleport import: replay ended at %s, not the declared head %s", head, b.Head)
	}
	return ExportTeleport(l, trace, b.Arm)
}

// ForkTeleport mints a new trace whose head points at source's current head, and
// reports the prefix the two now share. Nothing is copied: the fork is a second
// name for one immutable prefix, which is why forking is cheap and why the two
// traces are nonetheless two sessions to every accountant downstream.
//
// An empty target is minted from the source and the shared head, so the id an
// operator reads names its own lineage. Forking the same head twice yields two
// distinct traces — "two forks of one session are two sessions".
func ForkTeleport(l *sessionledger.Ledger, source, target string) (TeleportFork, error) {
	source = strings.TrimSpace(source)
	if l == nil || source == "" {
		return TeleportFork{}, errors.New("teleport fork: a ledger and a source trace are required")
	}
	head := l.Head(source)
	if head == "" {
		return TeleportFork{}, fmt.Errorf("teleport fork: source trace %q not found", source)
	}
	target = strings.TrimSpace(target)
	if target == "" {
		target = mintForkTrace(l, source, head)
	}
	if target == source {
		return TeleportFork{}, fmt.Errorf("teleport fork: a trace cannot fork onto itself (%q)", source)
	}
	if l.Head(target) != "" {
		return TeleportFork{}, fmt.Errorf("teleport fork: target trace %q already holds history at %s", target, l.Head(target))
	}
	shared, err := l.Fork(source, target)
	if err != nil {
		return TeleportFork{}, fmt.Errorf("teleport fork: %w", err)
	}
	return TeleportFork{TraceID: source, ForkTraceID: target, SharedPrefix: shared}, nil
}

// mintForkTrace derives a readable, deterministic id for a fork of source at head,
// stepping an ordinal until it finds one this ledger is not already using — so a
// second fork of the same head gets its own trace instead of colliding with the
// first. The scan is bounded; the unsuffixed name is the last resort, and
// ForkTeleport's own occupied-target check refuses it rather than clobbering.
func mintForkTrace(l *sessionledger.Ledger, source string, head sessionledger.Hash) string {
	short := string(head)
	if len(short) > 12 {
		short = short[:12]
	}
	base := source + "-fork-" + short
	if l.Head(base) == "" {
		return base
	}
	for i := 2; i < 1000; i++ {
		if c := fmt.Sprintf("%s.%d", base, i); l.Head(c) == "" {
			return c
		}
	}
	return base
}

// teleportWindowDoc is the canonical shape of a next-turn resident window. It is a
// struct, not a map, so the encoding is field-ordered and therefore comparable byte
// for byte across hosts.
type teleportWindowDoc struct {
	Schema  string                `json:"schema"`
	TraceID string                `json:"trace_id"`
	Head    sessionledger.Hash    `json:"head"`
	Entries []sessionledger.Entry `json:"entries"`
	Arm     TeleportArm           `json:"arm"`
}

// TeleportWindow renders the resident window the next turn on this host would be
// built from: the verified chain in order, plus the re-arm state.
//
// It is a PURE function of the durable closure, and that is the whole point of the
// witness. "Did the session survive the hop" is otherwise an opinion; here it is a
// byte comparison. If the window host B renders after an import is byte-identical
// to the one host A renders before the export, then B's next turn is built from
// exactly the material A's would have been, and the teleport moved the session
// rather than an approximation of it.
//
// It deliberately re-walks the ledger instead of reading a bundle, so A (which
// never made a bundle) and B (which did) are both answering from their own durable
// state — the comparison would be circular otherwise.
func TeleportWindow(l *sessionledger.Ledger, trace string, arm TeleportArm) ([]byte, error) {
	trace = strings.TrimSpace(trace)
	if l == nil || trace == "" {
		return nil, errors.New("teleport window: a ledger and a trace are required")
	}
	entries, err := l.Chain(trace)
	if err != nil {
		return nil, fmt.Errorf("teleport window: %w", err)
	}
	if err := sessionledger.Verify(entries); err != nil {
		return nil, fmt.Errorf("teleport window: trace %q does not verify: %w", trace, err)
	}
	return json.Marshal(teleportWindowDoc{
		Schema:  TeleportSchema,
		TraceID: trace,
		Head:    l.Head(trace),
		Entries: entries,
		Arm:     arm,
	})
}

// ---------------------------------------------------------------------------
// control-plane verbs
// ---------------------------------------------------------------------------

// handleTeleportVerb serves the fork/export/import verbs on the session control
// subtree. It reports whether it claimed the verb, so handleFakSession can fall
// through to the generic drive-state control path for everything else.
//
// It is mounted ahead of the generic path for the same reason steer is: these three
// verbs act on the durable chain rather than on the drive table, so they take a
// different body and answer a different document.
func (s *Server) handleTeleportVerb(w http.ResponseWriter, r *http.Request, traceID, verb string) bool {
	switch verb {
	case "fork", "export", "import":
	default:
		return false
	}
	l, err := teleportLedger()
	if err != nil {
		writeErrCode(w, http.StatusServiceUnavailable, "teleport_no_ledger",
			"TELEPORT_NO_LEDGER: the durable session ledger could not be opened: "+err.Error())
		return true
	}

	switch verb {
	case "export":
		var req TeleportArm
		// The arm is optional: a caller that only wants the chain may POST an empty
		// body. A malformed one is still a 400 — silently exporting a zero budget
		// would understate what the receiving host is allowed to spend.
		if r.ContentLength != 0 && !decodeRequestBody(w, r, &req) {
			return true
		}
		b, err := ExportTeleport(l, traceID, req)
		if err != nil {
			writeErrCode(w, http.StatusNotFound, "teleport_not_exportable", "TELEPORT_NOT_EXPORTABLE: "+err.Error())
			return true
		}
		s.logf("gateway: session %s export -> head %s (%d entries)", traceID, b.Head, len(b.Entries))
		writeJSON(w, http.StatusOK, b)

	case "import":
		var b TeleportBundle
		if !decodeRequestBody(w, r, &b) {
			return true
		}
		// The path names the trace being re-armed; a bundle for a different trace on
		// that path is a routing mistake, not a silent re-home.
		if strings.TrimSpace(b.TraceID) != "" && strings.TrimSpace(b.TraceID) != traceID {
			writeErrCode(w, http.StatusBadRequest, "teleport_trace_mismatch",
				fmt.Sprintf("TELEPORT_TRACE_MISMATCH: the bundle carries trace %q but the path names %q", b.TraceID, traceID))
			return true
		}
		b.TraceID = traceID
		got, err := ImportTeleport(l, b)
		if err != nil {
			status, code := http.StatusUnprocessableEntity, "teleport_unverified"
			if errors.Is(err, errTeleportResident) {
				status, code = http.StatusConflict, "teleport_trace_resident"
			}
			writeErrCode(w, status, code, strings.ToUpper(code)+": "+err.Error())
			return true
		}
		s.logf("gateway: session %s import <- head %s (%d entries re-derived)", traceID, got.Head, len(got.Entries))
		writeJSON(w, http.StatusOK, got)

	case "fork":
		var req TeleportFork
		if r.ContentLength != 0 && !decodeRequestBody(w, r, &req) {
			return true
		}
		f, err := ForkTeleport(l, traceID, req.ForkTraceID)
		if err != nil {
			writeErrCode(w, http.StatusNotFound, "teleport_not_forkable", "TELEPORT_NOT_FORKABLE: "+err.Error())
			return true
		}
		s.logf("gateway: session %s fork -> %s at shared prefix %s", traceID, f.ForkTraceID, f.SharedPrefix)
		writeJSON(w, http.StatusOK, f)
	}
	return true
}
