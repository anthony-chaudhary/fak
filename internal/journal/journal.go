// Package journal is the durable, append-only, tamper-evident DECISION JOURNAL —
// the regulated-audit (AUD) surface of the trust floor. The kernel already
// fans out a lifecycle Event on every adjudication (abi.RegisterEmitter), but
// the three shipped emitters (harvest, vdso, ifc) are all IN-MEMORY: none is a
// durable record of "what did the kernel decide, when, over which bytes, and
// why". This package closes that gap.
//
// WHAT IT GUARANTEES.
//
//   - DURABLE: a persisting abi.Emitter writes one JSONL row per
//     EvDecide / EvDeny / EvResultDeny / EvQuarantine / EvVDSOHit / EvCapFault / EvCapEvict /
//     EvCapVersionBind (the vDSO-served hit is included, so a cache hit is audited exactly
//     like an engine call; capability lifecycle events are the C6 witness surface).
//     Rows are appended and flushed per write, so a process crash loses nothing already
//     returned to the caller.
//   - TIME/SEQUENCE ANCHORED: the journal stamps its OWN monotonic Seq (1-based)
//     and a wall-clock timestamp on every row. The anchor lives in the row, not in
//     abi.Event — the frozen ABI is untouched.
//   - TAMPER-EVIDENT: every row carries the hash of the previous row's hash
//     chained with its own content (a hash chain / WORM ledger). Verify re-reads
//     the file and recomputes the chain; a single flipped byte breaks the link at
//     that row and fails the check. The journal does not PREVENT a privileged
//     edit, but it makes one DETECTABLE — the property an auditor underwrites.
//   - LIVE: in-process subscribers (and the gateway's /v1/fak/events stream) see
//     each row as it is committed; Recent serves a bounded tail without re-reading
//     the file.
//
// ENABLEMENT. The journal is off by default: writing to disk on every
// adjudication is a deployment choice, not something a benchmark or a unit test
// should pay for. Two ways to turn it on, both registering ONE persisting emitter
// against the frozen ABI:
//
//   - FAK_AUDIT_JOURNAL=/path/to/journal.jsonl — the package's init enables it at
//     boot (the FAK_IFC-style env toggle the rest of the kernel uses).
//   - journal.Enable(path) — the programmatic equivalent, for a front door (fak
//     guard) that decides AFTER init to default the audit trail on. Idempotent and
//     boot-env-respecting (see Enable).
//
// Unset and never Enabled => no emitter is registered and this package is inert.
package journal

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// Row is one durable audit record. It is the on-disk JSONL schema AND the live
// stream element. Field order is the hash-chain pre-image order — do not reorder
// without bumping the chain (it would invalidate every existing journal).
type Row struct {
	Seq          uint64 `json:"seq"`          // monotonic 1-based order anchor
	TSUnixNano   int64  `json:"ts_unix_nano"` // wall-clock time anchor
	Kind         string `json:"kind"`         // DECIDE | DENY | RESULT_DENY | QUARANTINE | VDSO_HIT | CAP_FAULT | CAP_EVICT | CAP_VERSION_BIND
	Tool         string `json:"tool,omitempty"`
	TraceID      string `json:"trace_id,omitempty"`
	Verdict      string `json:"verdict,omitempty"`
	Reason       string `json:"reason,omitempty"`
	By           string `json:"by,omitempty"`            // which adjudicator decided
	Taint        string `json:"taint,omitempty"`         // result provenance taint
	ArgsDigest   string `json:"args_digest,omitempty"`   // content hash of the call args
	ResultDigest string `json:"result_digest,omitempty"` // content hash of the result payload
	PrevHash     string `json:"prev_hash"`               // hash of the previous row ("" at genesis)
	Hash         string `json:"hash"`                    // chainHash(PrevHash, this row)

	// Correlation / bounded-disclosure fields — recorded and streamed, but NOT part
	// of the hash-chain pre-image (chainHash lists the chained fields explicitly, so
	// these are appended after Hash and EXISTING journals verify unchanged). The
	// tamper-evidence guarantee covers only the decision fields above; these are a
	// debugging convenience layered on top.
	CallSeq   uint64 `json:"call_seq,omitempty"`   // the kernel's per-call submission id (ToolCall.SeqNo): the join key tying a call's DECIDE to its later QUARANTINE
	Witness   string `json:"witness,omitempty"`    // the bounded-disclosure claim the verdict surfaced (offending self-modify glob / tool.arg bound / require-witness claim)
	ArgsLabel string `json:"args_label,omitempty"` // bounded, redacted call shape label for operator diagnosis; never raw args
	// DenyRule is the closed-vocabulary id of the policy RUNG that refused
	// (abi.DenyRuleID) — the machine-routable key Witness only ever carried as the
	// leading words of free-text prose. Verdict.Reason names the refusal's class;
	// this names the rule inside it, so a consumer can separate the seven gitgate
	// laws that all cite POLICY_BLOCK, or the recursive-delete rung from the
	// out-of-tree-write rung, without parsing a 400-character claim (#5863).
	//
	// It is the ONE field on this row whose value space is a compile-time closed
	// set: rowFromEvent re-validates whatever the rung stamped through
	// abi.DenyRuleID and drops a non-member WHOLE. Nothing is filtered, trimmed, or
	// truncated into the set, so — unlike a scrub-and-bound field such as ArgsLabel,
	// which fused an env-assignment value into a label before #5863 — no input byte
	// can appear here at all. Do not "relax" that to a character class.
	DenyRule string `json:"deny_rule,omitempty"`

	// Override fields (for SECURITY_OVERRIDE: audit trail for IFC and quarantine model overrides).
	// Not part of the hash-chain pre-image; appended for operator and audit convenience.
	OverrideType string `json:"override_type,omitempty"`

	// Capability fields (for C6: witness + audit surface). These are populated for
	// CAP_FAULT / CAP_EVICT / CAP_VERSION_BIND events to track capability lifecycle.
	// Fields is the carrier; these are NOT part of the hash-chain pre-image.
	CapKind   string `json:"cap_kind,omitempty"`   // capability kind: skill | mcp-tool | a2a-agent | ...
	CapName   string `json:"cap_name,omitempty"`   // capability name
	CapDigest string `json:"cap_digest,omitempty"` // capability content digest
	CapFrom   string `json:"cap_from,omitempty"`   // source version (for CAP_VERSION_BIND)
	CapTo     string `json:"cap_to,omitempty"`     // target version (for CAP_VERSION_BIND)

	// Crash fields (for CHILD_CRASH: the supervised-child abnormal-exit witness). A
	// crash is NOT a kernel decision — it happens outside the adjudication path when
	// the wrapped agent (or guard itself) dies — so it never flows through the ABI
	// emitter; AppendCrash writes it directly, like Cut's boundary anchor. These are
	// NOT part of the hash-chain pre-image (chainHash lists the chained fields
	// explicitly, so appending them here leaves every existing journal verifying
	// byte-for-byte). The chained forensic identity of a crash — its Kind, the agent
	// (Tool), the session (TraceID), and the closed-vocabulary class (Reason) — rides
	// the frozen decision fields above; ExitCode is a debugging convenience layered
	// on top.
	ExitCode  int               `json:"exit_code,omitempty"` // the child's exit code (-1 when signaled); 0 omitted
	ChildExit *ChildExitWitness `json:"child_exit,omitempty"`

	// Restart-chain field (for RESTART_HOP: the budget-restart continuity
	// witness, #3057). Like a crash, a restart is supervision — not a kernel
	// decision — so AppendRestartHop writes it directly through the chain. NOT
	// part of the hash-chain pre-image (chainHash lists the chained fields
	// explicitly, so existing journals verify byte-for-byte); the chained
	// forensic identity of a hop — Kind, agent (Tool), guard session (TraceID),
	// continuity class (Reason) — rides the frozen decision fields above, and
	// this carries the full correlated record layered on top.
	Restart *RestartHop `json:"restart,omitempty"`

	// Livelock field (for LIVELOCK: the result-side repeat-loop witness). A livelock
	// trip is a gateway observation, not a kernel decision — the result-side detector
	// crossed its repeat threshold on genuinely re-issued calls — so AppendLivelock
	// writes it directly through the chain, like a crash. NOT part of the hash-chain
	// pre-image (chainHash lists the chained fields explicitly, so appending it here
	// leaves every existing journal verifying byte-for-byte); the chained forensic
	// identity of a trip — Kind, the tool (Tool), the session (TraceID), the failure
	// class (Reason) — rides the frozen decision fields above, and this carries the
	// content-free repeat detail layered on top.
	Livelock *LivelockRow `json:"livelock,omitempty"`

	// Config-swap field (for CONFIG_SWAP: the capability-floor / route-manifest
	// hot-swap witness, #3959). A swap changes the live security boundary but is
	// supervision, not a kernel decision, so AppendConfigSwap writes it directly
	// through the chain, like a restart. NOT part of the hash-chain pre-image
	// (chainHash lists the chained fields explicitly, so appending it here leaves
	// every existing journal verifying byte-for-byte); the chained forensic
	// identity of a swap — Kind, the swapped surface (Tool), the outcome class
	// (Reason) — rides the frozen decision fields above, and this carries the full
	// correlated record (source path + sha256 of the installed bytes) on top.
	ConfigSwap *ConfigSwapRow `json:"config_swap,omitempty"`

	// Quality-quarantine field (for QUALITY_QUARANTINE: the flaky-quality-eval
	// retry-policy witness, #4569). A quarantine decision is quality supervision,
	// not a per-call kernel decision, so AppendQualityQuarantine writes it directly
	// through the chain, like a config swap. NOT part of the hash-chain pre-image
	// (chainHash lists the chained fields explicitly, so appending it here leaves
	// every existing journal verifying byte-for-byte); the chained forensic identity
	// of a case — Kind, the case id (Tool), the folded verdict (Verdict) and its
	// mirrored class (Reason) — rides the frozen decision fields above, and this
	// carries the full correlated record (provenance + tier/cost + attempt log) on top.
	Quality *QualityQuarantineRow `json:"quality,omitempty"`

	// Relay-provenance field (for RELAY_MSG: the cross-platform relayed-message
	// witness, #2851). Relaying a message is transport, not a kernel decision, so
	// RelayChain.Append writes it directly through the chain, like a restart. NOT
	// part of the hash-chain pre-image (chainHash lists the chained fields
	// explicitly, so appending it here leaves every existing journal verifying
	// byte-for-byte); the chained forensic identity of a relayed message — Kind,
	// the platform (Tool), the session key (TraceID), the adjudication verdict
	// (Verdict) and its refusal class (Reason), the direction (By), and the body
	// digest (ArgsDigest) — rides the frozen decision fields above, and this
	// carries the full correlated record (user id, turn id, per-session
	// predecessor link) layered on top.
	Relay *RelayProvenance `json:"relay,omitempty"`

	// Capability-grant field (for CAPABILITY_GRANT: the GATED-WIDEN
	// provenance witness, #5178). A grant loosens the live security boundary
	// but is supervision, not a kernel decision, so AppendCapabilityGrant
	// writes it directly through the chain, like a config swap. NOT part of
	// the hash-chain pre-image (chainHash lists the chained fields explicitly,
	// so appending it here leaves every existing journal verifying
	// byte-for-byte); the chained forensic identity of a grant — Kind, the
	// widened knob (Tool), the gated channel (Reason) and the actor (By) —
	// rides the frozen decision fields above, and this carries the full
	// correlated record (old→new values, amendment class, source) on top.
	Grant *CapabilityGrantRow `json:"capability_grant,omitempty"`
}

// Journal is a hash-chained append-only ledger with an in-process live stream.
// The zero value is not usable; construct with Open (file-backed) or OpenMemory
// (in-memory, for tests of the stream/verify logic).
type Journal struct {
	mu        sync.Mutex
	bw        *bufio.Writer
	f         *os.File         // nil for an in-memory journal
	path      string           // file path ("" for an in-memory journal)
	seq       uint64           // last committed seq
	lastHash  string           // last committed row hash (the chain head)
	clock     func() time.Time // injectable for deterministic tests
	subs      map[int]chan Row // live subscribers (best-effort fan-out)
	nextSub   int
	recent    []Row // bounded tail for Recent (full history is on disk)
	maxRecent int
	dropped   uint64 // live-stream sends dropped (slow consumer)
	writeErr  uint64 // append failures (file-backed)
}

const defaultMaxRecent = 1024

// Open opens (creating if absent) a file-backed journal in append mode. If the
// file already holds rows, Open recovers the chain head (seq + last hash) so a
// restart CONTINUES the same tamper-evident chain instead of forking it. A
// corrupt tail is reported via Verify, not here — Open stays robust so a damaged
// log never bricks startup; the auditor runs Verify to learn the chain is broken.
func Open(path string) (*Journal, error) {
	// Recover the chain head from any existing content first.
	seq, last, err := recoverHead(path)
	if err != nil {
		return nil, err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return nil, fmt.Errorf("journal: open %s: %w", path, err)
	}
	j := newJournal()
	j.f = f
	j.bw = bufio.NewWriter(f)
	j.path = path
	j.seq = seq
	j.lastHash = last
	return j, nil
}

// OpenMemory builds an in-memory journal (no file). Rows still chain, stream, and
// land in Recent; VerifyRows validates the chain over Recent without touching
// disk. Used by tests of the stream/verify logic.
func OpenMemory() *Journal { return newJournal() }

func newJournal() *Journal {
	return &Journal{
		clock:     time.Now,
		subs:      map[int]chan Row{},
		maxRecent: defaultMaxRecent,
	}
}

// Emit implements abi.Emitter: it is the kernel's per-event tap. It records every
// audit-relevant lifecycle event as a durable, chained row. It never blocks the
// kernel on a slow live consumer (best-effort fan-out) and never panics; a write
// failure is counted (WriteErrors) rather than propagated, since the fan-out
// contract is fire-and-forget.
func (j *Journal) Emit(ev abi.Event) {
	row, ok := rowFromEvent(ev)
	if !ok {
		return
	}
	j.append(row)
}

// append stamps the order/time anchor + chain hash and commits the row.
func (j *Journal) append(row Row) {
	j.mu.Lock()
	j.appendLocked(row)
	j.mu.Unlock()
}

// appendLocked is the commit core; the caller must already hold j.mu. It stamps
// the order/time anchor + chain hash onto the row and commits it (disk write,
// recent tail, best-effort live fan-out). Cut appends its boundary anchor through
// this so the anchor obeys every Journal invariant (chain head, recent tail,
// subscribers) while Cut keeps the lock across the whole rotation.
func (j *Journal) appendLocked(row Row) {
	j.seq++
	row.Seq = j.seq
	row.TSUnixNano = j.clock().UnixNano()
	row.PrevHash = j.lastHash
	row.Hash = chainHash(row.PrevHash, row)
	j.lastHash = row.Hash

	if j.bw != nil {
		if err := writeRow(j.bw, row); err != nil {
			j.writeErr++
		}
	}
	// Bounded recent tail (full history is the file).
	j.recent = append(j.recent, row)
	if len(j.recent) > j.maxRecent {
		j.recent = j.recent[len(j.recent)-j.maxRecent:]
	}
	// Best-effort live fan-out: never block the kernel on a slow subscriber.
	for _, ch := range j.subs {
		select {
		case ch <- row:
		default:
			j.dropped++
		}
	}
}

// Subscribe returns a live channel of rows committed AFTER the call, plus a
// cancel that unsubscribes and closes the channel. Sends are best-effort: a
// subscriber that falls behind drops rows (counted in Dropped) — the FILE is the
// durable record, the stream is a convenience.
func (j *Journal) Subscribe() (<-chan Row, func()) {
	j.mu.Lock()
	id := j.nextSub
	j.nextSub++
	ch := make(chan Row, 256)
	j.subs[id] = ch
	j.mu.Unlock()
	var once sync.Once
	cancel := func() {
		once.Do(func() {
			j.mu.Lock()
			if c, ok := j.subs[id]; ok {
				delete(j.subs, id)
				close(c)
			}
			j.mu.Unlock()
		})
	}
	return ch, cancel
}

// Recent returns up to the last n committed rows (most recent last). n<=0 returns
// the whole bounded tail. It serves the gateway endpoint without re-reading disk.
func (j *Journal) Recent(n int) []Row {
	j.mu.Lock()
	defer j.mu.Unlock()
	if n <= 0 || n > len(j.recent) {
		n = len(j.recent)
	}
	out := make([]Row, n)
	copy(out, j.recent[len(j.recent)-n:])
	return out
}

// Stats reports the journal's live counters (head seq, dropped stream sends,
// write errors) for /healthz-style introspection.
func (j *Journal) Stats() (seq, dropped, writeErrors uint64) {
	j.mu.Lock()
	defer j.mu.Unlock()
	return j.seq, j.dropped, j.writeErr
}

// Path returns the file the journal appends to, or "" for an in-memory journal.
// It lets a caller that enabled the journal report WHERE the durable trail lives
// (the fak guard banner / exit summary) and what to run `fak audit verify` over.
//
// A NIL receiver reports "" instead of panicking. Across cmd/fak a nil *Journal is the ordinary
// encoding of "journaling is off" — maybeRecordGuardSessionIndex (cmd/fak/guard_sessions.go)
// tests `audit == nil` before reaching for the path for precisely that reason. A sibling call
// site two lines later (the guardOwnsInteractiveTerminal row in cmd/fak/guard.go) omitted that
// check, so every DEFAULT `fak guard` launch — the attended path, which enables no journal —
// nil-dereferenced j.mu here and panicked. cmdGuard then exits 2, which reds the entire
// guard / precompact / stop-hook test family at once from one missing nil test. Making this
// accessor TOTAL retires that whole class: "no journal at all" and "journal with no file" now
// answer the same "" the doc line above already promises for the in-memory case.
func (j *Journal) Path() string {
	if j == nil {
		return ""
	}
	j.mu.Lock()
	defer j.mu.Unlock()
	return j.path
}

// ExportTo writes the journal as JSONL to w — the durable full history for a
// file-backed journal (re-read from disk after flushing buffered rows), or the
// bounded recent tail for an in-memory one. Returns the number of rows written.
// The output round-trips through Verify-style parsing, so an export of a sound
// journal is itself a sound journal.
func (j *Journal) ExportTo(w io.Writer) (int, error) {
	j.mu.Lock()
	if j.bw != nil {
		if err := j.bw.Flush(); err != nil {
			j.mu.Unlock()
			return 0, err
		}
	}
	path, mem := j.path, append([]Row(nil), j.recent...)
	j.mu.Unlock()

	if path == "" { // in-memory: export the recent tail
		n := 0
		for _, row := range mem {
			b, err := json.Marshal(row)
			if err != nil {
				return n, err
			}
			if _, err := w.Write(append(b, '\n')); err != nil {
				return n, err
			}
			n++
		}
		return n, nil
	}
	f, err := os.Open(path)
	if err != nil {
		return 0, fmt.Errorf("journal: export %s: %w", path, err)
	}
	defer f.Close()
	n := 0
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	for sc.Scan() {
		if len(sc.Bytes()) == 0 {
			continue
		}
		if _, err := w.Write(append(sc.Bytes(), '\n')); err != nil {
			return n, err
		}
		n++
	}
	return n, sc.Err()
}

// Flush pushes buffered bytes to the OS (durable across a process crash, not a
// power loss). Append already flushes per row; Flush is for explicit sync points.
func (j *Journal) Flush() error {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.bw != nil {
		return j.bw.Flush()
	}
	return nil
}

// Close flushes, fsyncs, and closes the file. Safe on an in-memory journal.
func (j *Journal) Close() error {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.bw != nil {
		if err := j.bw.Flush(); err != nil {
			return err
		}
	}
	if j.f != nil {
		_ = j.f.Sync()
		err := j.f.Close()
		j.f, j.bw = nil, nil
		return err
	}
	return nil
}

// writeRow appends one JSONL row and flushes it (per-row durability).
func writeRow(bw *bufio.Writer, row Row) error {
	b, err := json.Marshal(row)
	if err != nil {
		return err
	}
	if _, err := bw.Write(b); err != nil {
		return err
	}
	if err := bw.WriteByte('\n'); err != nil {
		return err
	}
	return bw.Flush()
}

// chainHash is the tamper-evident link: sha256 over the previous row's hash
// chained with this row's content fields (Seq..ResultDigest, in declaration
// order). PrevHash and Hash are excluded from the pre-image (PrevHash is the
// chained-in prefix; Hash is the output). A unit separator (0x1f) delimits fields
// so no concatenation collision is possible.
func chainHash(prev string, r Row) string {
	h := sha256.New()
	io.WriteString(h, prev)
	fmt.Fprintf(h, "\x1f%d\x1f%d\x1f%s\x1f%s\x1f%s\x1f%s\x1f%s\x1f%s\x1f%s\x1f%s\x1f%s",
		r.Seq, r.TSUnixNano, r.Kind, r.Tool, r.TraceID, r.Verdict,
		r.Reason, r.By, r.Taint, r.ArgsDigest, r.ResultDigest)
	return hex.EncodeToString(h.Sum(nil))
}

// rowFromEvent projects a lifecycle Event into an audit row, returning false for
// a non-audit kind (EvSubmit/EvDispatch/EvComplete/EvRungLabel are operational,
// not decisions). Digests come from the frozen Ref.Digest (the content hash the
// vDSO + provenance already maintain) — the emitter never resolves blob bytes,
// so it stays cheap and leaks no payload into the log.
func rowFromEvent(ev abi.Event) (Row, bool) {
	var kind string
	switch ev.Kind {
	case abi.EvDecide:
		// A DENY (and a Submit-path require-witness intermediate) is ALSO emitted as a
		// dedicated follow-up event the kernel always pairs with this EvDecide.
		// Recording this EvDecide row too would write the SAME decision twice into the
		// durable hash-chained journal, double-counting it in every consumer that folds
		// rows back — the `fak guard` exit summary's "decision(s) appended" count and
		// the guard-RSI verdict-quality metric (which keys on the `verdict` field, so a
		// DECIDE(DENY)+DENY pair counts as two denials). Skip the redundant re-emit so
		// each decision lands ONCE; abi.RedundantDecisionEvent is the shared rule the
		// decision-stream folders agree on (rungobs is the reference consumer).
		if abi.RedundantDecisionEvent(ev) {
			return Row{}, false
		}
		if ev.Fields != nil && ev.Fields["event"] == "security_override" {
			kind = KindSecurityOverride
		} else {
			kind = "DECIDE"
		}
	case abi.EvDeny:
		kind = "DENY"
	case abi.EvResultDeny:
		kind = "RESULT_DENY"
	case abi.EvQuarantine:
		kind = "QUARANTINE"
	case abi.EvVDSOHit:
		kind = "VDSO_HIT"
	case abi.EvCapFault:
		kind = "CAP_FAULT"
	case abi.EvCapEvict:
		kind = "CAP_EVICT"
	case abi.EvCapVersionBind:
		kind = "CAP_VERSION_BIND"
	default:
		return Row{}, false
	}
	row := Row{Kind: kind}
	if c := ev.Call; c != nil {
		row.Tool = c.Tool
		row.TraceID = c.TraceID
		row.ArgsDigest = refDigest(c.Args)
		row.CallSeq = c.SeqNo // join key: same call's DECIDE and QUARANTINE share it
		row.ArgsLabel = argsLabel(c.Args)
		if row.ArgsLabel == "" {
			row.ArgsLabel = argsLabelFromMeta(c.Meta)
		}
		if row.ArgsLabel == "" {
			row.ArgsLabel = fallbackArgsLabel(c.Tool)
		}
	}
	if v := ev.Verdict; v != nil {
		row.Verdict = verdictName(v.Kind)
		row.Reason = abi.ReasonName(v.Reason)
		row.By = v.By
		row.Witness = witnessOf(v)
		row.DenyRule = denyRuleOf(v)
	}
	if r := ev.Result; r != nil {
		row.ResultDigest = refDigest(r.Payload)
		row.Taint = taintName(r.Payload.Taint)
	}
	// Populate capability fields from Event.Fields (the carrier for C6 events)
	// Fields carries: {cap_kind, cap_name, cap_digest, cap_from, cap_to, reason}
	if ev.Fields != nil {
		if ot, ok := ev.Fields["override_type"].(string); ok && ot != "" {
			row.OverrideType = ot
		}
		if row.Tool == "" {
			if tool, ok := ev.Fields["tool"].(string); ok && tool != "" {
				row.Tool = tool
			}
		}
		if row.TraceID == "" {
			if trace, ok := ev.Fields["trace_id"].(string); ok && trace != "" {
				row.TraceID = trace
			}
		}
		if or, ok := ev.Fields["override_reason"].(string); ok && or != "" && row.Witness == "" {
			row.Witness = or
		}
		if ck, ok := ev.Fields["cap_kind"].(string); ok {
			row.CapKind = ck
		}
		if cn, ok := ev.Fields["cap_name"].(string); ok {
			row.CapName = cn
		}
		if cd, ok := ev.Fields["cap_digest"].(string); ok {
			row.CapDigest = cd
		}
		if cf, ok := ev.Fields["cap_from"].(string); ok {
			row.CapFrom = cf
		}
		if ct, ok := ev.Fields["cap_to"].(string); ok {
			row.CapTo = ct
		}
		if rs, ok := ev.Fields["reason"].(string); ok && row.Reason == "" {
			row.Reason = rs // allow override for capability events
		}
	}
	return row, true
}

// refDigest is the audit identity of a Ref's bytes WITHOUT resolving them: the
// content hash if the backend stamped one, else a hash of the inline bytes, else
// empty. Never materializes a blob (no resolver dependency on the hot path).
// witnessOf extracts the bounded-disclosure claim a verdict surfaced — the
// offending self-modify glob / arg bound (WitnessPayload.Claim) or the
// require-witness gate's claim (Meta["claim"]). "" when the verdict disclosed
// nothing. This is the one forensic field the live wire carried but the durable
// row used to drop, leaving an audit unable to say WHICH glob/arg tripped a deny.
//
// The claim is bounded and scrubbed HERE (boundWitness) because the journal is
// the SOLE scrubber on this wire: internal/guardcorpus copies row.Witness
// verbatim into the exported dataset and is documented as never re-deriving
// redaction. Both sources are untrusted for this purpose — Meta is a string map
// any rung can write, and even the typed Payload.Claim is concatenated from
// call-derived bytes by several live rungs (see boundWitness).
//
// Unlike DenyRule (#5863) this field CANNOT be a closed-vocabulary set-membership
// test: its value is open-ended remedial prose, 156 distinct values across this
// host's corpus, and that prose is what keeps a refused agent alive. So it gets
// the weaker-but-lossless treatment — a generous bound plus a value-targeted
// scrub — rather than being dropped whole on a non-member.
func witnessOf(v *abi.Verdict) string {
	if v == nil {
		return ""
	}
	if wp, ok := v.Payload.(abi.WitnessPayload); ok && wp.Claim != "" {
		return boundWitness(wp.Claim)
	}
	return boundWitness(v.Meta["claim"])
}

// denyRuleOf extracts the refusing rung's closed-vocabulary rule id from the
// verdict, or "" when the rung stamped none. The journal is the SOLE scrubber on
// this wire (internal/guardcorpus copies these fields verbatim into the exported
// dataset), so it never trusts Meta — a string map any rung, and in principle a
// model-influenced one, can write. abi.DenyRuleID is a set-membership test over a
// compile-time literal vocabulary: a non-member is dropped whole rather than
// scrubbed down into the set, so this field cannot carry an arg value, a path, a
// host, or a secret no matter what is stamped. That property is the reason the
// offending TOKEN is deliberately NOT recorded alongside it: a token is user data
// by construction and has no closed vocabulary to be validated against.
func denyRuleOf(v *abi.Verdict) string {
	if v == nil || v.Meta == nil {
		return ""
	}
	id, ok := abi.DenyRuleID(v.Meta[abi.MetaDenyRule])
	if !ok {
		return ""
	}
	return id
}

func refDigest(r abi.Ref) string {
	if r.Digest != "" {
		return r.Digest
	}
	if r.Kind == abi.RefInline && len(r.Inline) > 0 {
		sum := sha256.Sum256(r.Inline)
		return "sha256:" + hex.EncodeToString(sum[:])
	}
	return ""
}

const (
	// MetaArgsLabel carries the same bounded, redacted label row.ArgsLabel records when
	// call args are blob-backed. It is internal telemetry only; the raw args stay out of
	// the journal.
	MetaArgsLabel   = "fak_args_label"
	maxArgsLabelLen = 96
)

// ArgsLabelForBytes returns a bounded, redacted shape label for JSON tool args. It is
// exported so gateways can preserve the label before large args are lowered to blob refs.
func ArgsLabelForBytes(b []byte) string {
	var obj map[string]any
	if err := json.Unmarshal(b, &obj); err != nil || len(obj) == 0 {
		return ""
	}
	return argsLabelForObject(obj)
}

func argsLabel(r abi.Ref) string {
	if r.Kind != abi.RefInline || len(r.Inline) == 0 {
		return ""
	}
	return ArgsLabelForBytes(r.Inline)
}

func argsLabelFromMeta(meta map[string]string) string {
	if meta == nil {
		return ""
	}
	return safeProvidedArgsLabel(meta[MetaArgsLabel])
}

func fallbackArgsLabel(tool string) string {
	if atom := safeAtom(tool); atom != "" {
		return "tool=" + atom
	}
	return ""
}

func argsLabelForObject(obj map[string]any) string {
	parts := []string{}
	if s := firstStringField(obj, "command", "cmd", "script"); s != "" {
		if stem := commandStem(s); stem != "" {
			parts = append(parts, "command="+stem)
		}
	}
	if s := firstStringField(obj, "path", "file_path", "filepath", "file", "workdir", "cwd"); s != "" {
		if stem := pathStem(s); stem != "" {
			parts = append(parts, "path="+stem)
		}
	}
	if s := firstStringField(obj, "action", "operation", "method"); s != "" {
		if atom := safeAtom(s); atom != "" {
			parts = append(parts, "action="+atom)
		}
	}
	if len(parts) == 0 {
		if keys := safeObjectKeys(obj); len(keys) > 0 {
			parts = append(parts, "keys="+strings.Join(keys, ","))
		}
	}
	return boundArgsLabel(strings.Join(parts, " "))
}

func safeProvidedArgsLabel(label string) string {
	label = strings.TrimSpace(label)
	if label == "" || secretish(label) {
		return ""
	}
	var b strings.Builder
	for _, r := range label {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
			b.WriteRune(r)
		case r == '_' || r == '-' || r == '.' || r == '/' || r == '=' || r == ',' || r == ' ':
			b.WriteRune(r)
		}
	}
	return boundArgsLabel(b.String())
}

func firstStringField(obj map[string]any, names ...string) string {
	for _, name := range names {
		v, ok := obj[name]
		if !ok {
			continue
		}
		if s, ok := v.(string); ok && strings.TrimSpace(s) != "" {
			return s
		}
	}
	return ""
}

// navStems names command stems that only position the shell — they select a
// directory or load an environment, they are not the operative command. Labeling
// one of them names the navigation prefix of a compound command and throws the
// command that actually ran away ("cd fak && go test ./..." -> "cd fak"), which
// collapsed unrelated failures onto one undiagnosable label (#5863). Their
// argument is a path, so the stem cannot simply be dropped in favor of the second
// token either — the whole segment has to be stepped over.
var navStems = map[string]bool{
	"cd":           true,
	"chdir":        true,
	"pushd":        true,
	"popd":         true,
	"set-location": true,
	"source":       true,
	".":            true,
}

// commandStem names the operative command of a shell command line as at most two
// scrubbed atoms. It walks the connector-separated segments and steps over leading
// navigation segments so the label names what ran, not how the shell got there. The
// label stays bounded and scrubbed exactly as before — this re-aims the same two
// atoms, it never emits more of them. If every segment is navigation there is no
// operative command to name, so the first segment's label is kept.
func commandStem(command string) string {
	first := ""
	for _, seg := range commandSegments(command) {
		label, stem := segmentLabel(seg)
		if label == "" {
			continue
		}
		if first == "" {
			first = label
		}
		if navStems[strings.ToLower(stem)] {
			continue
		}
		return label
	}
	return first
}

// commandSegments splits a command line on the connectors that separate one command
// from the next (";", "&&", "||", "|", "&", newline). Quoting is deliberately not
// parsed: the result only ever feeds a bounded, scrubbed label, never execution, and
// a mis-split can only produce a shorter atom, never a longer or less-redacted one.
func commandSegments(command string) []string {
	segs := []string{}
	var b strings.Builder
	flush := func() {
		if s := strings.TrimSpace(b.String()); s != "" {
			segs = append(segs, s)
		}
		b.Reset()
	}
	for i := 0; i < len(command); i++ {
		c := command[i]
		switch c {
		case ';', '|', '&', '\n', '\r':
			flush()
			if i+1 < len(command) && command[i+1] == c { // collapse "&&" / "||"
				i++
			}
		default:
			b.WriteByte(c)
		}
	}
	flush()
	return segs
}

// segmentLabel returns the bounded label for one segment plus the bare stem it was
// built from, so the caller can tell a navigation segment from an operative one.
func segmentLabel(seg string) (label, stem string) {
	fields := strings.Fields(seg)
	for i, raw := range fields {
		tok := cleanToken(raw)
		if skipStemToken(tok) {
			continue
		}
		s := pathStem(tok)
		if s == "" {
			continue
		}
		if next := commandSecond(fields[i+1:]); next != "" {
			return boundArgsLabel(s + " " + next), s
		}
		return s, s
	}
	return "", ""
}

// skipStemToken reports whether a token cannot be the operative command's stem. An
// environment assignment ("VAR=value", "env", "export") is skipped so the label
// names the command the environment was set up for — and so an assignment's value
// can never reach the label, which it could before (#5863).
func skipStemToken(tok string) bool {
	if tok == "" || tok == "&" {
		return true
	}
	if strings.HasPrefix(tok, "-") || strings.Contains(tok, "=") {
		return true
	}
	switch strings.ToLower(tok) {
	case "env", "export":
		return true
	}
	return strings.HasPrefix(strings.ToLower(tok), "$env:")
}

func commandSecond(fields []string) string {
	for _, raw := range fields {
		tok := cleanToken(raw)
		if tok == "" || tok == "&" || strings.HasPrefix(tok, "-") || strings.HasPrefix(strings.ToLower(tok), "$env:") {
			continue
		}
		if strings.Contains(tok, "=") {
			continue
		}
		if strings.ContainsAny(tok, `/\`) {
			return pathStem(tok)
		}
		return safeAtom(tok)
	}
	return ""
}

func pathStem(s string) string {
	s = strings.TrimSpace(strings.ReplaceAll(s, "\\", "/"))
	if s == "" || secretish(s) {
		return ""
	}
	parts := strings.Split(strings.Trim(s, "/"), "/")
	if len(parts) == 0 {
		return safeAtom(s)
	}
	last := cleanToken(parts[len(parts)-1])
	if last == "" || secretish(last) {
		return ""
	}
	return safeAtom(last)
}

func safeObjectKeys(obj map[string]any) []string {
	keys := make([]string, 0, len(obj))
	for k := range obj {
		k = safeAtom(k)
		if k == "" || secretish(k) {
			continue
		}
		keys = append(keys, k)
	}
	sort.Strings(keys)
	if len(keys) > 4 {
		keys = keys[:4]
	}
	return keys
}

func cleanToken(s string) string {
	return strings.Trim(strings.TrimSpace(s), `"'`)
}

func safeAtom(s string) string {
	s = cleanToken(s)
	if s == "" || secretish(s) {
		return ""
	}
	var b strings.Builder
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
			b.WriteRune(r)
		case r == '_' || r == '-' || r == '.' || r == '/':
			b.WriteRune(r)
		}
	}
	return boundArgsLabel(b.String())
}

func secretish(s string) bool {
	x := strings.ToLower(s)
	for _, needle := range []string{"authorization", "bearer", "api_key", "apikey", "secret", "token", "password", "sk-"} {
		if strings.Contains(x, needle) {
			return true
		}
	}
	return false
}

func boundArgsLabel(s string) string {
	s = strings.TrimSpace(s)
	if len(s) <= maxArgsLabelLen {
		return s
	}
	return s[:maxArgsLabelLen] + "..."
}

const (
	// maxWitnessLen bounds row.Witness. It is DELIBERATELY not maxArgsLabelLen (96).
	// Measured over this host's whole corpus (513 .dispatch-runs/guard-audit/*.jsonl,
	// 82k rows, ~630 with a witness, ~156 distinct values — the corpus is live, so
	// repeated scans saw 622-631 as peers appended and the reaper pruned): witness
	// length p50 53, p90 65, p99 447, max 447. Bounding at 96 truncates 57 rows (9.05%) and discards
	// 26.22% of all witness prose bytes — and the long values are precisely the
	// gate's own remedy text (the 447-byte OFF_TRUNK refusal names the sanctioned
	// `fak worktree worker prepare` route only in its second half). Truncating that
	// converts a recoverable refusal into a dead end, which costs more than the
	// disclosure it prevents. 512 sits above the observed maximum, so the measured
	// truncation cost today is ZERO while the field stops being structurally
	// unbounded.
	maxWitnessLen = 512
	boundEllipsis = "..."
	redactedValue = "[redacted]"
)

// boundWitness makes the Witness field satisfy the contract internal/guardcorpus
// states for it. That package copies row.Witness VERBATIM into the exported
// GUARD-SESSION dataset under a comment asserting the journal "already bounded
// and scrubbed" it — an assertion that was true of ArgsLabel and false of
// Witness, which reached the wire raw.
//
// The field is NOT rung-authored by construction, which is why a bound is needed
// rather than just a corrected comment. Live producers concatenate call-derived
// bytes into the claim: internal/adjudicator/egresslist.go builds
// "egress restricted, host not allowlisted: "+dests[0] from a host parsed out of
// the call args, decide.go does the same for the WebFetch host and scheme, and
// internal/adjudicator/lintwrites.go builds one from the target path plus a
// Go/JSON parser message that embeds the offending SOURCE TOKEN verbatim.
func boundWitness(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	s = redactSecretAssignments(s)
	if len(s) <= maxWitnessLen {
		return s
	}
	return s[:maxWitnessLen] + boundEllipsis
}

// redactSecretAssignments replaces the VALUE of any `key=value` / `key: value`
// atom whose KEY is secretish, and leaves every other byte untouched.
//
// It is deliberately NOT safeProvidedArgsLabel's whole-string secretish() drop.
// That rule keys on the SUBSTRING "secret" anywhere in the string, so reusing it
// here would blank the witness on exactly the SECRET_EXFIL refusals an operator
// most needs to read — measured: it destroys 2 distinct corpus values / 4 rows,
// including "ctxmmu SECRET_EXFIL secret_pattern quarantine_id=q1". Keying on the
// assignment KEY instead alters ZERO real witness rows — verified by replaying
// boundWitness over every row of the live corpus (the only assignment keys
// present are quarantine_id, core.hooksPath and if, none secretish) — while
// still redacting a value a rung concatenated behind a secret-shaped name.
func redactSecretAssignments(s string) string {
	if !secretish(s) {
		return s // fast path: no needle anywhere, so no assignment can be secretish
	}
	var b strings.Builder
	b.Grow(len(s) + len(redactedValue))
	i := 0
	for i < len(s) {
		if s[i] != '=' && s[i] != ':' {
			b.WriteByte(s[i])
			i++
			continue
		}
		b.WriteByte(s[i])
		sep := i
		i++
		// KEY: the identifier run immediately LEFT of the separator.
		k := sep
		for k > 0 && isKeyByte(s[k-1]) {
			k--
		}
		if k == sep || !secretish(s[k:sep]) {
			continue
		}
		// VALUE: the next whitespace-delimited atom, after optional separator spacing.
		v := i
		for v < len(s) && (s[v] == ' ' || s[v] == '\t') {
			v++
		}
		e := v
		for e < len(s) && !isWitnessSpace(s[e]) {
			e++
		}
		if e == v {
			continue // nothing assigned; leave the separator as it stands
		}
		b.WriteString(s[i:v]) // preserve the original spacing verbatim
		b.WriteString(redactedValue)
		i = e
	}
	return b.String()
}

func isKeyByte(c byte) bool {
	return c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' ||
		c == '_' || c == '-' || c == '.'
}

func isWitnessSpace(c byte) bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r'
}

func verdictName(k abi.VerdictKind) string {
	switch k {
	case abi.VerdictAllow:
		return "ALLOW"
	case abi.VerdictDeny:
		return "DENY"
	case abi.VerdictTransform:
		return "TRANSFORM"
	case abi.VerdictQuarantine:
		return "QUARANTINE"
	case abi.VerdictRequireWitness:
		return "WITNESS"
	case abi.VerdictDefer:
		return "DEFER"
	case abi.VerdictIndeterminate:
		return "INDETERMINATE"
	}
	return fmt.Sprintf("K%d", k)
}

func taintName(t abi.TaintLabel) string {
	switch t {
	case abi.TaintTrusted:
		return "trusted"
	case abi.TaintTainted:
		return "tainted"
	case abi.TaintQuarantined:
		return "quarantined"
	}
	return ""
}

// recoverHead scans an existing journal to recover the chain head (last seq + last
// hash) so an append continues the same chain. A missing file is the genesis case
// (seq 0, empty hash). It does NOT validate the chain (that is Verify's job) so a
// damaged log never blocks startup.
func recoverHead(path string) (seq uint64, lastHash string, err error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return 0, "", nil
		}
		return 0, "", fmt.Errorf("journal: stat %s: %w", path, err)
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var r Row
		if err := json.Unmarshal(line, &r); err != nil {
			// A torn final line (crash mid-write) is tolerated: stop at the last
			// well-formed row. Verify will catch genuine corruption.
			break
		}
		seq = r.Seq
		lastHash = r.Hash
	}
	if err := sc.Err(); err != nil {
		return 0, "", fmt.Errorf("journal: scan %s: %w", path, err)
	}
	return seq, lastHash, nil
}

// ReadRows reads all committed rows from a journal file, in order — the READ side
// of the durable log for a CONSUMER (a live guard-tail pane, an exporter) that wants
// the rows as data, not an integrity check (use Verify for that). It is deliberately
// robust for a live reader: a MISSING file is the empty journal (nil, nil) — tailing
// a not-yet-written journal is a valid "no rows yet" state, not an error — and a torn
// final line (a crash mid-append) is tolerated by stopping at the last well-formed
// row (mirroring recoverHead), so a reader never errors on a half-written tail.
// Genuine I/O errors (permission, a read fault) are returned. Verify, not ReadRows,
// is the surface that detects in-the-middle tampering.
//
// It reads the LIVE SEGMENT ONLY. Once a journal has been cut (rotation is armed in
// production at 64 MB), the live file is a TAIL and this returns a short slice that
// is indistinguishable from a whole small journal — which is how a roll-up ends up
// reporting a tail as a total (#6488). So: a consumer that produces a total, a rate,
// or a roll-up uses ReadAllSegments (fold it through WithoutCutAnchors for a count
// identical to the same journal unrotated); a consumer that genuinely wants only the
// recent rows uses ReadTail, which returns the same rows PLUS the TailOmission that
// says what it skipped. Plain ReadRows is for a caller that already knows it is
// looking at one specific segment file.
func ReadRows(path string) ([]Row, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("journal: read %s: %w", path, err)
	}
	defer f.Close()
	var out []Row
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var r Row
		if err := json.Unmarshal(line, &r); err != nil {
			break // torn final line: stop at the last well-formed row (Verify catches real corruption)
		}
		out = append(out, r)
	}
	if err := sc.Err(); err != nil {
		return out, fmt.Errorf("journal: scan %s: %w", path, err)
	}
	return out, nil
}

// Verify re-reads a journal file and validates the hash chain end to end. It
// returns the number of rows checked and a non-nil error naming the FIRST broken
// link (a recomputed hash mismatch, a prev-hash discontinuity, or a sequence
// gap). A journal that passes Verify has not been edited since it was written.
func Verify(path string) (n int, err error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, fmt.Errorf("journal: open %s: %w", path, err)
	}
	defer f.Close()
	return verifyReader(f)
}

func verifyReader(r io.Reader) (int, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	var (
		prev    string
		wantSeq uint64
		n       int
	)
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var row Row
		if err := json.Unmarshal(line, &row); err != nil {
			return n, fmt.Errorf("journal: row %d: malformed JSON: %w", n+1, err)
		}
		wantSeq++
		next, err := verifyStep(prev, wantSeq, row)
		if err != nil {
			return n, err
		}
		prev = next
		n++
	}
	if err := sc.Err(); err != nil {
		return n, fmt.Errorf("journal: scan: %w", err)
	}
	return n, nil
}

// VerifyRows validates the hash chain over an in-memory slice (e.g. Recent() of
// an in-memory journal). Same checks as Verify, returning the first broken link.
func VerifyRows(rows []Row) (int, error) {
	var (
		prev    string
		wantSeq uint64
	)
	for i, row := range rows {
		wantSeq++
		next, err := verifyStep(prev, wantSeq, row)
		if err != nil {
			return i, err
		}
		prev = next
	}
	return len(rows), nil
}

// verifyStep checks one row against the running chain head + expected sequence and
// returns the new chain head. It is the single source of truth for "is this row
// authentic and in order", shared by file and in-memory verification.
func verifyStep(prev string, wantSeq uint64, row Row) (string, error) {
	if row.Seq != wantSeq {
		return "", fmt.Errorf("journal: sequence gap: seq=%d want %d", row.Seq, wantSeq)
	}
	if row.PrevHash != prev {
		return "", fmt.Errorf("journal: broken chain at seq %d: prev_hash=%q want %q", row.Seq, row.PrevHash, prev)
	}
	if got := chainHash(row.PrevHash, row); got != row.Hash {
		return "", fmt.Errorf("journal: tampered row at seq %d: hash=%q recomputed %q", row.Seq, row.Hash, got)
	}
	return row.Hash, nil
}

// ---------------------------------------------------------------------------
// Registered instance — opt-in via FAK_AUDIT_JOURNAL.
// ---------------------------------------------------------------------------

var (
	activeMu sync.Mutex
	active   *Journal
)

// Active returns the registered durable journal, or nil if none was enabled
// (FAK_AUDIT_JOURNAL unset at boot and no programmatic Enable). The gateway uses
// this to serve /v1/fak/events or 404.
func Active() *Journal {
	activeMu.Lock()
	defer activeMu.Unlock()
	return active
}

// ResetActiveForTest clears the process-active journal pointer so a later Enable opens a
// FRESH journal instead of returning a prior test's idempotent one. It is TEST-ONLY plumbing:
// a test that programmatically Enables the global journal (e.g. the `fak guard --replay-trace`
// smoke test) must call this in cleanup so the global state does not leak into a sibling test
// that assumes a clean boot. It closes and nils the active journal; it does NOT unregister the
// emitter (the ABI has no unregister), so pair it with abi.ResetForTest when the ABI itself is
// being reset. A no-op when nothing is active.
func ResetActiveForTest() {
	activeMu.Lock()
	j := active
	active = nil
	activeMu.Unlock()
	if j != nil {
		_ = j.Close()
	}
}

// Enable turns the durable decision journal ON at path AFTER init has run — the
// programmatic equivalent of FAK_AUDIT_JOURNAL, for a front door (fak guard) that
// decides to default the audit trail on. It creates the parent directory, opens
// (creating, or CONTINUING an existing chain) a file-backed journal, registers it
// as ONE persisting emitter against the frozen ABI, and returns it.
//
// It is IDEMPOTENT and boot-env-respecting: if a journal is already active
// (FAK_AUDIT_JOURNAL won at boot, or a prior Enable ran) Enable is a no-op that
// returns the existing journal — the emitter is never double-registered (the ABI
// has no unregister) and the first/boot enablement always wins. A genuine open
// failure is returned (never silently swallowed) so the caller can decide whether
// to proceed without the trail.
func Enable(path string) (*Journal, error) {
	activeMu.Lock()
	defer activeMu.Unlock()
	if active != nil {
		return active, nil
	}
	if dir := filepath.Dir(path); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			return nil, fmt.Errorf("journal: create dir %s: %w", dir, err)
		}
	}
	j, err := Open(path)
	if err != nil {
		return nil, err
	}
	active = j
	abi.RegisterEmitter(j)
	return j, nil
}

func init() {
	path := os.Getenv("FAK_AUDIT_JOURNAL")
	if path == "" {
		return // off unless a front door (fak guard) programmatically Enables it
	}
	if _, err := Enable(path); err != nil {
		// Fail loud but do not brick the kernel: a missing audit sidecar must not
		// stop adjudication (the in-memory counters still hold). An auditor who
		// requires fail-closed wires that as a separate posture (issue #12).
		fmt.Fprintf(os.Stderr, "fak: audit journal disabled — %v\n", err)
		return
	}
	fmt.Fprintf(os.Stderr, "fak: audit journal -> %s (durable, hash-chained)\n", path)
}

// ReadRowsFrom decodes complete JSONL rows beginning at byte offset and returns
// the next safe offset. A partial trailing row is left for the next poll.
func ReadRowsFrom(path string, offset int64) ([]Row, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, offset, err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, offset, err
	}
	if offset < 0 || offset > info.Size() {
		offset = 0
	}
	if _, err := f.Seek(offset, io.SeekStart); err != nil {
		return nil, offset, err
	}
	r := bufio.NewReader(f)
	var rows []Row
	next := offset
	for {
		line, err := r.ReadBytes('\n')
		if len(line) > 0 && line[len(line)-1] == '\n' {
			var row Row
			if json.Unmarshal(bytes.TrimSpace(line), &row) == nil {
				rows = append(rows, row)
			}
			next += int64(len(line))
		}
		if err != nil {
			if err == io.EOF {
				return rows, next, nil
			}
			return rows, next, err
		}
	}
}
