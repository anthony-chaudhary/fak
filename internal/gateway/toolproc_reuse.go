package gateway

// toolproc_reuse.go — the GATEWAY SEAM for internal/toolproc's live reuse cache
// (#5119; the caller repeatarm.go labels as "the labeled next step" and the one
// toolprocgate.ReuseArm was built for but never had).
//
// WHAT WAS ALREADY THERE, AND WHAT THIS ADDS. #4764 landed the decision spine
// (ClassifyRepeats/Normalize), #5122 the armed byte cache (toolproc.ArmedCache),
// and #5407 the verdict vocabulary at the consumer (toolprocgate.ReuseArm.Serve /
// Offer, REUSE_* codes 1090-1095). Every one of those was library-only: nothing
// on the live wire ever called Serve. This file is the caller — the two halves of
// one loop, at the two seams the gateway already owns:
//
//	SERVE   adjudicateProposedServed, per proposed tool call, before dispatch.
//	DEPOSIT admitInboundResults, per ADMITTED inbound tool_result.
//
// WHY THE FILE NAME MATTERS. This seam first appeared as `reuse_arm.go`, and that
// name is why it never ran: Go reads a trailing `_arm` as the GOARCH build
// constraint, so the file was silently dropped from every amd64 build (`go list`
// reported it under IgnoredGoFiles) and `go build ./internal/gateway` stayed green
// while referencing Server fields that did not exist. A seam nothing compiles is
// indistinguishable from a seam nothing calls; the name is load-bearing.
//
// WHY IT IS NOT THE vDSO PATH. The vDSO served-inline probe next door gates on
// the tool NAME (readOnlyPrefix: get_/read_/search_/list_/lookup_/find_/calc).
// served_inline_guardtrace_test.go witnesses the consequence: on native Claude
// Code names (Read/Bash/Grep/Glob) that gate matches NOTHING, so the dominant
// path serves 0 for a structural reason. toolproc keys on the command CONTENT
// instead — `read:<path>@<digest>` / `query:<canon>` — so it reaches exactly the
// family the name gate cannot, and the two probes are complementary rather than
// redundant. A reuse miss is byte-identical to today: the call falls through to
// normal adjudication, which is why arming this is safe.
//
// THE THREE SAFETY PROPERTIES, all enforced here at the boundary (the leaf's own
// admission rules in repeatreuse.go/repeatarm.go are the backstop, not the only
// gate):
//
//   - IMMUTABLE READ serves only under a WITNESSED content digest. This is
//     stricter than the leaf: toolproc.Normalize falls back to a path-only
//     identity when no digest was observed, which is right for OFFLINE analytics
//     of a finished rollout but wrong on a live wire, where a file mutated
//     between two reads would then be served stale. reuseRecordFor refuses a read
//     whose digest cannot be witnessed right now, so a serve is always keyed on
//     (resolved path + current content digest) and a mutation forms a new key.
//   - MUTABLE QUERY coalesces only inside its freshness window, only when the
//     operator opted in (ArmedConfig.CoalesceQueries), and every hit carries its
//     stale-age and source on the served line.
//   - WRITE and UNKNOWN are never served and never retained. Two independent
//     refusals: the command class (fail-closed for anything outside toolproc's
//     closed registry) and vdso.IsWriteShaped on the tool NAME, so a write-shaped
//     tool cannot deposit bytes even if its argument line reads like a `cat`.
//
// WHY IT IS OPT-IN (SetToolprocReuse, default unarmed ⇒ inert). Serving a read
// locally is only sound where the gateway can witness the SAME filesystem the
// tool executor reads — fak's own guard/native path, not an arbitrary remote
// client whose files this process cannot stat. The operator arms it exactly
// where that holds; unarmed, every function here is one map probe and a nil check.
//
// WHY THE STATE IS OUT OF LINE (reuseSeams, not Server fields). The armed state is
// a wiring-time opt-in that only two call sites read, and keeping it in this file
// keeps the whole seam — state, arming, refusals, and both halves of the loop — a
// single-file add and a single-file revert. The Server struct is a shared-trunk
// hot spot with concurrent editors; a seam that can be armed and disarmed without
// touching it is one fewer merge collision for a knob that is off by default.

import (
	"context"
	"encoding/json"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/toolproc"
	"github.com/anthony-chaudhary/fak/internal/toolprocgate"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// The served-line metadata a toolproc reuse hit adds on top of the two keys
// toolprocgate.ReuseArm.Serve already stamps (reuse_reason / reuse_source). The
// identity IS the digest key the acceptance gate asks a receipt to show.
const (
	// ReuseIdentityMetaKey carries the reuse key the hit was served under —
	// "read:<path>@sha256:<hex>" for an immutable read, "query:<canonical>" for a
	// coalesced status poll.
	ReuseIdentityMetaKey = "reuse_identity"
	// ReuseSavedBytesMetaKey carries the tool-result bytes this hit did NOT
	// re-fetch. Measured from the bytes actually served, not from the decision
	// store's tally (the store learns a size at Admit time, which on this seam
	// runs BEFORE the fetch it is sizing).
	ReuseSavedBytesMetaKey = "reuse_saved_bytes"
	// ReuseServedByVerdict is the WireVerdict.By a locally answered repeat cites,
	// distinguishing it from the "vdso" served-inline hit on the same seam.
	ReuseServedByVerdict = "toolproc"
)

// reuseShellArgKeys are the argument names whose value is a COMMAND LINE. Their
// value is handed to toolproc verbatim, which is what lets `cat X`, `git status`
// and `dispatch_status.py` classify by content under any tool name (Bash,
// shell_command, run_terminal_cmd, ...).
var reuseShellArgKeys = []string{"command", "cmd", "script", "shell_command"}

// reusePathArgKeys are the argument names whose value is a FILE PATH. toolproc's
// readerTools fold ("read", "read_file", "get-content") turns the bare path into
// an immutable-read identity; under any other tool name the path tokenizes to an
// unregistered program and falls through to CmdUnknown — fail-closed, which is
// why Write/Edit (also carrying file_path) can never be served or retained.
var reusePathArgKeys = []string{"file_path", "path", "filename", "file", "target_file"}

// reuseSeam is one Server's armed reuse state. Held out of line (see the file
// header) in reuseSeams, keyed by Server, so arming this knob never edits the
// Server struct. The *ReuseArm serializes its own admission; this struct is
// written once at wiring time and only read on the request path.
type reuseSeam struct {
	arm      *toolprocgate.ReuseArm
	digest   toolproc.DigestFn
	coalesce bool
}

// reuseSeams maps an armed *Server to its seam. A sync.Map because the key set is
// write-once-per-Server and read on every proposed call — the load-heavy shape it
// is built for. An unarmed Server has no entry, so the request path pays one
// lock-free probe.
var reuseSeams sync.Map // *Server -> *reuseSeam

// SetToolprocReuse arms the toolproc reuse seam (#5119), or re-arms it with fresh
// config (which drops the previously retained bytes with the old cache). It is a
// wiring-time seam — the same inject-after-New posture as SetToolMaxAge and the
// other host-set knobs: set it before Serve accepts turns.
//
// digest witnesses a read target's CURRENT content, normally
// toolproc.FileDigest(repoRoot). A nil digest leaves immutable-read serving
// permanently off (no witness ⇒ no key ⇒ no serve) and arms only the
// freshness-window coalescing, if cfg opted into it.
func (s *Server) SetToolprocReuse(cfg toolproc.ArmedConfig, digest toolproc.DigestFn) {
	if s == nil {
		return
	}
	reuseSeams.Store(s, &reuseSeam{
		arm:      toolprocgate.NewReuseArm(cfg),
		digest:   digest,
		coalesce: cfg.CoalesceQueries,
	})
}

// DisarmToolprocReuse removes the reuse seam, restoring the byte-identical
// pre-#5119 path and dropping every retained body. Wiring-time, like
// SetToolprocReuse; also the cleanup a test registers so one Server's armed bytes
// can never outlive it.
func (s *Server) DisarmToolprocReuse() {
	if s == nil {
		return
	}
	reuseSeams.Delete(s)
}

// reuseSeamOf returns this Server's armed seam, or nil when the operator never
// armed it (the default). Every function on this path treats nil as "not armed",
// which is the same fall-through a cache miss takes.
func (s *Server) reuseSeamOf() *reuseSeam {
	if s == nil {
		return nil
	}
	v, ok := reuseSeams.Load(s)
	if !ok {
		return nil
	}
	sm, _ := v.(*reuseSeam)
	return sm
}

// reuseRawFor extracts the INVOCATION LINE toolproc classifies from a proposed
// call's JSON arguments: a command line if the call carries one, else a file path.
// ok is false when the args are not a JSON object, carry neither shape, or carry an
// empty value — all of which mean "nothing to classify", never "safe to serve".
//
// A command line is handed over VERBATIM (`cat X`, `git status`) — toolproc reads
// its first token as the program. A file path is rendered as `<tool> "<path>"`
// instead of bare, and that shape is load-bearing: toolproc.matchSpec consumes the
// FIRST token as the program name and takes the read path from what FOLLOWS, so a
// bare `skill.md` classifies as an immutable read whose Path is EMPTY (the one
// token was eaten as the program). An empty Path has no digest to witness, so this
// seam would refuse it — a host `Read` tool that silently never serves, which is
// precisely the family the issue exists to serve. The quotes keep a path with
// spaces one token (toolproc's tokenizer strips them); a path that carries its own
// quote is refused rather than re-quoted, since a mis-split path would key on the
// wrong file.
func reuseRawFor(tool, args string) (string, bool) {
	var m map[string]json.RawMessage
	if json.Unmarshal([]byte(strings.TrimSpace(args)), &m) != nil {
		return "", false
	}
	str := func(keys []string) (string, bool) {
		for _, k := range keys {
			raw, ok := m[k]
			if !ok {
				continue
			}
			var v string
			if json.Unmarshal(raw, &v) != nil {
				continue
			}
			if v = strings.TrimSpace(v); v != "" {
				return v, true
			}
		}
		return "", false
	}
	if v, ok := str(reuseShellArgKeys); ok {
		return v, true
	}
	if v, ok := str(reusePathArgKeys); ok {
		if strings.ContainsAny(v, "\"'") {
			return "", false
		}
		return tool + ` "` + v + `"`, true
	}
	return "", false
}

// reuseRecordFor builds the CallRecord for one live tool call, applying the
// boundary refusals this seam adds on top of the leaf's own. ok is false whenever
// the call must not participate in reuse at all — in serve AND in deposit, so the
// two halves can never disagree about what is eligible.
//
// outputBytes is the size of an observed result (deposit side) or 0 when the
// result does not exist yet (serve side).
func (sm *reuseSeam) reuseRecordFor(tool, args string, atMS, outputBytes int64) (toolproc.CallRecord, toolproc.NormalCall, bool) {
	var zeroR toolproc.CallRecord
	var zeroN toolproc.NormalCall
	if sm == nil || sm.arm == nil {
		return zeroR, zeroN, false
	}
	// Un-bypassable tool-NAME backstop, mirroring fillVDSOFromResult: a
	// write-shaped tool never serves and never deposits, whatever its argument
	// line happens to tokenize to.
	if vdso.IsWriteShaped(tool) {
		return zeroR, zeroN, false
	}
	raw, ok := reuseRawFor(tool, args)
	if !ok {
		return zeroR, zeroN, false
	}
	rec := toolproc.CallRecord{Tool: tool, Raw: raw, AtMS: atMS, OutputBytes: outputBytes}
	nc := toolproc.Normalize(rec, toolproc.RepeatConfig{})
	switch nc.Class {
	case toolproc.CmdImmutableRead:
		// A live serve MUST be content-keyed. Witness the digest now; no witness
		// (nil DigestFn, unreadable or absent file) ⇒ refuse, rather than fall back
		// to the leaf's path-only identity, which on a live wire would serve the
		// bytes of a file that has since changed.
		if nc.Path == "" || sm.digest == nil {
			return zeroR, zeroN, false
		}
		d := sm.digest(nc.Path)
		if d == "" {
			return zeroR, zeroN, false
		}
		rec.Digest = d
		nc = toolproc.Normalize(rec, toolproc.RepeatConfig{})
	case toolproc.CmdMutableQuery:
		// Coalescing a mutable status poll is separately opt-in; without it the
		// decision layer would still emit advisory receipts, but this seam must not
		// answer the call, so refuse before Admit rather than after.
		if !sm.coalesce {
			return zeroR, zeroN, false
		}
	default: // idempotent write, unknown — fail-closed, never served, never retained
		return zeroR, zeroN, false
	}
	return rec, nc, true
}

// reuseServe probes the armed reuse cache for one proposed tool call. On a hit it
// returns the bytes to fold into the turn plus the served-line metadata; on any
// miss (unarmed, ineligible, no bytes on hand, screened, over the operator's
// max-age ceiling) it returns ok=false and the caller proceeds exactly as it does
// today.
func (s *Server) reuseServe(ctx context.Context, tool, args string) ([]byte, map[string]string, bool) {
	sm := s.reuseSeamOf()
	rec, nc, ok := sm.reuseRecordFor(tool, args, time.Now().UnixMilli(), 0)
	if !ok {
		return nil, nil, false
	}
	v := sm.arm.Serve(ctx, nil, rec)
	if !v.Served() {
		// Either an ordinary miss (first fetch, digest changed, window expired) or
		// the fail-closed unnamed-verdict branch. Both mean: run the real call.
		return nil, nil, false
	}
	meta := v.Result.Meta
	if meta == nil {
		meta = map[string]string{}
	}
	meta[ReuseIdentityMetaKey] = nc.Identity
	// A freshness-window hit is genuinely stale by StaleAgeMS; a digest-keyed hit
	// is keyed on the content as it is RIGHT NOW, so it carries no age at all and
	// the served line renders without an age clause (and no max-age ceiling can
	// apply to it, which is correct — there is nothing to be stale about).
	if v.Receipt.StaleAgeMS > 0 {
		meta["age_ms"] = strconv.FormatInt(v.Receipt.StaleAgeMS, 10)
	}
	body := resolveBytes(ctx, v.Result.Payload)
	if len(body) == 0 {
		return nil, nil, false
	}
	meta[ReuseSavedBytesMetaKey] = strconv.Itoa(len(body))
	// Same operator TTL ceiling the vDSO hit respects (#1349), and the same
	// poisoned-bytes screen: a served repeat is held to every guard a served
	// vDSO answer is, because it enters the transcript through the same door.
	if s.servedHitOverMaxAge(tool, meta) {
		return nil, nil, false
	}
	if _, held := ctxmmu.ScreenBytes(body); held {
		return nil, nil, false
	}
	return body, meta, true
}

// reuseOffer deposits one ADMITTED inbound tool_result under the identity the
// decision blesses, so a LATER repeat of the same read (or status poll) can be
// answered locally. Eligibility is the SAME reuseRecordFor gate the serve side
// uses, so nothing can be retained that would not have been servable; the armed
// cache's own budget and fail-closed admission are the backstop.
func (s *Server) reuseOffer(tool, args, result string) {
	sm := s.reuseSeamOf()
	rec, _, ok := sm.reuseRecordFor(tool, args, time.Now().UnixMilli(), int64(len(result)))
	if !ok {
		return
	}
	sm.arm.Offer(rec, []byte(result))
}
