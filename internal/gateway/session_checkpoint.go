package gateway

// session_checkpoint.go — POST /v1/fak/session/{trace_id}/checkpoint (#2425): the wire
// half of the two-axis checkpoint. It mints (and, on request, re-checks) the pair
// {ledger_head_hash, tree_witness} bound in ONE append-only sessionledger record, so
// "this conversation state corresponds to this workspace state" becomes a claim a peer
// can check instead of a courtesy it has to extend.
//
// WHY THE CALLER SUPPLIES THE TREE WITNESS. The gateway does not read the workspace: it
// takes the HEAD SHA and dirty set in the body and folds them with the ledger head. Two
// reasons, both load-bearing. (1) The gateway serves sessions whose workspace it may not
// share — a served session can be on another host entirely, where "the working tree" is
// not a thing this process can see. (2) Minting the witness here would put a `git`
// subprocess on the request path, which is exactly the per-decide subprocess boundary
// DIRECTION.md exists to remove. The client that OWNS the tree reads it (cmd/fak's
// `fak session checkpoint-witness`); the gateway binds and persists.
//
// That split does not weaken the claim: the checkpoint's value is not that the gateway
// vouched for the tree, it is that the pair is bound by a hash any peer can re-derive
// from the ledger and the repo. A session that lies about its tree writes a checkpoint
// that fails its own verify the moment anyone checks.

import (
	"errors"
	"net/http"

	"github.com/anthony-chaudhary/fak/internal/sessionledger"
)

// CheckpointRequest is the body of POST /v1/fak/session/{trace_id}/checkpoint: the git
// half of the witness, plus the bit that asks for a re-check instead of a mint.
type CheckpointRequest struct {
	// HeadSHA is the workspace's committed anchor (`git rev-parse HEAD`). Required.
	HeadSHA string `json:"head_sha"`
	// Dirty is the working set as the caller observed it: one entry per path git
	// reported dirty, each carrying the digest of its bytes. Empty = a clean tree.
	Dirty []sessionledger.DirtyEntry `json:"dirty,omitempty"`
	// Verify asks for a re-check of the trace's LATEST checkpoint against the supplied
	// witness instead of minting a new one. A peer checking someone else's claim sends
	// this; the session recording its own state does not.
	Verify bool `json:"verify,omitempty"`
}

// CheckpointResponse is the answer to both shapes: the receipt, plus the verdict when a
// verify was asked for. Axis names the half that moved on a failure — "tree" (the
// workspace drifted from the conversation that described it) or "transcript" (the ledger
// no longer matches the record) — so a caller branches on data, not on prose.
type CheckpointResponse struct {
	TraceID    string                       `json:"trace_id"`
	Checkpoint sessionledger.Checkpoint     `json:"checkpoint"`
	Verified   *bool                        `json:"verified,omitempty"`
	Axis       sessionledger.CheckpointAxis `json:"axis,omitempty"`
	Detail     string                       `json:"detail,omitempty"`
}

// handleSessionCheckpointVerb serves the checkpoint verb on the session control subtree.
// It reports whether it claimed the verb, so handleFakSession falls through to the
// generic drive-state control path for everything else — the same dispatch shape
// handleTeleportVerb uses, and for the same reason: this verb acts on the durable CHAIN
// rather than on the drive table, so it takes its own body and answers its own document.
func (s *Server) handleSessionCheckpointVerb(w http.ResponseWriter, r *http.Request, traceID, verb string) bool {
	if verb != "checkpoint" {
		return false
	}
	var req CheckpointRequest
	// An absent body is not decoded (the same optional-body posture the teleport verbs
	// take); it then fails the witness check below with the specific reason, rather than
	// a generic decode error.
	if r.ContentLength != 0 && !decodeRequestBody(w, r, &req) {
		return true
	}
	tree, err := sessionledger.NewTreeWitness(req.HeadSHA, req.Dirty)
	if err != nil {
		writeErrCode(w, http.StatusBadRequest, "checkpoint_no_tree_witness",
			"CHECKPOINT_NO_TREE_WITNESS: a checkpoint binds the ledger head to a git tree witness, "+
				"so head_sha is required: "+err.Error())
		return true
	}
	// The durable chain these control-plane verbs act on — the same ledger the teleport
	// verbs open, so a checkpoint and an export see one history, not two.
	l, err := teleportLedger()
	if err != nil {
		writeErrCode(w, http.StatusServiceUnavailable, "checkpoint_no_ledger",
			"CHECKPOINT_NO_LEDGER: the durable session ledger could not be opened: "+err.Error())
		return true
	}

	if req.Verify {
		cp, err := l.LatestCheckpoint(traceID)
		if err != nil {
			writeErrCode(w, http.StatusNotFound, "checkpoint_absent", "CHECKPOINT_ABSENT: "+err.Error())
			return true
		}
		resp := CheckpointResponse{TraceID: traceID, Checkpoint: cp}
		err = l.VerifyCheckpoint(cp, tree)
		ok := err == nil
		resp.Verified = &ok
		if !ok {
			var mm *sessionledger.CheckpointMismatch
			if errors.As(err, &mm) {
				resp.Axis, resp.Detail = mm.Axis, mm.Detail
			} else {
				resp.Detail = err.Error()
			}
			s.logf("gateway: session %s checkpoint %s FAILED on the %s axis: %s", traceID, cp.Hash, resp.Axis, resp.Detail)
			// 409, not 422: the request is well formed and the answer is authoritative —
			// the WORLD conflicts with the claim.
			writeJSON(w, http.StatusConflict, resp)
			return true
		}
		writeJSON(w, http.StatusOK, resp)
		return true
	}

	cp, err := l.Checkpoint(traceID, tree)
	if err != nil {
		writeErrCode(w, http.StatusServiceUnavailable, "checkpoint_not_recorded",
			"CHECKPOINT_NOT_RECORDED: "+err.Error())
		return true
	}
	s.logf("gateway: session %s checkpoint %s (ledger head %s + tree %s/%s)",
		traceID, cp.Hash, cp.LedgerHead, cp.Tree.HeadSHA, cp.Tree.DirtySHA256)
	writeJSON(w, http.StatusOK, CheckpointResponse{TraceID: traceID, Checkpoint: cp})
	return true
}
