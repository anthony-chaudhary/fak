package main

import (
	"io"
	"os"
	"strings"
)

func dispatchOfflineSessionVerb(stdout, stderr io.Writer, verb string, args []string) (int, bool) {
	if verb == "new" {
		return runSessionNew(stdout, stderr, args), true
	}
	if verb == "open" {
		return runSessionOpen(stdout, stderr, args), true
	}
	if verb == "move" {
		return runSessionMove(stdout, stderr, args), true
	}
	if verb == "discovery-serve" {
		return runSessionDiscoveryServe(stdout, stderr, args), true
	}
	if verb == "journal-audit" {
		return runSessionJournalAudit(stdout, stderr, args), true
	}

	// reset-diff (#1575) is the one offline verb in this surface: a pure JSON-in,
	// diff-out render over internal/sessionreset.DiffReset that never dials a live
	// gateway, so it is dispatched here before the gateway-shaped arity/flag table
	// below (which assumes every verb talks to a sessionClient).
	if verb == "log" {
		return runSessionLog(stdout, stderr, args), true
	}
	if verb == "reset-diff" {
		return runSessionResetDiff(os.Stdin, stdout, stderr, args), true
	}
	// branch (#1200) is an offline fork of a checkpoint into a new durable id — a pure
	// image-in, image-out move over internal/sessionimage.BranchDir that never dials a live
	// gateway, so it is dispatched here alongside the other offline verbs.
	if verb == "branch" {
		return runSessionBranch(stdout, stderr, args), true
	}
	// checkpoint (#2760) takes an on-demand durable snapshot of a session — an offline
	// image-in, image-out capture over internal/sessionimage.SnapshotDir (preserving the id,
	// source read-only) that never dials a live gateway, dispatched alongside branch.
	if verb == "checkpoint" {
		return runSessionCheckpoint(stdout, stderr, args), true
	}
	// checkpoint-witness (#2425) mints the OTHER checkpoint: not an image copy but the
	// two-axis hash pair {ledger_head_hash, tree_witness}, bound in one append-only
	// sessionledger record so a peer can check "this conversation state corresponds to
	// this workspace state" without trusting the session that claimed it. Offline; the
	// only thing it dials is git.
	if verb == "checkpoint-witness" {
		return runSessionCheckpointWitness(stdout, stderr, args), true
	}
	// fork (#2761) snapshot-and-branches a session into a divergent continuation — an offline
	// image-in, image-out move over internal/sessionimage.ForkDir that pins the branch point
	// (checkpoint, #2760) then diverges from it (branch, #1200), source read-only, never
	// dialing a live gateway. Dispatched alongside its sibling lifecycle verbs.
	if verb == "fork" {
		// Two forks share this verb, told apart by shape (see session_teleport.go):
		// #2761's image fork requires --out/--checkpoint, while #2419's ledger fork
		// takes a bare <trace> and mints a new trace over the shared prefix. The bare
		// form used to be a usage error, so nothing that parsed before now re-routes.
		if teleportIsLedgerFork(args) {
			return runTeleportFork(stdout, stderr, args), true
		}
		return runSessionFork(stdout, stderr, args), true
	}
	// export/import (#2419) move a session between hosts as a verified hash closure
	// over the durable ledger — offline, like their sibling lifecycle verbs.
	if verb == "export" {
		return runTeleportExport(stdout, stderr, args), true
	}
	if verb == "import" {
		return runTeleportImport(os.Stdin, stdout, stderr, args), true
	}
	if verb == "audit" {
		return runSessionAuditAlias(stdout, stderr, args), true
	}
	if verb == "recover" {
		return runSessionRecover(stdout, stderr, args), true
	}
	// compact-audit (#4763) mines native Codex rollout JSONL for compaction health —
	// offline, streaming, no gateway — so it dispatches with the other offline verbs.
	if verb == "compact-audit" {
		return runSessionCompactAudit(stdout, stderr, args), true
	}
	// observe is the zero-configuration user view over the same deterministic rollout
	// telemetry: active Codex profile, current workspace, four calendar days.
	if verb == "observe" {
		return runSessionObserve(stdout, stderr, args), true
	}
	// gate-fatigue (#4427) folds the guard-stop ledger into a per-gate
	// approval-without-inspection rate — offline, read-only, no gateway — so it
	// dispatches with the other offline verbs.
	if verb == "gate-fatigue" {
		return runFatigue(stdout, stderr, args), true
	}
	// subscribe (#2767) is the re-attach drain of one session's event stream; it
	// carries its own cursor flag surface, so it dispatches ahead of the shared
	// gateway-verb flag table (session_subscribe_cmd.go).
	if verb == "subscribe" {
		return runSessionSubscribe(stdout, stderr, args), true
	}
	if verb == "envelope" && (len(args) == 0 || strings.HasPrefix(args[0], "-")) {
		return runSessionEnvelope(stdout, stderr, args), true
	}

	return 0, false
}
