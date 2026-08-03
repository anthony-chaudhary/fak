package main

// wip_sync.go — `fak wip sync`, the OPT-IN replication verb for refs/fak/wip/* (#5479,
// child C8 of #3871). It is the git-I/O shell over internal/wipref/sync.go, which owns
// the refspecs, the mirror grammar, and the replication verdict; the rationale for the
// push-then-fetch order, for the separate mirror namespace, and for why only part of
// leaseref.Sync's reasoning transfers lives in that file's doc comment.
//
// The verb is never run automatically. A checkpoint is a TREE-WIDE capture of a dirty
// working tree, so publishing it off-machine is a privacy and bandwidth decision an
// operator makes deliberately rather than inherits — the ticket is explicit that opt-in
// is the right default, and nothing here, in the guard hooks, or in the loop tick calls
// it on anyone's behalf.

import (
	"context"
	"flag"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/wipref"
)

func runWipSync(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("wip sync", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "wip")
	repo := fs.String("C", "", "run in this git repo (default: cwd)")
	remote := fs.String("remote", "origin", "the git remote (name or URL) to replicate to")
	pushOnly := fs.Bool("push-only", false, "publish this clone's checkpoints only — never download a peer host's captured trees")
	fetchOnly := fs.Bool("fetch-only", false, "import the remote's checkpoints into the read-only mirror only (skip the push)")
	asJSON := fs.Bool("json", false, "emit the sync result as JSON")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	if *pushOnly && *fetchOnly {
		fmt.Fprintln(stderr, "fak wip sync: --push-only and --fetch-only are mutually exclusive")
		return 2
	}

	res, err := wipSync(context.Background(), *repo, *remote, !*fetchOnly, !*pushOnly)
	if err != nil {
		fmt.Fprintf(stderr, "fak wip sync: %v\n", err)
		if *asJSON {
			// Emit the partial result for a ledger, but NEVER let a successful encode
			// launder a failed replication into exit 0 — a caller that reads exit codes
			// must see that its WIP did not make it off this machine.
			_ = encodeJSONOrFail(stdout, stderr, res, "fak wip sync")
		}
		return 1
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, res, "fak wip sync")
	}
	fmt.Fprintf(stdout, "synced with %s: pushed=%v fetched=%v (%d published, %d mirrored)\n",
		res.Remote, res.Pushed, res.Fetched, res.Published, res.Mirrored)
	fmt.Fprintln(stdout, wipReplicationSummary(res.Replicated, res.StaleRemote, res.LocalOnly))
	return 0
}

// wipSync converges this clone's checkpoint namespace with remote: publish the local
// refs, then refresh the read-only mirror the status column reads. See
// internal/wipref/sync.go for why the push runs first and why the fetch lands outside
// refs/fak/wip/*. Errors are INFRASTRUCTURE only (git not executable, a non-zero
// push/fetch exit — network, auth, a missing remote); there is no policy verdict here,
// because moving refs is transport, not admission.
func wipSync(ctx context.Context, repo, remote string, doPush, doFetch bool) (wipref.SyncResult, error) {
	res := wipref.SyncResult{Remote: remote}
	if !wipref.ValidRemote(remote) {
		return res, fmt.Errorf("invalid remote %q (must be one safe git argv token)", remote)
	}
	if !doPush && !doFetch {
		return res, fmt.Errorf("sync with neither push nor fetch does nothing — enable at least one direction")
	}

	local, err := wipListRecords(ctx, repo)
	if err != nil {
		return res, err
	}

	if doPush {
		res.PushRefspec = wipref.PushRefspec
		// An EMPTY namespace short-circuits. leaseref/sync.go states that "a wildcard
		// refspec matching ZERO refs is a successful no-op in git" — that holds for
		// fetch, but NOT for push: git 2.x answers a zero-match push refspec with
		// "No refs in common and none specified; doing nothing." and exit 1, so a
		// clone that has never checkpointed would see its sync fail for having nothing
		// to say. Skipping the subprocess is both correct and cheaper.
		if len(local) == 0 {
			res.Pushed = true
		} else {
			// A failed push STOPS the sync: nothing local has changed yet, so the
			// operator gets one unambiguous failure and a mirror that still describes
			// the last SUCCESSFUL publication, rather than an error plus a silently
			// reclassified status column.
			if err := wipSyncDirection(ctx, repo, "push", remote, wipref.PushRefspec,
				" — sync stopped before fetch; your checkpoints are still LOCAL_ONLY"); err != nil {
				return res, err
			}
			// The push itself is the evidence. A completed push of PushRefspec put every
			// local ref on the remote at its current object, so this clone's own listing
			// IS the post-push remote state for its sessions. Recording it is what lets
			// --push-only report REPLICATED without downloading a single peer byte; a
			// later fetch re-derives the mirror from the remote and corrects it if the
			// remote has since moved.
			if err := wipWriteMirror(ctx, repo, remote, local); err != nil {
				return res, err
			}
			res.Pushed = true
			res.Published = len(local)
		}
	}

	if doFetch {
		res.FetchRefspec = wipref.FetchRefspec(remote)
		if err := wipSyncDirection(ctx, repo, "fetch", remote, res.FetchRefspec, ""); err != nil {
			return res, err
		}
		res.Fetched = true
	}

	mirror, err := wipMirrorIndex(ctx, repo, remote)
	if err != nil {
		return res, err
	}
	rep := wipref.FoldWithMirror(local, mirror, time.Now().Unix())
	res.Mirrored = len(mirror)
	res.Replicated, res.StaleRemote, res.LocalOnly = rep.Replicated, rep.StaleRemote, rep.LocalOnly
	return res, nil
}

// wipSyncDirection runs one transport direction (git <verb> <remote> <refspec>) and
// normalizes its outcome: a non-executable git is an infrastructure error, a non-zero
// exit is a verb-tagged message carrying git's own stderr plus an optional
// direction-specific suffix. Mirrors leaseref's runSyncDirection.
func wipSyncDirection(ctx context.Context, repo, verb, remote, refspec, extra string) error {
	_, errStr, code, err := gitWip(ctx, repo, nil, verb, remote, refspec)
	if err != nil {
		return fmt.Errorf("git not executable: %w", err)
	}
	if code != 0 {
		return fmt.Errorf("git %s %s %s exited %d: %s%s", verb, remote, refspec, code, strings.TrimSpace(errStr), extra)
	}
	return nil
}

// wipWriteMirror records, in this clone's mirror for remote, the objects a COMPLETED
// push proved the remote now holds. It is a LAST-KNOWN claim rather than a live probe:
// the remote could later be reset or pruned, and the next `fak wip sync` fetch is what
// re-derives the mirror from reality. Writing it is ONE `git update-ref --stdin` batch
// however many checkpoints there are, holding the same O(1)-in-refs spawn discipline the
// listing path had to fight for (#5336) — the live namespace routinely carries thousands
// of refs. A session id that is not one safe ref segment is skipped rather than
// smuggled into a ref name.
func wipWriteMirror(ctx context.Context, repo, remote string, recs []wipref.RefRecord) error {
	var b strings.Builder
	n := 0
	for _, r := range recs {
		sess := wipref.SessionFromRef(r.Ref)
		if !wipref.ValidSession(sess) || r.Object == "" {
			continue
		}
		fmt.Fprintf(&b, "update %s %s\n", wipref.MirrorSessionRef(remote, sess), r.Object)
		n++
	}
	if n == 0 {
		return nil
	}
	_, errStr, code, err := gitWipStdin(ctx, repo, b.String(), "update-ref", "--stdin")
	if err != nil {
		return fmt.Errorf("git not executable: %w", err)
	}
	if code != 0 {
		return fmt.Errorf("record %d published checkpoint(s) in the %s mirror: update-ref --stdin exited %d: %s",
			n, remote, code, strings.TrimSpace(errStr))
	}
	return nil
}

// wipMirrorIndex reads the replication evidence for one remote: the session->object
// lookup behind the REPLICATED / STALE_REMOTE / LOCAL_ONLY column. It is a LOCAL ref
// read — `fak wip status` must never make a network round trip to answer "is my work
// off this machine", both because status runs at boundaries where the network may be
// exactly what failed, and because an agent reading it at a compaction boundary has no
// budget for a remote probe.
func wipMirrorIndex(ctx context.Context, repo, remote string) (map[string]string, error) {
	recs, err := wipListMirrorRecords(ctx, repo, remote)
	if err != nil {
		return nil, err
	}
	return wipref.MirrorIndex(remote, recs), nil
}

// wipListMirrorRecords lists this clone's mirror for one remote. Unlike the live
// listing it asks for the ref and object ONLY — a mirrored checkpoint's stamp is never
// read here, because the mirror answers one question ("does the remote hold this
// object") and decoding a peer's stamp would invite the live verbs to start treating
// mirrored refs as local checkpoints. Fields are NUL-separated in one for-each-ref, so
// the read stays O(1) in git spawns however many refs the mirror holds.
func wipListMirrorRecords(ctx context.Context, repo, remote string) ([]wipref.RefRecord, error) {
	pattern := strings.TrimSuffix(wipref.MirrorNamespace(remote), "/")
	out, errStr, code, err := gitWip(ctx, repo, nil,
		"for-each-ref", "--format=%(refname)%00%(objectname)%00", pattern)
	if err != nil {
		return nil, fmt.Errorf("git for-each-ref: %w", err)
	}
	if code != 0 {
		return nil, fmt.Errorf("git for-each-ref exited %d: %s", code, strings.TrimSpace(errStr))
	}
	fields := strings.Split(out, "\x00")
	var recs []wipref.RefRecord
	for i := 0; i+1 < len(fields); i += 2 {
		ref := strings.TrimSpace(fields[i])
		obj := strings.TrimSpace(fields[i+1])
		if ref == "" || obj == "" {
			continue
		}
		recs = append(recs, wipref.RefRecord{Ref: ref, Object: obj})
	}
	return recs, nil
}

// wipReplicationSummary is the one line that makes the distinction legible in plain
// output. It exists because "checkpointed" and "safe" used to be the same word: a
// LOCAL_ONLY checkpoint survives THIS SESSION dying, which is what autocheckpoint
// protects against, and does not survive THIS MACHINE going away, which is the
// correlated failure an agent at a compaction boundary cannot otherwise tell apart.
func wipReplicationSummary(replicated, stale, localOnly int) string {
	line := fmt.Sprintf("replication: %d REPLICATED, %d STALE_REMOTE, %d LOCAL_ONLY", replicated, stale, localOnly)
	switch {
	case stale > 0:
		return line + "\n  STALE_REMOTE: an OLDER checkpoint is on the remote, this delta is not — `fak wip sync` to publish it."
	case localOnly > 0:
		return line + "\n  LOCAL_ONLY survives this session dying, not this machine — `fak wip sync` to replicate."
	default:
		return line
	}
}
