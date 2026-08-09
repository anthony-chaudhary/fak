package main

// wip_adopt.go — `fak wip reconcile adopt|resume|receipt`: the GIT-WITNESSED half of
// checkpoint adoption (#5998). internal/wiprecon/adopt.go owns the ownership rule; this
// file supplies the four facts that rule needs from the repo (the checkpoint object, its
// reconciliation verdict, the incumbent's liveness, the clock), performs the
// compare-and-swap that makes "exactly one successor wins" a property of git, and
// materializes the adopted delta somewhere a shared tree cannot be hurt by it.
//
// THE RECEIPT LIVES IN GIT, under refs/fak/checkpointadopt/<session>, pointing at a BLOB
// whose body carries the marker-line JSON. Three reasons, in order of load-bearing-ness:
//
//   - CAS. `git update-ref <ref> <new> <old>` is an atomic compare-and-swap against a
//     ref's current value, and it is the only such primitive every fak host already
//     shares. Two successors that both read "no receipt" both attempt a swap from the
//     zero OID; git lets exactly one through and the loser gets ADOPT_LOST_RACE having
//     mutated nothing. The same guard the checkpoint anchor uses (wipAnchorCAS).
//   - DURABILITY. The claim outlives the successor process, which is the entire point:
//     the successor is exactly as mortal as the session it is rescuing.
//   - A NAMESPACE THAT CANNOT COLLIDE. refs/fak/checkpointadopt/* deliberately does NOT
//     nest under refs/fak/wip/*, for the reason wipref/sync.go spells out at length: every
//     wip verb's sweep is written against the "refs/fak/wip" prefix AS A STRING, and one
//     careless HasPrefix would feed receipts to `wip reap`, which is a deleter. It follows
//     refs/fak/checkpointsync/* (the mirror stamp) instead.
//
// ORDER OF OPERATIONS, and why it is this order. The receipt is journaled BEFORE the first
// byte is materialized. That costs one extra ref write per adoption and buys the property
// the ticket actually asks for: a successor that dies mid-materialization leaves a receipt
// that says "ADOPTED, target T" rather than an unattributable half-written directory. The
// resume path then re-materializes into that same T and verifies bytes, so a torn write is
// repaired rather than inherited. The reverse order — materialize, then claim — has a
// window in which crashed work belongs to nobody.
//
// WHERE THE BYTES GO, and why never into the shared tree by default. This tree runs
// concurrent agents with dirty working copies. Writing a stranger's recovered delta into
// it would land on peers' live edits, which is the exact reason `wip reconcile` was
// advisory in the first place. So adoption materializes into ONE of:
//
//   - an EXPLICIT PATCH TARGET (the default): <git-dir>/fak-adopt/<session>.patch, a file
//     no build reads, from which an operator applies deliberately; or
//   - an ISOLATED WORKER PATH (--into <dir>), refused if it resolves inside this repo's
//     working tree, into which the checkpoint's post-image blobs are written and then
//     READ BACK AND HASHED against the bytes git holds for that object.
//
// Landing (--land) is opt-in and goes through the existing wipLandWith, which already
// refuses TREE_DIVERGED / TREE_WIDE_SNAPSHOT rather than clobbering. NOTHING here ever
// deletes refs/fak/wip/<session>: the checkpoint is preserved until `fak wip reap` sees
// its delta in HEAD, so a failed landing is still a recoverable checkpoint.

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/wiprecon"
	"github.com/anthony-chaudhary/fak/internal/wipref"
)

// wipAdoptNamespace is where adoption receipts live. See the file header for why it is a
// sibling of refs/fak/wip/* rather than a child.
const wipAdoptNamespace = "refs/fak/checkpointadopt/"

// Refusal tokens beyond the closed wiprecon.AdoptVerdict vocabulary. Each names a fact the
// repo witnessed, so a caller branches on the token rather than on prose.
const (
	// wipReasonAdoptNoCheckpoint: the session has no checkpoint ref to adopt.
	wipReasonAdoptNoCheckpoint = "NO_CHECKPOINT"
	// wipReasonAdoptLostRace: the compare-and-swap failed — a concurrent successor moved
	// the receipt between this one's read and its write. Nothing was materialized.
	wipReasonAdoptLostRace = "ADOPT_LOST_RACE"
	// wipReasonAdoptDeltaMismatch: the checkpoint's patch bytes no longer hash to the
	// digest the receipt bound. Refuse rather than write bytes the claim does not cover.
	wipReasonAdoptDeltaMismatch = "DELTA_DIGEST_MISMATCH"
	// wipReasonAdoptTargetInTree: --into resolves inside this repo's working tree, where
	// materializing a stranger's delta would land on concurrent peers' edits.
	wipReasonAdoptTargetInTree = "TARGET_INSIDE_TREE"
	// wipReasonAdoptUnsafePath: the delta names a path that escapes the target directory.
	wipReasonAdoptUnsafePath = "UNSAFE_DELTA_PATH"
	// wipReasonAdoptEmptyDelta: the checkpoint's delta is empty; there is nothing to adopt.
	wipReasonAdoptEmptyDelta = "EMPTY_DELTA"
	// wipReasonAdoptNoReceipt: `receipt` was asked for a claim that does not exist.
	wipReasonAdoptNoReceipt = "NO_RECEIPT"
)

// wipAdoptResult is the JSON/plain outcome of one adoption attempt. Verdict is always set
// — a granted verdict from wiprecon or one of the tokens above — so a scripted caller
// never has to distinguish "refused" from "failed" by reading prose.
type wipAdoptResult struct {
	Session       string            `json:"session"`
	Successor     string            `json:"successor,omitempty"`
	Verdict       string            `json:"verdict"`
	Reason        string            `json:"reason,omitempty"`
	CheckpointRef string            `json:"checkpoint_ref,omitempty"`
	CheckpointSHA string            `json:"checkpoint_sha,omitempty"`
	Receipt       *wiprecon.Receipt `json:"receipt,omitempty"`
	Target        string            `json:"target,omitempty"`
	TargetKind    string            `json:"target_kind,omitempty"` // patch | worker
	Materialized  []string          `json:"materialized,omitempty"`
	Deleted       []string          `json:"deleted,omitempty"` // in the delta as deletions; not written
	Verified      int               `json:"verified"`          // files whose written bytes were re-hashed and matched
	Landed        bool              `json:"landed"`
	LandedSHA     string            `json:"landed_sha,omitempty"`
	Land          *wipLandResult    `json:"land,omitempty"`
	// Preserved is the standing guarantee, reported rather than assumed: the checkpoint
	// ref still exists. Adoption never drops a checkpoint, on any path, including refusal.
	Preserved bool `json:"checkpoint_preserved"`
}

// wipAdoptOptions is one adoption bid. Now is injected so the TTL/takeover rule is
// testable without sleeping through a real one.
type wipAdoptOptions struct {
	Session   string
	Successor string
	Host      string
	Into      string // isolated worker path; mutually exclusive with PatchOut
	PatchOut  string // explicit patch target
	TTL       int64
	Land      bool
	Now       time.Time

	// raceHook runs between the receipt READ and the compare-and-swap. Production never
	// sets it; a test injects a concurrent successor's write there to witness the losing
	// side refusing with ADOPT_LOST_RACE, which is the only deterministic way to exercise
	// a race whose whole point is that it is decided by git rather than by timing.
	raceHook func()
}

// runWipReclaim is the CLI shell for the adoption sub-verbs. It is reached as
// `fak wip reconcile adopt|resume|receipt <session>` (see runWipReconcile).
//
// `adopt` and `resume` are the SAME operation and deliberately so: an adoption is
// idempotent under the receipt, so the verb a caller types only has to match what it
// believes is true, and being wrong about that is not an error. The queue prints whichever
// one matches the row's receipt so the printed command reads honestly.
func runWipReclaim(stdout, stderr io.Writer, verb string, argv []string) int {
	fs := flag.NewFlagSet("wip reconcile "+verb, flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "wip")
	repo := fs.String("C", "", "run in this git repo (default: cwd)")
	session := fs.String("session", "", "the CRASHED session whose checkpoint is being adopted")
	successor := fs.String("successor", "", "the adopting session id (default: $CLAUDE_CODE_SESSION_ID, else $FAK_SESSION_ID)")
	into := fs.String("into", "", "materialize the delta into this ISOLATED worker directory (refused if it is inside this repo's working tree)")
	patchOut := fs.String("patch-out", "", "write the delta to this patch file instead of the default <git-dir>/fak-adopt/<session>.patch")
	ttl := fs.Int64("ttl", wiprecon.DefaultTTLSeconds, "seconds this claim is honoured without progress before a liveness-checked takeover may replace it")
	land := fs.Bool("land", false, "after materializing, land the checkpoint through `fak wip land`'s existing scope and divergence refusals")
	asJSON := fs.Bool("json", false, "emit the adoption result as JSON")
	if code, ok := parseFlagsOrHelp(fs, argv); !ok {
		return code
	}

	sess := strings.TrimSpace(*session)
	if rest := fs.Args(); len(rest) > 0 {
		if len(rest) != 1 {
			fmt.Fprintf(stderr, "fak wip reconcile %s: at most one <session> argument (flags must precede it)\n", verb)
			return 2
		}
		sess = strings.TrimSpace(rest[0])
	}
	if sess == "" {
		fmt.Fprintf(stderr, "fak wip reconcile %s: no session id (pass <session> or --session)\n", verb)
		return 2
	}
	if *into != "" && *patchOut != "" {
		fmt.Fprintf(stderr, "fak wip reconcile %s: --into and --patch-out are mutually exclusive (one isolated worker path OR one patch file)\n", verb)
		return 2
	}

	ctx := context.Background()
	if verb == "receipt" {
		return runWipAdoptReceipt(ctx, stdout, stderr, *repo, sess, *asJSON)
	}

	succ := firstNonEmpty(strings.TrimSpace(*successor), wipAdoptSuccessorDefault())
	if succ == "" {
		fmt.Fprintf(stderr, "fak wip reconcile %s: no successor id (pass --successor or set $CLAUDE_CODE_SESSION_ID)\n", verb)
		return 2
	}

	host, _ := os.Hostname()
	res, code, err := wipAdoptRun(ctx, *repo, wipAdoptOptions{
		Session:   sess,
		Successor: succ,
		Host:      host,
		Into:      strings.TrimSpace(*into),
		PatchOut:  strings.TrimSpace(*patchOut),
		TTL:       *ttl,
		Land:      *land,
		Now:       time.Now(),
	})
	if err != nil && code == 1 {
		fmt.Fprintf(stderr, "fak wip reconcile %s: %v\n", verb, err)
	}
	if *asJSON {
		if jc := encodeJSONOrFail(stdout, stderr, res, "fak wip reconcile "+verb); jc != 0 {
			return jc
		}
		return code
	}
	wipAdoptRender(stdout, res)
	return code
}

// runWipAdoptReceipt prints the stored claim, read-only. A missing receipt is exit 3 with
// NO_RECEIPT rather than an error: "nobody has claimed this" is an answer.
func runWipAdoptReceipt(ctx context.Context, stdout, stderr io.Writer, repo, session string, asJSON bool) int {
	rec, _, has, err := wipReadReceipt(ctx, repo, session)
	if err != nil {
		fmt.Fprintf(stderr, "fak wip reconcile receipt: %v\n", err)
		return 1
	}
	res := wipAdoptResult{Session: session, Preserved: wipCheckpointPresent(ctx, repo, session)}
	if !has {
		res.Verdict, res.Reason = wipReasonAdoptNoReceipt, fmt.Sprintf("no adoption receipt for %s", session)
	} else {
		res.Verdict, res.Successor, res.Receipt = string(rec.Phase), rec.Successor, &rec
		res.CheckpointRef, res.CheckpointSHA = rec.CheckpointRef, rec.CheckpointSHA
		res.Target = rec.Target
	}
	if asJSON {
		if jc := encodeJSONOrFail(stdout, stderr, res, "fak wip reconcile receipt"); jc != 0 {
			return jc
		}
	} else if !has {
		fmt.Fprintf(stdout, "%s\n", res.Reason)
	} else {
		fmt.Fprintf(stdout, "%s: %s holds phase %s (attempt %d) over %s\n",
			session, rec.Successor, rec.Phase, rec.Attempt, rec.CheckpointSHA)
		if rec.Target != "" {
			fmt.Fprintf(stdout, "  target: %s\n", rec.Target)
		}
		for _, ev := range rec.Audit {
			from := ""
			if ev.From != "" {
				from = " from " + ev.From
			}
			fmt.Fprintf(stdout, "  %d %s by %s%s %s\n", ev.At, ev.Event, ev.Actor, from, ev.Detail)
		}
	}
	if !has {
		return 3
	}
	return 0
}

// wipAdoptRun is the whole adoption, from classification through an optional landing.
//
// Returns (result, exitCode, err): 0 granted (or an idempotent no-op), 3 a checkable
// refusal that mutated nothing, 1 a runtime failure, 2 an argument fault.
//
// Preserved is stamped HERE, from a re-read of the checkpoint ref AFTER the attempt, on
// every path — grant, refusal, and runtime failure alike. The standing guarantee that
// adoption never drops a checkpoint is worth exactly as much as the observation behind it,
// and a hardcoded `true` would be the self-report this whole subsystem exists to refuse:
// it would print "the checkpoint is preserved" for a session that has no checkpoint.
func wipAdoptRun(ctx context.Context, repo string, opts wipAdoptOptions) (wipAdoptResult, int, error) {
	res, code, err := wipAdoptAttempt(ctx, repo, opts)
	res.Preserved = wipCheckpointPresent(ctx, repo, opts.Session)
	return res, code, err
}

// wipCheckpointPresent reports whether one session's checkpoint ref still resolves. An
// unreadable ref answers FALSE: "preserved" is a promise, and a promise this process could
// not witness must never be printed as one it kept.
func wipCheckpointPresent(ctx context.Context, repo, session string) bool {
	if !wipref.ValidSession(session) {
		return false
	}
	_, has, err := wipCurrentOID(ctx, repo, wipref.SessionRef(session))
	return err == nil && has
}

// wipAdoptAttempt is the adoption itself. Split from wipAdoptRun only so the preservation
// witness above can cover every one of its exits without threading a defer through them.
func wipAdoptAttempt(ctx context.Context, repo string, opts wipAdoptOptions) (wipAdoptResult, int, error) {
	res := wipAdoptResult{Session: opts.Session, Successor: opts.Successor}
	if !wipref.ValidSession(opts.Session) {
		return res, 2, fmt.Errorf("invalid session id %q (must be one safe ref segment)", opts.Session)
	}
	if !wipref.ValidSession(opts.Successor) {
		return res, 2, fmt.Errorf("invalid successor id %q (must be one safe ref segment)", opts.Successor)
	}
	if opts.Now.IsZero() {
		opts.Now = time.Now()
	}
	now := opts.Now.Unix()

	ref := wipref.SessionRef(opts.Session)
	res.CheckpointRef = ref
	obj, has, err := wipCurrentOID(ctx, repo, ref)
	if err != nil {
		return res, 1, err
	}
	if !has {
		res.Verdict = wipReasonAdoptNoCheckpoint
		res.Reason = fmt.Sprintf("no checkpoint ref %s — nothing to adopt", ref)
		return res, 3, nil
	}
	res.CheckpointSHA = obj

	// The reconciliation verdict for THIS checkpoint, derived exactly as the queue derives
	// it. Adoption may never be more permissive than classification: routing through the
	// same facts is what keeps QUARANTINE un-adoptable by construction rather than by a
	// second, drift-prone copy of the rule.
	live, err := wipLiveSessions(ctx, repo)
	if err != nil {
		return res, 1, err
	}
	action, err := wipAdoptAction(ctx, repo, opts.Session, ref, obj, live)
	if err != nil {
		return res, 1, err
	}

	patch, digest, err := wipAdoptDelta(ctx, repo, obj)
	if err != nil {
		return res, 1, err
	}

	cur, curOID, hasReceipt, err := wipReadReceipt(ctx, repo, opts.Session)
	if err != nil {
		// An unreadable incumbent claim must NOT read as "unclaimed" — that is exactly the
		// double-adoption this whole file prevents. Fail closed.
		return res, 1, fmt.Errorf("read adoption receipt for %s: %w", opts.Session, err)
	}
	var incumbent *wiprecon.Receipt
	req := wiprecon.AdoptRequest{
		Session:       opts.Session,
		Action:        action,
		CheckpointRef: ref,
		CheckpointSHA: obj,
		DeltaDigest:   digest,
		Successor:     opts.Successor,
		SuccessorHost: opts.Host,
		Now:           now,
		TTLSeconds:    opts.TTL,
	}
	if hasReceipt {
		incumbent = &cur
		res.Receipt = &cur
		req.IncumbentLive = live[cur.Successor]
	}

	decision := wiprecon.DecideAdopt(incumbent, req)
	res.Verdict, res.Reason = string(decision.Verdict), decision.Reason
	if !decision.Verdict.Granted() {
		if decision.Verdict == wiprecon.AdoptSettled {
			return res, 0, nil // an already-landed checkpoint is a benign no-op
		}
		return res, 3, nil
	}
	if strings.TrimSpace(patch) == "" {
		res.Verdict, res.Reason = wipReasonAdoptEmptyDelta, fmt.Sprintf("checkpoint %s holds an empty delta", obj)
		return res, 3, nil
	}

	next, ok := wiprecon.ApplyAdopt(incumbent, req, decision.Verdict)
	if !ok {
		return res, 1, fmt.Errorf("internal: no receipt for granted verdict %s", decision.Verdict)
	}
	target, kind, code, terr := wipAdoptResolveTarget(ctx, repo, opts, next)
	if code == 1 {
		return res, 1, terr
	}
	if terr != nil {
		res.Verdict, res.Reason = wipReasonAdoptTargetInTree, terr.Error()
		return res, code, nil
	}
	next.Target, res.Target, res.TargetKind = target, target, kind

	// JOURNAL BEFORE MUTATION. Everything above this line is a read; everything below it
	// can leave bytes on disk. The swap is the moment ownership becomes a fact other
	// processes can see, and a crash one instruction later is fully resumable.
	if opts.raceHook != nil {
		opts.raceHook()
	}
	receiptOID, won, err := wipWriteReceipt(ctx, repo, next, curOID, hasReceipt)
	if err != nil {
		return res, 1, err
	}
	if !won {
		res.Verdict = wipReasonAdoptLostRace
		res.Reason = fmt.Sprintf("a concurrent successor claimed %s between this bid's read and its write — nothing was materialized", opts.Session)
		res.Receipt = nil
		return res, 3, nil
	}
	res.Receipt = &next

	mat, mcode, merr := wipAdoptMaterialize(ctx, repo, obj, digest, kind, target)
	res.Materialized, res.Deleted, res.Verified = mat.Files, mat.Deleted, mat.Verified
	if merr != nil {
		if mcode == 3 {
			res.Verdict, res.Reason = mat.Reason, merr.Error()
			return res, 3, nil
		}
		return res, 1, merr
	}

	next = wiprecon.MarkPhase(next, wiprecon.PhaseMaterialized, now, wiprecon.EventMaterialized,
		fmt.Sprintf("%d file(s) verified into %s", mat.Verified, target))
	receiptOID, won, err = wipWriteReceipt(ctx, repo, next, receiptOID, true)
	if err != nil {
		return res, 1, err
	}
	if !won {
		// Someone displaced this claim while it was writing bytes. The bytes are in an
		// isolated target, so nothing shared was hurt; report it as the race it is.
		res.Verdict = wipReasonAdoptLostRace
		res.Reason = fmt.Sprintf("the %s adoption receipt moved while this successor materialized; its target %s is left in place for inspection", opts.Session, target)
		return res, 3, nil
	}
	res.Receipt = &next

	if !opts.Land {
		return res, 0, nil
	}

	land, lcode, lerr := wipLandWith(ctx, repo, opts.Session, wipLandOptions{})
	res.Land = &land
	if lcode != 0 || !land.Committed {
		// The receipt stays at MATERIALIZED and the checkpoint ref stays put, so the
		// landing is retryable from exactly here. A refusal is not a lost checkpoint —
		// which is why the land's own token replaces the adoption verdict rather than
		// the adoption reporting a success it did not finish.
		res.Verdict = firstNonEmpty(land.Reason, "LAND_FAILED")
		if lerr != nil {
			res.Reason = lerr.Error()
		}
		return res, lcode, lerr
	}
	res.Landed, res.LandedSHA = true, land.SHA
	next.LandedSHA = land.SHA
	next = wiprecon.MarkPhase(next, wiprecon.PhaseLanded, now, wiprecon.EventLanded,
		fmt.Sprintf("committed %s", land.SHA))
	if _, _, err := wipWriteReceipt(ctx, repo, next, receiptOID, true); err != nil {
		return res, 1, err
	}
	res.Receipt = &next
	return res, 0, nil
}

// wipAdoptAction classifies ONE checkpoint with the same three facts wipReconcileAt folds
// over every checkpoint: liveness, landing, clean-apply. Deriving it per-session keeps
// adoption O(1) in the number of unrelated checkpoints while using the identical rule.
func wipAdoptAction(ctx context.Context, repo, session, ref, obj string, live map[string]bool) (wiprecon.Action, error) {
	c := wiprecon.Candidate{Session: session, Owner: wiprecon.OwnerCrashed}
	if live[session] {
		c.Owner = wiprecon.OwnerLive
		return wiprecon.Decide(c).Action, nil
	}
	rec, err := wipRecordAt(ctx, repo, ref, obj)
	if err != nil {
		return "", err
	}
	st, err := wipOwnerState(ctx, repo, rec)
	if err != nil {
		return "", err
	}
	c.Landed = st == wipref.OwnerLanded
	if !c.Landed {
		c.Applies = wipDeltaApplies(ctx, repo, rec)
	}
	return wiprecon.Decide(c).Action, nil
}

// wipAdoptDelta returns the checkpoint's RAW patch bytes and their content digest. The
// digest is the third leg of the binding (ref, SHA, bytes): a resume proves it is about to
// write the delta it claimed, not merely that a ref still resolves to the same object.
func wipAdoptDelta(ctx context.Context, repo, obj string) (string, string, error) {
	patch, errStr, code, err := gitWip(ctx, repo, nil, "diff", obj+"^", obj)
	if err != nil {
		return "", "", fmt.Errorf("git diff %s: %w", obj, err)
	}
	if code != 0 {
		return "", "", fmt.Errorf("git diff %s exited %d: %s", obj, code, strings.TrimSpace(errStr))
	}
	return patch, wipAdoptDigest(patch), nil
}

func wipAdoptDigest(b string) string {
	sum := sha256.Sum256([]byte(b))
	return "sha256:" + hex.EncodeToString(sum[:])
}

// wipAdoptRef is where one session's adoption receipt lives.
func wipAdoptRef(session string) string { return wipAdoptNamespace + session }

// wipReadReceipt reads the stored claim. (receipt, its object id, whether one exists).
// A ref that exists but whose body will not decode is an ERROR, never "unclaimed".
func wipReadReceipt(ctx context.Context, repo, session string) (wiprecon.Receipt, string, bool, error) {
	oid, has, err := wipCurrentOID(ctx, repo, wipAdoptRef(session))
	if err != nil || !has {
		return wiprecon.Receipt{}, "", false, err
	}
	body, err := gitWipOut(ctx, repo, nil, "cat-file", "blob", oid)
	if err != nil {
		return wiprecon.Receipt{}, oid, true, err
	}
	rec, err := wiprecon.DecodeReceipt(body)
	if err != nil {
		return wiprecon.Receipt{}, oid, true, err
	}
	return rec, oid, true, nil
}

// wipWriteReceipt hashes the receipt into a blob and compare-and-swaps the claim ref onto
// it. Returns (new object id, whether the swap won, error). A lost swap is NOT an error:
// it is the whole point of the mechanism, and the loser must report it and stop.
func wipWriteReceipt(ctx context.Context, repo string, rec wiprecon.Receipt, oldOID string, had bool) (string, bool, error) {
	body, err := wiprecon.EncodeReceipt(rec)
	if err != nil {
		return "", false, err
	}
	out, errStr, code, err := gitWipStdin(ctx, repo, body, "hash-object", "-w", "--stdin")
	if err != nil {
		return "", false, fmt.Errorf("hash adoption receipt: %w", err)
	}
	if code != 0 {
		return "", false, fmt.Errorf("hash adoption receipt exited %d: %s", code, strings.TrimSpace(errStr))
	}
	oid := strings.TrimSpace(out)
	won, err := wipCasUpdateRef(ctx, repo, wipAdoptRef(rec.Session), oid, oldOID, had)
	if err != nil {
		return "", false, err
	}
	return oid, won, nil
}

// wipAdoptSuccessorDefault is the adopting session's own id, from the same environment
// the checkpoint verb reads.
func wipAdoptSuccessorDefault() string {
	return firstNonEmpty(os.Getenv("CLAUDE_CODE_SESSION_ID"), os.Getenv("FAK_SESSION_ID"))
}

// wipAdoptResolveTarget picks where the delta materializes: an explicit --into worker
// directory, an explicit --patch-out file, the target a prior attempt recorded (so a
// resume finds its own bytes rather than orphaning them), or the default patch path under
// the git dir. Returns (target, kind, exitCode, err); exit 3 with an err is the checkable
// TARGET_INSIDE_TREE refusal.
func wipAdoptResolveTarget(ctx context.Context, repo string, opts wipAdoptOptions, rec wiprecon.Receipt) (string, string, int, error) {
	switch {
	case opts.Into != "":
		abs, err := filepath.Abs(opts.Into)
		if err != nil {
			return "", "", 1, err
		}
		inside, err := wipAdoptInsideTree(ctx, repo, abs)
		if err != nil {
			return "", "", 1, err
		}
		if inside {
			return "", "", 3, fmt.Errorf("--into %s resolves inside this repo's working tree; a recovered stranger's delta written there would land on concurrent peers' edits — pick a directory outside the tree", abs)
		}
		return abs, "worker", 0, nil
	case opts.PatchOut != "":
		abs, err := filepath.Abs(opts.PatchOut)
		if err != nil {
			return "", "", 1, err
		}
		inside, err := wipAdoptInsideTree(ctx, repo, abs)
		if err != nil {
			return "", "", 1, err
		}
		if inside {
			return "", "", 3, fmt.Errorf("--patch-out %s resolves inside this repo's working tree, where an unignored patch file becomes a peer's dirty path — pick a path outside the tree", abs)
		}
		return abs, "patch", 0, nil
	case rec.Target != "":
		kind := "patch"
		if !strings.HasSuffix(rec.Target, ".patch") {
			kind = "worker"
		}
		return rec.Target, kind, 0, nil
	}
	gd, err := gitWipOut(ctx, repo, nil, "rev-parse", "--absolute-git-dir")
	if err != nil {
		return "", "", 1, err
	}
	return filepath.Join(strings.TrimSpace(gd), "fak-adopt", rec.Session+".patch"), "patch", 0, nil
}

// wipAdoptInsideTree reports whether an absolute path lies at or under this repo's working
// tree root. Comparison is case-insensitive so a Windows path that differs only in case is
// still recognized as inside — the over-strict direction, which REFUSES, is the safe way
// to be wrong about "may I write here". An unresolvable root is an error, not a pass.
func wipAdoptInsideTree(ctx context.Context, repo, abs string) (bool, error) {
	top, err := gitWipOut(ctx, repo, nil, "rev-parse", "--show-toplevel")
	if err != nil {
		return false, err
	}
	root, err := filepath.Abs(strings.TrimSpace(top))
	if err != nil {
		return false, err
	}
	a, r := filepath.Clean(abs), filepath.Clean(root)
	if strings.EqualFold(a, r) {
		return true, nil
	}
	return strings.HasPrefix(strings.ToLower(a), strings.ToLower(r)+string(filepath.Separator)), nil
}

// wipAdoptMaterialization is what one materialization actually wrote.
type wipAdoptMaterialization struct {
	Files    []string
	Deleted  []string
	Verified int
	Reason   string // a closed refusal token when the caller must exit 3
}

// wipAdoptMaterialize writes the adopted delta to the target and VERIFIES the bytes
// against the checkpoint object.
//
// Verification is a read-back, not a trust of the write call: every byte written is read
// off disk again and hashed against what git holds for that object. That is the difference
// between "os.WriteFile returned nil" and "the successor's copy is the checkpoint", and on
// a resume after a torn write it is the only thing that can tell those apart.
//
// The checkpoint ref is re-resolved first: a checkpoint that moved between the claim and
// the write is CHECKPOINT_MOVED, because the receipt authorized specific bytes.
func wipAdoptMaterialize(ctx context.Context, repo, obj, digest, kind, target string) (wipAdoptMaterialization, int, error) {
	var out wipAdoptMaterialization

	patch, gotDigest, err := wipAdoptDelta(ctx, repo, obj)
	if err != nil {
		return out, 1, err
	}
	if gotDigest != digest {
		out.Reason = wipReasonAdoptDeltaMismatch
		return out, 3, fmt.Errorf("checkpoint %s now hashes %s, not the adopted %s", obj, gotDigest, digest)
	}

	if kind == "patch" {
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			return out, 1, err
		}
		if err := os.WriteFile(target, []byte(patch), 0o644); err != nil {
			return out, 1, err
		}
		back, err := os.ReadFile(target)
		if err != nil {
			return out, 1, err
		}
		if wipAdoptDigest(string(back)) != digest {
			out.Reason = wipReasonAdoptDeltaMismatch
			return out, 3, fmt.Errorf("the patch written to %s does not hash to the adopted delta %s", target, digest)
		}
		out.Files, out.Verified = []string{target}, 1
		return out, 0, nil
	}

	changes, err := wipAdoptChanges(ctx, repo, obj)
	if err != nil {
		return out, 1, err
	}
	for _, ch := range changes {
		if !wipAdoptSafeRelPath(ch.Path) {
			out.Reason = wipReasonAdoptUnsafePath
			return out, 3, fmt.Errorf("the %s delta names %q, which escapes the target directory", obj, ch.Path)
		}
		if ch.Deleted {
			// A deletion has no post-image to write. Recorded, never applied: an isolated
			// worker path is a materialization of the checkpoint's CONTENT, and removing a
			// file the successor never created is not this verb's call to make.
			out.Deleted = append(out.Deleted, ch.Path)
			continue
		}
		want, err := wipAdoptBlob(ctx, repo, obj, ch.Path)
		if err != nil {
			return out, 1, err
		}
		dst := filepath.Join(target, filepath.FromSlash(ch.Path))
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			return out, 1, err
		}
		if err := os.WriteFile(dst, []byte(want), 0o644); err != nil {
			return out, 1, err
		}
		back, err := os.ReadFile(dst)
		if err != nil {
			return out, 1, err
		}
		if wipAdoptDigest(string(back)) != wipAdoptDigest(want) {
			out.Reason = wipReasonAdoptDeltaMismatch
			return out, 3, fmt.Errorf("materialized %s does not match the bytes checkpoint %s holds for it", dst, obj)
		}
		out.Files = append(out.Files, ch.Path)
		out.Verified++
	}
	return out, 0, nil
}

// wipAdoptChange is one path in a checkpoint's delta.
type wipAdoptChange struct {
	Path    string
	Deleted bool
}

// wipAdoptChanges enumerates the delta with NUL-delimited name-status output, so a path
// containing a space or a quote parses exactly. --no-renames keeps every entry a single
// (status, path) pair: a rename arrives as a delete plus an add, which is what a content
// materialization wants anyway.
func wipAdoptChanges(ctx context.Context, repo, obj string) ([]wipAdoptChange, error) {
	raw, errStr, code, err := gitWip(ctx, repo, nil, "diff", "--name-status", "--no-renames", "-z", obj+"^", obj)
	if err != nil {
		return nil, fmt.Errorf("git diff --name-status: %w", err)
	}
	if code != 0 {
		return nil, fmt.Errorf("git diff --name-status exited %d: %s", code, strings.TrimSpace(errStr))
	}
	fields := strings.Split(strings.TrimSuffix(raw, "\x00"), "\x00")
	var out []wipAdoptChange
	for i := 0; i+1 < len(fields); i += 2 {
		status, path := fields[i], fields[i+1]
		if status == "" || path == "" {
			continue
		}
		out = append(out, wipAdoptChange{Path: path, Deleted: strings.HasPrefix(status, "D")})
	}
	return out, nil
}

// wipAdoptBlob reads one path's post-image bytes out of the checkpoint object, RAW.
func wipAdoptBlob(ctx context.Context, repo, obj, path string) (string, error) {
	out, errStr, code, err := gitWip(ctx, repo, nil, "cat-file", "blob", obj+":"+path)
	if err != nil {
		return "", fmt.Errorf("git cat-file %s:%s: %w", obj, path, err)
	}
	if code != 0 {
		return "", fmt.Errorf("git cat-file %s:%s exited %d: %s", obj, path, code, strings.TrimSpace(errStr))
	}
	return out, nil
}

// wipAdoptSafeRelPath rejects a delta path that could write outside the target: absolute
// paths, volume-qualified paths, and any ".." segment. git does not produce these, so a
// hit here means the object was crafted — refuse rather than reason about it.
func wipAdoptSafeRelPath(p string) bool {
	if p == "" || strings.ContainsRune(p, 0) {
		return false
	}
	if filepath.IsAbs(p) || strings.HasPrefix(p, "/") || filepath.VolumeName(p) != "" {
		return false
	}
	for _, seg := range strings.Split(strings.ReplaceAll(p, "\\", "/"), "/") {
		if seg == ".." {
			return false
		}
	}
	return true
}

// wipAdoptRender prints one adoption in the shape an operator can act on: the verdict, the
// sentence behind it, and — always — the standing guarantee that the checkpoint ref is
// still there.
func wipAdoptRender(stdout io.Writer, res wipAdoptResult) {
	fmt.Fprintf(stdout, "%s\t%s\t%s\n", res.Verdict, res.Session, res.Reason)
	if res.Receipt != nil {
		fmt.Fprintf(stdout, "  receipt: %s holds phase %s (attempt %d) over %s\n",
			res.Receipt.Successor, res.Receipt.Phase, res.Receipt.Attempt, res.CheckpointSHA)
	}
	if res.Target != "" {
		fmt.Fprintf(stdout, "  target (%s): %s — %d file(s) verified against the checkpoint\n", res.TargetKind, res.Target, res.Verified)
	}
	if len(res.Deleted) > 0 {
		fmt.Fprintf(stdout, "  the delta also DELETES %d path(s), not applied here: %s\n", len(res.Deleted), strings.Join(res.Deleted, ", "))
	}
	if res.Landed {
		fmt.Fprintf(stdout, "  landed %s; the checkpoint ref is preserved until `fak wip reap` witnesses it in HEAD\n", res.LandedSHA)
	}
	switch {
	case res.Preserved:
		fmt.Fprintf(stdout, "  checkpoint %s is preserved; adoption never drops a checkpoint\n", res.CheckpointRef)
	case res.CheckpointSHA != "":
		// The ref resolved at the start of this run and does not now. Nothing on this path
		// deletes one, so a disappearance is a CONCURRENT deleter — and the operator has to
		// hear it, because the materialized target may now be the only copy of that delta.
		fmt.Fprintf(stdout, "  WARNING: checkpoint %s held %s when this run began and is GONE now; adoption never deletes one, so something else did — %s\n",
			res.CheckpointRef, shortWipSHA(res.CheckpointSHA), firstNonEmpty(res.Target, "and nothing was materialized"))
	}
}
