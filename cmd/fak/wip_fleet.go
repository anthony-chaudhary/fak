package main

// wip_fleet.go — the git-I/O shell for FLEET-VISIBLE checkpoints (#3880, child C8b of
// #3871): the enumeration behind `fak wip status --fleet` and the object-size gate
// `fak wip sync` applies before it publishes. internal/wipref/fleet.go owns every
// decision here (which delta is withheld, how a mirrored ref is classified, what the
// refspecs are); this file only runs git and hands the listings back.
//
// It is a separate file from wip.go and wip_sync.go for the reason wip_reconcile.go was
// split out: wip.go is already against the god-file ceiling the tree enforces
// (internal/hooks/gate_godfile.go, godFileMaxLines = 1500), and the fleet surface is a
// self-contained cut whose helpers serve nothing else.
//
// THE ONE THING TO KNOW BEFORE CHANGING THE PUBLISH PATH. A checkpoint is a TREE-WIDE
// capture and can be megabytes; the coordinator clone accumulates every host's. So the
// push is no longer unconditionally "send the namespace": an over-bound delta publishes a
// STUB — a parentless commit over the empty tree carrying the same stamp, marked
// MetadataOnly — at the same refs/fak/wip/<session>. The session stays enumerable
// fleet-wide, its bytes stay here. Two consequences that are easy to break:
//
//   - THE MIRROR MUST RECORD WHAT WAS ACTUALLY PUBLISHED, not the local object. The mirror
//     is the evidence ClassifyReplication grades against, and recording the local object
//     for a gated session would report REPLICATED for a delta that never left the machine —
//     the precise overstatement the whole replication column exists to prevent. A gated
//     session records its STUB, which grades STALE_REMOTE: the remote does hold this
//     session's ref, at a different object, and this delta is not off-machine. That is the
//     honest reading, and it is the safe direction to be wrong in.
//
//   - THE STUB MUST BE DETERMINISTIC. commit-tree stamps the wall clock into the object, so
//     a naive mint would produce a new stub object on every sync, churn the remote ref, and
//     leave the mirror chasing a hash that changes for no reason. Pinning the author and
//     committer identity and dating both from the checkpoint's own CheckpointedAt makes the
//     stub a pure function of the stamp: re-syncing an unchanged checkpoint is a no-op.

import (
	"context"
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/wipref"
)

// wipStubIdentity is the fixed author/committer a metadata stub is minted under. It is a
// placeholder identity, never a person: the stub is machinery, and binding it to whichever
// clone happened to run the sync would make the object non-deterministic across the fleet
// for no gain. The .invalid TLD is reserved by RFC 2606 precisely so it can never resolve.
const wipStubIdentity = "fak wip <wip@fak.invalid>"

// wipLocalHost resolves THIS machine's stable node id for the checkpoint stamp. It reuses
// leaseref's node identity rather than minting a second convention, so "which machine holds
// this lease" and "which machine stranded this WIP" are answered in the same vocabulary. A
// hostname it cannot resolve degrades to leaseref.NodeUnknown, which the fleet fold reads
// as HostUnknown — a classification, never an error, and never a reason to fail a capture.
func wipLocalHost(repo string) string {
	root := repo
	if root == "" {
		root = "."
	}
	return leaseref.LocalNodeID(root)
}

// wipDeltaBytes measures, in uncompressed blob bytes, what a checkpoint's tree introduces
// over its base — the number the publish gate reads. TWO git spawns however many files
// changed: one `diff --raw` for the post-image blob ids, one `cat-file --batch-check` for
// their sizes. It runs at CAPTURE time, once, because measuring at push time would be a
// spawn per ref over a namespace that routinely holds thousands (#5336).
//
// It is BEST-EFFORT by contract and returns 0 — which the gate reads as UNMEASURED, i.e.
// publishable — on any failure. A checkpoint exists to not lose work at a risky boundary;
// a size measurement is an optimization for someone else's disk, and it may never be the
// reason a capture fails.
func wipDeltaBytes(ctx context.Context, repo, base, tree string) int64 {
	raw, _, code, err := gitWip(ctx, repo, nil, "diff", "--raw", "--no-renames", "--abbrev=40", base, tree)
	if err != nil || code != 0 {
		return 0
	}
	var ids strings.Builder
	seen := map[string]bool{}
	for _, ln := range strings.Split(raw, "\n") {
		// ":<srcmode> <dstmode> <srcsha> <dstsha> <status>\t<path>" — the post-image id is
		// the fourth space-separated field. A deletion carries an all-zero dstsha and adds
		// nothing to the remote, so it is skipped rather than counted as a missing object.
		if !strings.HasPrefix(ln, ":") {
			continue
		}
		f := strings.Fields(strings.SplitN(ln, "\t", 2)[0])
		if len(f) < 4 {
			continue
		}
		id := f[3]
		if id == "" || strings.Trim(id, "0") == "" || seen[id] {
			continue
		}
		seen[id] = true
		ids.WriteString(id)
		ids.WriteString("\n")
	}
	if ids.Len() == 0 {
		return 0
	}
	out, _, code, err := gitWipStdin(ctx, repo, ids.String(), "cat-file", "--batch-check")
	if err != nil || code != 0 {
		return 0
	}
	var total int64
	for _, ln := range strings.Split(out, "\n") {
		f := strings.Fields(ln)
		// "<oid> blob <size>"; a "<oid> missing" line has two fields and is skipped.
		if len(f) != 3 || f[1] != "blob" {
			continue
		}
		if n, err := strconv.ParseInt(f[2], 10, 64); err == nil {
			total += n
		}
	}
	return total
}

// wipEmptyTree writes (idempotently) and returns this repo's empty tree object id. It is
// asked of git rather than hardcoded to 4b825dc… because that constant is the SHA-1 empty
// tree, and a repo initialized with sha256 object format has a different one.
func wipEmptyTree(ctx context.Context, repo string) (string, error) {
	out, errStr, code, err := gitWipStdin(ctx, repo, "", "hash-object", "-w", "-t", "tree", "--stdin")
	if err != nil {
		return "", fmt.Errorf("git not executable: %w", err)
	}
	if code != 0 {
		return "", fmt.Errorf("write the empty tree: hash-object exited %d: %s", code, strings.TrimSpace(errStr))
	}
	return strings.TrimSpace(out), nil
}

// wipMintStub mints the metadata-only object a size-gated session publishes in place of
// its delta: a parentless commit over the empty tree whose message carries the checkpoint's
// stamp marked MetadataOnly and naming the withheld object. Deterministic — see the file
// header for why that matters.
func wipMintStub(ctx context.Context, repo, emptyTree string, rec wipref.RefRecord) (string, error) {
	msg, err := wipref.EncodeStamp(wipref.MetadataStamp(rec.Stamp, rec.Object))
	if err != nil {
		return "", fmt.Errorf("encode the metadata stamp for %s: %w", rec.Ref, err)
	}
	when := rec.Stamp.CheckpointedAt
	if when < 0 {
		when = 0
	}
	date := fmt.Sprintf("@%d +0000", when)
	env := []string{
		"GIT_AUTHOR_NAME=fak wip", "GIT_AUTHOR_EMAIL=wip@fak.invalid", "GIT_AUTHOR_DATE=" + date,
		"GIT_COMMITTER_NAME=fak wip", "GIT_COMMITTER_EMAIL=wip@fak.invalid", "GIT_COMMITTER_DATE=" + date,
	}
	obj, errStr, code, err := gitWip(ctx, repo, env, "commit-tree", emptyTree, "-m", msg)
	if err != nil {
		return "", fmt.Errorf("git not executable: %w", err)
	}
	if code != 0 {
		return "", fmt.Errorf("mint the metadata stub for %s: commit-tree exited %d: %s", rec.Ref, code, strings.TrimSpace(errStr))
	}
	return strings.TrimSpace(obj), nil
}

// wipPublish runs the GATED push: plan every local checkpoint against the bound, mint a
// stub for each over-bound one, push the resulting refspecs, and report both the plan and
// the object each session actually published (which is what the mirror must record).
//
// An UNGATED plan takes the single-glob path that predates the gate — one push, no stub
// mint, no per-ref argv — so a fleet whose deltas are all normal-sized pays nothing for
// the gate existing.
func wipPublish(ctx context.Context, repo, remote string, local []wipref.RefRecord, maxDeltaBytes int64) (wipref.PublishPlan, map[string]string, error) {
	plan := wipref.PlanPublish(local, maxDeltaBytes)
	published := make(map[string]string, len(plan.Entries))
	for _, e := range plan.Entries {
		published[e.Session] = e.Object
	}
	if len(plan.Entries) == 0 {
		return plan, published, nil
	}

	stubs := map[string]string{}
	if plan.Gated() {
		emptyTree, err := wipEmptyTree(ctx, repo)
		if err != nil {
			return plan, published, err
		}
		byRef := map[string]wipref.RefRecord{}
		for _, r := range local {
			byRef[r.Ref] = r
		}
		for _, e := range plan.Entries {
			if e.Class != wipref.PublishMetadataOnly {
				continue
			}
			// PlanPublish keys entries off the REF NAME, so this lookup is total for every
			// entry it emitted. Checking anyway rather than minting from a zero record: a
			// stub built from an empty stamp would publish a ref that names no session, no
			// host and no withheld object — a row in the fleet view that says nothing while
			// looking exactly like one that says something.
			rec, ok := byRef[e.Ref]
			if !ok {
				return plan, published, fmt.Errorf("plan named %s but the local listing has no such ref", e.Ref)
			}
			stub, err := wipMintStub(ctx, repo, emptyTree, rec)
			if err != nil {
				return plan, published, err
			}
			stubs[e.Session] = stub
			published[e.Session] = stub
		}
	}

	batches, err := wipref.PushRefspecs(plan, stubs)
	if err != nil {
		return plan, published, err
	}
	for _, specs := range batches {
		args := append([]string{"push", remote}, specs...)
		_, errStr, code, err := gitWip(ctx, repo, nil, args...)
		if err != nil {
			return plan, published, fmt.Errorf("git not executable: %w", err)
		}
		if code != 0 {
			return plan, published, fmt.Errorf("git push %s (%d refspec(s)) exited %d: %s — sync stopped before fetch; your checkpoints are still LOCAL_ONLY",
				remote, len(specs), code, strings.TrimSpace(errStr))
		}
	}
	return plan, published, nil
}

// wipSyncBounded is the size-gated push path used when fleet visibility is enabled.
// Fetch-only calls retain the ordinary sync implementation. An ungated plan also
// delegates unchanged so the common path keeps the one glob refspec.
func wipSyncBounded(ctx context.Context, repo, remote string, doPush, doFetch bool, maxDeltaBytes int64) (wipref.SyncResult, error) {
	if !doPush {
		return wipSync(ctx, repo, remote, false, doFetch)
	}
	local, err := wipListRecords(ctx, repo)
	if err != nil {
		return wipref.SyncResult{}, err
	}
	plan := wipref.PlanPublish(local, maxDeltaBytes)
	if !plan.Gated() {
		res, err := wipSync(ctx, repo, remote, true, doFetch)
		res.MaxDeltaBytes = maxDeltaBytes
		return res, err
	}
	pub, objects, err := wipPublish(ctx, repo, remote, local, maxDeltaBytes)
	if err != nil {
		return wipref.SyncResult{}, err
	}
	res := wipref.SyncResult{Remote: remote, Pushed: true, Published: len(pub.Entries), MaxDeltaBytes: maxDeltaBytes, MetadataOnly: pub.MetadataOnly}
	published := wipPublishedRecords(pub, objects)
	if err := wipWriteMirror(ctx, repo, remote, published); err != nil {
		return res, err
	}
	if doFetch {
		fetch, err := wipSync(ctx, repo, remote, false, true)
		if err != nil {
			return res, err
		}
		res.Fetched, res.FetchRefspec, res.Mirrored = fetch.Fetched, fetch.FetchRefspec, fetch.Mirrored
		res.SyncedAt, res.Source = fetch.SyncedAt, fetch.Source
	}
	return res, nil
}

// wipPublishedRecords projects the plan plus the per-session published object into the ref
// records wipWriteMirror stores. The OBJECT is the published one, never the local one — see
// the file header for why a gated session recording its local object would report a delta
// as REPLICATED that never left this machine.
func wipPublishedRecords(plan wipref.PublishPlan, published map[string]string) []wipref.RefRecord {
	out := make([]wipref.RefRecord, 0, len(plan.Entries))
	for _, e := range plan.Entries {
		obj := published[e.Session]
		if obj == "" {
			obj = e.Object
		}
		out = append(out, wipref.RefRecord{Ref: e.Ref, Object: obj})
	}
	return out
}

// ---- the fleet listing ----

// wipListFleetRecords lists this clone's mirror of remote WITH each mirrored stamp decoded.
// It is deliberately a SEPARATE reader from wipListMirrorRecords, which asks for ref+object
// only: that function answers "does the remote hold this object" for the replication column
// and its comment is explicit that decoding a peer's stamp there would invite the live verbs
// to start treating mirrored refs as local checkpoints. This reader exists for the ONE
// caller that legitimately wants a peer's metadata — the fleet enumeration — and its output
// never reaches reap, land, reconcile or attribute.
//
// One `for-each-ref` however many refs the mirror holds, same NUL-separated triple grammar
// (and same O(1)-in-spawns discipline) as wipListRecords.
func wipListFleetRecords(ctx context.Context, repo, remote string) ([]wipref.RefRecord, error) {
	pattern := strings.TrimSuffix(wipref.MirrorNamespace(remote), "/")
	out, errStr, code, err := gitWip(ctx, repo, nil,
		"for-each-ref", "--format=%(refname)%00%(objectname)%00%(contents)%00", pattern)
	if err != nil {
		return nil, fmt.Errorf("git for-each-ref: %w", err)
	}
	if code != 0 {
		return nil, fmt.Errorf("git for-each-ref exited %d: %s", code, strings.TrimSpace(errStr))
	}
	fields := strings.Split(out, "\x00")
	var recs []wipref.RefRecord
	for i := 0; i+2 < len(fields); i += 3 {
		ref := strings.TrimSpace(fields[i])
		obj := strings.TrimSpace(fields[i+1])
		if ref == "" || obj == "" {
			continue
		}
		// An UNDECODABLE stamp leaves the zero Stamp, which the fold labels from the ref
		// name and reports as HostUnknown — a peer's ref never vanishes from the listing
		// just because this clone could not read its metadata.
		stamp, _ := wipref.DecodeStamp(fields[i+2])
		recs = append(recs, wipref.RefRecord{Ref: ref, Object: obj, Stamp: stamp})
	}
	return recs, nil
}

// wipObjectsPresent reports which of objs this clone can actually read, in ONE
// `cat-file --batch-check`. It is what separates "a peer's delta is recoverable from here"
// from "a peer's ref is here and its bytes are not" — a distinction a ref listing alone
// cannot make, and the one that stops the fleet view promising work it cannot deliver.
func wipObjectsPresent(ctx context.Context, repo string, objs []string) (map[string]bool, error) {
	present := map[string]bool{}
	if len(objs) == 0 {
		return present, nil
	}
	var in strings.Builder
	seen := map[string]bool{}
	for _, o := range objs {
		if o == "" || seen[o] {
			continue
		}
		seen[o] = true
		in.WriteString(o)
		in.WriteString("\n")
	}
	out, errStr, code, err := gitWipStdin(ctx, repo, in.String(), "cat-file", "--batch-check")
	if err != nil {
		return nil, fmt.Errorf("git cat-file --batch-check: %w", err)
	}
	if code != 0 {
		return nil, fmt.Errorf("git cat-file --batch-check exited %d: %s", code, strings.TrimSpace(errStr))
	}
	for _, ln := range strings.Split(out, "\n") {
		f := strings.Fields(ln)
		// "<oid> <type> <size>" is present; "<oid> missing" is not.
		if len(f) == 3 {
			present[f[0]] = true
		}
	}
	return present, nil
}

// wipFleetStatus is the whole read side of `fak wip status --fleet`: enumerate the mirror
// of remote, learn which sessions are this clone's own and which mirrored objects are
// actually here, and fold. Every read is LOCAL — the fleet view never makes a network round
// trip, for the same reason the replication column does not: an operator asking "whose work
// is stranded" is very often asking it because the network is what failed. What a `git
// fetch` last imported is the evidence, and the attached MirrorView says how old it is.
func wipFleetStatus(ctx context.Context, repo, remote string, nowUnix int64) (wipref.FleetReport, error) {
	if !wipref.ValidRemote(remote) {
		return wipref.FleetReport{}, fmt.Errorf("invalid remote %q (must be one safe git argv token)", remote)
	}
	mirrored, err := wipListFleetRecords(ctx, repo, remote)
	if err != nil {
		return wipref.FleetReport{}, err
	}
	local, err := wipListRecords(ctx, repo)
	if err != nil {
		return wipref.FleetReport{}, err
	}
	localSessions := make(map[string]bool, len(local))
	for _, r := range local {
		sess := r.Stamp.SessionID
		if sess == "" {
			sess = wipref.SessionFromRef(r.Ref)
		}
		localSessions[sess] = true
	}
	objs := make([]string, 0, len(mirrored))
	for _, r := range mirrored {
		objs = append(objs, r.Object)
	}
	present, err := wipObjectsPresent(ctx, repo, objs)
	if err != nil {
		return wipref.FleetReport{}, err
	}
	rep := wipref.FoldFleet(remote, mirrored, localSessions, present, nowUnix)
	// The provenance is not decoration here. "No peer has stranded work" is a claim about
	// OTHER HOSTS, and this clone may only make it after a real fetch inside tolerance —
	// see MirrorView.EmptyIsAbsence.
	view, err := wipMirrorView(ctx, repo, remote, len(mirrored), nowUnix, 0)
	if err != nil {
		return wipref.FleetReport{}, err
	}
	rep.Mirror = &view
	return rep, nil
}

// wipFleetRender prints the fleet listing for PLAIN output: one row per mirrored
// checkpoint, then the census. It deliberately does NOT print the mirror's provenance —
// the caller does, once, for the whole status call, because the fleet listing and the
// replication column were graded against the same mirror and a caveat printed twice is a
// caveat that gets skimmed. The empty case is the dangerous one, so it still says which
// remote produced no rows rather than printing nothing at all.
func wipFleetRender(stdout io.Writer, rep wipref.FleetReport) {
	for _, r := range rep.Rows {
		size := ""
		if r.DeltaBytes > 0 {
			size = fmt.Sprintf("\t%dB", r.DeltaBytes)
		}
		fmt.Fprintf(stdout, "%s\t%s\t%s\tage=%ds\t%d leaves%s\t%s\n",
			r.Host, r.Session, shortWipSHA(r.Object), r.AgeSeconds, len(r.Leaves), size, r.Disposition)
	}
	if rep.Count == 0 {
		fmt.Fprintf(stdout, "no checkpoints mirrored from %s\n", rep.Remote)
	}
	fmt.Fprintln(stdout, wipref.FleetSummary(rep))
}
