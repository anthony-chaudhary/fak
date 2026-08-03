package wipref

// sync.go is the REPLICATION half of the checkpoint substrate (#5479, child C8 of
// #3871): the pure decisions behind `fak wip sync` and behind the REPLICATED /
// LOCAL_ONLY / STALE_REMOTE column `fak wip status` prints. Like the rest of this
// package it carries ZERO git I/O — cmd/fak/wip_sync.go runs the push/fetch and
// hands the ref listings back here — so the refspecs, the mirror grammar, and the
// replication verdict are unit-testable without a repo or a remote.
//
// WHY THIS EXISTS. The sibling lease namespace refs/fak/locks/* has a full
// replication story (internal/leaseref/sync.go, a publish path, an operator
// converge verb, an expiry delete-push, a garden-tick class); refs/fak/wip/* had
// none of it. So the EPHEMERAL record — a lease, worthless the moment its session
// dies — survived a machine loss, and the VALUABLE one — an uncommitted delta
// reachable from no branch — lived on exactly one disk. Worse, "checkpointed" and
// "safe" were the same word: at a compaction boundary an agent could not tell
// whether it was protected against *this session dies* or *this machine goes away*.
//
// WHAT TRANSFERS FROM leaseref.Sync, AND WHAT DOES NOT.
//
//   - THE FORCE REFSPEC TRANSFERS, by a different structural route. Lease refs point
//     at BLOBS, so there is no ancestry and every update of an existing ref needs the
//     force. Checkpoint refs point at COMMITS — but each checkpoint is minted
//     `commit-tree <tree> -p HEAD` (cmd/fak/wip.go), parented on the then-current
//     HEAD and never on the previous checkpoint, so two successive checkpoints of one
//     session are SIBLINGS, not ancestor and descendant. A non-forced update is
//     rejected here exactly as it is there.
//
//   - PUSH-BEFORE-FETCH TRANSFERS; ITS SAFETY RATIONALE DOES NOT. leaseref force-fetches
//     into its OWN live namespace, so fetching first would force-reset a just-acquired
//     local lease to the remote's stale generation — the ordering is a data-loss guard.
//     Here the fetch lands in a MIRROR namespace (below), so it structurally cannot
//     regress a local checkpoint. The ordering survives as a FRESHNESS rule instead:
//     the mirror is the evidence `status` reads offline, and refreshing it after the
//     push is what makes one `fak wip sync` leave behind evidence that describes the
//     state the caller just created. A failed push still stops the sync, so a sync that
//     errors has changed nothing locally and the operator sees one unambiguous failure
//     rather than an error plus a quietly-reclassified status column.
//
//   - THE FETCH DESTINATION DOES NOT TRANSFER, and this is the load-bearing divergence.
//     leaseref force-fetches peers' lease refs into refs/fak/locks/* because a lease is
//     globally meaningful: machine B genuinely wants machine A's lease inside its own
//     arbiter's view. A checkpoint is not. EVERY live reader of refs/fak/wip/* in this
//     repo is TREE-RELATIVE — `wip reap` DELETES refs whose delta has landed in *this*
//     HEAD, `wip reconcile` grades RECLAIM/QUARANTINE by test-applying the delta to
//     *this* working tree, `wip attribute` and `wip sweep-guard` attribute *this*
//     tree's dirty hunks, `wip land` materializes into *this* working tree. Force-
//     fetching a peer host's checkpoints into the live namespace would hand all five a
//     population they cannot reason about, and one of the five is a deleter. So the
//     fetch lands under RemoteMirrorNamespace, read-only evidence that no existing
//     verb enumerates, and the live namespace stays exactly what it is today: this
//     clone's own checkpoints.
//
//   - DELETIONS STILL DO NOT RIDE A REFSPEC, for the same reason as the sibling: a
//     glob push/fetch transports existing refs only, and a prune-style sync would
//     delete live state on the losing side of a window. internal/gitgate already
//     refuses `git push --prune` on precisely that history (#5360).
//
// OPT-IN, DELIBERATELY. Nothing in this package or its cmd shell runs a sync on its
// own. Replicating a dirty working tree off-machine is a real privacy and bandwidth
// decision an operator makes deliberately rather than inherits — a checkpoint is a
// TREE-WIDE capture, so it carries whatever a shared tree happened to be holding.

import (
	"sort"
	"strings"
)

// RemoteMirrorNamespace is where the FETCH side of a sync lands a remote's checkpoint
// refs: refs/fak/remotewip/<remote-segment>/<session>. It deliberately does NOT nest
// under RefNamespace. `git for-each-ref refs/fak/wip` matches a literal pattern
// completely or up to a slash, so a sibling named refs/fak/wip-remote/... would be safe
// by that rule alone — but every wip verb's ref sweep is written against the "refs/fak/wip"
// prefix as a string, and one careless HasPrefix elsewhere would silently feed a peer
// host's checkpoints to the reaper. A prefix that cannot collide by accident is worth
// more than the naming symmetry.
const RemoteMirrorNamespace = "refs/fak/remotewip/"

// mirrorSegmentMax caps the sanitized remote segment. Loose refs are FILES, so an
// unbounded segment from a long remote URL turns into a long path on a platform that
// still has path limits. 64 bytes is well past any real remote name.
const mirrorSegmentMax = 64

// Replication is how one local checkpoint stands against the last synced evidence for
// a remote. Three states, because the middle one is the one that bites: an operator who
// synced once and checkpointed again has an OLDER delta on the remote, and reading that
// as "replicated" is exactly the false comfort this column exists to remove.
type Replication string

const (
	// ReplicationLocalOnly: no evidence this session's checkpoint is on the remote.
	// This checkpoint survives the session dying. It does not survive the machine.
	ReplicationLocalOnly Replication = "LOCAL_ONLY"
	// ReplicationStaleRemote: the remote holds this session's ref at a DIFFERENT object
	// — an earlier checkpoint made it off-machine, the current delta did not.
	ReplicationStaleRemote Replication = "STALE_REMOTE"
	// ReplicationReplicated: the remote holds this session's ref at THIS EXACT object.
	ReplicationReplicated Replication = "REPLICATED"
)

// ValidRemote rejects a remote that cannot safely be one git argv token: empty, a
// leading dash (would misparse as a flag), or an embedded whitespace/control byte. A
// remote NAME (origin) and a remote URL (ssh://..., https://...) both pass — this is
// argv hygiene, not URL validation; git itself decides whether the remote exists. Same
// rule and same reasoning as leaseref's validRemote, restated here rather than imported
// because this package must stay free of the I/O-bearing sibling.
func ValidRemote(remote string) bool {
	if remote == "" || strings.HasPrefix(remote, "-") {
		return false
	}
	for _, c := range []byte(remote) {
		if c <= ' ' || c == 0x7f {
			return false
		}
	}
	return true
}

// MirrorSegment folds a remote (a name or a whole URL) into ONE safe ref path
// component. Everything outside [A-Za-z0-9_-] becomes '-', which kills git's refname
// hazards — a leading '.', an embedded "..", a ".lock" suffix — in a single stroke
// rather than three special cases, at the cost of mangling a URL into something only
// roughly readable. That trade is right because the segment is a lookup key, not a
// display string. Two different remotes CAN mangle to the same segment (an operator
// syncing to two hosts differing only in punctuation); they would then share a mirror
// and the later sync's evidence wins. Naming remotes is the cure, and `origin` — the
// only remote the fleet actually uses — is unchanged by this function.
func MirrorSegment(remote string) string {
	var b strings.Builder
	for _, c := range []byte(remote) {
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9', c == '_', c == '-':
			b.WriteByte(c)
		default:
			b.WriteByte('-')
		}
		if b.Len() >= mirrorSegmentMax {
			break
		}
	}
	if s := b.String(); s != "" {
		return s
	}
	return "remote"
}

// MirrorNamespace is the ref directory a remote's mirrored checkpoints live under,
// trailing slash included so it composes like RefNamespace.
func MirrorNamespace(remote string) string {
	return RemoteMirrorNamespace + MirrorSegment(remote) + "/"
}

// MirrorSessionRef is where one session's mirrored checkpoint lives for a remote.
func MirrorSessionRef(remote, session string) string {
	return MirrorNamespace(remote) + session
}

// SessionFromMirrorRef recovers the session id from a mirrored ref. A ref outside the
// remote's mirror namespace is returned unchanged, matching SessionFromRef's contract.
func SessionFromMirrorRef(remote, ref string) string {
	return strings.TrimPrefix(ref, MirrorNamespace(remote))
}

// PushRefspec is the one refspec the push side ever uses: the whole checkpoint
// namespace, forced, confined to refs/fak/wip/* at BOTH ends. It is NOT safe to hand
// this to git when the namespace is empty: unlike fetch, a zero-match PUSH refspec is
// answered with "No refs in common and none specified; doing nothing." and exit 1, so
// the caller must short-circuit an empty namespace instead of reporting a spurious
// failure on every clone that has not checkpointed yet (see wipSync).
const PushRefspec = "+" + RefNamespace + "*:" + RefNamespace + "*"

// FetchRefspec is the fetch side: the remote's whole checkpoint namespace, forced, into
// THIS clone's read-only mirror for that remote. The source side is refs/fak/wip/* — a
// peer publishes with PushRefspec, so that is where its checkpoints are.
func FetchRefspec(remote string) string {
	return "+" + RefNamespace + "*:" + MirrorNamespace(remote) + "*"
}

// SyncResult reports what one sync actually did — which directions ran, the exact
// refspecs used, and the replication summary the caller can print without a second
// pass. Same role as leaseref.SyncResult: a loop's ledger can record the convergence
// action without re-deriving it.
type SyncResult struct {
	Remote       string `json:"remote"`
	Pushed       bool   `json:"pushed"`
	Fetched      bool   `json:"fetched"`
	PushRefspec  string `json:"push_refspec,omitempty"`
	FetchRefspec string `json:"fetch_refspec,omitempty"`
	// Published is how many local checkpoint refs the push covered — the count whose
	// mirror entries the successful push justifies writing.
	Published int `json:"published"`
	// Mirrored is how many refs the mirror holds for this remote after the sync.
	Mirrored int `json:"mirrored"`
	// Replicated / StaleRemote / LocalOnly summarize the LOCAL checkpoints after the
	// sync, so `fak wip sync` answers "am I off this machine now" in one call.
	Replicated  int `json:"replicated"`
	StaleRemote int `json:"stale_remote"`
	LocalOnly   int `json:"local_only"`
}

// MirrorIndex folds mirrored ref records into the session->object lookup the
// replication verdict reads. Keyed by session id rather than by full ref so a local
// record and a mirrored one compare directly.
func MirrorIndex(remote string, recs []RefRecord) map[string]string {
	if len(recs) == 0 {
		return nil
	}
	idx := make(map[string]string, len(recs))
	for _, r := range recs {
		sess := SessionFromMirrorRef(remote, r.Ref)
		if sess == "" || r.Object == "" {
			continue
		}
		idx[sess] = r.Object
	}
	return idx
}

// ClassifyReplication grades one local checkpoint against the mirror. A nil/empty
// mirror yields LOCAL_ONLY for everything, which is the correct answer when no sync has
// ever run and the SAFE direction to be wrong in when a caller simply did not read the
// mirror: this column may never overstate durability.
func ClassifyReplication(rec RefRecord, mirror map[string]string) (Replication, string) {
	sess := rec.Stamp.SessionID
	if sess == "" {
		sess = SessionFromRef(rec.Ref)
	}
	remoteObj, ok := mirror[sess]
	if !ok || remoteObj == "" {
		return ReplicationLocalOnly, ""
	}
	if remoteObj == rec.Object {
		return ReplicationReplicated, remoteObj
	}
	return ReplicationStaleRemote, remoteObj
}

// FoldWithMirror is Fold plus the replication verdict: the same deterministic,
// session-sorted projection, with each row graded against the mirror and the report
// carrying the three counts. Fold delegates here with a nil mirror, so every existing
// caller keeps its exact behaviour and reads LOCAL_ONLY — which is what it is.
func FoldWithMirror(recs []RefRecord, mirror map[string]string, nowUnix int64) StatusReport {
	out := make([]SessionStatus, 0, len(recs))
	rep := StatusReport{}
	for _, r := range recs {
		sess := r.Stamp.SessionID
		if sess == "" {
			sess = SessionFromRef(r.Ref)
		}
		age := nowUnix - r.Stamp.CheckpointedAt
		if age < 0 {
			age = 0
		}
		leaves := r.Stamp.Leaves
		if leaves == nil {
			leaves = []string{}
		}
		state, remoteObj := ClassifyReplication(r, mirror)
		switch state {
		case ReplicationReplicated:
			rep.Replicated++
		case ReplicationStaleRemote:
			rep.StaleRemote++
		default:
			rep.LocalOnly++
		}
		out = append(out, SessionStatus{
			Session:      sess,
			Object:       r.Object,
			StartSHA:     r.Stamp.StartSHA,
			Leaves:       leaves,
			Buildable:    r.Stamp.Buildable,
			AgeSeconds:   age,
			Replication:  string(state),
			RemoteObject: remoteObj,
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Session < out[j].Session })
	rep.Count = len(out)
	rep.Sessions = out
	return rep
}
