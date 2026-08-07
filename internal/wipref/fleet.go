package wipref

// fleet.go is the FLEET-VISIBILITY half of the checkpoint substrate (#3880, child C8b
// of #3871): the pure decisions behind `fak wip status --fleet` and behind the
// object-size gate `fak wip sync` applies before it publishes. Like the rest of this
// package it carries ZERO git I/O — cmd/fak/wip_fleet.go runs the ref reads and the
// pushes and hands the listings back here.
//
// WHAT #3880 ACTUALLY ADDS, AND WHAT WAS ALREADY THERE. sync.go already publishes
// refs/fak/wip/* on a push refspec and imports a peer's namespace into
// refs/fak/remotewip/<remote>/* on a fetch — the transport half of "fleet-visible" has
// shipped. Two things were missing and both are here:
//
//   - NOBODY ENUMERATED THE MIRROR. Every existing reader of the mirror uses it as a
//     session->object LOOKUP to grade THIS clone's checkpoints (ClassifyReplication), and
//     wipListMirrorRecords deliberately does not even decode a mirrored stamp. So a peer
//     host's crashed-session WIP could be sitting in refs/fak/remotewip/origin/* and no
//     verb in the repo would say so. FoldFleet is the enumeration: every mirrored ref as a
//     row, keyed by OWNER HOST rather than by this clone's replication question.
//
//   - EVERY OBJECT RODE THE PUSH. A lease is a blob of a few hundred bytes; a checkpoint
//     is a TREE-WIDE capture of a dirty working tree and can be megabytes. Publishing all
//     of them unconditionally bloats the one clone the whole fleet shares. PlanPublish is
//     the gate: a delta over the bound publishes a METADATA-ONLY ref — a real ref at
//     refs/fak/wip/<session> carrying the full stamp over an EMPTY tree — so the session
//     stays enumerable fleet-wide while its bytes stay on the owner host.
//
// WHY THE METADATA REF IS A COMMIT OVER AN EMPTY TREE, AND NOT A SMALLER TRICK. git has
// no way to publish a ref without publishing the object it points at, so "ref + metadata
// but not the delta" has to be a DIFFERENT, cheap object. A parentless commit over the
// empty tree transfers exactly one new object (the empty tree is universal), carries the
// stamp in its message the same way a real checkpoint does — so the fleet reader needs no
// second grammar — and is a commit, so nothing downstream has to special-case a ref that
// points at a blob. The stamp it carries is marked MetadataOnly and records
// DeltaObject, the object id the owner host kept, which is what a later cross-host
// reclaim (out of scope here, #C4) needs to ask that host for.
//
// WHY THE FLEET VIEW REFUSES TO GRADE RECLAIMABILITY THE WAY reconcile DOES. `fak wip
// reconcile` grades RECLAIM/QUARANTINE by test-applying the delta to THIS working tree.
// A fleet row is a PEER's delta against a peer's base; test-applying it here would answer
// a question nobody asked and would be wrong the moment the two clones' HEADs differ.
// FleetDisposition is therefore a narrower, closed vocabulary over facts the fleet view
// can actually witness: whose it is, whether the bytes made it here, and nothing more.

import (
	"fmt"
	"sort"
	"strings"
)

// HostUnknown is the owner-host label for a mirrored checkpoint whose stamp carries no
// host — every checkpoint minted before #3880, and any stamp that failed to decode. It
// is a CLASSIFICATION, never an error, matching leaseref.NodeUnknown's tolerance rule:
// the convention is adopted by writers going forward and never enforced retroactively on
// refs already in the namespace.
const HostUnknown = "host-unknown"

// DefaultMaxDeltaBytes is the object-size bound a sync adopts when the operator states
// none: 1 MiB of changed blob content per checkpoint.
//
// It is a bound on the COORDINATOR CLONE's growth, not a judgement about a delta. The
// fleet's shared clone accumulates every host's checkpoints and is never gc-pruned of
// them while the refs live, so the bound has to be small enough that a runaway capture —
// a vendored dump, a build artifact that slipped past .gitignore, a multi-megabyte log —
// cannot silently become everyone's disk. 1 MiB is comfortably above a normal source
// delta (this repo's largest ordinary working-tree delta is tens of KB) and comfortably
// below the artifacts that motivate the gate. An operator who genuinely wants a big
// delta off-machine raises it, or passes 0 for unbounded: the gate is opt-OUT on size,
// never a refusal.
const DefaultMaxDeltaBytes int64 = 1 << 20

// pushRefspecBatch caps how many explicit refspecs ride one `git push` argv. The live
// namespace routinely carries thousands of refs and Windows caps a command line at
// ~32 KB; at ~100 bytes per refspec a batch of 128 is ~13 KB, which leaves room for the
// remote and the verb on every platform this runs on. Batching only ever happens on the
// gated path — an ungated plan still pushes the single glob refspec (see PushRefspecs).
const pushRefspecBatch = 128

// PublishClass is what a sync decides to do with ONE local checkpoint. Two values,
// because the gate is about bytes and never about admission: a checkpoint is always
// published, the only question is whether its delta rides along.
type PublishClass string

const (
	// PublishFull: the delta is within the bound (or its size is unknown) — push the
	// checkpoint object itself, exactly as every sync did before the gate existed.
	PublishFull PublishClass = "FULL"
	// PublishMetadataOnly: the delta is over the bound — publish a metadata-only ref at
	// the same refs/fak/wip/<session> and leave the delta's objects on the owner host.
	// The session stays enumerable fleet-wide; its bytes do not travel.
	PublishMetadataOnly PublishClass = "METADATA_ONLY"
)

// ClassifyPublish grades one checkpoint's stamp against the bound.
//
// maxDeltaBytes <= 0 is UNBOUNDED — the operator opting every delta into the push.
//
// An UNKNOWN size (DeltaBytes 0: a stamp minted before #3880 recorded one, or a delta
// that genuinely changed no bytes) classifies FULL. That is the direction that never
// loses data and the direction that changes nothing for a namespace full of pre-gate
// refs: the gate may not suppress a delta it cannot prove is oversized. The ambiguity
// costs nothing in the other branch either — a zero-byte delta is precisely the object
// there is no point withholding.
func ClassifyPublish(s Stamp, maxDeltaBytes int64) PublishClass {
	if maxDeltaBytes <= 0 || s.DeltaBytes <= 0 || s.DeltaBytes <= maxDeltaBytes {
		return PublishFull
	}
	return PublishMetadataOnly
}

// MetadataStamp derives the stamp a metadata-only ref carries from the real checkpoint's
// stamp: everything a fleet reader needs to identify and rank the work (host, session,
// base, scope, leaves, age, size) plus the two fields that stop it being mistaken for the
// delta itself — MetadataOnly, and DeltaObject naming the object the owner host kept.
func MetadataStamp(s Stamp, object string) Stamp {
	s.MetadataOnly = true
	s.DeltaObject = object
	return s
}

// PublishEntry is one session's publish decision — the row a ledger records and the
// input the refspec builder consumes.
type PublishEntry struct {
	Session    string       `json:"session"`
	Ref        string       `json:"ref"`
	Object     string       `json:"object"`
	Class      PublishClass `json:"class"`
	DeltaBytes int64        `json:"delta_bytes,omitempty"`
}

// PublishPlan is the whole gated decision for one push: the bound it was reached under,
// every session's entry, and the two counts a caller prints without re-tallying.
type PublishPlan struct {
	MaxDeltaBytes int64          `json:"max_delta_bytes"`
	Entries       []PublishEntry `json:"entries"`
	Full          int            `json:"full"`
	MetadataOnly  int            `json:"metadata_only"`
}

// Gated reports whether the plan withheld any delta — i.e. whether the push has to build
// explicit refspecs at all. A plan that gates nothing takes the single-glob path that
// predates #3880, byte for byte.
func (p PublishPlan) Gated() bool { return p.MetadataOnly > 0 }

// PlanPublish grades every local checkpoint against the bound, session-sorted so a plan
// is byte-stable across runs and two clones' ledgers diff cleanly. A record whose session
// is not one safe ref segment, or that carries no object, is DROPPED rather than smuggled
// into a refspec — the same rule wipWriteMirror holds.
func PlanPublish(recs []RefRecord, maxDeltaBytes int64) PublishPlan {
	plan := PublishPlan{MaxDeltaBytes: maxDeltaBytes, Entries: []PublishEntry{}}
	for _, r := range recs {
		sess := r.Stamp.SessionID
		if sess == "" {
			sess = SessionFromRef(r.Ref)
		}
		if !ValidSession(sess) || r.Object == "" {
			continue
		}
		cls := ClassifyPublish(r.Stamp, maxDeltaBytes)
		if cls == PublishMetadataOnly {
			plan.MetadataOnly++
		} else {
			plan.Full++
		}
		plan.Entries = append(plan.Entries, PublishEntry{
			Session:    sess,
			Ref:        SessionRef(sess),
			Object:     r.Object,
			Class:      cls,
			DeltaBytes: r.Stamp.DeltaBytes,
		})
	}
	sort.Slice(plan.Entries, func(i, j int) bool { return plan.Entries[i].Session < plan.Entries[j].Session })
	return plan
}

// PushRefspecs renders the plan as the batches of refspecs a caller hands `git push`,
// one batch per push invocation.
//
// An UNGATED plan returns exactly one batch holding the single glob PushRefspec — the
// identical argv every sync used before the gate existed, so the common case gains no
// per-ref argv and no extra spawn. A GATED plan cannot use the glob (git has no negative
// refspec, so a glob would carry the very objects being withheld), so it names every
// session explicitly: a FULL entry pushes its own ref, a METADATA_ONLY entry pushes the
// substitute object metaObjects[session] to the same destination.
//
// A METADATA_ONLY entry with no minted substitute is an ERROR rather than a silent skip
// or a silent full push: the first would drop a session out of the fleet view without
// saying so, and the second would publish the exact bytes the gate exists to withhold.
func PushRefspecs(plan PublishPlan, metaObjects map[string]string) ([][]string, error) {
	if len(plan.Entries) == 0 {
		return nil, nil
	}
	if !plan.Gated() {
		return [][]string{{PushRefspec}}, nil
	}
	specs := make([]string, 0, len(plan.Entries))
	for _, e := range plan.Entries {
		src := e.Ref
		if e.Class == PublishMetadataOnly {
			obj := strings.TrimSpace(metaObjects[e.Session])
			if obj == "" {
				return nil, fmt.Errorf("session %s was gated to %s but no metadata object was minted for it", e.Session, PublishMetadataOnly)
			}
			src = obj
		}
		specs = append(specs, "+"+src+":"+e.Ref)
	}
	var batches [][]string
	for i := 0; i < len(specs); i += pushRefspecBatch {
		end := i + pushRefspecBatch
		if end > len(specs) {
			end = len(specs)
		}
		batches = append(batches, specs[i:end])
	}
	return batches, nil
}

// FleetDisposition is the closed vocabulary a mirrored checkpoint is classified into.
// It answers only what the fleet view can witness — see this file's header for why it
// deliberately does not reuse reconcile's RECLAIM/QUARANTINE grades.
type FleetDisposition string

const (
	// FleetOwnLocal: the session is one THIS clone still holds in its live namespace.
	// It is this host's own work coming back over the mirror, not a peer's stranded WIP,
	// and `fak wip status` already grades its durability. Classified by SESSION, not by
	// object, so a clone whose local checkpoint has moved on since the last sync still
	// recognizes the row as its own rather than reporting itself as a strandee.
	FleetOwnLocal FleetDisposition = "OWN_LOCAL"
	// FleetMetadataOnly: the owner host published this session's ref but withheld its
	// delta under the size gate. The work is VISIBLE fleet-wide and is NOT recoverable
	// from the coordinator — recovering it means going to the owner host.
	FleetMetadataOnly FleetDisposition = "METADATA_ONLY"
	// FleetObjectMissing: the ref is here and claims a real delta, but the object is not
	// in this clone (a partial fetch, a pruned coordinator, a stamp naming an object the
	// push never carried). Fail-safe: never call a delta reclaimable on the strength of a
	// ref alone.
	FleetObjectMissing FleetDisposition = "OBJECT_MISSING"
	// FleetReclaimable: a PEER host's checkpoint whose delta object is present in this
	// clone — the fleet-wide crash-recovery candidate this whole ticket exists to make
	// enumerable. Landing it is a separate, later act (#C4 `land`); this is the worklist,
	// not the lander.
	FleetReclaimable FleetDisposition = "RECLAIMABLE"
)

// FleetRow is one mirrored checkpoint's public row: whose it is, what it captured, and
// what this clone may do about it.
type FleetRow struct {
	Host        string           `json:"host"`
	Session     string           `json:"session"`
	Ref         string           `json:"ref"`
	Object      string           `json:"object"`
	StartSHA    string           `json:"start_sha,omitempty"`
	Leaves      []string         `json:"leaves"`
	Scope       []string         `json:"scope,omitempty"`
	Buildable   bool             `json:"buildable"`
	AgeSeconds  int64            `json:"age_seconds"`
	DeltaBytes  int64            `json:"delta_bytes,omitempty"`
	Disposition FleetDisposition `json:"disposition"`
	Reason      string           `json:"reason"`
	// DeltaObject is the object the owner host kept back under the size gate, carried on
	// a METADATA_ONLY row so a later cross-host reclaim knows what to ask that host for.
	DeltaObject string `json:"delta_object,omitempty"`
}

// FleetReport is the deterministic fold of a remote's whole mirrored namespace, sorted by
// (host, session) so a fleet snapshot is byte-stable and two syncs diff cleanly.
type FleetReport struct {
	Remote string     `json:"remote"`
	Count  int        `json:"count"`
	Rows   []FleetRow `json:"rows"`
	// The disposition census over Rows. Sums to Count.
	Reclaimable   int `json:"reclaimable"`
	MetadataOnly  int `json:"metadata_only"`
	OwnLocal      int `json:"own_local"`
	ObjectMissing int `json:"object_missing"`
	// Hosts is the sorted set of owner hosts the mirror carries, so an operator sees the
	// fleet's SHAPE without reading every row.
	Hosts []string `json:"hosts"`
	// Mirror is the PROVENANCE of this whole listing (mirrorstamp.go). It is load-bearing
	// HERE in a way it is only advisory in StatusReport: a fleet reader's natural reading
	// of an empty listing is "no peer has stranded work", which is a claim about OTHER
	// HOSTS this clone is entitled to make only after a real FETCH inside tolerance.
	// Consult MirrorView.EmptyIsAbsence before reading Count == 0 as a fact.
	Mirror *MirrorView `json:"mirror,omitempty"`
}

// FleetHost is the owner host a mirrored record reports, falling back to HostUnknown for
// a stamp minted before hosts were recorded.
func FleetHost(s Stamp) string {
	if h := strings.TrimSpace(s.Host); h != "" {
		return h
	}
	return HostUnknown
}

// ClassifyFleet grades one mirrored record. localSessions is the set of session ids this
// clone still holds in refs/fak/wip/*; objectPresent reports whether the record's object
// is readable in this clone. Precedence is ownership first (this clone's own work is
// never a peer strandee), then the owner's own metadata-only declaration, then presence,
// and only then the reclaimable candidate — so every path to RECLAIMABLE has positively
// witnessed both that the work belongs to someone else and that its bytes are here.
func ClassifyFleet(rec RefRecord, localSessions map[string]bool, objectPresent bool) (FleetDisposition, string) {
	sess := rec.Stamp.SessionID
	if sess == "" {
		sess = SessionFromMirrorRefAny(rec.Ref)
	}
	switch {
	case localSessions[sess]:
		return FleetOwnLocal, "this clone's own session — `fak wip status` grades its durability"
	case rec.Stamp.MetadataOnly:
		return FleetMetadataOnly, "owner published the ref but withheld the delta under the size gate — recover from the owner host"
	case !objectPresent:
		return FleetObjectMissing, "ref present, delta object not in this clone — re-sync before treating it as recoverable"
	default:
		return FleetReclaimable, "a peer host's checkpoint with its delta present here — a cross-host recovery candidate"
	}
}

// SessionFromMirrorRefAny recovers the session id from a mirrored ref WITHOUT knowing
// which remote's mirror it came from: the last path component. SessionFromMirrorRef needs
// the remote to strip an exact prefix, which the fold does have — this is the fallback for
// a record whose stamp lost its session id, where a label of last resort beats an empty
// key that would collide every unstamped row onto one another.
func SessionFromMirrorRefAny(ref string) string {
	if i := strings.LastIndex(ref, "/"); i >= 0 {
		return ref[i+1:]
	}
	return ref
}

// FoldFleet projects a remote's mirrored records into the sorted fleet report. Pure — the
// cmd shell reads the mirror, the live session set, and object presence, and this fold
// turns them into the enumeration. A negative age (clock skew between two hosts, which is
// the NORMAL case across a fleet rather than an anomaly) is clamped to 0, matching Fold.
func FoldFleet(remote string, recs []RefRecord, localSessions map[string]bool, objectPresent map[string]bool, nowUnix int64) FleetReport {
	rep := FleetReport{Remote: remote, Rows: make([]FleetRow, 0, len(recs))}
	hosts := map[string]bool{}
	for _, r := range recs {
		sess := r.Stamp.SessionID
		if sess == "" {
			sess = SessionFromMirrorRef(remote, r.Ref)
		}
		if sess == "" || sess == r.Ref {
			sess = SessionFromMirrorRefAny(r.Ref)
		}
		disp, reason := ClassifyFleet(r, localSessions, objectPresent[r.Object])
		switch disp {
		case FleetOwnLocal:
			rep.OwnLocal++
		case FleetMetadataOnly:
			rep.MetadataOnly++
		case FleetObjectMissing:
			rep.ObjectMissing++
		default:
			rep.Reclaimable++
		}
		age := nowUnix - r.Stamp.CheckpointedAt
		if age < 0 || r.Stamp.CheckpointedAt == 0 {
			age = 0
		}
		leaves := r.Stamp.Leaves
		if leaves == nil {
			leaves = []string{}
		}
		host := FleetHost(r.Stamp)
		hosts[host] = true
		rep.Rows = append(rep.Rows, FleetRow{
			Host:        host,
			Session:     sess,
			Ref:         r.Ref,
			Object:      r.Object,
			StartSHA:    r.Stamp.StartSHA,
			Leaves:      leaves,
			Scope:       r.Stamp.Scope,
			Buildable:   r.Stamp.Buildable,
			AgeSeconds:  age,
			DeltaBytes:  r.Stamp.DeltaBytes,
			Disposition: disp,
			Reason:      reason,
			DeltaObject: r.Stamp.DeltaObject,
		})
	}
	sort.Slice(rep.Rows, func(i, j int) bool {
		if rep.Rows[i].Host != rep.Rows[j].Host {
			return rep.Rows[i].Host < rep.Rows[j].Host
		}
		return rep.Rows[i].Session < rep.Rows[j].Session
	})
	rep.Count = len(rep.Rows)
	rep.Hosts = make([]string, 0, len(hosts))
	for h := range hosts {
		rep.Hosts = append(rep.Hosts, h)
	}
	sort.Strings(rep.Hosts)
	return rep
}

// FleetSummary is the one line that makes a fleet listing readable in plain output, and
// the one place the RECLAIMABLE count is allowed to be stated as a finding. It mirrors
// wipReplicationSummary's job on the durability axis: name the count, then name the lever.
func FleetSummary(rep FleetReport) string {
	line := fmt.Sprintf("fleet: %d checkpoint(s) across %d host(s) on %s — %d RECLAIMABLE, %d METADATA_ONLY, %d OWN_LOCAL, %d OBJECT_MISSING",
		rep.Count, len(rep.Hosts), rep.Remote, rep.Reclaimable, rep.MetadataOnly, rep.OwnLocal, rep.ObjectMissing)
	switch {
	case rep.Reclaimable > 0:
		return line + "\n  RECLAIMABLE: a peer host's uncommitted delta is sitting here — `fak wip reconcile` grades it against THIS tree before anything lands."
	case rep.MetadataOnly > 0:
		return line + "\n  METADATA_ONLY: the owner host withheld the delta under the size gate — its bytes are still on that host, not here."
	default:
		return line
	}
}
