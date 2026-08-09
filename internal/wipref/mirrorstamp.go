package wipref

// mirrorstamp.go is the PROVENANCE half of the remote mirror (#5556, deferred from
// #5479): a per-remote record of WHEN this clone last refreshed
// refs/fak/remotewip/<remote>/* and BY WHICH DIRECTION, plus the verdict a reader must
// consult before it reads an empty mirror as a fact. Like the rest of this package it
// carries ZERO git I/O — cmd/fak/wip_sync.go writes and reads the stamp ref and hands the
// decoded stamp back here.
//
// WHY A TIMESTAMP IS LOAD-BEARING AND NOT DECORATION. `fak wip status --remote R` grades
// replication from the mirror, and the mirror is a LOCAL ref read — deliberately never a
// network probe (see wipMirrorIndex). That makes the mirror EVIDENCE, and evidence has an
// as-of date. Without one, a mirror that holds nothing for a session is two completely
// different facts wearing the same face:
//
//   - ABSENCE. This clone fetched R minutes ago and R genuinely holds no checkpoint for
//     that session. Rendering "no checkpoints" is then correct.
//   - IGNORANCE. Nobody on this clone has fetched R since Tuesday, or ever. The mirror is
//     silent because nobody asked, not because there was nothing to hear.
//
// ClassifyReplication already refuses to overstate DURABILITY: an unread mirror grades
// every local checkpoint LOCAL_ONLY, the pessimistic answer, so a caller that forgot to
// read the mirror is told its work is at risk rather than told it is safe. A reader
// looking OUTWARD across remotes has the mirror image of that problem and inherits none
// of the guard: its natural rendering of an empty mirror is "that peer has no
// checkpoints", which overstates KNOWLEDGE instead of durability, and it does so most
// confidently exactly when the clone is least informed. MirrorView.EmptyIsAbsence is the
// one field such a reader must consult before printing a zero, and it is false unless a
// real FETCH landed inside the caller's stated tolerance.
//
// WHY THE SOURCE MATTERS AS MUCH AS THE TIME. wipWriteMirror (cmd/fak/wip_sync.go)
// populates the mirror after a successful PUSH without ever asking the remote what else
// it holds: a completed push of PushRefspec proves the remote now carries THIS clone's
// objects, and proves precisely nothing about a peer's. So a clone that only ever runs
// `fak wip sync --push-only` accumulates a mirror that is perfectly fresh and describes
// only itself. A timestamp alone would present that mirror as an authoritative survey of
// the whole remote. Source is what stops a publication from masquerading as a census, and
// it is why the stamp is two fields rather than one.
//
// WHY IT IS ITS OWN REF NAMESPACE, AND WHY THAT NAME. The stamp cannot live under
// MirrorNamespace: everything there is enumerated by wipListMirrorRecords and keyed by
// session id, so a stamp ref would appear as a phantom peer session in the replication
// index. It also must not be a STRING extension of "refs/fak/wip" or "refs/fak/remotewip"
// — sync.go chose refs/fak/remotewip/ precisely so that one careless HasPrefix elsewhere
// could not feed a peer's checkpoints to the reaper, and a sibling like
// refs/fak/wipstamp/ would hand that hazard straight back. refs/fak/checkpointsync/ is a
// prefix neither existing sweep can reach by accident in either matching style.
//
// The stamp is LOCAL-ONLY by construction: PushRefspec is confined to refs/fak/wip/* at
// both ends, so nothing here is ever published, and a peer's stamp — which describes that
// peer's picture, not this one's — can never arrive over a fetch.

import (
	"encoding/json"
	"fmt"
	"strings"
)

// MirrorStampNamespace is where this clone records each remote's mirror provenance:
// refs/fak/checkpointsync/<remote-segment>, one ref per remote, pointing at a blob
// holding the stamp JSON. See the file comment for why the prefix is deliberately
// unreachable from both checkpoint namespaces' sweeps.
const MirrorStampNamespace = "refs/fak/checkpointsync/"

// mirrorStampMarker prefixes the single stamp line inside the blob. The blob is written
// by this clone alone, so the marker is not there to tolerate prose the way
// wipref.stampMarker is — it is there so DecodeMirrorStamp REFUSES a ref that was pointed
// at some other object, rather than decoding whatever JSON it happens to find and
// reporting a fabricated sync time.
const mirrorStampMarker = "fak-wip-mirror: "

// DefaultMirrorMaxAgeSeconds is the staleness tolerance a reader gets when it states
// none. It is not a taste call: it is the repo's one registered background refresh
// cadence, `fak garden tick`'s gardenTickIntervalSeconds = 3600 (cmd/fak/garden.go). A
// mirror older than that has outlived a full tick without being refreshed, so whatever
// was supposed to keep this clone's picture current did not run or did not fetch. A
// reader with a different appetite passes its own maxAge; nothing in this file treats
// 3600 as a truth about how fast checkpoints change.
const DefaultMirrorMaxAgeSeconds int64 = 3600

// MirrorSource names what put the mirror into the state a stamp describes. The two
// values are not two flavours of the same act — they cover different populations, which
// is the whole reason the field exists.
type MirrorSource string

const (
	// MirrorFromFetch: a completed `git fetch` of FetchRefspec. It surveyed the remote's
	// WHOLE checkpoint namespace, so the mirror afterwards is evidence about every session
	// on that remote, this clone's and every peer's — including the sessions that are not
	// there.
	MirrorFromFetch MirrorSource = "FETCH"
	// MirrorFromPush: a completed push of PushRefspec, recorded from this clone's own ref
	// listing without asking the remote what else it holds. Evidence about THIS clone's
	// sessions only. An empty mirror under this source says nothing whatsoever about a
	// peer, however recent the stamp is.
	MirrorFromPush MirrorSource = "PUSH"
)

// MirrorStamp is one remote's mirror provenance as this clone last recorded it.
type MirrorStamp struct {
	// Remote is the remote the recording sync named. It is checked against the remote
	// being asked about, because MirrorSegment can fold two different remotes onto one
	// ref segment (see its doc comment) and a stamp inherited from the other one is not
	// evidence about this one.
	Remote string `json:"remote"`
	// Source is FETCH or PUSH — see MirrorSource. An unrecognized value is treated as
	// unknown provenance and never grants EmptyIsAbsence.
	Source MirrorSource `json:"source"`
	// SyncedAt is the unix second the recording sync completed. Zero or negative means
	// "no usable stamp", which classifies as NEVER_SYNCED rather than as the epoch.
	SyncedAt int64 `json:"synced_at"`
	// Refs is how many refs the mirror held when the stamp was written — what the sync
	// actually observed, so a later reader can tell "the fetch saw nothing" from "the
	// mirror was emptied after the fetch".
	Refs int `json:"refs"`
}

// EncodeMirrorStamp renders s as the blob body the stamp ref points at: the marker
// followed by compact JSON and a trailing newline. Round-trips through DecodeMirrorStamp.
func EncodeMirrorStamp(s MirrorStamp) (string, error) {
	b, err := json.Marshal(s)
	if err != nil {
		return "", err
	}
	return mirrorStampMarker + string(b) + "\n", nil
}

// DecodeMirrorStamp extracts a MirrorStamp from a stamp blob's body. ok=false means the
// body carried no parseable marker line — the caller must then treat the mirror as
// NEVER_SYNCED, never as "synced at time zero".
func DecodeMirrorStamp(body string) (MirrorStamp, bool) {
	for _, line := range strings.Split(body, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, mirrorStampMarker) {
			continue
		}
		var s MirrorStamp
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, mirrorStampMarker)), &s); err == nil {
			return s, true
		}
	}
	return MirrorStamp{}, false
}

// MirrorStampRef is the ref one remote's stamp lives at. Keyed by the SAME sanitized
// segment as MirrorNamespace, so the stamp and the mirror it describes cannot drift onto
// different keys.
func MirrorStampRef(remote string) string {
	return MirrorStampNamespace + MirrorSegment(remote)
}

// MirrorFreshness is how old this clone's picture of a remote is. Three states, and the
// first one is the point of the whole file: "I have never looked" must never render as
// "there is nothing there".
type MirrorFreshness string

const (
	// MirrorNeverSynced: no usable stamp for this remote — no sync has ever completed
	// here, or the stamp is unreadable, or it belongs to a different remote that folds
	// onto the same ref segment. Whatever the mirror holds or fails to hold under this
	// verdict is ignorance, not measurement.
	MirrorNeverSynced MirrorFreshness = "NEVER_SYNCED"
	// MirrorFresh: a sync completed within the caller's tolerance.
	MirrorFresh MirrorFreshness = "FRESH"
	// MirrorStale: a sync completed, but longer ago than the caller is willing to treat
	// as current. The mirror is still the best evidence available — it is simply evidence
	// about the past, and a gap in it may be a gap in this clone's attention.
	MirrorStale MirrorFreshness = "STALE"
)

// MirrorView is one remote's provenance row: what this clone knows about its OWN picture
// of a remote, kept strictly separate from anything that picture contains.
type MirrorView struct {
	Remote    string `json:"remote"`
	Namespace string `json:"namespace"`
	// Freshness / Source / SyncedAt describe the last completed sync, when there was one.
	Freshness MirrorFreshness `json:"freshness"`
	Source    MirrorSource    `json:"source,omitempty"`
	SyncedAt  int64           `json:"synced_at,omitempty"`
	// AgeSeconds is -1, never 0, when there is no usable stamp. A 0 there would render as
	// "just now" in every naive consumer — the precise overstatement this type exists to
	// prevent — whereas a negative age is unmistakably not an age.
	AgeSeconds int64 `json:"age_seconds"`
	// MaxAgeSeconds is the tolerance this verdict was reached under, carried so a
	// downstream reader can see which yardstick produced FRESH rather than assume one.
	MaxAgeSeconds int64 `json:"max_age_seconds"`
	// Mirrored is how many refs the mirror holds for this remote right now. It is a COUNT,
	// not a finding: consult EmptyIsAbsence before saying anything about what a zero means.
	Mirrored int `json:"mirrored"`
	// StampedRemote is set only when the stamp found at this remote's ref names a
	// DIFFERENT remote — two remotes sanitized onto one segment. It is reported rather
	// than silently ignored because the operator's cure (name the remotes distinctly)
	// cannot be found from a bare NEVER_SYNCED.
	StampedRemote string `json:"stamped_remote,omitempty"`
	// EmptyIsAbsence is the single question a reader must ask before rendering Mirrored as
	// a fact about the remote: may a missing entry here be reported as "the remote does not
	// have it"? True only when a real FETCH — which surveys the remote's whole namespace —
	// completed inside MaxAgeSeconds. False under NEVER_SYNCED (never looked), under STALE
	// (looked, long ago), and under a PUSH-sourced stamp (published, never surveyed).
	EmptyIsAbsence bool `json:"empty_is_absence"`
}

// ClassifyMirror grades one remote's mirror provenance. ok=false — no stamp ref, or a
// blob that did not decode — is NEVER_SYNCED, the same safe direction ClassifyReplication
// takes for a mirror it could not read. maxAgeSeconds <= 0 adopts
// DefaultMirrorMaxAgeSeconds.
//
// A stamp dated in the FUTURE is clamped to age 0 rather than rejected, matching Fold's
// treatment of a future checkpoint stamp: the stamp is written by this clone, so a
// forward-dated one is this machine's clock moving, not a claim from elsewhere.
func ClassifyMirror(remote string, st MirrorStamp, ok bool, mirrored int, nowUnix, maxAgeSeconds int64) MirrorView {
	if maxAgeSeconds <= 0 {
		maxAgeSeconds = DefaultMirrorMaxAgeSeconds
	}
	v := MirrorView{
		Remote:        remote,
		Namespace:     MirrorNamespace(remote),
		Freshness:     MirrorNeverSynced,
		AgeSeconds:    -1,
		MaxAgeSeconds: maxAgeSeconds,
		Mirrored:      mirrored,
	}
	if !ok || st.SyncedAt <= 0 {
		return v
	}
	if st.Remote != "" && st.Remote != remote {
		v.StampedRemote = st.Remote
		return v
	}
	age := nowUnix - st.SyncedAt
	if age < 0 {
		age = 0
	}
	v.Source = st.Source
	v.SyncedAt = st.SyncedAt
	v.AgeSeconds = age
	if age > maxAgeSeconds {
		v.Freshness = MirrorStale
	} else {
		v.Freshness = MirrorFresh
	}
	v.EmptyIsAbsence = v.Freshness == MirrorFresh && st.Source == MirrorFromFetch
	return v
}

// MirrorCaveat is the sentence a reader MUST print next to any mirror-derived count that
// cannot be read as a fact about the remote, and the empty string when it can. It exists
// so the distinction survives into PLAIN output: a JSON-only EmptyIsAbsence would leave
// the human-facing column saying "no checkpoints" with the qualification one flag away,
// which is the same shape of failure as "checkpointed" and "safe" having been one word.
//
// Each branch names the cure, because the states differ in what the operator should do:
// a colliding segment wants the remotes renamed, a push-only mirror wants a fetch, and a
// stale one wants a re-sync.
func MirrorCaveat(v MirrorView) string {
	if v.EmptyIsAbsence {
		return ""
	}
	switch {
	case v.StampedRemote != "":
		return fmt.Sprintf("the mirror ref for %q was last written for remote %q — two remotes fold onto one ref segment, so nothing here describes %q; give the remotes distinct names and re-sync.",
			v.Remote, v.StampedRemote, v.Remote)
	case v.Freshness == MirrorNeverSynced:
		return fmt.Sprintf("this clone has never synced with %s, so an empty mirror is IGNORANCE, not absence — run `fak wip sync --remote %s`.", v.Remote, v.Remote)
	case v.Source != MirrorFromFetch:
		return fmt.Sprintf("this clone has only PUBLISHED to %s (last sync %ds ago), never surveyed it: the mirror describes this clone's own sessions and says nothing about a peer's — run `fak wip sync --remote %s` to fetch.",
			v.Remote, v.AgeSeconds, v.Remote)
	case v.Freshness == MirrorStale:
		return fmt.Sprintf("this clone last fetched %s %ds ago, past the %ds tolerance — a checkpoint missing from the mirror may be staleness, not absence; re-sync before reading a zero as a fact.",
			v.Remote, v.AgeSeconds, v.MaxAgeSeconds)
	default:
		// Unreachable from ClassifyMirror — every unlicensed verdict it produces is one of
		// the four above. It is here so a HAND-BUILT view that withholds the licence still
		// gets a caveat rather than silently reading as licensed prose.
		return fmt.Sprintf("this clone's picture of %s cannot be treated as a survey; re-sync before reading anything missing from the mirror as missing from the remote.", v.Remote)
	}
}
