package gpulease

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"time"
)

// ScaleOutVerdict is the closed set of bounded outcomes a second GPU-heavy
// launch may receive when it could not acquire the exclusive lease (#1549).
//
// The premise is that a bare refusal ("stop the holder process and retry") is
// not a decision: it forces an operator to choose between killing a legitimate
// peer and giving up. A bounded decision names one of three honest moves and
// carries the evidence for it. The vocabulary is closed so a caller can switch
// on it without parsing prose.
type ScaleOutVerdict string

const (
	// ScaleOutAttachOwner: the GPU-owning serving process is known to be
	// accepting sessions, so the launch is admitted as a client/peer rather
	// than as a second GPU owner. This is the only verdict that does NOT
	// require the caller to retry or give up.
	ScaleOutAttachOwner ScaleOutVerdict = "ATTACH_OWNER"
	// ScaleOutWaitWithBound: the holder is making progress and is expected to
	// release, and the caller supplied a finite bound it is willing to queue.
	// RetryAfter names that bound.
	ScaleOutWaitWithBound ScaleOutVerdict = "WAIT_WITH_BOUND"
	// ScaleOutRefuse: no attachable owner and the wait cannot be bounded (or
	// the evidence is unreadable). The refusal names the holder and the reason;
	// it is never a silent CPU fallback.
	ScaleOutRefuse ScaleOutVerdict = "TRUTHFUL_REFUSE"
)

// OwnerCapacity is the read-only description of the GPU-owning serving process.
// It mirrors NPULeaseManager.CurrentResident: a read-only surface that reports
// what is resident now, never mutating the lease.
type OwnerCapacity struct {
	Held     bool           `json:"held"`
	PID      int            `json:"pid"`
	Progress HolderProgress `json:"progress"`
	// Attachable is true when the owner exposes a documented co-serving path
	// AND has at least one free session slot. Default false: the flock is not
	// by itself an attach surface. A held lease with no published capacity is
	// NOT attachable.
	Attachable bool `json:"attachable"`
	// Capacity is the total session slots the owner reports (0 = unknown/none).
	Capacity int `json:"capacity"`
	// InUse is the number of slots currently admitted (0 = unknown).
	InUse int `json:"in_use"`
	// Detail is a one-line human explanation of the evidence considered.
	Detail string `json:"detail,omitempty"`
}

// FreeSlots is the number of session slots the owner can still admit. It is 0
// when capacity is unknown/none, which is the fail-closed reading.
func (c OwnerCapacity) FreeSlots() int {
	if c.Capacity <= 0 || c.InUse < 0 {
		return 0
	}
	free := c.Capacity - c.InUse
	if free < 0 {
		return 0
	}
	return free
}

// scaleOutSchema is the schema stamped on every ScaleOutReceipt. It is a stable
// family prefix a consumer can switch on across generations.
const scaleOutSchema = "fak.gpulease-scaleout/1"

// DefaultWaitBound is the wait a WAIT_WITH_BOUND decision advertises when the
// caller's own bound is larger. A holder is expected to release within this
// window; a caller willing to wait longer still queues in bounded chunks so it
// can re-decide rather than block forever.
const DefaultWaitBound = 120 * time.Second

// ScaleOutReceipt is the observable, serializable record of a bounded decision.
// It is the typed replacement for the bare refusal text: every field is either
// evidence read from the lease/sidecar or a value derived from it.
type ScaleOutReceipt struct {
	Schema            string          `json:"schema"`
	Verdict           ScaleOutVerdict `json:"verdict"`
	Reason            string          `json:"reason"`
	Path              string          `json:"path"`
	Holder            OwnerCapacity   `json:"holder"`
	RetryAfter        string          `json:"retry_after,omitempty"`
	RetryAfterSeconds int             `json:"retry_after_seconds,omitempty"`
	FreeSlots         int             `json:"free_slots,omitempty"`
	At                time.Time       `json:"at"`
}

// JSON renders the receipt as stable single-line JSON. It is single-line (not
// MarshalIndent) so it can be appended to one error line and still round-trip
// through json.Unmarshal.
func (r ScaleOutReceipt) JSON() (string, error) {
	b, err := json.Marshal(r)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// String is the one-line operator summary appended to the launcher refusal.
func (r ScaleOutReceipt) String() string {
	base := fmt.Sprintf("gpulease scale-out decision %s: %s", r.Verdict, r.Reason)
	switch r.Verdict {
	case ScaleOutWaitWithBound:
		return fmt.Sprintf("%s (retry after %s)", base, r.RetryAfter)
	case ScaleOutAttachOwner:
		return fmt.Sprintf("%s (attach to %s; %d free slot(s))", base, FormatHolderPID(r.Holder.PID), r.FreeSlots)
	default:
		return base
	}
}

// ScaleOutOptions configures DecideScaleOut.
type ScaleOutOptions struct {
	// Path is the lockfile to inspect. Empty means DefaultPath().
	Path string
	// WaitBound is the maximum the caller is willing to queue. Zero means the
	// caller cannot bound the wait, so a progressing-but-unattachable holder
	// yields TRUTHFUL_REFUSE rather than WAIT_WITH_BOUND. This is the fail-closed
	// default: a caller that cannot wait must not be told to.
	WaitBound time.Duration
	// OwnerCapacityProbe reads the co-serving attach metadata a serving owner
	// publishes (see PublishOwnerCapacity). Nil falls back to ReadOwnerCapacity;
	// a nil probe is NOT the same as "no attach surface" because a real owner
	// that publishes is always discoverable.
	OwnerCapacityProbe func(path string) (OwnerCapacity, bool)
	// Now overrides the clock (tests only). Nil means time.Now.
	Now func() time.Time
	// Progress overrides the holder probe (tests only). Nil means the real
	// ProbeHolderProgress with the real clock. When set, DecideScaleOut does NOT
	// itself sample the holder; the injected probe is the sole evidence.
	Progress func(path string) HolderProbe
}

// DecideScaleOut returns the typed, bounded decision for a second GPU-heavy
// launch that could not acquire the exclusive lease. It is READ-ONLY: it never
// takes, releases, or otherwise mutates the lease. err may be the ErrBusy from
// Acquire (used to name the holder); when err is nil it probes the live lease.
//
// Decision rules (deterministic, fail-closed):
//
//  1. Probe the holder. When err is nil and the lease is not held, or the probe
//     reports DEAD, the holder is gone: the only truthful advice is retry now.
//     WAIT_WITH_BOUND with a 0-second bound when the caller can wait at all,
//     else TRUTHFUL_REFUSE naming "retry". ATTACH_OWNER is wrong because there
//     is no owner to attach to.
//  2. ATTACH_OWNER requires positive evidence: a probe reports Attachable with
//     at least one free slot. A held lease with no published capacity is NOT
//     attachable, so this never fires on the flock alone.
//  3. WAIT_WITH_BOUND requires a holder that is LIVE_PROGRESSING AND a caller
//     that supplied WaitBound > 0. The advertised bound is the smaller of the
//     caller's bound and DefaultWaitBound.
//  4. Everything else (STALLED, UNKNOWN, unreadable holder, progressing but no
//     bound) is TRUTHFUL_REFUSE. An unknown/unreadable holder must never be
//     upgraded to ATTACH_OWNER or WAIT: missing evidence is not evidence.
func DecideScaleOut(err error, opts ScaleOutOptions) ScaleOutReceipt {
	nowFn := opts.Now
	if nowFn == nil {
		nowFn = time.Now
	}
	path := opts.Path
	if path == "" {
		path = DefaultPath()
	}

	probe := probeForScaleOut(path, opts)

	rec := ScaleOutReceipt{
		Schema: scaleOutSchema,
		Path:   path,
		Holder: OwnerCapacity{Held: probe.Held, PID: probe.PID, Progress: probe.Verdict},
		At:     nowFn().UTC(),
	}

	// Rung 1: holder gone. err==nil means the caller is deciding against a live
	// lease rather than a BusyError, so a free lease is the retry-now case.
	deadOrFree := probe.Verdict == HolderProgressDead || (err == nil && !probe.Held)
	if deadOrFree {
		if opts.WaitBound > 0 {
			rec.Verdict = ScaleOutWaitWithBound
			rec.Reason = "holder is dead/free; retry immediately"
			rec.RetryAfter = "0s"
			rec.RetryAfterSeconds = 0
			return rec
		}
		rec.Verdict = ScaleOutRefuse
		rec.Reason = "holder gone, retry"
		return rec
	}

	// Rung 2: a documented attach path with capacity is the only positive
	// attachable evidence. Checked before the wait bound because attaching is
	// strictly better than queueing.
	if owner, ok := scaleOutOwnerCapacity(path, opts); ok && owner.Attachable && owner.FreeSlots() > 0 {
		rec.Verdict = ScaleOutAttachOwner
		rec.Holder = owner
		rec.Holder.Held = true
		rec.Holder.PID = probe.PID
		rec.Holder.Progress = probe.Verdict
		rec.FreeSlots = owner.FreeSlots()
		rec.Reason = fmt.Sprintf("owner %s is accepting sessions with %d free slot(s)%s",
			FormatHolderPID(probe.PID), owner.FreeSlots(), ownerDetailSuffix(owner.Detail))
		return rec
	}

	// Rung 3: a progressing holder plus a caller-supplied bound is a bounded wait.
	if probe.Verdict == HolderProgressLiveProgressing && opts.WaitBound > 0 {
		bound := opts.WaitBound
		if bound > DefaultWaitBound {
			bound = DefaultWaitBound
		}
		rec.Verdict = ScaleOutWaitWithBound
		rec.Reason = fmt.Sprintf("holder %s is progressing (%s); queued within a %s bound",
			FormatHolderPID(probe.PID), probe.Detail, bound)
		rec.RetryAfter = bound.String()
		rec.RetryAfterSeconds = int(bound / time.Second)
		return rec
	}

	// Rung 4: fail-closed refusal. Name the holder, the progress verdict, and the
	// concrete reason no attach/wait decision was possible. Never claim a silent
	// CPU fallback: the caller must decide explicitly.
	rec.Verdict = ScaleOutRefuse
	rec.Reason = fmt.Sprintf("holder %s (%s): %s",
		FormatHolderPID(probe.PID), probe.Verdict, scaleOutRefusalWhy(probe, opts))
	return rec
}

// probeForScaleOut returns the holder evidence, preferring an injected probe so
// tests can drive every verdict without a live holder. The probe is read-only in
// every branch.
func probeForScaleOut(path string, opts ScaleOutOptions) HolderProbe {
	if opts.Progress != nil {
		return opts.Progress(path)
	}
	return ProbeHolderProgress(HolderProgressOptions{Path: path})
}

// scaleOutOwnerCapacity reads the owner's published capacity, falling back to the
// sidecar reader. A nil probe is not "no attach path": a real owner that publishes
// is always discoverable, so wiring ReadOwnerCapacity here is what makes an
// unmodified serving owner attachable.
func scaleOutOwnerCapacity(path string, opts ScaleOutOptions) (OwnerCapacity, bool) {
	if opts.OwnerCapacityProbe != nil {
		return opts.OwnerCapacityProbe(path)
	}
	rec, ok := ReadOwnerCapacity(path)
	if !ok {
		return OwnerCapacity{}, false
	}
	return rec.AsOwnerCapacity(), true
}

func ownerDetailSuffix(detail string) string {
	if detail == "" {
		return ""
	}
	return "; " + detail
}

// scaleOutRefusalWhy names the concrete reason no attach/wait decision was
// reachable, keeping the three causes distinct so an operator can act.
func scaleOutRefusalWhy(probe HolderProbe, opts ScaleOutOptions) string {
	switch probe.Verdict {
	case HolderProgressLiveProgressing:
		if opts.WaitBound <= 0 {
			return "holder is progressing but the caller supplied no wait bound, and no owner advertised an attach surface; wait for it to exit or re-run with a wait bound"
		}
	case HolderProgressStalled:
		return "holder looks stalled and no owner advertised an attach surface; inspect it with 'fak doctor serve' before releasing"
	case HolderProgressUnknown:
		return "holder progress is unreadable and no owner advertised an attach surface; cannot bound the wait"
	}
	return "no attachable owner and the wait cannot be bounded"
}

// OwnerCapacitySuffix is the sidecar a serving OWNER writes beside its lease to
// advertise live session capacity and a documented co-serving attach endpoint.
const OwnerCapacitySuffix = ".owner"

// OwnerCapacityStaleAfter is the age past which a capacity sidecar is treated as
// stale: a serving owner that stopped updating is no longer evidence of a live
// attach surface, so a stale sidecar must not make a dead owner look attachable.
const OwnerCapacityStaleAfter = 30 * time.Second

// OwnerCapacityPath returns the sidecar path for the lockfile at path.
func OwnerCapacityPath(path string) string {
	if path == "" {
		path = DefaultPath()
	}
	return path + OwnerCapacitySuffix
}

// OwnerCapacityRecord is the JSON sidecar a serving OWNER writes next to the
// lease to advertise its live session capacity + a documented attach endpoint.
type OwnerCapacityRecord struct {
	Schema     string    `json:"schema"`
	PID        int       `json:"pid"`
	Capacity   int       `json:"capacity"`
	InUse      int       `json:"in_use"`
	AttachHint string    `json:"attach_hint,omitempty"`
	UpdatedAt  time.Time `json:"updated_at"`
}

// ownerCapacitySchema is the schema stamped on every published sidecar.
const ownerCapacitySchema = "fak.gpulease-owner-capacity/1"

// AsOwnerCapacity converts a published record into the read-only OwnerCapacity a
// decision consumes. Attachable requires BOTH a positive capacity with a free
// slot AND a documented attach hint: capacity without an endpoint is not an
// attach surface.
func (r OwnerCapacityRecord) AsOwnerCapacity() OwnerCapacity {
	return OwnerCapacity{
		Held:       true,
		PID:        r.PID,
		Capacity:   r.Capacity,
		InUse:      r.InUse,
		Attachable: r.AttachHint != "" && r.Capacity > 0 && r.Capacity-r.InUse > 0,
		Detail:     "owner capacity sidecar",
	}
}

// PublishOwnerCapacity writes the sidecar atomically (tmp+rename). Best-effort:
// a failure only means a later decision probes closer to unknown, never that the
// lease itself is affected.
func PublishOwnerCapacity(path string, rec OwnerCapacityRecord) error {
	if rec.Schema == "" {
		rec.Schema = ownerCapacitySchema
	}
	if rec.UpdatedAt.IsZero() {
		rec.UpdatedAt = time.Now().UTC()
	}
	b, err := json.Marshal(rec)
	if err != nil {
		return err
	}
	dst := OwnerCapacityPath(path)
	tmp := dst + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	if err := os.Rename(tmp, dst); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}

// ClearOwnerCapacity removes the sidecar. A missing file is success: the
// post-condition (no sidecar) already holds.
func ClearOwnerCapacity(path string) error {
	if err := os.Remove(OwnerCapacityPath(path)); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return nil
}

// ReadOwnerCapacity reads and validates the sidecar. ok=false on any error, a
// schema mismatch, or a stale UpdatedAt (older than OwnerCapacityStaleAfter).
// It is read-only and fail-closed: a corrupt or stale sidecar never yields an
// attachable owner.
func ReadOwnerCapacity(path string) (OwnerCapacityRecord, bool) {
	b, err := os.ReadFile(OwnerCapacityPath(path))
	if err != nil {
		return OwnerCapacityRecord{}, false
	}
	var rec OwnerCapacityRecord
	if err := json.Unmarshal(b, &rec); err != nil {
		return OwnerCapacityRecord{}, false
	}
	if rec.Schema != ownerCapacitySchema {
		return OwnerCapacityRecord{}, false
	}
	if rec.UpdatedAt.IsZero() || time.Since(rec.UpdatedAt) > OwnerCapacityStaleAfter {
		return OwnerCapacityRecord{}, false
	}
	if rec.Capacity < 0 || rec.InUse < 0 {
		return OwnerCapacityRecord{}, false
	}
	return rec, true
}
