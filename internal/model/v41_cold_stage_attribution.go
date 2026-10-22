package model

import (
	"errors"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// V41ColdStage names one exclusive region of the cold (full-history) V4.1
// forward. Regions never nest: entering one closes the previous, so the stage
// rows plus Unassigned partition every counter movement and host nanosecond
// the session observed since attribution was enabled.
type V41ColdStage uint8

const (
	V41ColdStageEmbedding V41ColdStage = iota
	V41ColdStageLayerSetup
	V41ColdStageEngram
	V41ColdStageMHCPre
	V41ColdStageQKV
	V41ColdStageCompressor
	V41ColdStageIndexer
	V41ColdStagePublish
	V41ColdStageAttention
	V41ColdStageOutProj
	V41ColdStageFFNIn
	V41ColdStageRouter
	V41ColdStageRouted
	V41ColdStageSharedPost
	V41ColdStageLayerHidden
	V41ColdStageHead
	V41ColdStageCount
)

const v41ColdStageNone = V41ColdStageCount

var v41ColdStageNames = [V41ColdStageCount]string{
	"embedding", "layer_setup", "engram", "mhc_pre", "qkv", "compressor", "indexer", "attn_publish",
	"attention", "out_proj", "ffn_in", "router", "routed", "shared_post", "layer_hidden", "head",
}

func (s V41ColdStage) String() string {
	if s < V41ColdStageCount {
		return v41ColdStageNames[s]
	}
	return "unassigned"
}

// V41ColdStageCounters is one region's accumulated host wall and backend
// deltas. Entries counts region entries; Unassigned has none. Counters follow
// compute.BackendCounterSnapshot; DeviceSubmits is the backend's always-on
// submit count; DeviceDispatches and SubmitWaitNanos are the always-on
// compute.DeviceWork reads.
type V41ColdStageCounters struct {
	Entries          uint64
	HostWallNanos    int64
	Counters         compute.BackendCounterSnapshot
	DeviceSubmits    uint64
	DeviceDispatches uint64
	SubmitWaitNanos  uint64
}

// V41ColdStageAttribution is a cumulative, closed partition: Stages plus
// Unassigned sum to every delta and host nanosecond between the enabling read
// and this snapshot's read. Unobserved families read zero, never partial.
type V41ColdStageAttribution struct {
	// CountersObserved covers Counters; DispatchCountsLive additionally
	// covers its dispatch and submit fields, which a profile-off Vulkan
	// process leaves at zero.
	CountersObserved   bool
	DispatchCountsLive bool
	// DeviceSubmitsObserved covers DeviceSubmits, which unlike
	// Counters.DispatchSubmits stays live with the dispatch profile off.
	DeviceSubmitsObserved bool
	// DeviceWorkObserved covers DeviceDispatches and SubmitWaitNanos;
	// SubmitWaitComplete mirrors compute.DeviceWork.SubmitWaitObserved, so
	// false makes SubmitWaitNanos a lower bound.
	DeviceWorkObserved bool
	SubmitWaitComplete bool
	Stages             [V41ColdStageCount]V41ColdStageCounters
	Unassigned         V41ColdStageCounters
}

// ErrV41ColdStageWindow refuses a window difference whose endpoints come from
// different observability or whose cumulative rows went backwards.
var ErrV41ColdStageWindow = errors.New("model: V4.1 cold-stage attribution window is inconsistent")

// Total sums every stage row and Unassigned.
func (a V41ColdStageAttribution) Total() V41ColdStageCounters {
	total := a.Unassigned
	for _, row := range a.Stages {
		total = total.add(row)
	}
	return total
}

// Sub returns the rows accumulated between before and a.
func (a V41ColdStageAttribution) Sub(before V41ColdStageAttribution) (V41ColdStageAttribution, error) {
	if a.CountersObserved != before.CountersObserved || a.DispatchCountsLive != before.DispatchCountsLive ||
		a.DeviceSubmitsObserved != before.DeviceSubmitsObserved || a.DeviceWorkObserved != before.DeviceWorkObserved || a.SubmitWaitComplete != before.SubmitWaitComplete {
		return V41ColdStageAttribution{}, ErrV41ColdStageWindow
	}
	out := a
	var ok bool
	for i := range a.Stages {
		if out.Stages[i], ok = a.Stages[i].sub(before.Stages[i]); !ok {
			return V41ColdStageAttribution{}, ErrV41ColdStageWindow
		}
	}
	if out.Unassigned, ok = a.Unassigned.sub(before.Unassigned); !ok {
		return V41ColdStageAttribution{}, ErrV41ColdStageWindow
	}
	return out, nil
}

func (c V41ColdStageCounters) add(o V41ColdStageCounters) V41ColdStageCounters {
	return V41ColdStageCounters{
		Entries:          c.Entries + o.Entries,
		HostWallNanos:    c.HostWallNanos + o.HostWallNanos,
		Counters:         compute.AddBackendCounters(c.Counters, o.Counters),
		DeviceSubmits:    c.DeviceSubmits + o.DeviceSubmits,
		DeviceDispatches: c.DeviceDispatches + o.DeviceDispatches,
		SubmitWaitNanos:  c.SubmitWaitNanos + o.SubmitWaitNanos,
	}
}

func (c V41ColdStageCounters) sub(before V41ColdStageCounters) (V41ColdStageCounters, bool) {
	counters, err := compute.SubBackendCounters(c.Counters, before.Counters)
	if err != nil || c.Entries < before.Entries || c.HostWallNanos < before.HostWallNanos || c.DeviceSubmits < before.DeviceSubmits ||
		c.DeviceDispatches < before.DeviceDispatches || c.SubmitWaitNanos < before.SubmitWaitNanos {
		return V41ColdStageCounters{}, false
	}
	return V41ColdStageCounters{
		Entries: c.Entries - before.Entries, HostWallNanos: c.HostWallNanos - before.HostWallNanos, Counters: counters,
		DeviceSubmits:    c.DeviceSubmits - before.DeviceSubmits,
		DeviceDispatches: c.DeviceDispatches - before.DeviceDispatches, SubmitWaitNanos: c.SubmitWaitNanos - before.SubmitWaitNanos,
	}, true
}

// v41ColdStageRecorder is session-owned and single-owner like the session it
// observes. A nil recorder is the disabled path: every method returns at once.
type v41ColdStageRecorder struct {
	owner   *Session
	backend compute.Backend
	read    func() (compute.BackendCounterRead, bool)
	now     func() int64
	open    V41ColdStage
	lastAt  int64
	last    compute.BackendCounterRead
	rows    V41ColdStageAttribution
}

var v41ColdStageEpoch = time.Now()

func v41ColdStageNow() int64 { return int64(time.Since(v41ColdStageEpoch)) }

func newV41ColdStageRecorder(read func() (compute.BackendCounterRead, bool), now func() int64) *v41ColdStageRecorder {
	r := &v41ColdStageRecorder{read: read, now: now, open: v41ColdStageNone}
	r.lastAt = now()
	if read == nil {
		return r
	}
	var ok bool
	if r.last, ok = read(); ok {
		r.rows.CountersObserved = true
		r.rows.DispatchCountsLive = r.last.DispatchCountsLive
		r.rows.DeviceSubmitsObserved = r.last.SubmitsObserved
		r.rows.DeviceWorkObserved = r.last.WorkObserved
		r.rows.SubmitWaitComplete = r.last.WorkObserved && r.last.Work.SubmitWaitObserved
	}
	return r
}

// EnableV41ColdStageAttribution starts per-stage attribution of this session's
// cold V4.1 forward. It is idempotent and keeps an existing recorder.
func (s *Session) EnableV41ColdStageAttribution() {
	if s == nil || s.v41ColdStages != nil {
		return
	}
	backend := s.Backend
	var read func() (compute.BackendCounterRead, bool)
	if backend != nil {
		read = func() (compute.BackendCounterRead, bool) {
			counters, ok, err := compute.ReadBackendCounters(backend)
			return counters, ok && err == nil
		}
	}
	r := newV41ColdStageRecorder(read, v41ColdStageNow)
	r.owner, r.backend = s, backend
	s.v41ColdStages = r
}

// V41ColdStageAttribution folds everything since the previous read into the
// open region and returns the cumulative rows. ok is false until enabled.
func (s *Session) V41ColdStageAttribution() (V41ColdStageAttribution, bool) {
	if s == nil || s.v41ColdStages == nil {
		return V41ColdStageAttribution{}, false
	}
	return s.v41ColdStages.snapshot(), true
}

func (st *v41ForwardState) coldStageRecorder() *v41ColdStageRecorder {
	if st == nil || st.callbackOwner == nil {
		return nil
	}
	return st.callbackOwner.v41ColdStages
}

func (r *v41ColdStageRecorder) enter(stage V41ColdStage) {
	if r == nil {
		return
	}
	r.fold()
	r.open = stage
	r.rows.Stages[stage].Entries++
}

func (r *v41ColdStageRecorder) leave() {
	if r == nil {
		return
	}
	r.fold()
	r.open = v41ColdStageNone
}

func (r *v41ColdStageRecorder) snapshot() V41ColdStageAttribution {
	r.fold()
	return r.rows
}

func (r *v41ColdStageRecorder) fold() {
	at := r.now()
	row := &r.rows.Unassigned
	if r.open < V41ColdStageCount {
		row = &r.rows.Stages[r.open]
	}
	if wall := at - r.lastAt; wall > 0 {
		row.HostWallNanos += wall
	}
	r.lastAt = at
	if !r.rows.CountersObserved && !r.rows.DeviceSubmitsObserved && !r.rows.DeviceWorkObserved {
		return
	}
	var cur compute.BackendCounterRead
	ok := r.owner == nil || r.owner.Backend == r.backend
	if ok {
		cur, ok = r.read()
	}
	if !ok || cur.DispatchCountsLive != r.last.DispatchCountsLive {
		r.dropCounters()
		r.dropDeviceSubmits()
		r.dropDeviceWork()
		return
	}
	if r.rows.CountersObserved {
		delta, err := compute.SubBackendCounters(cur.Counters, r.last.Counters)
		if err != nil {
			r.dropCounters()
		} else {
			row.Counters = compute.AddBackendCounters(row.Counters, delta)
		}
	}
	if r.rows.DeviceSubmitsObserved {
		if !cur.SubmitsObserved || cur.Submits < r.last.Submits {
			r.dropDeviceSubmits()
		} else {
			row.DeviceSubmits += cur.Submits - r.last.Submits
		}
	}
	if r.rows.DeviceWorkObserved {
		if !cur.WorkObserved || cur.Work.Dispatches < r.last.Work.Dispatches || cur.Work.SubmitWaitNS < r.last.Work.SubmitWaitNS {
			r.dropDeviceWork()
		} else {
			row.DeviceDispatches += cur.Work.Dispatches - r.last.Work.Dispatches
			row.SubmitWaitNanos += cur.Work.SubmitWaitNS - r.last.Work.SubmitWaitNS
		}
	}
	r.last = cur
}

// dropCounters, dropDeviceSubmits and dropDeviceWork fail closed: once one
// read of a family is missing, inconsistent or from a rebound backend, no row
// may keep a partial total of that family as observed.
func (r *v41ColdStageRecorder) dropCounters() {
	r.rows.CountersObserved, r.rows.DispatchCountsLive = false, false
	for i := range r.rows.Stages {
		r.rows.Stages[i].Counters = compute.BackendCounterSnapshot{}
	}
	r.rows.Unassigned.Counters = compute.BackendCounterSnapshot{}
}

func (r *v41ColdStageRecorder) dropDeviceSubmits() {
	r.rows.DeviceSubmitsObserved = false
	for i := range r.rows.Stages {
		r.rows.Stages[i].DeviceSubmits = 0
	}
	r.rows.Unassigned.DeviceSubmits = 0
}

func (r *v41ColdStageRecorder) dropDeviceWork() {
	r.rows.DeviceWorkObserved, r.rows.SubmitWaitComplete = false, false
	for i := range r.rows.Stages {
		r.rows.Stages[i].DeviceDispatches, r.rows.Stages[i].SubmitWaitNanos = 0, 0
	}
	r.rows.Unassigned.DeviceDispatches, r.rows.Unassigned.SubmitWaitNanos = 0, 0
}
