package ggufload

import (
	"reflect"
	"sync"
	"sync/atomic"
)

// Inkling2GraphResetter is the minimal device surface Inkling2Device needs to
// destroy process-wide CUDA graph execs, so unit tests can substitute a fake
// device without CUDA hardware.
type Inkling2GraphResetter interface {
	Ink2GraphReset()
}

// Inkling2GraphExecKey identifies one captured graph exec by geometry label,
// cohort/batch width, and buffer-layout fingerprint.
type Inkling2GraphExecKey struct {
	Geometry    string
	Width       int
	Fingerprint uint64
}

type inkling2OwnerBox struct {
	token any
}

// Inkling2Device is the single per-device owner of process-wide CUDA graph
// execs and the active owner token. Both scalar resident sessions and cohort
// engines claim ownership and reset through it, so a cohort reset clears the
// owner token and the exec table at the same time.
type Inkling2Device struct {
	owner       atomic.Pointer[inkling2OwnerBox]
	legacyOwner *atomic.Pointer[inkling2Resident]

	mu       sync.Mutex
	resetter Inkling2GraphResetter
	execs    map[Inkling2GraphExecKey]any
}

// NewInkling2Device creates a per-device graph owner bound to resetter.
func NewInkling2Device(resetter Inkling2GraphResetter) *Inkling2Device {
	return &Inkling2Device{
		resetter: resetter,
		execs:    make(map[Inkling2GraphExecKey]any),
	}
}

var (
	defaultInkling2Device = &Inkling2Device{
		legacyOwner: &inkling2GraphOwner,
		execs:       make(map[Inkling2GraphExecKey]any),
	}
	inkling2DevicesMu sync.Mutex
	inkling2Devices   = make(map[any]*Inkling2Device)
)

// inkling2DeviceFor returns the shared Inkling2Device for dev, or the process
// default when dev is nil or not comparable as a map key.
func inkling2DeviceFor(dev Inkling2GraphResetter) *Inkling2Device {
	if dev == nil {
		return defaultInkling2Device
	}
	if !reflect.TypeOf(dev).Comparable() {
		return defaultInkling2Device
	}
	inkling2DevicesMu.Lock()
	defer inkling2DevicesMu.Unlock()
	if d, ok := inkling2Devices[dev]; ok {
		return d
	}
	d := NewInkling2Device(dev)
	inkling2Devices[dev] = d
	return d
}

// Claim ensures token is the current graph owner for this device. It returns
// true when ownership switched from another owner (or from nil after a reset),
// in which case the caller must drop any stale local capture flags before
// running and any exec entries recorded for the previous owner are cleared.
func (d *Inkling2Device) Claim(token any) (switched bool) {
	if token == nil {
		return false
	}
	d.mu.Lock()
	defer d.mu.Unlock()

	cur := d.owner.Load()
	same := cur != nil && inkling2SameOwner(cur.token, token)
	if same && d.legacyOwner != nil {
		if r, ok := token.(*inkling2Resident); ok && d.legacyOwner.Load() != r {
			same = false
		}
	}
	if same {
		return false
	}
	d.owner.Store(&inkling2OwnerBox{token: token})
	if d.legacyOwner != nil {
		if r, ok := token.(*inkling2Resident); ok {
			d.legacyOwner.Store(r)
		} else {
			d.legacyOwner.Store(nil)
		}
	}
	clear(d.execs)
	return true
}

// Reset destroys the device's captured graph execs, clears the exec table, and
// clears the owner token atomically under the device owner.
func (d *Inkling2Device) Reset() {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.resetter != nil {
		d.resetter.Ink2GraphReset()
	}
	clear(d.execs)
	d.owner.Store(nil)
	if d.legacyOwner != nil {
		d.legacyOwner.Store(nil)
	}
}

// Owner returns the current owner token, or nil after a Reset.
func (d *Inkling2Device) Owner() any {
	cur := d.owner.Load()
	if cur == nil {
		return nil
	}
	if d.legacyOwner != nil {
		if _, ok := cur.token.(*inkling2Resident); ok && d.legacyOwner.Load() == nil {
			return nil
		}
	}
	return cur.token
}

// RecordExec records a captured exec handle under key for the current owner.
func (d *Inkling2Device) RecordExec(key Inkling2GraphExecKey, handle any) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.execs == nil {
		d.execs = make(map[Inkling2GraphExecKey]any)
	}
	d.execs[key] = handle
}

// LookupExec returns the recorded exec handle for key, if present.
func (d *Inkling2Device) LookupExec(key Inkling2GraphExecKey) (any, bool) {
	d.mu.Lock()
	defer d.mu.Unlock()
	h, ok := d.execs[key]
	return h, ok
}

// ExecCount returns the number of recorded execs currently held by the device.
func (d *Inkling2Device) ExecCount() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	return len(d.execs)
}

func inkling2SameOwner(a, b any) bool {
	if a == nil || b == nil {
		return a == b
	}
	ta, tb := reflect.TypeOf(a), reflect.TypeOf(b)
	if ta != tb || !ta.Comparable() {
		return false
	}
	return a == b
}
