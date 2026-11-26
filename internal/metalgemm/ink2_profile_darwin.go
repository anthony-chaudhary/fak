//go:build darwin && arm64 && cgo

package metalgemm

/*
#include "ink2_device.h"
*/
import "C"

import (
	"fmt"
	"time"
	"unsafe"
)

type ink2ProfileKey struct {
	scope, kernel, detail string
	grid, perGroup        [3]int
	threads               bool
}

type ink2Profiler struct {
	mode, requested Ink2ProfileMode
	scope, detail   string
	rows            map[ink2ProfileKey]*Ink2ProfileRow
	// open holds the row of each pass encoded on the open command buffer.
	open      []*Ink2ProfileRow
	seconds   []float64
	bufferGPU float64
	unsampled uint64
}

// newInk2Device opens a device in the given profile mode, falling back from
// pass to commit when the GPU cannot sample at pass boundaries.
func newInk2Device(mode Ink2ProfileMode) (*Ink2Device, error) {
	d, err := openInk2Device()
	if err != nil || mode == Ink2ProfileOff {
		return d, err
	}
	d.prof.requested, d.prof.mode = mode, mode
	d.prof.rows = map[ink2ProfileKey]*Ink2ProfileRow{}
	if mode == Ink2ProfilePass && C.mg_ink2_profile_passes(d.ctx) == 0 {
		d.prof.mode = Ink2ProfileCommit
	}
	return d, nil
}

// ProfileMode is the mode in effect, which differs from the one requested
// after a fallback.
func (d *Ink2Device) ProfileMode() Ink2ProfileMode { return d.prof.mode }

// SetProfileScope labels the dispatches that follow, until the next call.
func (d *Ink2Device) SetProfileScope(scope string) {
	if d.prof.mode == Ink2ProfileOff {
		return
	}
	d.mu.Lock()
	d.prof.scope = scope
	d.mu.Unlock()
}

// profileDetail labels the next dispatch only.
func (d *Ink2Device) profileDetail(format string, args ...any) {
	if d.prof.mode == Ink2ProfileOff {
		return
	}
	d.mu.Lock()
	d.prof.detail = fmt.Sprintf(format, args...)
	d.mu.Unlock()
}

// profileDispatch records the launch just encoded; caller holds d.mu.
func (d *Ink2Device) profileDispatch(kernel string, size, perGroup [3]int, threads bool, encode time.Duration) error {
	pr := &d.prof
	key := ink2ProfileKey{pr.scope, kernel, pr.detail, size, perGroup, threads}
	pr.detail = ""
	row := pr.rows[key]
	if row == nil {
		row = &Ink2ProfileRow{Scope: key.scope, Kernel: kernel, Detail: key.detail, Grid: size, PerGroup: perGroup, Threads: threads}
		pr.rows[key] = row
	}
	row.Dispatches++
	row.EncodeSeconds += encode.Seconds()
	if pr.mode == Ink2ProfilePass {
		pr.open = append(pr.open, row)
		return nil
	}
	if err := d.flushLocked(); err != nil {
		return err
	}
	var s C.mg_ink2_stats
	C.mg_ink2_get_stats(d.ctx, &s)
	row.GPUSeconds += float64(s.last_gpu_seconds)
	return nil
}

// profileFlushed charges the passes of the command buffer just flushed; caller
// holds d.mu.
func (d *Ink2Device) profileFlushed(ok bool) {
	pr := &d.prof
	if pr.mode == Ink2ProfileOff {
		return
	}
	var s C.mg_ink2_stats
	C.mg_ink2_get_stats(d.ctx, &s)
	if ok {
		pr.bufferGPU += float64(s.last_gpu_seconds)
	}
	if pr.mode != Ink2ProfilePass {
		return
	}
	open := pr.open
	pr.open = pr.open[:0]
	if !ok || len(open) == 0 {
		return
	}
	if cap(pr.seconds) < len(open) {
		pr.seconds = make([]float64, len(open))
	}
	sec := pr.seconds[:len(open)]
	if n := int(C.mg_ink2_profile_take(d.ctx, (*C.double)(unsafe.Pointer(&sec[0])), C.int(len(sec)))); n != len(open) {
		pr.unsampled += uint64(len(open))
		return
	}
	for i, row := range open {
		if sec[i] < 0 {
			pr.unsampled++
			continue
		}
		row.GPUSeconds += sec[i]
	}
}

// Profile returns the attribution collected since the last reset. Dispatches
// still pending are counted but carry no GPU time until a fence runs them.
func (d *Ink2Device) Profile() Ink2Profile {
	d.mu.Lock()
	defer d.mu.Unlock()
	p := Ink2Profile{Mode: d.prof.mode, Requested: d.prof.requested, BufferGPUSeconds: d.prof.bufferGPU, Unsampled: d.prof.unsampled}
	for _, r := range d.prof.rows {
		p.Rows = append(p.Rows, *r)
	}
	ink2ProfileRank(p.Rows)
	return p
}

// ResetProfile fences, so no pending dispatch is charged to the next window,
// and clears the collected rows.
func (d *Ink2Device) ResetProfile() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.prof.mode == Ink2ProfileOff || d.ctx == nil {
		return nil
	}
	err := d.flushLocked()
	clear(d.prof.rows)
	d.prof.bufferGPU, d.prof.unsampled, d.prof.open = 0, 0, d.prof.open[:0]
	return err
}
