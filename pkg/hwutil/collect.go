package hwutil

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"strings"
	"time"
)

// Options tunes one collection pass.
type Options struct {
	// Interval is the window between two polls of a family; deltas are
	// computed over it. Zero selects DefaultInterval.
	Interval time.Duration
	// Samples is how many windows to average. One (the default) takes two
	// polls separated by Interval.
	Samples int
	// Recorder supplies the tier-residency rows merged into the snapshot;
	// nil drains the process-wide Default() recorder. The drain consumes
	// whatever was recorded since the previous drain, so each snapshot is an
	// interval view of residency and movement.
	Recorder *Recorder
}

// DefaultInterval is the standard delta window.
const DefaultInterval = 500 * time.Millisecond

// DefaultOptions returns the plane's defaults: one 500ms window.
func DefaultOptions() Options { return Options{Interval: DefaultInterval, Samples: 1} }

// normalize fills defaults and rejects malformed options.
func (o Options) normalize() (Options, error) {
	if o.Interval <= 0 {
		o.Interval = DefaultInterval
	}
	if o.Samples <= 0 {
		o.Samples = 1
	}
	return o, nil
}

// familyResults carries each collected family plus per-family failure notes.
// A nil family with a note is the degraded (HWUTIL_PARTIAL) state; code,
// when set, names a different closed code for the whole host plane.
type familyResults struct {
	cpu   *CPU
	dram  *DRAM
	disks []Disk
	net   []NetIfc
	links []Link
	errs  []string
	code  ErrorCode
}

// Collect takes one hardware-usage sample for this host. Each family is
// polled once per window; a family whose source is unread degrades to nil and
// contributes a HWUTIL_PARTIAL note instead of aborting the others. The total
// budget is enforced through ctx. Tiers and transfers come from the
// in-process tier-residency emitter (the Options.Recorder drain, Default()
// when unset): empty when nothing was recorded since the last drain, never
// zero-filled from host counters.
func Collect(ctx context.Context, opts Options) (*Snapshot, error) {
	o, err := opts.normalize()
	if err != nil {
		return nil, err
	}
	started := time.Now()
	snap := &Snapshot{
		Schema: Schema,
		Host:   hostname(),
		Arch:   runtime.GOARCH,
		Os:     runtime.GOOS,
		At:     started.UTC(),
	}
	fam := collectFamilies(ctx, o)
	snap.CPU = fam.cpu
	snap.DRAM = fam.dram
	snap.Disks = fam.disks
	snap.Net = fam.net
	snap.Links = fam.links
	snap.GPUs = collectGPUs(ctx)
	rec := o.Recorder
	if rec == nil {
		rec = Default()
	}
	snap.Tiers, snap.Transfers = rec.Drain()
	if len(fam.errs) > 0 {
		code := fam.code
		if code == "" {
			code = CodePartial
		}
		snap.Error = string(code) + ": " + strings.Join(fam.errs, "; ")
	}
	snap.DurationSeconds = elapsedSeconds(started)
	return snap, nil
}

// hostname returns the host identity, always non-empty.
func hostname() string {
	h, err := os.Hostname()
	if err != nil || strings.TrimSpace(h) == "" {
		return "unknown-host"
	}
	return h
}

func elapsedSeconds(start time.Time) float64 {
	return round6(time.Since(start).Seconds())
}

// sleepCtx waits d or until ctx fires; it reports whether the full wait held.
func sleepCtx(ctx context.Context, d time.Duration) bool {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-t.C:
		return true
	}
}

// avg divides num by den, returning nil when den is zero (no movement window
// produced a delta — an unread value, never an invented zero rate).
func avg(num float64, den float64) *float64 {
	if den <= 0 {
		return nil
	}
	v := round6(num / den)
	return &v
}

func round6(v float64) float64 {
	if v == 0 {
		return 0
	}
	// Keep wire values compact: six significant decimals is well past any
	// counter resolution while staying JSON-clean.
	const prec = 1e6
	r := v * prec
	if r >= 0 {
		r += 0.5
	} else {
		r -= 0.5
	}
	return float64(int64(r)) / prec
}

// f64 and i64 are pointer helpers for building nil-capable rows.
func f64(v float64) *float64 { x := round6(v); return &x }
func i64(v int64) *int64     { return &v }

// safeDelta subtracts two monotonic counters; a backwards step (counter
// reset or wrap) reports ok=false so the rate stays nil instead of wrapping
// into a nonsense positive number.
func safeDelta(prev, next uint64) (uint64, bool) {
	if next < prev {
		return 0, false
	}
	return next - prev, true
}

// parseUintFile reads a single-integer sysfs/proc file; missing or malformed
// content yields ok=false (the null-on-error contract at the source edge).
func parseUintFile(path string) (uint64, bool) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, false
	}
	var v uint64
	if _, err := fmt.Sscanf(strings.TrimSpace(string(b)), "%d", &v); err != nil {
		return 0, false
	}
	return v, true
}

func parseFloatFile(path string) (float64, bool) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, false
	}
	var v float64
	if _, err := fmt.Sscanf(strings.TrimSpace(string(b)), "%g", &v); err != nil {
		return 0, false
	}
	return v, true
}
