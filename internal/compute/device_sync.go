package compute

import "sort"

// Host-sync sites a backend reports. A site names the outermost entry point
// that made the host wait on device completion.
const (
	SyncSiteSubmitWait    = "submit_wait"
	SyncSiteBatchFlush    = "batch_flush"
	SyncSiteFlushStatus   = "flush_status"
	SyncSiteReadback      = "readback"
	SyncSiteStepSuspend   = "step_suspend"
	SyncSiteStepEnd       = "step_end"
	SyncSitePipelineDrain = "pipeline_drain"
	SyncSiteUpload        = "upload"
	// SyncStageNone is the stage of a wait whose path recorded no compute
	// dispatch since its previous wait (a pure transfer).
	SyncStageNone = "none"
)

// DeviceSyncSite is one cumulative (site, stage) cell of host waits on device
// completion. Stage is the last compute kernel recorded on the waited path.
// WaitNS is host wall from submit (or wait start, for a drained pipelined
// batch) to the observed signal; D2HBytes is the readback payload the wait
// delivered.
type DeviceSyncSite struct {
	Site     string
	Stage    string
	Syncs    uint64
	WaitNS   uint64
	D2HBytes uint64
}

// HostSyncs sums w's sync cells.
func (w DeviceWork) HostSyncs() (syncs, waitNS uint64) {
	for _, s := range w.SyncSites {
		syncs += s.Syncs
		waitNS += s.WaitNS
	}
	return syncs, waitNS
}

// DeviceIdleDelta is the device-idle wall between two snapshots of one backend.
// ok is false unless both observed syncs, the idle clock did not go backwards,
// and no pipelined batch was submitted, drained or outstanding in between: idle
// behind an undrained pipelined batch is not on the clock.
func DeviceIdleDelta(before, after DeviceWork) (ns uint64, ok bool) {
	if !before.SyncObserved || !after.SyncObserved || after.DeviceIdleNS < before.DeviceIdleNS ||
		before.PipelinedSubmits != before.PipelinedDrains ||
		after.PipelinedSubmits != before.PipelinedSubmits || after.PipelinedDrains != before.PipelinedDrains {
		return 0, false
	}
	return after.DeviceIdleNS - before.DeviceIdleNS, true
}

type deviceSyncKey struct{ site, stage string }

// DeviceSyncSitesDelta returns after minus before per (site, stage), keeping
// only cells that moved, sorted by Site then Stage. ok is false when a cell
// went backwards or disappeared: the counters were reset, not observed.
func DeviceSyncSitesDelta(before, after []DeviceSyncSite) ([]DeviceSyncSite, bool) {
	prev := make(map[deviceSyncKey]DeviceSyncSite, len(before))
	for _, s := range before {
		prev[deviceSyncKey{s.Site, s.Stage}] = s
	}
	out := make([]DeviceSyncSite, 0, len(after))
	seen := 0
	for _, a := range after {
		b, had := prev[deviceSyncKey{a.Site, a.Stage}]
		if had {
			seen++
		}
		if a.Syncs < b.Syncs || a.WaitNS < b.WaitNS || a.D2HBytes < b.D2HBytes {
			return nil, false
		}
		if d := (DeviceSyncSite{Site: a.Site, Stage: a.Stage, Syncs: a.Syncs - b.Syncs, WaitNS: a.WaitNS - b.WaitNS, D2HBytes: a.D2HBytes - b.D2HBytes}); d.Syncs != 0 || d.WaitNS != 0 || d.D2HBytes != 0 {
			out = append(out, d)
		}
	}
	if seen != len(prev) {
		return nil, false
	}
	sortDeviceSyncSites(out)
	return out, true
}

// DeviceSyncSitesAdd returns the per (site, stage) sum of a and b, sorted by
// Site then Stage. Neither input is modified.
func DeviceSyncSitesAdd(a, b []DeviceSyncSite) []DeviceSyncSite {
	idx := make(map[deviceSyncKey]int, len(a)+len(b))
	out := make([]DeviceSyncSite, 0, len(a)+len(b))
	for _, in := range [2][]DeviceSyncSite{a, b} {
		for _, s := range in {
			k := deviceSyncKey{s.Site, s.Stage}
			if i, ok := idx[k]; ok {
				out[i].Syncs += s.Syncs
				out[i].WaitNS += s.WaitNS
				out[i].D2HBytes += s.D2HBytes
				continue
			}
			idx[k] = len(out)
			out = append(out, s)
		}
	}
	sortDeviceSyncSites(out)
	return out
}

func sortDeviceSyncSites(s []DeviceSyncSite) {
	sort.Slice(s, func(i, j int) bool {
		if s[i].Site != s[j].Site {
			return s[i].Site < s[j].Site
		}
		return s[i].Stage < s[j].Stage
	})
}
