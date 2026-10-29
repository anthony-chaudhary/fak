package compute

// The always-on CUDA device-work counters, the CUDA peer of the Vulkan shim's
// always-on dispatch, idle and host-sync counters. The shim (cuda/kernel_profile.cuh)
// counts with the host clock only, never CUDA events, so the counters stay on with the
// timestamp profile off:
//
//   - Dispatches: every g_stream launch, cuBLAS call and async copy/fill (each is an
//     FCUDA_TS site; TestCUDATimestampSourceCoversEveryLaunch keeps that true).
//   - SyncSites: every host wait on device completion in the shim's translation unit
//     (stream, device and event synchronize, synchronous H2D/D2H copies), keyed by site
//     and by the last launch since the previous wait. WaitNS is the wall the host spent
//     blocked inside the call, not submit-to-signal as on Vulkan.
//   - ReadbackBytes: the D2H bytes those waits returned.
//   - DeviceIdleNS: host wall from a wait that proved g_stream drained to the next launch,
//     including an interval still open at the read. Only a drain proven by a g_stream,
//     legacy-stream or device wait, or a synchronous D2H copy, opens it, so it is a lower
//     bound on device idle.
//
// SyncObserved is false for the rest of the process once any launch or wait bypassed the
// counters (a CUDA graph replay, or a caller of fcuda_dw_note_unobserved), so a window is
// reported unavailable rather than under-counted. SubmitWaitObserved stays false: there is
// no submit-to-signal clock. This file is the build-tag-free fold; the cgo reader that
// feeds it lives in cuda.go.

// Indices of the shim's fcuda_dw_counters output.
const (
	cudaDWDispatches = iota
	cudaDWReadbackBytes
	cudaDWIdleNS
	cudaDWUnobserved
	cudaDWCounterCount
)

// cudaSyncSiteByID maps the shim's site ids (kDwSite* in cuda/kernel_profile.cuh) to the
// shared site names. Order is ABI.
var cudaSyncSiteByID = [...]string{SyncSiteSubmitWait, SyncSiteReadback, SyncSiteUpload, SyncSitePipelineDrain}

// Reserved shim stage ids.
const (
	cudaSyncStageNoneID  uint32 = 0
	cudaSyncStageOtherID uint32 = 1
)

// cudaSyncCell is the Go value of one fcuda_dw_site_rows row.
type cudaSyncCell struct {
	Site, Stage             uint32
	Syncs, WaitNS, D2HBytes uint64
}

// cudaSyncStageName resolves a shim stage id to a bounded label. kernel returns the shim's
// name for an assigned id and "" for an unassigned one; ok is false then, because the cell
// names a stage the reader cannot resolve. A name outside the label grammar folds to "other",
// so a kernel name can never mint an unbounded or malformed series.
func cudaSyncStageName(id uint32, kernel func(uint32) string) (string, bool) {
	switch id {
	case cudaSyncStageNoneID:
		return SyncStageNone, true
	case cudaSyncStageOtherID:
		return HostFallbackOtherStage, true
	}
	name := kernel(id)
	if name == "" {
		return "", false
	}
	if !hostFallbackStageName.MatchString(name) {
		return HostFallbackOtherStage, true
	}
	return name, true
}

// cudaDeviceWorkFromShim folds one shim snapshot into DeviceWork. n is the cell count the
// shim reported for the filled rows; a torn read (len(cells) != n) or a cell naming an
// unknown site or stage leaves the sync fields unavailable (SyncSites nil,
// SyncObserved false) while keeping the dispatch and readback counters, which do not
// depend on the cell table.
func cudaDeviceWorkFromShim(counters [cudaDWCounterCount]uint64, n int, cells []cudaSyncCell, kernel func(uint32) string) DeviceWork {
	w := DeviceWork{
		Dispatches:    counters[cudaDWDispatches],
		ReadbackBytes: counters[cudaDWReadbackBytes],
	}
	if n < 0 || len(cells) != n {
		return w
	}
	sites := make([]DeviceSyncSite, 0, n)
	for _, c := range cells {
		if int(c.Site) >= len(cudaSyncSiteByID) {
			return w
		}
		stage, ok := cudaSyncStageName(c.Stage, kernel)
		if !ok {
			return w
		}
		if c.Syncs == 0 && c.WaitNS == 0 && c.D2HBytes == 0 {
			continue
		}
		sites = append(sites, DeviceSyncSite{Site: cudaSyncSiteByID[c.Site], Stage: stage, Syncs: c.Syncs, WaitNS: c.WaitNS, D2HBytes: c.D2HBytes})
	}
	// Two stage ids can fold to the same label ("other"); merge them so each
	// (site, stage) cell appears once, sorted, as DeviceSyncSitesDelta expects.
	w.SyncSites = DeviceSyncSitesAdd(sites, nil)
	w.DeviceIdleNS = counters[cudaDWIdleNS]
	w.SyncObserved = counters[cudaDWUnobserved] == 0
	return w
}
