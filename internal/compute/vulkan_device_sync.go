//go:build vulkan && (windows || linux) && cgo

package compute

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo !v41_indexer_witness,!v41_bf16_boundary_witness,!v41_logit_scale_witness,!v41_tail_rope_exact_witness LDFLAGS: -L${SRCDIR} -lfakvulkan
#cgo v41_indexer_witness,!v41_bf16_boundary_witness,!v41_logit_scale_witness,!v41_tail_rope_exact_witness LDFLAGS: -L${SRCDIR} -lfakvulkan_v41_indexer_witness
#cgo v41_bf16_boundary_witness,!v41_indexer_witness,!v41_logit_scale_witness,!v41_tail_rope_exact_witness LDFLAGS: -L${SRCDIR} -lfakvulkan_v41_bf16_boundary_witness
#cgo v41_logit_scale_witness,!v41_indexer_witness,!v41_bf16_boundary_witness,!v41_tail_rope_exact_witness LDFLAGS: -L${SRCDIR} -lfakvulkan_v41_logit_scale_witness
#cgo v41_tail_rope_exact_witness,!v41_indexer_witness,!v41_bf16_boundary_witness,!v41_logit_scale_witness LDFLAGS: -L${SRCDIR} -lfakvulkan_v41_tail_rope_exact_witness
#include <stdlib.h>
#include "vulkan_backend.h"
*/
import "C"

import "sync"

var vulkanSyncNames = sync.OnceValues(func() (sites, stages []string) {
	sites = make([]string, int(C.fvk_sync_site_count()))
	for i := range sites {
		sites[i] = C.GoString(C.fvk_sync_site_name(C.int(i)))
	}
	stages = make([]string, int(C.fvk_sync_stage_count()))
	for i := range stages {
		stages[i] = C.GoString(C.fvk_sync_stage_name(C.int(i)))
	}
	return sites, stages
})

// readVulkanDeviceSync fills w's idle clock and host-sync cells from the shim's
// always-on atomics. Like the rest of the device-work read it takes no vulkanMu.
func readVulkanDeviceSync(w *DeviceWork) {
	var submits, drains C.uint64_t
	C.fvk_sync_pipeline_counters(&submits, &drains)
	w.PipelinedSubmits, w.PipelinedDrains = uint64(submits), uint64(drains)
	w.DeviceIdleNS = uint64(C.fvk_device_idle_ns())
	sites, stages := vulkanSyncNames()
	n := int(C.fvk_sync_site_rows(nil, nil, nil, nil, nil, 0))
	for {
		capRows := n + 8
		ids := make([]C.uint32_t, 2*capRows)
		vals := make([]C.uint64_t, 3*capRows)
		n = int(C.fvk_sync_site_rows(&ids[0], &ids[capRows], &vals[0], &vals[capRows], &vals[2*capRows], C.int(capRows)))
		if n > capRows {
			continue
		}
		w.SyncSites = make([]DeviceSyncSite, 0, n)
		for i := 0; i < n; i++ {
			site, stage := int(ids[i]), int(ids[capRows+i])
			if site >= len(sites) || stage >= len(stages) {
				w.SyncSites = nil
				return
			}
			w.SyncSites = append(w.SyncSites, DeviceSyncSite{
				Site: sites[site], Stage: stages[stage],
				Syncs: uint64(vals[i]), WaitNS: uint64(vals[capRows+i]), D2HBytes: uint64(vals[2*capRows+i]),
			})
		}
		sortDeviceSyncSites(w.SyncSites)
		w.SyncObserved = true
		return
	}
}
