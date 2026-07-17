package model

import "github.com/anthony-chaudhary/fak/internal/compute"

// #4974: reproduce the witnessed `numactl --interleave=all` weight placement IN-PROCESS so the
// ordinary CPU Q4_K decode path selects it without an external wrapper. The 64-worker half of the
// witnessed regime is already auto-selected by compute.Q4KDecodeWorkers (#4625); this is the
// placement half. compute.PlanDecodeInterleave decides (a pure verdict over the host snapshot,
// overrideable via FAK_NUMA_INTERLEAVE), and compute.ApplyDecodeInterleave mbinds each resident
// weight region to MPOL_INTERLEAVE across the online NUMA nodes (linux/amd64 only, no-op elsewhere).
//
// The regions are the resident raw Q4_K super-block slabs — q4kw (the FFN gate/up/down, v/o_proj
// and, when tied-raw, lm_head) plus a separately pinned q4khead — which hold 0.5625 B/param and are
// the bytes the decode GEMV streams every step. Restriping THEM across nodes is what turns the
// default first-touch node-0 placement into the witnessed interleave regime.

// residentDecodeRegions collects the resident raw Q4_K weight slabs that the decode GEMV streams,
// deduplicated by backing-array base pointer so a head held in both q4kw and q4khead (or an aliased
// slab) is placed exactly once. Empty/nil slabs are skipped by ApplyDecodeInterleave.
func (m *Model) residentDecodeRegions() [][]byte {
	regions := make([][]byte, 0, len(m.q4kw)+1)
	seen := make(map[*byte]bool, len(m.q4kw)+1)
	add := func(raw []byte) {
		if len(raw) == 0 {
			return
		}
		base := &raw[0]
		if seen[base] {
			return
		}
		seen[base] = true
		regions = append(regions, raw)
	}
	for _, qt := range m.q4kw {
		if qt != nil {
			add(qt.raw)
		}
	}
	if m.q4khead != nil { // a raw lm_head that q4kHeadName()/the q4kw map may not surface
		add(m.q4khead.raw)
	}
	return regions
}

// ApplyDecodeNUMAInterleave runs the #4974 verdict and, when it says apply, mbinds every resident
// Q4_K weight region to MPOL_INTERLEAVE across the online NUMA nodes — the in-process equivalent of
// launching under `numactl --interleave=all`. It caches and returns the decision label (e.g.
// "interleave=applied(reason=eligible,nodes=0-7,regions=339)" or "interleave=skipped(reason=...)").
// Safe to call once after resident load and before decode; a no-op label off linux/amd64, on a
// single-node/constrained host, or under FAK_NUMA_INTERLEAVE=off.
func (m *Model) ApplyDecodeNUMAInterleave() string {
	res := compute.ApplyDecodeInterleave(m.residentDecodeRegions())
	m.numaInterleaveLabel = res.Label()
	return m.numaInterleaveLabel
}

// NUMAInterleaveLabel returns the cached label from the last ApplyDecodeNUMAInterleave call, or
// "interleave=unrun" if placement has not been attempted on this model. It lets a decode witness /
// bench line report the placement decision that was in force for the run.
func (m *Model) NUMAInterleaveLabel() string {
	if m.numaInterleaveLabel == "" {
		return "interleave=unrun"
	}
	return m.numaInterleaveLabel
}
