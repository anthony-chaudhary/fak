package compute

import (
	"sort"
	"strings"
)

// DeviceAllocClass is a device-allocation ledger class. Ordinals are the ABI of
// cuda/alloc_ledger.h (enum fak_alloc_class); the contract test pins the order.
type DeviceAllocClass int

const (
	AllocWeights DeviceAllocClass = iota
	AllocWeightsDequantShadow
	AllocQ4CoeffCache
	AllocKVCache
	AllocRecurrentState
	AllocActivationsScratch
	AllocPrefillScratch
	AllocAttentionScratch
	AllocSampling
	AllocWorkspaceCuBLAS
	AllocGraphPool
	AllocIdlePool
	AllocOther
	deviceAllocClassCount
)

var deviceAllocClassNames = [deviceAllocClassCount]string{
	"weights", "weights_dequant_shadow", "q4_coeff_cache", "kv_cache", "recurrent_state",
	"activations_scratch", "prefill_scratch", "attention_scratch", "sampling",
	"workspace_cublas", "graph_pool", "idle_pool", "other",
}

func (c DeviceAllocClass) String() string {
	if c < 0 || c >= deviceAllocClassCount {
		return deviceAllocClassNames[AllocOther]
	}
	return deviceAllocClassNames[c]
}

// DeviceAllocKind is where the bytes live. Ordinals match enum fak_alloc_kind.
type DeviceAllocKind int

const (
	AllocDeviceLocal DeviceAllocKind = iota
	AllocManaged
	AllocPinnedHost
	deviceAllocKindCount
)

var deviceAllocKindNames = [deviceAllocKindCount]string{"device_local", "managed", "pinned_host"}

func (k DeviceAllocKind) String() string {
	if k < 0 || k >= deviceAllocKindCount {
		return deviceAllocKindNames[AllocDeviceLocal]
	}
	return deviceAllocKindNames[k]
}

// DeviceAllocCells is the number of (class, kind) cells in a ledger read.
const DeviceAllocCells = int(deviceAllocClassCount) * int(deviceAllocKindCount)

// DeviceAllocCell is one (class, kind) ledger cell.
type DeviceAllocCell struct {
	Class     DeviceAllocClass `json:"-"`
	Kind      DeviceAllocKind  `json:"-"`
	ClassName string           `json:"class"`
	KindName  string           `json:"kind"`
	LiveBytes int64            `json:"live_bytes"`
	PeakBytes int64            `json:"peak_bytes"`
	LiveCount int64            `json:"live_count"`
	Allocs    int64            `json:"allocs"`
	Frees     int64            `json:"frees"`
}

// DeviceAllocSite is one interned allocation site's live footprint.
type DeviceAllocSite struct {
	Site      string `json:"site"`
	Class     string `json:"class"`
	LiveBytes int64  `json:"live_bytes"`
	PeakBytes int64  `json:"peak_bytes"`
	LiveCount int64  `json:"live_count"`
	Allocs    int64  `json:"allocs"`
}

// DeviceAllocSnapshot is the ledger plus the driver's view of device memory.
// UnaccountedBytes = (TotalBytes - FreeBytes) - DeviceLocalLiveBytes when
// MemInfoKnown: context, module images, cuBLAS/graph internals and any
// allocation path the ledger does not yet tag.
type DeviceAllocSnapshot struct {
	Cells                []DeviceAllocCell `json:"cells"`
	Sites                []DeviceAllocSite `json:"sites,omitempty"`
	LiveBytes            int64             `json:"live_bytes"`
	DeviceLocalLiveBytes int64             `json:"device_local_live_bytes"`
	ManagedLiveBytes     int64             `json:"managed_live_bytes"`
	PinnedHostLiveBytes  int64             `json:"pinned_host_live_bytes"`
	LiveCount            int64             `json:"live_count"`
	UntrackedFrees       int64             `json:"untracked_frees"`
	MemInfoKnown         bool              `json:"mem_info_known"`
	TotalBytes           int64             `json:"total_bytes"`
	FreeBytes            int64             `json:"free_bytes"`
	UsedBytes            int64             `json:"used_bytes"`
	UnaccountedBytes     int64             `json:"unaccounted_bytes"`
}

// DeviceAllocLedgerReader is implemented by backends that carry a tagged
// device-allocation ledger.
type DeviceAllocLedgerReader interface {
	DeviceAllocLedger() DeviceAllocSnapshot
}

// BuildDeviceAllocSnapshot folds a raw ledger read (5 int64s per cell:
// live, peak, live_count, allocs, frees; class-major) and the driver memory
// view into a snapshot. Cells with no traffic are omitted; sites sort by live
// bytes, largest first.
func BuildDeviceAllocSnapshot(raw []int64, untrackedFrees int64, sites []DeviceAllocSite, total, free int64, known bool) DeviceAllocSnapshot {
	s := DeviceAllocSnapshot{UntrackedFrees: untrackedFrees, MemInfoKnown: known}
	for i := 0; i < DeviceAllocCells && (i+1)*5 <= len(raw); i++ {
		v := raw[i*5 : i*5+5]
		if v[0] == 0 && v[1] == 0 && v[3] == 0 {
			continue
		}
		cell := DeviceAllocCell{
			Class:     DeviceAllocClass(i / int(deviceAllocKindCount)),
			Kind:      DeviceAllocKind(i % int(deviceAllocKindCount)),
			LiveBytes: v[0], PeakBytes: v[1], LiveCount: v[2], Allocs: v[3], Frees: v[4],
		}
		cell.ClassName, cell.KindName = cell.Class.String(), cell.Kind.String()
		s.Cells = append(s.Cells, cell)
		s.LiveBytes += cell.LiveBytes
		s.LiveCount += cell.LiveCount
		switch cell.Kind {
		case AllocDeviceLocal:
			s.DeviceLocalLiveBytes += cell.LiveBytes
		case AllocManaged:
			s.ManagedLiveBytes += cell.LiveBytes
		case AllocPinnedHost:
			s.PinnedHostLiveBytes += cell.LiveBytes
		case deviceAllocKindCount:
			// sentinel bound; i % deviceAllocKindCount never yields it, so no bucket
		}
	}
	for _, site := range sites {
		if site.LiveBytes != 0 || site.Allocs != 0 {
			s.Sites = append(s.Sites, site)
		}
	}
	sort.SliceStable(s.Sites, func(i, j int) bool { return s.Sites[i].LiveBytes > s.Sites[j].LiveBytes })
	if known && total > 0 {
		s.TotalBytes, s.FreeBytes = total, free
		s.UsedBytes = total - free
		s.UnaccountedBytes = s.UsedBytes - s.DeviceLocalLiveBytes
	}
	return s
}

// ClassLiveBytes sums live bytes of class across every kind.
func (s DeviceAllocSnapshot) ClassLiveBytes(class DeviceAllocClass) int64 {
	var n int64
	for _, c := range s.Cells {
		if c.Class == class {
			n += c.LiveBytes
		}
	}
	return n
}

// DeviceAllocClassFor maps a backend MemoryClass and allocation site onto a
// ledger class. promptPanel marks prompt-panel (prefill) scratch.
func DeviceAllocClassFor(class MemoryClass, site string, promptPanel bool) DeviceAllocClass {
	lower := strings.ToLower(site)
	has := func(words ...string) bool {
		for _, w := range words {
			if strings.Contains(lower, w) {
				return true
			}
		}
		return false
	}
	switch class {
	case MemoryWeights, MemoryOffload:
		if has("dequant", "shadow") {
			return AllocWeightsDequantShadow
		}
		return AllocWeights
	case MemoryKVCache:
		if has("recurrent", "gdn-state", "conv-state", "ssm") {
			return AllocRecurrentState
		}
		return AllocKVCache
	case MemoryScratchpad, MemoryActivation:
		switch {
		case promptPanel || has("prefill", "panel", "sequence"):
			return AllocPrefillScratch
		case has("attention", "attn"):
			return AllocAttentionScratch
		case has("argmax", "sampl", "logit"):
			return AllocSampling
		default:
			return AllocActivationsScratch
		}
	default:
		return AllocOther
	}
}
