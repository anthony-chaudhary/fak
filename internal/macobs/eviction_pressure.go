package macobs

// EvictionPressure is the typed, receipt-ready signal that explains a silent
// macOS jetsam kill under memory pressure. It folds the kernel's memorystatus
// level and recovery count (the number of times the kernel has reclaimed
// memory pages) together with the wired-limit headroom so a receipt can name
// *why* a process was killed instead of surfacing an untyped hole.
//
// All fields are populated by BuildEvictionPressure, a pure function: the
// platform probe (sysctl kern.memorystatus_level, vm_stat recovery counters)
// lives in hardware.go and feeds this model. No syscalls occur here, which is
// what makes the classifier deterministic and unit-testable.
type EvictionPressure struct {
	// RecoveryCount is the kernel's count of memory recovery events
	// (reuse/reclaim). A rising value across samples is direct evidence of
	// pressure-driven eviction.
	RecoveryCount int `json:"recovery_count"`
	// MemorystatusLevel is kern.memorystatus_level, the integer percentage
	// of available memory. It is not the separate 1/2/4 pressure enum exposed
	// by kern.memorystatus_vm_pressure_level.
	MemorystatusLevel int `json:"memorystatus_level"`
	// WiredLimitBytes is the process/VM wired-memory ceiling used to model
	// resident-memory headroom. 0 means the ceiling is unknown/unavailable.
	WiredLimitBytes uint64 `json:"wired_limit_bytes"`
	// ResidentBytes is the resident memory attributed to the workload at the
	// moment of observation.
	ResidentBytes uint64 `json:"resident_bytes"`
	// HeadroomBytes is WiredLimitBytes - ResidentBytes, floored at 0 so a
	// resident footprint that exceeds the ceiling never wraps around uint64.
	HeadroomBytes uint64 `json:"headroom_bytes"`
	// HeadroomFraction is HeadroomBytes / WiredLimitBytes, or 0 when
	// WiredLimitBytes is 0 (guards division by zero).
	HeadroomFraction float64 `json:"headroom_fraction"`
	// AtRisk reports whether a positive available-memory percentage is below the
	// eviction floor or known headroom is below the configured reserve.
	AtRisk bool `json:"at_risk"`
}

// BuildEvictionPressure constructs an EvictionPressure from its raw inputs.
// It is pure: no syscalls, no globals, no clock. reserveBytes is the headroom
// that must remain free for the classifier to consider the workload healthy.
//
// The uint64 subtraction is underflow-guarded: when resident exceeds the wired
// limit the headroom is clamped to 0. A zero wired limit yields both zero
// headroom and a zero fraction, with no panic.
func BuildEvictionPressure(recoveryCount, memorystatusLevel int, wiredLimitBytes, residentBytes, reserveBytes uint64) EvictionPressure {
	var headroom uint64
	if wiredLimitBytes > residentBytes {
		headroom = wiredLimitBytes - residentBytes
	}

	fraction := 0.0
	if wiredLimitBytes > 0 {
		fraction = float64(headroom) / float64(wiredLimitBytes)
	}

	// The reserve rule only applies when the wired ceiling is known; an
	// unknown ceiling yields no evidence either way, so it defers entirely to
	// the kernel memorystatus level.
	atRisk := memorystatusLevel > 0 && memorystatusLevel < evictionAvailablePercentFloor
	if underReserve := evictionReserveKnown(wiredLimitBytes) && reserveBytes > 0 && headroom < reserveBytes; underReserve {
		atRisk = true
	}

	return EvictionPressure{
		RecoveryCount:     recoveryCount,
		MemorystatusLevel: memorystatusLevel,
		WiredLimitBytes:   wiredLimitBytes,
		ResidentBytes:     residentBytes,
		HeadroomBytes:     headroom,
		HeadroomFraction:  fraction,
		AtRisk:            atRisk,
	}
}

// IsEvictionRisk classifies whether the observed memory state is consistent
// with an imminent eviction/jetsam kill. It is a pure predicate.
//
// A workload is at risk when either its positive available-memory percentage
// from kern.memorystatus_level is below the eviction floor or the wired-limit
// headroom has fallen below the required reserve. A reserve of 0 means only
// the percentage is consulted.
func IsEvictionRisk(memorystatusLevel int, headroomBytes, reserveBytes uint64) bool {
	if memorystatusLevel > 0 && memorystatusLevel < evictionAvailablePercentFloor {
		return true
	}
	if reserveBytes > 0 && headroomBytes < reserveBytes {
		return true
	}
	return false
}

// evictionReserveKnown reports whether the wired-limit ceiling backing a
// headroom measurement is actually known. A zero wired limit means the
// platform probe could not establish a ceiling; a zero headroom in that case
// is missing data, not evidence of pressure, so the reserve rule abstains.
func evictionReserveKnown(wiredLimitBytes uint64) bool {
	return wiredLimitBytes > 0
}

// evictionAvailablePercentFloor is this classifier's minimum available-memory
// percentage. It is separate from analyzer.go's 15% degradation threshold and
// is not a kernel jetsam threshold. Non-positive inputs retain abstention from
// the percentage rule.
const evictionAvailablePercentFloor = 3

// Legacy pressure enum constants refer only to kern.memorystatus_vm_pressure_level,
// not the available-memory percentage consumed by the eviction helpers.
const (
	// MemorystatusLevelNormal is the vm-pressure enum's normal state.
	//
	// Deprecated: do not pass this enum to the percentage-based eviction helpers.
	MemorystatusLevelNormal = 1
	// MemorystatusLevelWarning is the vm-pressure enum's warning state.
	//
	// Deprecated: do not pass this enum to the percentage-based eviction helpers.
	MemorystatusLevelWarning = 2
	// MemorystatusLevelCritical is the vm-pressure enum's critical state.
	//
	// Deprecated: do not pass this enum to the percentage-based eviction helpers.
	MemorystatusLevelCritical = 4
)
