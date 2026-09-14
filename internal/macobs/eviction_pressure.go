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
	// MemorystatusLevel is the macOS memorystatus level: 1 = normal,
	// 2 = warning (foreground pressure), 4 = critical (jetsam killing).
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
	// AtRisk reports whether the observed headroom is below the configured
	// reserve, i.e. an eviction/jetsam kill is plausible.
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
	atRisk := memorystatusLevel >= MemorystatusLevelWarning
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
// A workload is at risk when either the kernel has escalated past the normal
// memorystatus level (level >= MemorystatusLevelWarning) or the wired-limit
// headroom has fallen below the required reserve. A reserve of 0 means only
// the memorystatus level is consulted.
func IsEvictionRisk(memorystatusLevel int, headroomBytes, reserveBytes uint64) bool {
	if memorystatusLevel >= MemorystatusLevelWarning {
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

// Memorystatus levels reported by the macOS kernel via
// sysctl kern.memorystatus_level. Exposed as named constants so receipts and
// tests share the closed vocabulary instead of bare literals.
const (
	// MemorystatusLevelNormal is the steady state: no pressure.
	MemorystatusLevelNormal = 1
	// MemorystatusLevelWarning indicates the system is under memory
	// pressure and the kernel is actively reclaiming pages.
	MemorystatusLevelWarning = 2
	// MemorystatusLevelCritical indicates severe pressure; the kernel is
	// evicting (jetsam-killing) processes to survive.
	MemorystatusLevelCritical = 4
)
