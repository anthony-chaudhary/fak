package memgate

// Pressure is the admission-relevant host memory pressure state.
type Pressure string

const (
	// PressureUnknown indicates that host memory capacity or availability could not be determined.
	PressureUnknown Pressure = "unknown"
	// PressureNormal indicates sufficient available host memory with nominal compressor and wired usage.
	PressureNormal Pressure = "normal"
	// PressureWarning indicates constrained available memory or moderate memory compressor activity.
	PressureWarning Pressure = "warning"
	// PressureCritical indicates severe memory exhaustion, high wired GPU usage, or heavy paging compression.
	PressureCritical Pressure = "critical"
)

// AdmissionSample is the byte-precise host input used by local admission.
// AllocatableBytes is already reduced by the platform safety margin.
type AdmissionSample struct {
	TotalBytes       int64    `json:"total_bytes"`
	AllocatableBytes int64    `json:"allocatable_bytes"`
	CompressedBytes  int64    `json:"compressed_bytes"`
	WiredBytes       int64    `json:"wired_bytes"`
	Pressure         Pressure `json:"pressure"`
}

// AdmissionSampleFor classifies a parsed host memory reading without rounding.
// Missing capacity is unknown and high wired/compressed memory is critical so
// callers can fail closed before invoking a loader.
func AdmissionSampleFor(mem Memory) AdmissionSample {
	s := AdmissionSample{
		TotalBytes:       mem.TotalBytes,
		AllocatableBytes: mem.AvailableBytes,
		CompressedBytes:  mem.CompressedBytes,
		WiredBytes:       mem.WiredBytes,
		Pressure:         PressureNormal,
	}
	if mem.TotalBytes <= 0 || mem.AvailableBytes <= 0 {
		s.Pressure = PressureUnknown
		return s
	}
	wired := float64(mem.WiredBytes) / float64(mem.TotalBytes)
	compressed := float64(mem.CompressedBytes) / float64(mem.TotalBytes)
	switch {
	case wired > HighWiredFraction || compressed >= 0.20:
		s.Pressure = PressureCritical
	case compressed >= 0.10 || mem.AvailableBytes < mem.TotalBytes/10:
		s.Pressure = PressureWarning
	}
	return s
}
