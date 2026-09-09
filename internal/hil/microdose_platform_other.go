//go:build !(darwin && arm64 && cgo)

package hil

func runPlatformMicroDose(kind DoseKind, hw HardwareInfo) MicroDoseResult {
	return runHostReferenceMicroDose(kind, hw)
}
