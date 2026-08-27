package model

import "fmt"

type UnifiedMemorySample struct {
	StorageMode        string `json:"storage_mode"`
	ResourceBytes      int64  `json:"resource_bytes"`
	CPUWriteBytes      int64  `json:"cpu_write_bytes"`
	GPUReadBytes       int64  `json:"gpu_read_bytes"`
	GPUWriteBytes      int64  `json:"gpu_write_bytes"`
	PageFaultBytes     int64  `json:"page_fault_bytes"`
	SLCResidentBytes   int64  `json:"slc_resident_bytes"`
	CommandNanoseconds int64  `json:"command_nanoseconds"`
	AcceptedTokens     int    `json:"accepted_tokens"`
}
type UnifiedMemoryReceipt struct {
	Schema                 string  `json:"schema"`
	Engine                 string  `json:"engine"`
	StorageMode            string  `json:"storage_mode"`
	SharedCapacityBytes    int64   `json:"shared_capacity_bytes"`
	EffectiveTransferBytes int64   `json:"effective_transfer_bytes"`
	PageFaultBytes         int64   `json:"page_fault_bytes"`
	SLCResidencyRatio      float64 `json:"slc_residency_ratio"`
	EffectiveGBps          float64 `json:"effective_gbps"`
	BytesPerAccepted       float64 `json:"bytes_per_accepted_token"`
	QualityConstraint      string  `json:"quality_constraint"`
	StopRule               string  `json:"stop_rule"`
	Rollback               string  `json:"rollback"`
}

func MeasureUnifiedMemory(s UnifiedMemorySample) (UnifiedMemoryReceipt, error) {
	if s.StorageMode == "" || s.ResourceBytes < 0 || s.CPUWriteBytes < 0 || s.GPUReadBytes < 0 || s.GPUWriteBytes < 0 || s.PageFaultBytes < 0 || s.SLCResidentBytes < 0 || s.SLCResidentBytes > s.ResourceBytes || s.CommandNanoseconds <= 0 || s.AcceptedTokens < 0 {
		return UnifiedMemoryReceipt{}, fmt.Errorf("model: invalid unified-memory sample")
	}
	r := UnifiedMemoryReceipt{Schema: "fak-unified-memory/1", Engine: "fak-native-metal", StorageMode: s.StorageMode, SharedCapacityBytes: s.ResourceBytes, EffectiveTransferBytes: s.CPUWriteBytes + s.GPUReadBytes + s.GPUWriteBytes + s.PageFaultBytes, PageFaultBytes: s.PageFaultBytes, QualityConstraint: "same artifact, logits, and accepted tokens", StopRule: "reject locality mode when page faults or command tail regress", Rollback: "restore prior Metal storage/resource mode"}
	if s.ResourceBytes > 0 {
		r.SLCResidencyRatio = float64(s.SLCResidentBytes) / float64(s.ResourceBytes)
	}
	r.EffectiveGBps = float64(r.EffectiveTransferBytes) / float64(s.CommandNanoseconds)
	if s.AcceptedTokens > 0 {
		r.BytesPerAccepted = float64(r.EffectiveTransferBytes) / float64(s.AcceptedTokens)
	}
	return r, nil
}
