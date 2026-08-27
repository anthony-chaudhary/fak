package qwen4exp

import "errors"

const MetalResidencySchema = "fak.qwen4exp.metal-residency/1"

type MetalResidency struct {
	Schema               string          `json:"schema"`
	Artifact             string          `json:"artifact"`
	ArtifactBytes        int64           `json:"artifact_bytes"`
	DType                string          `json:"dtype"`
	Chip                 string          `json:"chip"`
	UnifiedPhysicalBytes int64           `json:"unified_physical_bytes"`
	SystemReservedBytes  int64           `json:"system_reserved_bytes"`
	RuntimePeakBytes     int64           `json:"runtime_peak_bytes"`
	StreamedBytes        int64           `json:"streamed_bytes"`
	DiskFreeBytes        int64           `json:"disk_free_bytes"`
	Pressure             string          `json:"pressure"`
	Thermal              string          `json:"thermal"`
	Ops                  map[string]bool `json:"ops"`
	Engine               string          `json:"engine"`
	Fallback             string          `json:"fallback"`
}

func (p MetalResidency) Validate() error {
	if p.Schema != MetalResidencySchema {
		return errors.New("qwen4exp metal: invalid schema")
	}
	if p.Artifact == "" || p.ArtifactBytes <= 0 || p.DType == "" || p.Chip == "" || p.UnifiedPhysicalBytes <= 0 || p.SystemReservedBytes < 0 || p.RuntimePeakBytes <= 0 {
		return errors.New("qwen4exp metal: incomplete identity")
	}
	if p.Engine != "fak-native" || p.Fallback != "none" {
		return errors.New("qwen4exp metal: non-native or fallback plan")
	}
	usable := p.UnifiedPhysicalBytes - p.SystemReservedBytes
	if usable <= 0 || p.RuntimePeakBytes > usable {
		return errors.New("qwen4exp metal: unified-memory overcommit")
	}
	if p.StreamedBytes < 0 || p.StreamedBytes > p.ArtifactBytes {
		return errors.New("qwen4exp metal: invalid streaming")
	}
	resident := p.ArtifactBytes - p.StreamedBytes
	if resident+p.RuntimePeakBytes > usable {
		return errors.New("qwen4exp metal: artifact and runtime exceed usable memory")
	}
	if p.StreamedBytes > 0 && p.DiskFreeBytes < p.StreamedBytes {
		return errors.New("qwen4exp metal: insufficient streaming storage")
	}
	if p.Pressure == "" || p.Thermal == "" {
		return errors.New("qwen4exp metal: pressure/thermal envelope missing")
	}
	for _, op := range []string{"gdn", "qsa_top2048", "sparse_moe", "shared_expert", "ple_ngram"} {
		if !p.Ops[op] {
			return errors.New("qwen4exp metal: incomplete operation coverage")
		}
	}
	return nil
}
