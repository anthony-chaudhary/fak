// Package containment provides resource boundary accounting, slot capacity attribution,
// and hybrid cgroup vs parameter metadata max-pooling for unified memory APU inference.
package containment

// ModelMetadata declares deterministic model weight, context, and scratch geometry
// used to calculate an unforgeable capacity floor for APU memory attribution.
type ModelMetadata struct {
	ModelName       string `json:"model_name"`
	ParameterCount  int64  `json:"parameter_count"`
	WeightBytes     int64  `json:"weight_bytes"`
	ContextTokens   int    `json:"context_tokens"`
	KVBytesPerToken int    `json:"kv_bytes_per_token"`
	ScratchpadBytes int64  `json:"scratchpad_bytes"`
}

// EstimatedMemoryBytes returns the minimum expected memory footprint based on model parameters,
// active context window, and runtime scratchpad requirements.
func (m ModelMetadata) EstimatedMemoryBytes() int64 {
	kvBytes := int64(m.ContextTokens) * int64(m.KVBytesPerToken)
	return m.WeightBytes + kvBytes + m.ScratchpadBytes
}

// APUAllocationRecord details the hybrid capacity attribution for a model runner instance.
type APUAllocationRecord struct {
	InstanceID              string        `json:"instance_id"`
	ModelMeta               ModelMetadata `json:"model_meta"`
	CgroupBytes             int64         `json:"cgroup_bytes"`
	MetadataEstimateBytes   int64         `json:"metadata_estimate_bytes"`
	AttributedResidentBytes int64         `json:"attributed_resident_bytes"`
	APUUnderreportedBytes   int64         `json:"apu_underreported_bytes"`
	GTTBypassDetected       bool          `json:"gtt_bypass_detected"`
}

// EstimateAPUResidentMemory implements hybrid capacity attribution max-pooling:
//
//	resident_bytes = max(cgroup_bytes, model_metadata_estimate_bytes)
//
// On Linux APU architectures (such as AMD Strix Halo, Hawk Point, Phoenix), model allocations
// in GTT/TTM memory apertures can bypass container cgroup v2 memory.current counters.
// When cgroup telemetry is accurate, cgroup_bytes governs; when GTT allocations bypass the counter,
// model metadata acts as an unforgeable capacity floor.
func EstimateAPUResidentMemory(cgroupBytes int64, meta ModelMetadata, isAPU bool) int64 {
	if cgroupBytes < 0 {
		cgroupBytes = 0
	}
	if !isAPU {
		return cgroupBytes
	}
	modelEstimate := meta.EstimatedMemoryBytes()
	if cgroupBytes > modelEstimate {
		return cgroupBytes
	}
	return modelEstimate
}

// EvaluateAPUAllocation evaluates an instance's resident memory against cgroup telemetry and metadata.
func EvaluateAPUAllocation(instanceID string, cgroupBytes int64, meta ModelMetadata, isAPU bool) APUAllocationRecord {
	estimated := meta.EstimatedMemoryBytes()
	attributed := EstimateAPUResidentMemory(cgroupBytes, meta, isAPU)

	var underreported int64
	var bypass bool
	if isAPU && attributed > cgroupBytes {
		underreported = attributed - cgroupBytes
		bypass = true
	}

	return APUAllocationRecord{
		InstanceID:              instanceID,
		ModelMeta:               meta,
		CgroupBytes:             cgroupBytes,
		MetadataEstimateBytes:   estimated,
		AttributedResidentBytes: attributed,
		APUUnderreportedBytes:   underreported,
		GTTBypassDetected:       bypass,
	}
}

// ArbitrateAPUHeadroom calculates total attributed memory consumption across multiple model instances
// and reports remaining free headroom in the APU's unified memory aperture.
func ArbitrateAPUHeadroom(totalApertureBytes int64, allocations []APUAllocationRecord) (consumedBytes int64, freeBytes int64, hasCapacity bool) {
	for _, alloc := range allocations {
		consumedBytes += alloc.AttributedResidentBytes
	}
	freeBytes = totalApertureBytes - consumedBytes
	if freeBytes < 0 {
		freeBytes = 0
		hasCapacity = false
	} else {
		hasCapacity = true
	}
	return consumedBytes, freeBytes, hasCapacity
}
