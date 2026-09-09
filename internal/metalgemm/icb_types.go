package metalgemm

import (
	"errors"
	"fmt"
)

// ICBCommandType represents the type of command encoded in an MTLIndirectCommandBuffer.
type ICBCommandType string

const (
	ICBCommandTypeConcurrentDispatch        ICBCommandType = "MTLIndirectCommandTypeConcurrentDispatch"
	ICBCommandTypeConcurrentDispatchThreads ICBCommandType = "MTLIndirectCommandTypeConcurrentDispatchThreads"
)

// ICBStorageMode specifies the memory allocation mode for the indirect buffer.
type ICBStorageMode string

const (
	ICBStorageModeShared  ICBStorageMode = "MTLResourceStorageModeShared"
	ICBStorageModePrivate ICBStorageMode = "MTLResourceStorageModePrivate"
)

// ICBDescriptor specifies configuration for creating an MTLIndirectCommandBuffer.
type ICBDescriptor struct {
	CommandTypes             []ICBCommandType
	InheritBuffers           bool
	InheritPipelineState     bool
	MaxKernelBufferBindCount int
	MaxCallCount             int
	StorageMode              ICBStorageMode
}

// Validate verifies that the ICB descriptor satisfies Apple Silicon Metal constraints.
func (d *ICBDescriptor) Validate() error {
	if len(d.CommandTypes) == 0 {
		return errors.New("icb: command types must not be empty")
	}
	hasCompute := false
	for _, ct := range d.CommandTypes {
		if ct == ICBCommandTypeConcurrentDispatch || ct == ICBCommandTypeConcurrentDispatchThreads {
			hasCompute = true
			break
		}
	}
	if !hasCompute {
		return errors.New("icb: command types must contain concurrent compute dispatch")
	}
	if d.MaxKernelBufferBindCount < 1 || d.MaxKernelBufferBindCount > 31 {
		return fmt.Errorf("icb: max kernel buffer bind count %d out of range [1, 31]", d.MaxKernelBufferBindCount)
	}
	if d.MaxCallCount <= 0 {
		return fmt.Errorf("icb: max call count %d must be positive", d.MaxCallCount)
	}
	return nil
}

// OverheadModel computes projected execution latencies and throughput.
type OverheadModel struct {
	NumDispatches          int
	PerDispatchCPUEncodeUs float64
	ICBReplayOverheadUs    float64
	GPUBoundStepMs         float64
	BaselineCPUEncodeMs    float64
	ICBCPUEncodeMs         float64
	NetSavingsMs           float64
	BaselineTotalStepMs    float64
	ICBTotalStepMs         float64
	BaselineTokPerSec      float64
	ICBTokPerSec           float64
	ThroughputGainPercent  float64
}

// CalculateOverheadReduction evaluates CPU overhead reduction on Apple Silicon.
func CalculateOverheadReduction(numDispatches int, perDispatchUs, icbReplayUs, gpuStepMs float64) *OverheadModel {
	baselineEncodeMs := float64(numDispatches) * perDispatchUs / 1000.0
	icbEncodeMs := icbReplayUs / 1000.0
	netSavingsMs := baselineEncodeMs - icbEncodeMs

	baselineTotalMs := baselineEncodeMs + gpuStepMs
	icbTotalMs := icbEncodeMs + gpuStepMs

	baselineTokPerSec := 1000.0 / baselineTotalMs
	icbTokPerSec := 1000.0 / icbTotalMs
	gainPercent := ((icbTokPerSec - baselineTokPerSec) / baselineTokPerSec) * 100.0

	return &OverheadModel{
		NumDispatches:          numDispatches,
		PerDispatchCPUEncodeUs: perDispatchUs,
		ICBReplayOverheadUs:    icbReplayUs,
		GPUBoundStepMs:         gpuStepMs,
		BaselineCPUEncodeMs:    baselineEncodeMs,
		ICBCPUEncodeMs:         icbEncodeMs,
		NetSavingsMs:           netSavingsMs,
		BaselineTotalStepMs:    baselineTotalMs,
		ICBTotalStepMs:         icbTotalMs,
		BaselineTokPerSec:      baselineTokPerSec,
		ICBTokPerSec:           icbTokPerSec,
		ThroughputGainPercent:  gainPercent,
	}
}
