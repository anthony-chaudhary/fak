package model

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// moe_double_buffered.go — double-buffered prefill layer streaming with D2D hit compaction
// for offloaded MoE weights (Issue #11115).
//
// In large MoE models with offloaded expert weights (host RAM -> accelerator memory), prefill
// performance is bound by PCIe / host bus transfer bandwidth. Naive sequential execution
// idles GPU compute while waiting for layer L's expert weights to transfer over the host bus.
//
// MoEDoubleBufferedPipeline addresses this with two mechanisms:
//  1. Double-buffered staging window of 2 * E expert slots (BufferA and BufferB):
//     While layer L executes compute on buffer L % 2, asynchronous prefetch/DMA transfer of
//     layer L+1 into buffer (L+1) % 2 is overlapped in the background. Compute and DMA engines
//     run concurrently across alternating ping-pong buffers.
//  2. Device-to-Device (D2D) hit compaction:
//     If an expert is already resident on device from previous steps/layers (e.g. shared
//     experts across layers, persistent device cache, or resident in the alternating buffer),
//     it is transferred directly via high-bandwidth device-to-device copy, bypassing host bus
//     traffic. Staging plans compact D2D hits and issue host bus transfers only for genuine misses.
//  3. Clean memory release upon prefill completion:
//     All staging buffers, allocated device tensors, and resident entries are freed on completion.

// TransferSource identifies whether an expert was staged via host bus or device-to-device copy.
type TransferSource string

const (
	TransferSourceHost TransferSource = "host" // Host-to-Device (H2D) DMA transfer over host bus
	TransferSourceD2D  TransferSource = "d2d"  // Device-to-Device (D2D) copy in accelerator memory
)

// ExpertSlot represents one expert slot within a double-buffered staging window.
type ExpertSlot struct {
	Index        int            `json:"index"`
	Layer        int            `json:"layer"`
	ExpertID     int            `json:"expert_id"`
	WeightKey    string         `json:"weight_key"`
	Data         []float32      `json:"-"`
	Tensor       compute.Tensor `json:"-"`
	Bytes        int64          `json:"bytes"`
	Source       TransferSource `json:"source"`
	Resident     bool           `json:"resident"`
	PrefetchDone bool           `json:"prefetch_done"`
}

// MoEStagingBuffer represents one staging buffer (BufferA or BufferB) holding E expert slots.
type MoEStagingBuffer struct {
	mu             sync.Mutex
	ID             int           `json:"id"`
	Name           string        `json:"name"`
	Slots          []*ExpertSlot `json:"slots"`
	Layer          int           `json:"layer"`
	InUse          bool          `json:"in_use"`
	AllocatedBytes int64         `json:"allocated_bytes"`
}

func newMoEStagingBuffer(id int, name string, numExperts int, expertBytes int64) *MoEStagingBuffer {
	slots := make([]*ExpertSlot, numExperts)
	for i := 0; i < numExperts; i++ {
		slots[i] = &ExpertSlot{
			Index:    i,
			Layer:    -1,
			ExpertID: -1,
			Bytes:    expertBytes,
		}
	}
	return &MoEStagingBuffer{
		ID:    id,
		Name:  name,
		Slots: slots,
		Layer: -1,
	}
}

func (b *MoEStagingBuffer) release(backend compute.Backend) {
	b.mu.Lock()
	defer b.mu.Unlock()

	freed := make(map[compute.Buffer]bool)
	for _, slot := range b.Slots {
		if backend != nil && slot.Tensor.Buf() != nil && !freed[slot.Tensor.Buf()] {
			freed[slot.Tensor.Buf()] = true
			backend.Free(slot.Tensor)
		}
		slot.Data = nil
		slot.Tensor = compute.Tensor{}
		slot.Resident = false
		slot.PrefetchDone = false
		slot.Layer = -1
		slot.ExpertID = -1
		slot.WeightKey = ""
		slot.Source = ""
	}
	b.AllocatedBytes = 0
	b.Layer = -1
	b.InUse = false
}

// MoEPipelineConfig specifies parameters for MoEDoubleBufferedPipeline.
type MoEPipelineConfig struct {
	NumLayers              int                                                                                          // Total layers in prefill
	NumExperts             int                                                                                          // E: number of experts per layer
	ExpertBytes            int64                                                                                        // Memory size per expert in bytes
	HiddenSize             int                                                                                          // Dimension for activations/weights
	WeightKeyFn            func(layer, expertID int) string                                                             // Key generator for weight identification
	HostFetchFn            func(ctx context.Context, layer, expertID int) ([]float32, error)                            // Custom host weight provider
	D2DCopyFn              func(ctx context.Context, dstSlot, srcSlot *ExpertSlot) error                                // Custom D2D copy implementation
	ComputeFn              func(ctx context.Context, layer int, buf *MoEStagingBuffer, in []float32) ([]float32, error) // Custom layer compute function
	Backend                compute.Backend                                                                              // Optional backend for hardware device memory
	SimulatedTransferDelay time.Duration                                                                                // Simulated DMA transfer latency for tests
	SimulatedComputeDelay  time.Duration                                                                                // Simulated compute latency for tests
}

// DeviceResidentRecord tracks an expert currently resident in device memory.
type DeviceResidentRecord struct {
	WeightKey string
	BufferID  int // 0 for BufferA, 1 for BufferB, -1 for persistent device memory
	SlotIndex int
	Layer     int
	ExpertID  int
	Data      []float32
	Tensor    compute.Tensor
	Bytes     int64
}

// PipelineStats tracks performance and transfer metrics across prefill execution.
type PipelineStats struct {
	TotalLayersProcessed int   `json:"total_layers_processed"`
	HostFetches          int   `json:"host_fetches"`
	HostTransferBytes    int64 `json:"host_transfer_bytes"`
	D2DHits              int   `json:"d2d_hits"`
	D2DCopies            int   `json:"d2d_copies"`
	D2DTransferBytes     int64 `json:"d2d_transfer_bytes"`
	PrefetchCount        int   `json:"prefetch_count"`
	ComputeCount         int   `json:"compute_count"`
	PingPongSwaps        int   `json:"ping_pong_swaps"`
}

// CompactionRatio reports the proportion of expert requests satisfied via D2D rather than host bus.
func (s PipelineStats) CompactionRatio() float64 {
	total := s.D2DHits + s.HostFetches
	if total == 0 {
		return 0.0
	}
	return float64(s.D2DHits) / float64(total)
}

// StagingPlan holds the compacted transfer schedule for a layer.
type StagingPlan struct {
	Layer       int
	BufferID    int
	D2DHits     []*D2DTransfer
	HostFetches []*HostTransfer
}

// D2DTransfer describes an in-device copy between resident device memory and a staging slot.
type D2DTransfer struct {
	DstSlot  *ExpertSlot
	SrcSlot  *ExpertSlot
	Resident *DeviceResidentRecord
}

// HostTransfer describes an asynchronous DMA transfer from host memory to a staging slot.
type HostTransfer struct {
	DstSlot  *ExpertSlot
	Layer    int
	ExpertID int
	Key      string
}

// MoEDoubleBufferedPipeline coordinates double-buffered layer streaming with D2D hit compaction.
type MoEDoubleBufferedPipeline struct {
	mu sync.Mutex

	cfg MoEPipelineConfig

	// 2 * E expert slots: BufferA and BufferB
	BufferA *MoEStagingBuffer
	BufferB *MoEStagingBuffer

	residentMap map[string]*DeviceResidentRecord

	stats PipelineStats

	pendingPrefetchLayer int
	pendingPrefetchChan  <-chan error

	currentLayer int
	completed    bool
	closed       bool
}

// NewMoEDoubleBufferedPipeline instantiates a pipeline with 2 * E staging slots.
func NewMoEDoubleBufferedPipeline(cfg MoEPipelineConfig) (*MoEDoubleBufferedPipeline, error) {
	if cfg.NumExperts <= 0 {
		return nil, errors.New("moe_double_buffered: NumExperts must be positive")
	}
	if cfg.NumLayers <= 0 {
		return nil, errors.New("moe_double_buffered: NumLayers must be positive")
	}
	if cfg.ExpertBytes <= 0 {
		cfg.ExpertBytes = 4096
	}
	if cfg.HiddenSize <= 0 {
		cfg.HiddenSize = 64
	}

	p := &MoEDoubleBufferedPipeline{
		cfg:                  cfg,
		BufferA:              newMoEStagingBuffer(0, "BufferA", cfg.NumExperts, cfg.ExpertBytes),
		BufferB:              newMoEStagingBuffer(1, "BufferB", cfg.NumExperts, cfg.ExpertBytes),
		residentMap:          make(map[string]*DeviceResidentRecord),
		pendingPrefetchLayer: -1,
	}
	return p, nil
}

// GetBuffer returns the staging buffer for the given buffer index or layer (L % 2).
func (p *MoEDoubleBufferedPipeline) GetBuffer(idx int) *MoEStagingBuffer {
	if idx%2 == 0 {
		return p.BufferA
	}
	return p.BufferB
}

// TotalSlots returns the total capacity of the double-buffered window (2 * E).
func (p *MoEDoubleBufferedPipeline) TotalSlots() int {
	return len(p.BufferA.Slots) + len(p.BufferB.Slots)
}

func (p *MoEDoubleBufferedPipeline) weightKey(layer, expertID int) string {
	if p.cfg.WeightKeyFn != nil {
		return p.cfg.WeightKeyFn(layer, expertID)
	}
	return fmt.Sprintf("L%d_E%d", layer, expertID)
}

// RegisterDeviceResident manually registers an expert as resident on device
// (e.g. from prior inference steps or permanent device cache) to enable D2D hit compaction.
func (p *MoEDoubleBufferedPipeline) RegisterDeviceResident(weightKey string, data []float32, tensor compute.Tensor) {
	p.mu.Lock()
	defer p.mu.Unlock()

	bytes := int64(len(data) * 4)
	if bytes == 0 {
		bytes = p.cfg.ExpertBytes
	}
	p.residentMap[weightKey] = &DeviceResidentRecord{
		WeightKey: weightKey,
		BufferID:  -1, // Persistent device memory
		SlotIndex: -1,
		Layer:     -1,
		ExpertID:  -1,
		Data:      data,
		Tensor:    tensor,
		Bytes:     bytes,
	}
}

// PlanStaging partitions layer experts into D2D hits and compacted host bus fetches.
func (p *MoEDoubleBufferedPipeline) PlanStaging(layer int, targetBuf *MoEStagingBuffer) *StagingPlan {
	p.mu.Lock()
	defer p.mu.Unlock()

	plan := &StagingPlan{
		Layer:    layer,
		BufferID: targetBuf.ID,
	}

	numExperts := p.cfg.NumExperts
	for e := 0; e < numExperts; e++ {
		key := p.weightKey(layer, e)
		slot := targetBuf.Slots[e]

		rec, found := p.residentMap[key]
		if found {
			var srcSlot *ExpertSlot
			if rec.BufferID == 0 {
				srcSlot = p.BufferA.Slots[rec.SlotIndex]
			} else if rec.BufferID == 1 {
				srcSlot = p.BufferB.Slots[rec.SlotIndex]
			}

			plan.D2DHits = append(plan.D2DHits, &D2DTransfer{
				DstSlot:  slot,
				SrcSlot:  srcSlot,
				Resident: rec,
			})
		} else {
			plan.HostFetches = append(plan.HostFetches, &HostTransfer{
				DstSlot:  slot,
				Layer:    layer,
				ExpertID: e,
				Key:      key,
			})
		}
	}
	return plan
}

func (p *MoEDoubleBufferedPipeline) executePlan(ctx context.Context, plan *StagingPlan, targetBuf *MoEStagingBuffer) error {
	targetBuf.mu.Lock()
	defer targetBuf.mu.Unlock()

	targetBuf.Layer = plan.Layer
	targetBuf.InUse = true

	// Simulate asynchronous DMA bus transfer delay if configured
	if p.cfg.SimulatedTransferDelay > 0 {
		select {
		case <-time.After(p.cfg.SimulatedTransferDelay):
		case <-ctx.Done():
			return ctx.Err()
		}
	}

	// 1. Execute D2D hit compaction copies directly on device
	for _, hit := range plan.D2DHits {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		dst := hit.DstSlot
		dst.Layer = plan.Layer
		dst.ExpertID = hit.Resident.ExpertID
		dst.WeightKey = hit.Resident.WeightKey
		dst.Bytes = hit.Resident.Bytes
		dst.Source = TransferSourceD2D
		dst.Resident = true

		if p.cfg.D2DCopyFn != nil {
			if err := p.cfg.D2DCopyFn(ctx, dst, hit.SrcSlot); err != nil {
				return err
			}
		} else {
			if len(dst.Data) != len(hit.Resident.Data) {
				dst.Data = make([]float32, len(hit.Resident.Data))
			}
			copy(dst.Data, hit.Resident.Data)
			dst.Tensor = hit.Resident.Tensor
		}
		dst.PrefetchDone = true

		p.mu.Lock()
		p.stats.D2DHits++
		p.stats.D2DCopies++
		p.stats.D2DTransferBytes += dst.Bytes
		p.residentMap[dst.WeightKey] = &DeviceResidentRecord{
			WeightKey: dst.WeightKey,
			BufferID:  targetBuf.ID,
			SlotIndex: dst.Index,
			Layer:     dst.Layer,
			ExpertID:  dst.ExpertID,
			Data:      dst.Data,
			Tensor:    dst.Tensor,
			Bytes:     dst.Bytes,
		}
		p.mu.Unlock()
	}

	// 2. Execute compacted Host-to-Device transfers across the host bus
	for _, hf := range plan.HostFetches {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		dst := hf.DstSlot
		dst.Layer = hf.Layer
		dst.ExpertID = hf.ExpertID
		dst.WeightKey = hf.Key
		dst.Bytes = p.cfg.ExpertBytes
		dst.Source = TransferSourceHost
		dst.Resident = true

		var data []float32
		if p.cfg.HostFetchFn != nil {
			var err error
			data, err = p.cfg.HostFetchFn(ctx, hf.Layer, hf.ExpertID)
			if err != nil {
				return err
			}
		} else {
			elems := int(p.cfg.ExpertBytes / 4)
			if elems <= 0 {
				elems = p.cfg.HiddenSize
			}
			data = make([]float32, elems)
			val := float32((hf.Layer+1)*1000 + (hf.ExpertID + 1))
			for i := range data {
				data[i] = val + float32(i%10)/10.0
			}
		}
		dst.Data = data

		if p.cfg.Backend != nil {
			t := compute.NewF32(p.cfg.Backend, []int{len(data)}, data)
			dst.Tensor = p.cfg.Backend.Upload(t, compute.F32)
		}
		dst.PrefetchDone = true

		p.mu.Lock()
		p.stats.HostFetches++
		p.stats.HostTransferBytes += dst.Bytes
		p.residentMap[dst.WeightKey] = &DeviceResidentRecord{
			WeightKey: dst.WeightKey,
			BufferID:  targetBuf.ID,
			SlotIndex: dst.Index,
			Layer:     dst.Layer,
			ExpertID:  dst.ExpertID,
			Data:      dst.Data,
			Tensor:    dst.Tensor,
			Bytes:     dst.Bytes,
		}
		p.mu.Unlock()
	}

	// Calculate current buffer allocation
	var totalAlloc int64
	for _, s := range targetBuf.Slots {
		if s.Resident {
			totalAlloc += s.Bytes
		}
	}
	targetBuf.AllocatedBytes = totalAlloc

	p.mu.Lock()
	p.stats.PrefetchCount++
	p.mu.Unlock()

	return nil
}

// PrefetchLayerAsync initiates an asynchronous prefetch and staging of layer weights into
// its assigned buffer (layer % 2), returning a channel signaling completion.
func (p *MoEDoubleBufferedPipeline) PrefetchLayerAsync(ctx context.Context, layer int) (<-chan error, error) {
	if layer < 0 || layer >= p.cfg.NumLayers {
		return nil, fmt.Errorf("moe_double_buffered: layer %d out of bounds [0, %d)", layer, p.cfg.NumLayers)
	}

	targetBuf := p.GetBuffer(layer)
	plan := p.PlanStaging(layer, targetBuf)

	errCh := make(chan error, 1)
	go func() {
		err := p.executePlan(ctx, plan, targetBuf)
		errCh <- err
	}()

	return errCh, nil
}

// PrefetchLayer synchronously prefetches and stages layer weights.
func (p *MoEDoubleBufferedPipeline) PrefetchLayer(ctx context.Context, layer int) error {
	ch, err := p.PrefetchLayerAsync(ctx, layer)
	if err != nil {
		return err
	}
	return <-ch
}

// ExecuteLayer executes compute on layer L's buffer (L % 2) after verifying prefetch completion.
func (p *MoEDoubleBufferedPipeline) ExecuteLayer(ctx context.Context, layer int, input []float32) ([]float32, error) {
	if layer < 0 || layer >= p.cfg.NumLayers {
		return nil, fmt.Errorf("moe_double_buffered: layer %d out of bounds [0, %d)", layer, p.cfg.NumLayers)
	}

	buf := p.GetBuffer(layer)
	buf.mu.Lock()
	defer buf.mu.Unlock()

	if buf.Layer != layer {
		return nil, fmt.Errorf("moe_double_buffered: buffer %s layer is %d, want %d", buf.Name, buf.Layer, layer)
	}

	for _, s := range buf.Slots {
		if !s.PrefetchDone {
			return nil, fmt.Errorf("moe_double_buffered: slot %d prefetch not done for layer %d", s.Index, layer)
		}
	}

	// Simulate compute latency if configured
	if p.cfg.SimulatedComputeDelay > 0 {
		select {
		case <-time.After(p.cfg.SimulatedComputeDelay):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	var out []float32
	if p.cfg.ComputeFn != nil {
		var err error
		out, err = p.cfg.ComputeFn(ctx, layer, buf, input)
		if err != nil {
			return nil, err
		}
	} else {
		out = make([]float32, len(input))
		copy(out, input)
		scale := float32(1.0 / float32(len(buf.Slots)))
		for _, s := range buf.Slots {
			for i := 0; i < len(out) && i < len(s.Data); i++ {
				out[i] += s.Data[i] * 0.001 * scale
			}
		}
	}

	p.mu.Lock()
	p.stats.ComputeCount++
	p.currentLayer = layer
	p.mu.Unlock()

	return out, nil
}

// Step coordinates layer L compute with asynchronous prefetch of layer L+1.
// Compute executes on buffer L % 2 while prefetch streams into buffer (L+1) % 2.
func (p *MoEDoubleBufferedPipeline) Step(ctx context.Context, layer int, input []float32) ([]float32, error) {
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return nil, errors.New("moe_double_buffered: pipeline is closed")
	}
	pendingChan := p.pendingPrefetchChan
	pendingLayer := p.pendingPrefetchLayer
	p.pendingPrefetchChan = nil
	p.pendingPrefetchLayer = -1
	p.mu.Unlock()

	if pendingChan != nil && pendingLayer == layer {
		if err := <-pendingChan; err != nil {
			return nil, fmt.Errorf("moe_double_buffered: prefetch layer %d failed: %w", layer, err)
		}
	} else {
		buf := p.GetBuffer(layer)
		if buf.Layer != layer {
			if err := p.PrefetchLayer(ctx, layer); err != nil {
				return nil, err
			}
		}
	}

	// Overlap asynchronous prefetch of layer L+1 into buffer (L+1) % 2
	if layer+1 < p.cfg.NumLayers {
		nextPrefetchChan, err := p.PrefetchLayerAsync(ctx, layer+1)
		if err != nil {
			return nil, err
		}
		p.mu.Lock()
		p.pendingPrefetchLayer = layer + 1
		p.pendingPrefetchChan = nextPrefetchChan
		p.stats.PingPongSwaps++
		p.mu.Unlock()
	}

	// Execute layer L compute on buffer L % 2 concurrently
	out, err := p.ExecuteLayer(ctx, layer, input)
	if err != nil {
		return nil, err
	}

	return out, nil
}

// ExecutePrefill processes all layers in sequence with double-buffered pipelining and
// releases staging buffers upon completion.
func (p *MoEDoubleBufferedPipeline) ExecutePrefill(ctx context.Context, input []float32) ([]float32, error) {
	if p.cfg.NumLayers <= 0 {
		return nil, errors.New("moe_double_buffered: NumLayers must be positive")
	}

	defer func() {
		_ = p.Release()
	}()

	// Pre-stage initial layer 0 into BufferA
	if err := p.PrefetchLayer(ctx, 0); err != nil {
		return nil, fmt.Errorf("moe_double_buffered: initial prefetch of layer 0 failed: %w", err)
	}

	act := make([]float32, len(input))
	copy(act, input)

	for l := 0; l < p.cfg.NumLayers; l++ {
		var err error
		act, err = p.Step(ctx, l, act)
		if err != nil {
			return nil, fmt.Errorf("moe_double_buffered: step layer %d failed: %w", l, err)
		}
	}

	// Await any remaining pending prefetch
	p.mu.Lock()
	pendingChan := p.pendingPrefetchChan
	p.pendingPrefetchChan = nil
	p.pendingPrefetchLayer = -1
	p.mu.Unlock()
	if pendingChan != nil {
		if err := <-pendingChan; err != nil {
			return nil, err
		}
	}

	return act, nil
}

// Release frees all staging buffers and device tensors upon prefill completion.
func (p *MoEDoubleBufferedPipeline) Release() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.BufferA.release(p.cfg.Backend)
	p.BufferB.release(p.cfg.Backend)

	p.residentMap = make(map[string]*DeviceResidentRecord)
	p.pendingPrefetchChan = nil
	p.pendingPrefetchLayer = -1
	p.completed = true
	p.closed = true

	return nil
}

// IsReleased reports whether staging buffers have been released.
func (p *MoEDoubleBufferedPipeline) IsReleased() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.closed && p.BufferA.AllocatedBytes == 0 && p.BufferB.AllocatedBytes == 0
}

// IsCompleted reports whether prefill execution has finished.
func (p *MoEDoubleBufferedPipeline) IsCompleted() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.completed
}

// Stats returns current pipeline performance metrics.
func (p *MoEDoubleBufferedPipeline) Stats() PipelineStats {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.stats
}
