package ggufload

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// inkling2VRAMBudgetEnv caps the bytes bindInkling2Device may place on the
// device, overriding the free-memory probe.
const inkling2VRAMBudgetEnv = "FAK_INKLING2_VRAM_BUDGET_BYTES"

// inkling2VRAMHeadroomPct of the free reading, capped at
// inkling2VRAMHeadroomCap, is left for activations, KV cache, graph scratch and
// allocator rounding. The cap keeps a near-full card (Mini UD-Q3_K_XL on a
// 16 GiB card) from spilling experts that uploaded fine with no budget.
const (
	inkling2VRAMHeadroomPct = 15
	inkling2VRAMHeadroomCap = int64(1) << 30
)

// Inkling2DevicePlacement is where bindInkling2Device put the weights. A
// negative BudgetBytes means no budget was known and everything was uploaded.
// Cold expert stacks split into a RAM tier (prefaulted mapped pages) and an
// NVMe tier (pages streamed from the mapping on demand).
type Inkling2DevicePlacement struct {
	BudgetBytes   int64
	DenseBytes    int64
	ResidentBytes int64
	ColdBytes     int64
	// ResidentLayers / ColdLayers count layers whose expert stacks are on the
	// device / left as host (mmap) quant experts.
	ResidentLayers, ColdLayers int

	// RAMBytes / NVMeBytes are host (file) bytes of cold expert stacks in each
	// tier; RAMAvailableBytes is -1 when the host could not be probed.
	RAMBytes, NVMeBytes         int64
	RAMLayers, NVMeLayers       int
	RAMBudgetBytes              int64
	RAMAvailableBytes           int64
	RAMNeededForAllExpertsBytes int64
}

// FullyResident reports whether every expert stack is on the device.
func (p Inkling2DevicePlacement) FullyResident() bool { return p.ColdLayers == 0 }

// VRAMBytes is the planned device footprint: dense weights plus hot experts.
func (p Inkling2DevicePlacement) VRAMBytes() int64 { return p.DenseBytes + p.ResidentBytes }

func (p Inkling2DevicePlacement) String() string {
	tail := "no NVMe streaming"
	if p.NVMeBytes > 0 {
		tail = fmt.Sprintf("+%.0f GiB RAM would remove NVMe streaming", math.Ceil(inkling2GiB(p.RAMNeededForAllExpertsBytes)))
	}
	return fmt.Sprintf("inkling2 placement: vram %.1f GiB (dense+hot, %d expert layers), ram %.1f GiB (%d layers), nvme %.1f GiB (%d layers); %s",
		inkling2GiB(p.VRAMBytes()), p.ResidentLayers, inkling2GiB(p.RAMBytes), p.RAMLayers, inkling2GiB(p.NVMeBytes), p.NVMeLayers, tail)
}

func inkling2GiB(b int64) float64 { return float64(b) / (1 << 30) }

// Inkling2DenseOverBudgetError refuses a load whose dense weights alone exceed
// the device budget: experts can stay on the host, the dense path cannot.
type Inkling2DenseOverBudgetError struct {
	DenseBytes, BudgetBytes int64
}

func (e *Inkling2DenseOverBudgetError) Error() string {
	return fmt.Sprintf("inkling2: dense weights need %d bytes but the device budget is %d bytes (set %s to override)",
		e.DenseBytes, e.BudgetBytes, inkling2VRAMBudgetEnv)
}

// inkling2DeviceBudget is the byte budget for weight uploads: the env override
// when set, else free device memory minus headroom, else -1 (unbounded) for a
// backend that cannot report free memory, which keeps the upload-everything
// behaviour.
func inkling2DeviceBudget(be compute.Backend, lookup func(string) (string, bool)) (int64, error) {
	if v, ok := lookup(inkling2VRAMBudgetEnv); ok && strings.TrimSpace(v) != "" {
		n, err := strconv.ParseInt(strings.TrimSpace(v), 10, 64)
		if err != nil || n < 0 {
			return 0, fmt.Errorf("inkling2: %s=%q is not a non-negative byte count", inkling2VRAMBudgetEnv, v)
		}
		return n, nil
	}
	_, free, known := compute.DeviceMemoryInfo(be)
	if !known || free == compute.FreeUnknown || free < 0 {
		return -1, nil
	}
	return free - min(free*inkling2VRAMHeadroomPct/100, inkling2VRAMHeadroomCap), nil
}

// inkling2DeviceBytes estimates the device footprint of one uploaded matrix.
// Float weights are widened to F32; Q8_0 becomes int8 codes plus an f32 scale
// per 32 values; IQ4_NL rows are padded to 256 values. NVFP4 repacks to the
// same byte count it has in the GGUF.
func inkling2DeviceBytes(typ TensorType, rawLen, rows, cols int) int64 {
	n := int64(rows) * int64(cols)
	switch typ {
	case TensorF32, TensorF16, TensorBF16:
		return n * 4
	case TensorQ8_0:
		return n + n/32*4
	case TensorIQ4_NL:
		padK := (cols + 255) / 256 * 256
		return int64(rawLen) / int64(cols) * int64(padK)
	}
	return int64(rawLen)
}

// inkling2MatrixDeviceBytes is the device footprint of m, 0 when it stays host.
func inkling2MatrixDeviceBytes(m model.Inkling2LoopMatrix) int64 {
	q, ok := m.(*inkling2QuantMatrix)
	if !ok {
		return 0
	}
	switch q.typ {
	case TensorF32, TensorF16, TensorBF16:
		if q.rows*q.cols < inkling2HostDenseLimit {
			return 0
		}
	}
	return inkling2DeviceBytes(q.typ, len(q.raw), q.rows, q.cols)
}

func inkling2ExpertsDeviceBytes(e model.Inkling2LoopExperts) int64 {
	q, ok := e.(*inkling2QuantExperts)
	if !ok {
		return 0
	}
	return inkling2DeviceBytes(q.typ, len(q.raw), q.experts*q.rows, q.cols)
}

// inkling2PlanExpertResidency picks which layers' expert stacks go on the
// device. order is the residency priority, layerBytes[i] the combined
// gate_up+down bytes of layer i (both stacks move together so the fused MoE
// never sees a half-resident layer). Dense weights are placed first; a layer
// that does not fit is skipped and later, smaller layers may still fit. A
// negative budget places everything.
func inkling2PlanExpertResidency(order []int, layerBytes []int64, denseBytes, budget int64) (resident []bool, p Inkling2DevicePlacement, err error) {
	p = Inkling2DevicePlacement{BudgetBytes: budget, DenseBytes: denseBytes}
	if budget >= 0 && denseBytes > budget {
		return nil, p, &Inkling2DenseOverBudgetError{DenseBytes: denseBytes, BudgetBytes: budget}
	}
	resident = make([]bool, len(layerBytes))
	used := denseBytes
	for _, i := range order {
		b := layerBytes[i]
		if b == 0 {
			continue
		}
		if budget < 0 || used+b <= budget {
			resident[i] = true
			used += b
			p.ResidentBytes += b
			p.ResidentLayers++
			continue
		}
		p.ColdBytes += b
		p.ColdLayers++
	}
	return resident, p, nil
}

// planInkling2Device sizes every upload bindInkling2Device would make, plans
// expert residency under budget, then tiers the cold stacks into RAM / NVMe,
// returning the RAM-tier regions for bind to prefetch once uploads are done.
func planInkling2Device(m *Inkling2LoopModel, budget int64) ([]bool, Inkling2DevicePlacement, [][]byte, error) {
	w := m.Weights
	dense := inkling2MatrixDeviceBytes(w.Output)
	for i := range w.Layers {
		lw := &w.Layers[i]
		for _, mm := range []model.Inkling2LoopMatrix{lw.Wq, lw.Wk, lw.Wv, lw.Wo, lw.Gate, lw.Rel,
			lw.SinkProj, lw.Router, lw.KConvProj, lw.VConvProj} {
			dense += inkling2MatrixDeviceBytes(mm)
		}
	}
	for i := range w.Controllers {
		cw := &w.Controllers[i]
		for c := range cw.Down {
			dense += inkling2MatrixDeviceBytes(cw.Down[c]) + inkling2MatrixDeviceBytes(cw.Gain[c]) + inkling2MatrixDeviceBytes(cw.Forget[c])
		}
	}
	layerBytes := make([]int64, len(w.Layers))
	for i := range w.Layers {
		layerBytes[i] = inkling2ExpertsDeviceBytes(w.Layers[i].GateUp) + inkling2ExpertsDeviceBytes(w.Layers[i].Down)
	}
	resident, p, err := inkling2PlanExpertResidency(inkling2ExpertResidencyOrder(m.HP, len(w.Layers)), layerBytes, dense, budget)
	if err != nil {
		return nil, p, nil, err
	}
	ram, err := applyInkling2RAMTier(m, resident, &p)
	if err != nil {
		return nil, p, nil, err
	}
	return resident, p, ram, nil
}

// Inkling2DevicePlacementOf returns the placement of a device-bound model.
func Inkling2DevicePlacementOf(m *Inkling2LoopModel) (Inkling2DevicePlacement, bool) {
	if m == nil {
		return Inkling2DevicePlacement{}, false
	}
	s, ok := m.closer.(*inkling2DeviceSet)
	if !ok {
		return Inkling2DevicePlacement{}, false
	}
	return s.placement, true
}

// inkling2ExpertsAllResident reports whether every non-nil expert stack is a
// device stack; the resident forward has no host-expert fallback.
func inkling2ExpertsAllResident(w *model.Inkling2LoopWeights) bool {
	if w == nil || len(w.Layers) == 0 {
		return false
	}
	for i := range w.Layers {
		for _, e := range []model.Inkling2LoopExperts{w.Layers[i].GateUp, w.Layers[i].Down} {
			if e == nil {
				continue
			}
			if _, ok := e.(*inkling2DeviceExperts); !ok {
				return false
			}
		}
	}
	return true
}
