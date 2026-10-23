package model

import (
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

// v41CPUMissHaloProfile is the strix3 Q4 receipt's UMA picture
// (docs/benchmarks/receipts/halo-v41-100-30/v41-q4-cpu-miss-20261013-strix3.md):
// device H2D ~3.5 GB/s solo, host streaming ~25 GB/s under contention, one
// V4.1 Flash Q2_K/Q3_K expert slab per miss.
var v41CPUMissHaloProfile = MoEBandwidthProfile{
	PCIeBandwidthBytesPerSec: 3.5e9,
	CPUBandwidthBytesPerSec:  25e9,
	ExpertSizeBytes:          12810240,
	HostCPUEnabled:           true,
}

// ErrExpertCPUMissFlag refuses an unparseable FAK_EXPERT_CPU_MISS. It is not
// an ErrV41ForwardStage, so stepV41 raises it instead of silently replaying
// the token on the whole-history fallback.
var ErrExpertCPUMissFlag = errors.New("model: invalid " + ExpertCPUMissEnv)

// v41CPUMissArm computes a layer's device-ring misses on the host while the
// batched routed layer runs the resident picks. plan returns the pick slots the
// host takes and how many misses still stream to the device.
type v41CPUMissArm struct {
	err        error
	controller *ExpertCPUMissController
	plan       func(layer int, picks []routePick) (cpu []int, streamed int, err error)
}

func (s *Session) v41CPUMissArmFor(st *v41ForwardState) *v41CPUMissArm {
	if s == nil || s.M == nil || st == nil || st.routedLayer == nil {
		return nil
	}
	policy, err := ExpertCPUMissPolicyFromEnv(v41CPUMissHaloProfile)
	if err != nil {
		return &v41CPUMissArm{err: fmt.Errorf("%w: %w", ErrExpertCPUMissFlag, err)}
	}
	if !policy.Enabled() {
		return nil
	}
	arm := &v41CPUMissArm{controller: &ExpertCPUMissController{Policy: policy}}
	arm.plan = func(layer int, picks []routePick) ([]int, int, error) {
		return s.v41CPUMissPlan(arm.controller, layer, picks)
	}
	return arm
}

// v41CPUMissPlan classifies each pick against device residency and lets the
// controller partition the misses. A pick whose projections the device cannot
// stage is left to the device path, which owns that refusal.
func (s *Session) v41CPUMissPlan(c *ExpertCPUMissController, layer int, picks []routePick) ([]int, int, error) {
	if len(picks) == 0 {
		return nil, 0, nil
	}
	slots := map[int][]int{}
	var misses []MoEMissCandidate
	r := s.routedExpertRing(layerName(layer, "ffn.experts."+itoa(picks[0].expert)+".w1.weight"))
	done := s.ringEnter(r)
	for slot, pick := range picks {
		stem := "ffn.experts." + itoa(pick.expert)
		resident, stageable := true, true
		for _, leaf := range [3]string{".w1.weight", ".w3.weight", ".w2.weight"} {
			w, ok := s.resolveExpertWeight(layerName(layer, stem+leaf))
			if !ok {
				stageable = false
				break
			}
			key := w.halKey()
			var in bool
			if r != nil {
				_, in = r.pool.Get(polymodel.ModelID(key))
			} else {
				_, in = s.halW[key]
			}
			resident = resident && in
		}
		if !stageable || resident {
			continue
		}
		if len(slots[pick.expert]) == 0 {
			misses = append(misses, MoEMissCandidate{ExpertID: pick.expert})
		}
		slots[pick.expert] = append(slots[pick.expert], slot)
	}
	done()
	if len(misses) == 0 {
		return nil, 0, nil
	}
	plan, err := c.Plan(layer, misses)
	if err != nil {
		return nil, 0, err
	}
	var cpu []int
	for _, e := range plan.CPUExperts {
		cpu = append(cpu, slots[e.ExpertID]...)
	}
	streamed := 0
	for _, e := range plan.StreamExperts {
		streamed += len(slots[e.ExpertID])
	}
	return cpu, streamed, nil
}

// v41RoutedLayerStepCPUMiss runs the planned misses on the host concurrently
// with the batched device layer over the remaining picks, then adds the host
// rows into the device partial in v41RoutedSumOrder. A device decline discards
// the host rows so the per-pick loop sees every pick, as without the arm.
func (m *Model) v41RoutedLayerStepCPUMiss(l int, picks []routePick, x []float32, scratch *v41ProjScratch) (sum []float32, handled bool, err error) {
	arm := scratch.cpuMiss
	H := m.Cfg.HiddenSize
	if arm.err != nil {
		return nil, true, arm.err
	}
	cpuSlots, streamed, planErr := arm.plan(l, picks)
	if planErr != nil {
		return nil, true, v41StageErr(v41StageMoE, l, fmt.Errorf("%w: %w", ErrV41ForwardStage, planErr))
	}
	if len(cpuSlots) == 0 {
		s, outcome, cause := scratch.routedLayer(l, picks, x)
		return v41RoutedLayerResult(l, H, s, outcome, cause)
	}
	onCPU := make([]bool, len(picks))
	for _, slot := range cpuSlots {
		onCPU[slot] = true
	}
	devicePicks := make([]routePick, 0, len(picks)-len(cpuSlots))
	hostPicks := make([]routePick, 0, len(cpuSlots))
	for slot, pick := range picks {
		if onCPU[slot] {
			hostPicks = append(hostPicks, pick)
		} else {
			devicePicks = append(devicePicks, pick)
		}
	}

	type hostResult struct {
		rows  [][]float32
		err   error
		nanos int64
	}
	results := make(chan hostResult, 1)
	go func() {
		opened := time.Now()
		rows, rowsErr := m.v41CPUMissRows(l, hostPicks, x)
		results <- hostResult{rows: rows, err: rowsErr, nanos: time.Since(opened).Nanoseconds()}
	}()
	var host hostResult
	joined := false
	join := func() int64 {
		opened := time.Now()
		host = <-results
		joined = true
		return time.Since(opened).Nanoseconds()
	}
	defer func() {
		if !joined {
			join()
		}
	}()

	deviceHandled := true
	var deviceErr error
	sum = make([]float32, H)
	if len(devicePicks) > 0 {
		s, outcome, cause := scratch.routedLayer(l, devicePicks, x)
		sum, deviceHandled, deviceErr = v41RoutedLayerResult(l, H, s, outcome, cause)
	}
	wait := join()
	if deviceErr != nil {
		return nil, true, deviceErr
	}
	if !deviceHandled {
		return nil, false, nil
	}
	if host.err != nil {
		return nil, true, host.err
	}
	sum = append([]float32(nil), sum...)
	for _, slot := range v41RoutedSumOrder(hostPicks) {
		row := host.rows[slot]
		if len(row) != H {
			return nil, true, v41ExpertOperationErr(l, "sum", errV41ExpertResult)
		}
		for i, v := range row {
			sum[i] = float32(sum[i] + v)
			if !finite32(sum[i]) {
				return nil, true, v41ExpertOperationErr(l, "sum", errV41ExpertResult)
			}
		}
	}
	m.v41NoteCPUMiss(len(hostPicks), streamed, host.nanos, wait)
	return sum, true, nil
}

// v41CPUMissRows faults every projection of the host picks concurrently, then
// contracts the picks one after another so each GEMV takes the full decode
// worker budget. Row i is pick i's BF16-published routed-down output.
func (m *Model) v41CPUMissRows(l int, picks []routePick, x []float32) ([][]float32, error) {
	weights := make([][3]v41CPUMissWeight, len(picks))
	errs := make([]error, 3*len(picks))
	var wg sync.WaitGroup
	for slot, pick := range picks {
		stem := "ffn.experts." + itoa(pick.expert)
		for j, leaf := range [3]string{".w1.weight", ".w3.weight", ".w2.weight"} {
			wg.Add(1)
			go func(slot, j int, name string) {
				defer wg.Done()
				defer func() {
					if r := recover(); r != nil {
						errs[3*slot+j] = v41StageErr(v41StageMoE, l, fmt.Errorf("%w: CPU-miss tensor %s: %v", ErrV41ForwardStage, name, r))
					}
				}()
				weights[slot][j], errs[3*slot+j] = m.v41CPUMissResolve(l, name)
			}(slot, j, layerName(l, stem+leaf))
		}
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}
	rows := make([][]float32, len(picks))
	for slot, pick := range picks {
		row, err := m.v41CPUMissExpert(l, "ffn.experts."+itoa(pick.expert), x, pick.weight, weights[slot])
		if err != nil {
			return nil, err
		}
		rows[slot] = row
	}
	return rows, nil
}

// v41CPUMissWeight is one routed projection as the host GEMV reads it: packed
// k-quant or Q4_K bytes, or an f32 block when the model holds it unpacked.
type v41CPUMissWeight struct {
	expert expertWeight
	f32    []float32
}

func (m *Model) v41CPUMissResolve(l int, name string) (v41CPUMissWeight, error) {
	if qt := m.q4kw[name]; qt != nil {
		return v41CPUMissWeight{expert: expertWeight{name: name, q4: qt}}, nil
	}
	if qt := m.kqw[name]; qt != nil {
		return v41CPUMissWeight{expert: expertWeight{name: name, kq: qt}}, nil
	}
	if m.expertCheckpoint.Has(name) {
		w, err := m.expertCheckpoint.fault(name)
		if err != nil {
			return v41CPUMissWeight{}, v41StageErr(v41StageMoE, l, fmt.Errorf("%w: tensor %s: %w", ErrV41ForwardStage, name, err))
		}
		if w.kq != nil || w.q4 != nil {
			return v41CPUMissWeight{expert: w}, nil
		}
	}
	if w, ok := m.residentF32Mat(name); ok {
		return v41CPUMissWeight{f32: w}, nil
	}
	return v41CPUMissWeight{}, v41StageErr(v41StageMoE, l, fmt.Errorf("%w: missing tensor %s", ErrV41ForwardStage, name))
}

func v41CPUMissGEMV(w v41CPUMissWeight, x []float32, out, in int) ([]float32, bool) {
	y := make([]float32, out)
	switch {
	case w.expert.kq != nil:
		if w.expert.kq.out != out || w.expert.kq.in != in {
			return nil, false
		}
		kQuantMatRowsInto(w.expert.kq, x, y)
	case w.expert.q4 != nil:
		if w.expert.q4.out != out || w.expert.q4.in != in {
			return nil, false
		}
		q4kMatRowsInto(w.expert.q4, x, y)
	case len(w.f32) == out*in:
		y = parMatRows(w.f32, x, out, in)
	default:
		return nil, false
	}
	return y, true
}

// v41CPUMissExpert is the host routed expert over packed weights: the same
// BF16 gate/up publication, clamp-SwiGLU, route weight, down publication and
// order as the host triple arm of v41FullRoutedExpertPublished, with the GEMVs
// reading the packed bytes instead of an f32 expansion.
func (m *Model) v41CPUMissExpert(l int, stem string, x []float32, weight float32, w [3]v41CPUMissWeight) (out []float32, err error) {
	defer func() {
		if r := recover(); r != nil {
			out, err = nil, v41StageErr(v41StageMoE, l, fmt.Errorf("%w: CPU-miss expert %s: %v", ErrV41ForwardStage, stem, r))
		}
	}()
	cfg := m.Cfg
	H, I := cfg.HiddenSize, cfg.MoEIntermediateSize
	gateUp := func(layer int, _ string, xin []float32) ([]float32, v41ExpertGateUpOutcome, error) {
		g, ok := v41CPUMissGEMV(w[0], xin, I, H)
		if !ok {
			return nil, v41GateUpError, errV41ExpertResult
		}
		u, ok := v41CPUMissGEMV(w[1], xin, I, H)
		if !ok {
			return nil, v41GateUpError, errV41ExpertResult
		}
		gate, cause := v41RoutedBF16(layer, "gate/up", g, I)
		if cause != nil {
			return nil, v41GateUpError, cause
		}
		up, cause := v41RoutedBF16(layer, "gate/up", u, I)
		if cause != nil {
			return nil, v41GateUpError, cause
		}
		clampSwiGLUProjections(gate, up, float32(cfg.SwigluLimit))
		h := make([]float32, I)
		for i := range h {
			h[i] = silu(gate[i]) * up[i]
		}
		return h, v41GateUpHandled, nil
	}
	down := func(_ int, _ string, operand []float32) ([]float32, v41ExpertDownOutcome, error) {
		y, ok := v41CPUMissGEMV(w[2], operand, H, I)
		if !ok {
			return nil, v41DownError, errV41ExpertResult
		}
		return y, v41DownHandled, nil
	}
	unreachable := func() ([]float32, []float32, []float32, error) {
		return nil, nil, nil, v41ExpertOperationErr(l, "gate/up", errV41ExpertResult)
	}
	unreachableDown := func() ([]float32, error) {
		return nil, v41ExpertOperationErr(l, "down", errV41ExpertResult)
	}
	return m.v41FullRoutedExpertPublished(l, stem, x, weight, cfg, nil, gateUp, down, unreachable, unreachableDown, false, nil)
}
