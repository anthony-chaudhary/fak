package compute

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"time"
)

// Planner is an optional backend capability. Callers that do not find it continue to
// use Backend.MatMul directly.
type Planner interface {
	PrepareMatMul(w, x Tensor) (*MatMulPlan, error)
}

const matMulPlanRevision = 1

// MatMulPlanDescriptor is the immutable, inspectable compatibility envelope selected
// by PrepareMatMul. Device identifies the backend execution tier (host for cpu-ref).
type MatMulPlanDescriptor struct {
	Operation      string
	Backend        string
	Device         string
	WeightDtype    Dtype
	InputDtype     Dtype
	WeightShape    []int
	InputShape     []int
	WeightLayout   Layout
	InputLayout    Layout
	QuantBlock     int
	WorkspaceBytes int
	Preparation    time.Duration
}

// MatMulPlan is an immutable prepared Q8_0 matrix-vector operation. It may be reused
// concurrently provided each Run gets a distinct caller-owned workspace.
type MatMulPlan struct {
	desc     MatMulPlanDescriptor
	revision uint32
}

// Descriptor returns a copy; callers cannot mutate the plan's compatibility key.
func (p *MatMulPlan) Descriptor() MatMulPlanDescriptor {
	d := p.desc
	d.WeightShape = append([]int(nil), d.WeightShape...)
	d.InputShape = append([]int(nil), d.InputShape...)
	return d
}

// PlanRunDiagnostic captures one observed planned execution. Durations are observations,
// not a speedup claim.
type PlanRunDiagnostic struct {
	Operation      string
	Backend        string
	Device         string
	Duration       time.Duration
	WorkspaceBytes int
}

// PlanDiagnostics combines preparation and run observations. BreakEvenReuse is zero
// unless a supplied direct observation is slower than the planned-run mean.
type PlanDiagnostics struct {
	Preparation       time.Duration
	RunTimes          []time.Duration
	WorkspaceBytes    int
	BreakEvenReuse    int
	BreakEvenObserved bool
	SpeedupClaimed    bool
}

// Diagnostics builds an inspection record from two or more run observations. The direct
// duration must describe the same operation envelope; non-positive savings deliberately
// produce no break-even claim.
func (p *MatMulPlan) Diagnostics(direct time.Duration, runs ...PlanRunDiagnostic) PlanDiagnostics {
	d := PlanDiagnostics{Preparation: p.desc.Preparation, WorkspaceBytes: p.desc.WorkspaceBytes}
	var total time.Duration
	for _, run := range runs {
		d.RunTimes = append(d.RunTimes, run.Duration)
		total += run.Duration
	}
	if len(runs) == 0 {
		return d
	}
	mean := total / time.Duration(len(runs))
	saving := direct - mean
	if saving > 0 {
		d.BreakEvenObserved = true
		d.BreakEvenReuse = int((p.desc.Preparation + saving - 1) / saving)
	}
	return d
}

// Run executes the prepared operation using exactly the workspace size reported by
// Descriptor. Compatibility is checked before any output is allocated or written.
func (p *MatMulPlan) Run(be Backend, w, x Tensor, workspace []byte) (Tensor, PlanRunDiagnostic, error) {
	if p == nil {
		return Tensor{}, PlanRunDiagnostic{}, errors.New("compute: matmul plan is nil; prepare a plan or use Backend.MatMul")
	}
	if err := p.compatible(be, w, x, len(workspace)); err != nil {
		return Tensor{}, PlanRunDiagnostic{}, err
	}
	started := time.Now()
	cpu, ok := be.(*cpuBackend)
	if !ok { // guarded by compatible; retain fail-closed behavior if backend typing changes.
		return Tensor{}, PlanRunDiagnostic{}, fmt.Errorf("compute: matmul plan backend %q has no native planned runner", be.Name())
	}
	out, in := w.Shape[0], w.Shape[1]
	block := w.Quant.Block
	qx := workspace[:in]
	dx := workspace[in:]
	quantizeVecQ8Bytes(cpu.f32(x), block, qx, dx)
	y := make([]float32, out)
	wc, ws := cpu.i8(w), w.Quant.Scale
	nblk := in / block
	for o := 0; o < out; o++ {
		y[o] = qdot8scalarBytes(wc[o*in:o*in+in], ws[o*nblk:o*nblk+nblk], qx, dx, block)
	}
	duration := time.Since(started)
	return cpu.result([]int{out}, y), PlanRunDiagnostic{
		Operation: "matmul", Backend: p.desc.Backend, Device: p.desc.Device,
		Duration: duration, WorkspaceBytes: len(workspace),
	}, nil
}

func (p *MatMulPlan) compatible(be Backend, w, x Tensor, workspaceBytes int) error {
	if p.revision != matMulPlanRevision {
		return fmt.Errorf("compute: stale matmul plan revision %d (runtime requires %d); prepare again", p.revision, matMulPlanRevision)
	}
	if be == nil {
		return errors.New("compute: matmul plan backend is nil; prepare again for the active backend")
	}
	if be.Name() != p.desc.Backend || planDevice(be) != p.desc.Device {
		return fmt.Errorf("compute: matmul plan backend/device mismatch: prepared for %s/%s, got %s/%s; prepare again", p.desc.Backend, p.desc.Device, be.Name(), planDevice(be))
	}
	if w.Backend() != be || x.Backend() != be {
		return errors.New("compute: matmul plan tensor backend mismatch; upload tensors to the prepared backend or prepare again")
	}
	if w.Dtype != p.desc.WeightDtype || x.Dtype != p.desc.InputDtype {
		return fmt.Errorf("compute: matmul plan dtype mismatch: prepared for weight=%s input=%s, got weight=%s input=%s; prepare again", p.desc.WeightDtype, p.desc.InputDtype, w.Dtype, x.Dtype)
	}
	if !sameShape(w.Shape, p.desc.WeightShape) || !sameShape(x.Shape, p.desc.InputShape) {
		return fmt.Errorf("compute: matmul plan shape mismatch: prepared for weight=%v input=%v, got weight=%v input=%v; prepare again", p.desc.WeightShape, p.desc.InputShape, w.Shape, x.Shape)
	}
	if w.Layout != p.desc.WeightLayout || x.Layout != p.desc.InputLayout {
		return fmt.Errorf("compute: matmul plan layout mismatch; prepare again for weight=%d input=%d", w.Layout, x.Layout)
	}
	if w.Quant == nil || w.Quant.Block != p.desc.QuantBlock {
		got := 0
		if w.Quant != nil {
			got = w.Quant.Block
		}
		return fmt.Errorf("compute: matmul plan operation parameter mismatch: prepared quant block=%d, got %d; prepare again", p.desc.QuantBlock, got)
	}
	if workspaceBytes != p.desc.WorkspaceBytes {
		return fmt.Errorf("compute: matmul plan workspace size mismatch: need exactly %d bytes, got %d; allocate Descriptor().WorkspaceBytes", p.desc.WorkspaceBytes, workspaceBytes)
	}
	return nil
}

func planDevice(be Backend) string {
	if be.Name() == "cpu-ref" {
		return "host"
	}
	return be.Tier()
}

func sameShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func quantizeVecQ8Bytes(x []float32, block int, q, scales []byte) {
	for b := 0; b < len(x)/block; b++ {
		values := x[b*block : (b+1)*block]
		var amax float32
		for _, v := range values {
			if a := absf(v); a > amax {
				amax = a
			}
		}
		d := amax / 127
		binary.LittleEndian.PutUint32(scales[b*4:], math.Float32bits(d))
		if d == 0 {
			clear(q[b*block : (b+1)*block])
			continue
		}
		inv := 1 / d
		for i, v := range values {
			q[b*block+i] = byte(q8round(v * inv))
		}
	}
}

func qdot8scalarBytes(qw []int8, dw []float32, qx, scales []byte, block int) float32 {
	var acc float32
	for b := 0; b < len(qw)/block; b++ {
		wb, xb := qw[b*block:], qx[b*block:]
		var s0, s1, s2, s3 int32
		for i := 0; i < block; i += 4 {
			s0 += int32(wb[i]) * int32(int8(xb[i]))
			s1 += int32(wb[i+1]) * int32(int8(xb[i+1]))
			s2 += int32(wb[i+2]) * int32(int8(xb[i+2]))
			s3 += int32(wb[i+3]) * int32(int8(xb[i+3]))
		}
		dx := math.Float32frombits(binary.LittleEndian.Uint32(scales[b*4:]))
		acc += float32((s0+s1)+(s2+s3)) * dw[b] * dx
	}
	return acc
}
