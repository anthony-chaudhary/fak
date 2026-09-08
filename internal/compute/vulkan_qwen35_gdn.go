//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include <stdlib.h>
#include "vulkan_backend.h"
*/
import "C"

import (
	"fmt"
	"math"
	"unsafe"
)

const qwen35GDNVulkanPath = "vulkan/qwen35-gdn-ssm-decode-v1"

var _ VulkanQwen35GDNConvTiledChannelTransposer = (*vulkanBackend)(nil)

func (v *vulkanBackend) Qwen35GDNPath() string { return qwen35GDNVulkanPath }

// Qwen35GDNPreprojected runs the causal convolution and recurrent GDN panel on
// Vulkan-resident tensors. Both auxiliary states are updated in place. When
// vectorized GDN is disabled (via backend option or FAK_DISABLE_VECTOR_GDN),
// it routes to the scalar recurrence fallback path instead of the vectorized path.
// Qwen35GDNDecode composes the four input projections, the device-resident
// recurrent seam, and the output projection without making any host readback.
func (v *vulkanBackend) Qwen35GDNDecode(
	normalizedInput,
	inProjQKV, inProjZ, inProjB, inProjA,
	conv1D, aLog, dtBias, norm, outProj,
	convState, recurrentState Tensor,
	numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel int,
	rmsNormEpsilon float32,
) (output, nextConvState, nextRecurrentState Tensor, err error) {
	in := qwen35GDNInputs{normalizedInput, inProjQKV, inProjZ, inProjB, inProjA, conv1D, aLog, dtBias, norm, outProj, convState, recurrentState}
	if err := in.validateVulkanEntry(normalizedInput, numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel, rmsNormEpsilon); err != nil {
		return Tensor{}, Tensor{}, Tensor{}, err
	}

	mixed := v.MatMul(inProjQKV, normalizedInput)
	z := v.MatMul(inProjZ, normalizedInput)
	beta := v.MatMul(inProjB, normalizedInput)
	alpha := v.MatMul(inProjA, normalizedInput)
	core, err := v.Qwen35GDNPreprojected(
		mixed, z, beta, alpha, conv1D, aLog, dtBias, norm, convState, recurrentState,
		1, numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel, rmsNormEpsilon,
	)
	if err != nil {
		return Tensor{}, Tensor{}, Tensor{}, err
	}
	output = v.MatMul(outProj, core)
	return output, convState, recurrentState, nil
}

// validateVulkanEntry admits a decode-shaped GDN call onto the Vulkan path:
// shared geometry validation first, then a residency check over every operand
// with no host fallback.
func (in qwen35GDNInputs) validateVulkanEntry(
	input Tensor,
	numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel int,
	rmsNormEpsilon float32,
) error {
	_, _, _, _, operands, err := in.entry(input, numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel, rmsNormEpsilon)
	if err != nil {
		return err
	}
	for _, operand := range operands {
		if _, ok := operand.t.buf.(*vulkanBuf); !ok {
			return fmt.Errorf("compute: vulkan Qwen GDN %s is not Vulkan-resident; no host fallback", operand.name)
		}
	}
	return nil
}

func (v *vulkanBackend) Qwen35GDNPreprojected(
	mixed, z, beta, alpha, conv1D, aLog, dtBias, norm, convState, recurrentState Tensor,
	tokens, numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel int,
	eps float32,
) (Tensor, error) {
	if tokens <= 0 || numKeyHeads <= 0 || numValueHeads <= 0 || keyHeadDim <= 0 || valueHeadDim <= 0 || valueHeadDim > 1024 || convKernel <= 0 || numValueHeads%numKeyHeads != 0 {
		return Tensor{}, fmt.Errorf("compute: vulkan Qwen GDN invalid geometry")
	}
	convDim := 2*numKeyHeads*keyHeadDim + numValueHeads*valueHeadDim
	valueDim := numValueHeads * valueHeadDim
	want := func(name string, t Tensor, n int) error {
		if t.Dtype != F32 || t.Numel() != n {
			return fmt.Errorf("compute: vulkan Qwen GDN %s elements/dtype=%d/%s, want %d/F32", name, t.Numel(), t.Dtype, n)
		}
		if _, ok := t.buf.(*vulkanBuf); !ok {
			return fmt.Errorf("compute: vulkan Qwen GDN %s is not Vulkan-resident", name)
		}
		return nil
	}
	checks := []error{
		want("mixed", mixed, tokens*convDim), want("z", z, tokens*valueDim),
		want("beta", beta, tokens*numValueHeads), want("alpha", alpha, tokens*numValueHeads),
		want("conv1d", conv1D, convDim*convKernel), want("a_log", aLog, numValueHeads),
		want("dt_bias", dtBias, numValueHeads), want("norm", norm, valueHeadDim),
		want("conv_state", convState, (convKernel-1)*convDim),
		want("recurrent_state", recurrentState, numValueHeads*keyHeadDim*valueHeadDim),
	}
	for _, err := range checks {
		if err != nil {
			return Tensor{}, err
		}
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()

	if v.isVectorGDNDisabledLocked() {
		return v.qwen35GDNPreprojectedScalarLocked(
			mixed, z, beta, alpha, conv1D, aLog, dtBias, norm, convState, recurrentState,
			tokens, convDim, numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel, eps,
		)
	}

	buf := v.dallocTransient(tokens * valueDim * F32.Bytes())
	out := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{tokens, valueDim}, buf: buf, be: v}
	status := int(C.fvk_qwen35_gdn_preprojected_f32(
		v.vp(mixed), v.vp(z), v.vp(beta), v.vp(alpha), v.vp(conv1D), v.vp(aLog), v.vp(dtBias), v.vp(norm),
		v.vp(convState), v.vp(recurrentState), v.vp(out), C.int(tokens), C.int(convDim), C.int(numKeyHeads), C.int(numValueHeads),
		C.int(keyHeadDim), C.int(valueHeadDim), C.int(convKernel), C.float(eps)))
	if status != 0 {
		C.fvk_free(v.vp(out))
		return Tensor{}, fmt.Errorf("compute: vulkan Qwen GDN kernel failed closed (code %d); no CPU fallback", status)
	}
	v.vectorGDNCalls++
	v.transient = append(v.transient, buf)
	return out, nil
}

func (v *vulkanBackend) qwen35GDNPreprojectedScalarLocked(
	mixed, z, beta, alpha, conv1D, aLog, dtBias, norm, convState, recurrentState Tensor,
	tokens, convDim, numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel int,
	eps float32,
) (Tensor, error) {
	valueDim := numValueHeads * valueHeadDim
	wasBatch := bool(C.fvk_batch_active())
	if wasBatch {
		C.fvk_batch_flush()
	}

	readBuf := func(t Tensor) []float32 {
		b, ok := t.buf.(*vulkanBuf)
		if !ok || b == nil {
			return make([]float32, t.Numel())
		}
		data := make([]float32, t.Numel())
		if b.ptr != nil && len(data) > 0 {
			C.fvk_d2h(unsafe.Pointer(&data[0]), b.ptr, C.size_t(len(data)*4))
		}
		return data
	}

	mixedHost := readBuf(mixed)
	zHost := readBuf(z)
	betaHost := readBuf(beta)
	alphaHost := readBuf(alpha)
	convHost := readBuf(conv1D)
	aLogHost := readBuf(aLog)
	dtBiasHost := readBuf(dtBias)
	normHost := readBuf(norm)
	convStateHost := readBuf(convState)
	recStateHost := readBuf(recurrentState)

	outHost, nextCS, nextRS := qwen35GDNScalarRecurrence(
		mixedHost, zHost, betaHost, alphaHost, convHost, aLogHost, dtBiasHost, normHost,
		convStateHost, recStateHost,
		tokens, numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel, eps,
	)

	csBuf, _ := convState.buf.(*vulkanBuf)
	if csBuf != nil && csBuf.ptr != nil && len(nextCS) > 0 {
		C.fvk_h2d(csBuf.ptr, unsafe.Pointer(&nextCS[0]), C.size_t(len(nextCS)*4))
	}
	rsBuf, _ := recurrentState.buf.(*vulkanBuf)
	if rsBuf != nil && rsBuf.ptr != nil && len(nextRS) > 0 {
		C.fvk_h2d(rsBuf.ptr, unsafe.Pointer(&nextRS[0]), C.size_t(len(nextRS)*4))
	}

	buf := v.dallocTransient(tokens * valueDim * F32.Bytes())
	if buf != nil && buf.ptr != nil && len(outHost) > 0 {
		C.fvk_h2d(buf.ptr, unsafe.Pointer(&outHost[0]), C.size_t(len(outHost)*4))
	}
	if wasBatch {
		C.fvk_batch_begin()
	}
	out := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{tokens, valueDim}, buf: buf, be: v}
	v.scalarGDNCalls++
	v.transient = append(v.transient, buf)
	return out, nil
}

func qwen35GDNScalarRecurrence(
	mixed, z, beta, alpha, convW, aLog, dtBias, norm, convState, recurrent []float32,
	tokens, nK, nV, kHd, vHd, kernel int, eps float32,
) ([]float32, []float32, []float32) {
	convDim := 2*nK*kHd + nV*vHd
	keyDim := nK * kHd
	hist := kernel - 1
	cs := append([]float32(nil), convState...)
	rs := append([]float32(nil), recurrent...)
	out := make([]float32, tokens*nV*vHd)
	sig := func(x float32) float32 { return 1 / (1 + float32(math.Exp(float64(-x)))) }
	silu := func(x float32) float32 { return x * sig(x) }
	soft := func(x float32) float32 { return float32(math.Log1p(math.Exp(float64(x)))) }
	qScale := 1 / float32(math.Sqrt(float64(kHd)))
	for t := 0; t < tokens; t++ {
		co := make([]float32, convDim)
		for c := 0; c < convDim; c++ {
			s := mixed[t*convDim+c] * convW[c*kernel+hist]
			for k := 0; k < hist; k++ {
				s += cs[k*convDim+c] * convW[c*kernel+k]
			}
			for k := 0; k+1 < hist; k++ {
				cs[k*convDim+c] = cs[(k+1)*convDim+c]
			}
			if hist > 0 {
				cs[(hist-1)*convDim+c] = mixed[t*convDim+c]
			}
			co[c] = s * sig(s)
		}
		for h := 0; h < nV; h++ {
			kh := h / (nV / nK)
			q2, k2 := float32(0), float32(0)
			for i := 0; i < kHd; i++ {
				q := co[kh*kHd+i]
				k := co[keyDim+kh*kHd+i]
				q2 += q * q
				k2 += k * k
			}
			qi := (1 / float32(math.Sqrt(float64(q2)+1e-6))) * qScale
			ki := 1 / float32(math.Sqrt(float64(k2)+1e-6))
			av := float32(math.Exp(float64(-float32(math.Exp(float64(aLog[h]))) * soft(alpha[t*nV+h]+dtBias[h]))))
			bv := sig(beta[t*nV+h])
			vals := make([]float32, vHd)
			for d := 0; d < vHd; d++ {
				v := co[2*keyDim+h*vHd+d]
				kvmem := float32(0)
				for i := 0; i < kHd; i++ {
					k := co[keyDim+kh*kHd+i] * ki
					si := (h*kHd+i)*vHd + d
					rs[si] *= av
					kvmem += rs[si] * k
				}
				delta := (v - kvmem) * bv
				acc := float32(0)
				for i := 0; i < kHd; i++ {
					q := co[kh*kHd+i] * qi
					k := co[keyDim+kh*kHd+i] * ki
					si := (h*kHd+i)*vHd + d
					rs[si] += k * delta
					acc += q * rs[si]
				}
				vals[d] = acc
			}
			ss := float32(0)
			for _, x := range vals {
				ss += x * x
			}
			inv := 1 / float32(math.Sqrt(float64(ss/float32(vHd)+eps)))
			for d, x := range vals {
				zv := z[t*nV*vHd+h*vHd+d]
				out[t*nV*vHd+h*vHd+d] = norm[d] * (x * inv) * silu(zv)
			}
		}
	}
	return out, cs, rs
}

func checkVulkanBuffer(t Tensor, op string) *vulkanBuf {
	if t.buf == nil {
		panic(fmt.Sprintf("compute: vulkan %s tensor buffer is nil", op))
	}
	b, ok := t.buf.(*vulkanBuf)
	if !ok || b == nil || b.ptr == nil {
		panic(fmt.Sprintf("compute: vulkan %s tensor is not Vulkan-resident", op))
	}
	return b
}

func (v *vulkanBackend) checkVulkanBuffer(t Tensor, op string) *vulkanBuf {
	return checkVulkanBuffer(t, op)
}

// SplitQwen35QueryGate splits an interleaved query-gate projection panel [2*nHeads*headDim]
// into separated query [nHeads*headDim] and gate [nHeads*headDim] device tensors on Vulkan.
func (v *vulkanBackend) SplitQwen35QueryGate(qg Tensor, nHeads, headDim int) (Tensor, Tensor) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	checkVulkanBuffer(qg, "SplitQwen35QueryGate input")
	width := nHeads * headDim
	if width <= 0 || qg.Numel()%(2*width) != 0 {
		panic(fmt.Sprintf("compute: vulkan SplitQwen35QueryGate dimension mismatch: numel=%d want multiple of %d", qg.Numel(), 2*width))
	}
	tokens := qg.Numel() / (2 * width)
	qShape := []int{tokens, width}
	gateShape := []int{tokens, width}
	if tokens == 1 {
		qShape = []int{width}
		gateShape = []int{width}
	}
	startTr := len(v.transient)
	q, qb := v.devTr(qShape, F32)
	gate, gb := v.devTr(gateShape, F32)
	status := C.fvk_qwen35_split_qg_panel_f32(v.vp(qg), v.vp(q), v.vp(gate), C.int(tokens), C.int(nHeads), C.int(headDim))
	if status != 0 {
		if qb.ptr != nil {
			C.fvk_free(qb.ptr)
			qb.ptr = nil
		}
		if gb.ptr != nil {
			C.fvk_free(gb.ptr)
			gb.ptr = nil
		}
		v.transient = v.transient[:startTr]
		panic(fmt.Sprintf("compute: vulkan SplitQwen35QueryGate failed closed with status %d", int(status)))
	}
	return q, gate
}

// PartialRoPEQK applies rotary position embedding to the first rotaryDim dimensions
// of each head in Q and K on Vulkan without host round-trips.
func (v *vulkanBackend) PartialRoPEQK(q, k Tensor, pos, nQHeads, nKHeads, headDim, rotaryDim int, theta float64) (Tensor, Tensor) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	checkVulkanBuffer(q, "PartialRoPEQK Q")
	checkVulkanBuffer(k, "PartialRoPEQK K")
	qWidth := nQHeads * headDim
	kWidth := nKHeads * headDim
	if qWidth <= 0 || kWidth <= 0 || q.Numel()%qWidth != 0 {
		panic(fmt.Sprintf("compute: vulkan PartialRoPEQK dimension mismatch: q=%d want multiple of %d", q.Numel(), qWidth))
	}
	tokens := q.Numel() / qWidth
	if k.Numel() != tokens*kWidth {
		panic(fmt.Sprintf("compute: vulkan PartialRoPEQK token mismatch: q tokens=%d, k=%d want %d", tokens, k.Numel(), tokens*kWidth))
	}
	qShape := []int{tokens, qWidth}
	kShape := []int{tokens, kWidth}
	if tokens == 1 {
		qShape = []int{qWidth}
		kShape = []int{kWidth}
	}
	startTr := len(v.transient)
	qOut, qb := v.devTr(qShape, F32)
	kOut, kb := v.devTr(kShape, F32)
	status := C.fvk_qwen35_partial_rope_panel_f32(
		v.vp(q), v.vp(k), v.vp(qOut), v.vp(kOut),
		C.int(tokens), C.int(pos), C.int(nQHeads), C.int(nKHeads),
		C.int(headDim), C.int(rotaryDim), C.double(theta),
	)
	if status != 0 {
		if qb.ptr != nil {
			C.fvk_free(qb.ptr)
			qb.ptr = nil
		}
		if kb.ptr != nil {
			C.fvk_free(kb.ptr)
			kb.ptr = nil
		}
		v.transient = v.transient[:startTr]
		panic(fmt.Sprintf("compute: vulkan PartialRoPEQK failed closed with status %d", int(status)))
	}
	return qOut, kOut
}

// SigmoidMulInPlace computes x = x * sigmoid(gate) in place on Vulkan.
func (v *vulkanBackend) SigmoidMulInPlace(x, gate Tensor) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	checkVulkanBuffer(x, "SigmoidMulInPlace X")
	checkVulkanBuffer(gate, "SigmoidMulInPlace Gate")
	n := x.Numel()
	if gate.Numel() != n {
		panic(fmt.Sprintf("compute: vulkan SigmoidMulInPlace shape mismatch: x=%d gate=%d", n, gate.Numel()))
	}
	status := C.fvk_sigmoid_mul_f32(v.vp(x), v.vp(gate), C.int(n))
	if status != 0 {
		panic(fmt.Sprintf("compute: vulkan SigmoidMulInPlace failed closed with status %d", int(status)))
	}
}

// Qwen35GDNConvTiledChannelTranspose runs 2D block-tiled channel transpose and causal depthwise 1D
// convolution for DeltaNet conv-state concatenation on Vulkan / RDNA 3.5 (gfx1151).
// Memory reads are distributed across all 16 pseudo-channels with 256-bit bus alignment,
// and convolution state is staged in LDS using Pad-1 bank stride (65 floats) to eliminate
// 32-way bank conflicts on Wave32.
func (v *vulkanBackend) Qwen35GDNConvTiledChannelTranspose(
	mixed, conv1D, convState Tensor,
	tokens, convDim, convKernel int,
) (output, nextConvState Tensor, err error) {
	if tokens <= 0 || convDim <= 0 || convKernel <= 0 {
		return Tensor{}, Tensor{}, &Qwen35GDNGeometryError{
			Operand: "geometry",
			Reason:  fmt.Sprintf("invalid conv dimensions tokens=%d, convDim=%d, convKernel=%d", tokens, convDim, convKernel),
			Err:     ErrVulkanInvalidGeometry,
		}
	}
	strideBytes := convDim * F32.Bytes()
	if !ValidateDeltaNet256BitBusAlignment(strideBytes) {
		return Tensor{}, Tensor{}, &Qwen35GDNGeometryError{
			Operand: "convDim",
			Reason:  fmt.Sprintf("convDim %d (stride %d bytes) violates 256-bit bus alignment", convDim, strideBytes),
			Err:     ErrVulkanInvalidGeometry,
		}
	}

	want := func(name string, t Tensor, n int) error {
		if t.Dtype != F32 || t.Numel() != n {
			return &Qwen35GDNGeometryError{
				Operand: name,
				Got:     t.Shape,
				Reason:  fmt.Sprintf("elements/dtype=%d/%s, want %d/F32", t.Numel(), t.Dtype, n),
				Err:     ErrVulkanInvalidGeometry,
			}
		}
		if t.buf == nil {
			return &Qwen35GDNResidencyError{
				Operand: name,
				Reason:  "tensor buffer is nil",
				Err:     ErrVulkanInvalidGeometry,
			}
		}
		vb, ok := t.buf.(*vulkanBuf)
		if !ok || vb == nil || vb.ptr == nil {
			return &Qwen35GDNResidencyError{
				Operand: name,
				Reason:  "tensor is not Vulkan-resident",
				Err:     ErrVulkanInvalidGeometry,
			}
		}
		return nil
	}
	hist := convKernel - 1
	if err := want("mixed", mixed, tokens*convDim); err != nil {
		return Tensor{}, Tensor{}, err
	}
	if err := want("conv1d", conv1D, convDim*convKernel); err != nil {
		return Tensor{}, Tensor{}, err
	}
	if hist > 0 {
		if err := want("conv_state", convState, hist*convDim); err != nil {
			return Tensor{}, Tensor{}, err
		}
	}

	vulkanMu.Lock()
	defer vulkanMu.Unlock()

	if v.isVectorGDNDisabledLocked() {
		return v.qwen35GDNConvTiledScalarLocked(mixed, conv1D, convState, tokens, convDim, convKernel)
	}

	outBuf := v.dallocTransient(tokens * convDim * F32.Bytes())
	output = Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{tokens, convDim}, buf: outBuf, be: v}

	var statePtr unsafe.Pointer
	if hist > 0 {
		statePtr = v.vp(convState)
	} else {
		statePtr = v.vp(output)
	}

	status := int(C.fvk_qwen35_gdn_conv_tiled_transpose_f32(
		v.vp(mixed), v.vp(conv1D), statePtr, v.vp(output),
		C.int(tokens), C.int(convDim), C.int(convKernel),
	))
	if status != 0 {
		C.fvk_free(v.vp(output))
		return Tensor{}, Tensor{}, &Qwen35GDNKernelError{Stage: "qwen35_gdn_conv_tiled_transpose", Code: status}
	}
	v.vectorGDNCalls++
	v.transient = append(v.transient, outBuf)
	return output, convState, nil
}

func (v *vulkanBackend) qwen35GDNConvTiledScalarLocked(
	mixed, conv1D, convState Tensor,
	tokens, convDim, convKernel int,
) (output, nextConvState Tensor, err error) {
	wasBatch := bool(C.fvk_batch_active())
	if wasBatch {
		C.fvk_batch_flush()
	}

	readBuf := func(t Tensor) []float32 {
		b, ok := t.buf.(*vulkanBuf)
		if !ok || b == nil {
			return make([]float32, t.Numel())
		}
		data := make([]float32, t.Numel())
		if b.ptr != nil && len(data) > 0 {
			C.fvk_d2h(unsafe.Pointer(&data[0]), b.ptr, C.size_t(len(data)*4))
		}
		return data
	}

	mixedHost := readBuf(mixed)
	convHost := readBuf(conv1D)
	var convStateHost []float32
	if convKernel > 1 {
		convStateHost = readBuf(convState)
	}

	outHost, nextCS, _, err := Tiled16ChannelTransposeConcat(mixedHost, convHost, tokens, convDim, convKernel, convStateHost)
	if err != nil {
		return Tensor{}, Tensor{}, err
	}

	if convKernel > 1 && len(nextCS) > 0 {
		csBuf, _ := convState.buf.(*vulkanBuf)
		if csBuf != nil && csBuf.ptr != nil {
			C.fvk_h2d(csBuf.ptr, unsafe.Pointer(&nextCS[0]), C.size_t(len(nextCS)*4))
		}
	}

	outBuf := v.dallocTransient(tokens * convDim * F32.Bytes())
	if outBuf != nil && outBuf.ptr != nil && len(outHost) > 0 {
		C.fvk_h2d(outBuf.ptr, unsafe.Pointer(&outHost[0]), C.size_t(len(outHost)*4))
	}
	if wasBatch {
		C.fvk_batch_begin()
	}
	output = Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{tokens, convDim}, buf: outBuf, be: v}
	v.scalarGDNCalls++
	v.transient = append(v.transient, outBuf)
	return output, convState, nil
}
