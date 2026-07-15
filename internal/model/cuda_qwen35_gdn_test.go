//go:build cuda

package model

import (
	"math"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

const cudaGDNRequiredEnv = "FAK_CUDA_GDN_REQUIRED"

type cudaGDNParityBackend interface {
	compute.Backend
	Qwen35GDNBackend
	HostXferBytes() uint64
	ResetHostXfer()
	H2DXferBytes() uint64
	ResetH2DXfer()
	Qwen35GDNOperationCount() uint64
	ResetQwen35GDNOperationCount()
	Recycle()
}

func requiredCUDAGDNParityBackend(t *testing.T) cudaGDNParityBackend {
	t.Helper()
	be, ok := compute.Lookup("cuda")
	if !ok || be == nil || be.Name() != "cuda" {
		if os.Getenv(cudaGDNRequiredEnv) == "1" {
			t.Fatalf("%s=1: real CUDA GDN parity is required, but exact backend cuda is not registered", cudaGDNRequiredEnv)
		}
		t.Skip("exact cuda backend not registered (set FAK_CUDA_GDN_REQUIRED=1 on an acceptance node to prohibit skips)")
	}
	gdn, ok := be.(cudaGDNParityBackend)
	if !ok {
		t.Fatalf("registered cuda backend %T lacks the complete GDN operation/transfer witness surface", be)
	}
	return gdn
}

func uploadModelGDNFixture(t *testing.T, be compute.Backend, shape []int, data []float32, class compute.MemoryClass, site string) compute.Tensor {
	t.Helper()
	resident := uploadHostF32Class(be, shape, data, class, site)
	t.Cleanup(func() { be.Free(resident) })
	return resident
}

type qwen35GDNParityStats struct {
	cosine, maxAbs, referenceNorm, deviceNorm float64
}

func compareQwen35GDNVector(t *testing.T, label string, reference, device []float32) qwen35GDNParityStats {
	t.Helper()
	if len(reference) != len(device) {
		t.Fatalf("%s length = %d, want %d", label, len(device), len(reference))
	}
	var dot, referenceNorm2, deviceNorm2, maxAbs float64
	for i := range reference {
		r, d := float64(reference[i]), float64(device[i])
		if math.IsNaN(r) || math.IsInf(r, 0) {
			t.Fatalf("%s CPU reference[%d] is non-finite: %v", label, i, reference[i])
		}
		if math.IsNaN(d) || math.IsInf(d, 0) {
			t.Fatalf("%s CUDA result[%d] is non-finite: %v", label, i, device[i])
		}
		dot += r * d
		referenceNorm2 += r * r
		deviceNorm2 += d * d
		if delta := math.Abs(r - d); delta > maxAbs {
			maxAbs = delta
		}
	}
	if referenceNorm2 == 0 || deviceNorm2 == 0 {
		t.Fatalf("%s has degenerate zero norm: cpu=%g cuda=%g", label, math.Sqrt(referenceNorm2), math.Sqrt(deviceNorm2))
	}
	cosine := dot / math.Sqrt(referenceNorm2*deviceNorm2)
	if math.IsNaN(cosine) || math.IsInf(cosine, 0) || cosine < Qwen35GDNParityCosineMin {
		t.Fatalf("%s cosine %.9f < %.3f (max_abs=%.3e cpu_norm=%.6g cuda_norm=%.6g)",
			label, cosine, Qwen35GDNParityCosineMin, maxAbs, math.Sqrt(referenceNorm2), math.Sqrt(deviceNorm2))
	}
	return qwen35GDNParityStats{
		cosine: cosine, maxAbs: maxAbs,
		referenceNorm: math.Sqrt(referenceNorm2), deviceNorm: math.Sqrt(deviceNorm2),
	}
}

// TestCUDAQwen35GDNMultiStepMatchesModelCPU consumes the external #4789 CPU
// corpus verbatim. It never constructs a Model or invokes the CPU oracle on the
// CUDA node. Four distinct inputs start from the serialized zero state, fill and
// roll the convolution window, and reuse the same in-place device state across a
// Recycle boundary after every step.
func TestCUDAQwen35GDNMultiStepMatchesModelCPU(t *testing.T) {
	required := os.Getenv(cudaGDNRequiredEnv) == "1"
	corpusPath, skip, err := selectQwen35GDNCorpus(required, os.Getenv(qwen35GDNCorpusPathEnv))
	if err != nil {
		t.Fatal(err)
	}
	if skip {
		t.Skipf("external GDN corpus not configured (set %s=<corpus-dir>; %s=1 prohibits this skip)", qwen35GDNCorpusPathEnv, cudaGDNRequiredEnv)
	}
	corpus, err := loadQwen35GDNCorpus(corpusPath)
	if err != nil {
		t.Fatalf("load required external GDN corpus: %v", err)
	}
	be := requiredCUDAGDNParityBackend(t)
	t.Cleanup(be.Recycle)
	if Qwen35GDNCUDAPath != compute.Qwen35GDNCUDAPath || be.Qwen35GDNPath() != Qwen35GDNCUDAPath {
		t.Fatalf("GDN path mismatch: model=%q compute=%q backend=%q", Qwen35GDNCUDAPath, compute.Qwen35GDNCUDAPath, be.Qwen35GDNPath())
	}

	g := corpus.Metadata.Geometry
	nK, nV, kHd, vHd, valueDim, convDim := g.NumKeyHeads, g.NumValueHeads, g.KeyHeadDim, g.ValueHeadDim, g.ValueDim, g.ConvDim
	hidden, kernel := g.HiddenSize, g.ConvKernel
	tensor := func(name string) []float32 {
		data, ok := corpus.Tensors[name]
		if !ok {
			t.Fatalf("verified corpus omitted tensor %q", name)
		}
		return data
	}
	weight := func(name string, shape []int) compute.Tensor {
		return uploadModelGDNFixture(t, be, shape, tensor(name), compute.MemoryWeights, "qwen35-gdn-weight "+name)
	}

	inQKV := weight(qwen35GDNWeightInQKV, []int{convDim, hidden})
	inZ := weight(qwen35GDNWeightInZ, []int{valueDim, hidden})
	inB := weight(qwen35GDNWeightInB, []int{nV, hidden})
	inA := weight(qwen35GDNWeightInA, []int{nV, hidden})
	convW := weight(qwen35GDNWeightConv, []int{convDim, 1, kernel})
	aLog := weight(qwen35GDNWeightALog, []int{nV})
	dtBias := weight(qwen35GDNWeightDT, []int{nV})
	norm := weight(qwen35GDNWeightNorm, []int{vHd})
	outW := weight(qwen35GDNWeightOut, []int{hidden, valueDim})
	convState := uploadModelGDNFixture(t, be, []int{kernel - 1, convDim}, tensor(qwen35GDNInitialConvState), compute.MemoryKVCache, "qwen35-gdn-conv-state")
	recurrentState := uploadModelGDNFixture(t, be, []int{nV, kHd, vHd}, tensor(qwen35GDNInitialRecurrentState), compute.MemoryKVCache, "qwen35-gdn-recurrent-state")
	convIdentity, recurrentIdentity := convState.Buf(), recurrentState.Buf()

	for _, stepMeta := range corpus.Metadata.Steps {
		step := stepMeta.Index
		input := tensor(stepMeta.Input)
		wantOutput := tensor(stepMeta.Output.Tensor)
		wantConv := tensor(stepMeta.ConvState.Tensor)
		wantRecurrent := tensor(stepMeta.RecurrentState.Tensor)

		x := uploadModelGDNFixture(t, be, []int{hidden}, input, compute.MemoryActivation, "qwen35-gdn-step-input")
		be.ResetHostXfer()
		be.ResetH2DXfer()
		opsBefore := be.Qwen35GDNOperationCount()
		h2dBefore, d2hBefore := be.H2DXferBytes(), be.HostXferBytes()
		gotOutputDev, nextConv, nextRecurrent, err := be.Qwen35GDNDecode(
			x, inQKV, inZ, inB, inA, convW, aLog, dtBias, norm, outW,
			convState, recurrentState,
			nK, nV, kHd, vHd, kernel, float32(corpus.Metadata.Epsilon),
		)
		be.Free(x)
		if err != nil {
			t.Fatalf("step %d real CUDA GDN operation: %v", step, err)
		}
		opsDelta := be.Qwen35GDNOperationCount() - opsBefore
		h2dInside := be.H2DXferBytes() - h2dBefore
		d2hInside := be.HostXferBytes() - d2hBefore
		if opsDelta != 1 || h2dInside != 0 || d2hInside != 0 {
			t.Fatalf("step %d witness deltas: operations=%d H2D=%d D2H=%d, want 1/0/0", step, opsDelta, h2dInside, d2hInside)
		}
		if nextConv.Buf() != convIdentity || nextRecurrent.Buf() != recurrentIdentity {
			t.Fatalf("step %d state identity changed: conv=%p/%p recurrent=%p/%p", step, nextConv.Buf(), convIdentity, nextRecurrent.Buf(), recurrentIdentity)
		}
		if !gotOutputDev.Ready() {
			t.Fatalf("step %d output is not Ready after final stream synchronization", step)
		}

		gotOutput := be.Read(gotOutputDev)
		be.Recycle()
		if nextConv.Buf() != convIdentity || nextRecurrent.Buf() != recurrentIdentity || !nextConv.Ready() || !nextRecurrent.Ready() {
			t.Fatalf("step %d KV-cache state did not survive Recycle", step)
		}
		gotConv := be.Read(nextConv)
		gotRecurrent := be.Read(nextRecurrent)
		wantProofD2H := uint64(len(gotOutput)+len(gotConv)+len(gotRecurrent)) * uint64(compute.F32.Bytes())
		if got := be.HostXferBytes() - d2hBefore; got != wantProofD2H {
			t.Fatalf("step %d proof D2H bytes = %d, want %d", step, got, wantProofD2H)
		}
		if got := be.H2DXferBytes() - h2dBefore; got != 0 {
			t.Fatalf("step %d proof phase unexpectedly copied %d H2D bytes", step, got)
		}

		outputStats := compareQwen35GDNVector(t, "output", wantOutput, gotOutput)
		convStats := compareQwen35GDNVector(t, "conv_state", wantConv, gotConv)
		recurrentStats := compareQwen35GDNVector(t, "recurrent_state", wantRecurrent, gotRecurrent)
		t.Logf("step=%d corpus=%s manifest_sha256=%s path=%s operations=%d h2d_inside=%d d2h_inside=%d state_identity=true state_durable=true output_cosine=%.9f output_max_abs=%.3e output_norm_cpu=%.6g output_norm_cuda=%.6g conv_cosine=%.9f conv_max_abs=%.3e conv_norm_cpu=%.6g conv_norm_cuda=%.6g recurrent_cosine=%.9f recurrent_max_abs=%.3e recurrent_norm_cpu=%.6g recurrent_norm_cuda=%.6g",
			step, corpus.Metadata.Format, corpus.ManifestSHA256, be.Qwen35GDNPath(), opsDelta, h2dInside, d2hInside,
			outputStats.cosine, outputStats.maxAbs, outputStats.referenceNorm, outputStats.deviceNorm,
			convStats.cosine, convStats.maxAbs, convStats.referenceNorm, convStats.deviceNorm,
			recurrentStats.cosine, recurrentStats.maxAbs, recurrentStats.referenceNorm, recurrentStats.deviceNorm)

		convState, recurrentState = nextConv, nextRecurrent
	}
}
