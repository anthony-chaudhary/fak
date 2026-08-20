package agent

import (
	"bytes"
	"log"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type qwen35IdentityBackend struct{ compute.Backend }

func (b *qwen35IdentityBackend) Name() string { return "cuda" }
func (*qwen35IdentityBackend) Qwen35GDNPath() string {
	return model.Qwen35GDNCUDAPath
}
func (b *qwen35IdentityBackend) Qwen35GDNDecode(
	normalizedInput,
	inProjQKV, inProjZ, inProjB, inProjA,
	conv1D, aLog, dtBias, norm, outProj,
	convState, recurrentState compute.Tensor,
	numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel int,
	rmsNormEpsilon float32,
) (compute.Tensor, compute.Tensor, compute.Tensor, error) {
	return compute.Tensor{}, compute.Tensor{}, compute.Tensor{}, nil
}

func TestInKernelExecutionIdentityQwen35CUDA(t *testing.T) {
	m := model.NewSynthetic(model.Config{LayerTypes: []string{"linear_attention"}})
	p := &InKernelPlanner{m: m, backend: &qwen35IdentityBackend{Backend: compute.Default()}}
	backend, path := p.executionIdentity()
	if backend != "cuda" || path != model.Qwen35GDNCUDAPath {
		t.Fatalf("executionIdentity() = backend=%q path=%q, want cuda/%q", backend, path, model.Qwen35GDNCUDAPath)
	}
}

func TestInKernelExecutionIdentityCPUReference(t *testing.T) {
	m := model.NewSynthetic(model.Config{LayerTypes: []string{"linear_attention"}})
	backend, path := (&InKernelPlanner{m: m}).executionIdentity()
	if backend != "cpu-ref" || path != "cpu/qwen35-gdn-reference" {
		t.Fatalf("executionIdentity() = backend=%q path=%q", backend, path)
	}
}

func TestInKernelExecutionIdentityQwen35Metal(t *testing.T) {
	m := model.NewSynthetic(model.Config{LayerTypes: []string{"linear_attention"}})
	backend, path := (&InKernelPlanner{m: m, metal: true, q4k: true}).executionIdentity()
	if backend != "metal" || path != "metal/qwen35-hybrid-session-v1" {
		t.Fatalf("executionIdentity() = backend=%q path=%q, want Metal Qwen hybrid identity", backend, path)
	}
}

func TestInKernelExecutionIdentityGenericDevice(t *testing.T) {
	p := &InKernelPlanner{m: model.NewSynthetic(model.Config{}), backend: compute.Default()}
	backend, path := p.executionIdentity()
	if backend != compute.Default().Name() || path != "device/generic" {
		t.Fatalf("executionIdentity() = backend=%q path=%q", backend, path)
	}
}

func TestInKernelExecutionLogNamesQwen35CUDAPath(t *testing.T) {
	m := model.NewSynthetic(model.Config{LayerTypes: []string{"linear_attention"}})
	p := &InKernelPlanner{m: m, modelID: "Qwen3.6-27B-Q4_K_M", backend: &qwen35IdentityBackend{Backend: compute.Default()}}
	var buf bytes.Buffer
	old := log.Writer()
	log.SetOutput(&buf)
	t.Cleanup(func() { log.SetOutput(old) })
	p.logExecutionSummary("avx512", "+fused", 23, 0, 0, 23, 192.63, 0.1, 1, 7.08, 0.1)
	got := buf.String()
	for _, want := range []string{"backend=cuda", "forward_path=" + model.Qwen35GDNCUDAPath, "q8dec=avx512+fused/"} {
		if !strings.Contains(got, want) {
			t.Fatalf("execution log %q missing %q", got, want)
		}
	}
}

func TestInKernelExecutionLogNamesQwen35MetalPath(t *testing.T) {
	m := model.NewSynthetic(model.Config{LayerTypes: []string{"linear_attention"}})
	p := &InKernelPlanner{m: m, modelID: "Qwen3.8-27B-Q4_K_M", metal: true, q4k: true}
	var buf bytes.Buffer
	old := log.Writer()
	log.SetOutput(&buf)
	t.Cleanup(func() { log.SetOutput(old) })
	p.logExecutionSummary("neon", "", 22, 3, 0, 22, 6.41, 3.4, 3, 1.07, 2.8)
	got := buf.String()
	for _, want := range []string{"backend=metal", "forward_path=metal/qwen35-hybrid-session-v1", "q4k=true"} {
		if !strings.Contains(got, want) {
			t.Fatalf("execution log %q missing %q", got, want)
		}
	}
}
