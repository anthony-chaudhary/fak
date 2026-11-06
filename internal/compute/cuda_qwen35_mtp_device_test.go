//go:build cuda && cgo

package compute

import (
	"fmt"
	"math"
	"os"
	"testing"
)

// Required-device arithmetic witness; software source controls are separate.
// fak-test:runtime integration est=15s lane=optin
func TestCUDAMTPPrelayerMatchesComposition(t *testing.T) {
	required := os.Getenv("FAK_CUDA_MTP_REQUIRED") == "1"
	backend, ok := Lookup("cuda")
	if !ok || backend == nil {
		if required {
			t.Fatal("required real CUDA backend absent")
		}
		t.Skip("CUDA absent")
	}
	c, ok := backend.(*cudaBackend)
	if !ok {
		t.Fatalf("unexpected CUDA backend %T", backend)
	}
	if c.Qwen35MTPDraftPath() != Qwen35MTPDraftPath {
		if required {
			t.Fatal("set FAK_CUDA_QWEN35_MTP=1")
		}
		t.Skip("MTP pre-layer opt-in disabled")
	}
	for _, h := range []int{1, 255, 257, 5120} {
		t.Run(fmt.Sprintf("h%d", h), func(t *testing.T) {
			owner := c.ForwardExecutionOwner()
			owner.Lock()
			defer owner.Unlock()
			defer c.Recycle()
			var weights []Tensor
			defer func() {
				for _, x := range weights {
					c.Free(x)
				}
			}()
			upload := func(shape []int, values []float32) Tensor {
				x := c.UploadClass(NewF32(Default(), shape, values), F32, MemoryScratchpad, "mtp-device-witness")
				weights = append(weights, x)
				return x
			}
			ph, eh, hn, en := make([]float32, h), make([]float32, h), make([]float32, h), make([]float32, h)
			wh := make([]float32, 2*h*h)
			for i := 0; i < h; i++ {
				ph[i] = float32(i%17-8) / 9
				eh[i] = float32(i%13-6) / 7
				hn[i] = float32(i%11+1) / 8
				en[i] = float32(i%7+1) / 6
			}
			for i := range wh {
				wh[i] = float32(i%19-9) / 128
			}
			req := Qwen35MTPFuseRequest{PriorHidden: upload([]int{h}, ph), CurrentEmbedding: upload([]int{h}, eh),
				HiddenNorm: upload([]int{h}, hn), EmbeddingNorm: upload([]int{h}, en), FC: upload([]int{h, 2 * h}, wh), Epsilon: 1e-6}
			// Original five-stage device composition is independent of the candidate.
			e := c.RMSNorm(req.CurrentEmbedding, req.EmbeddingNorm, req.Epsilon)
			p := c.RMSNorm(req.PriorHidden, req.HiddenNorm, req.Epsilon)
			joined := upload([]int{2 * h}, make([]float32, 2*h))
			c.PromptPanelSetRow(joined, 0, h, e)
			c.PromptPanelSetRow(joined, 1, h, p)
			want := c.Read(c.MatMul(req.FC, joined))
			beforeH, beforeD := c.Qwen35MTPTransferBytes()
			got, err := c.Qwen35MTPFuse(req)
			if err != nil {
				t.Fatal(err)
			}
			afterH, afterD := c.Qwen35MTPTransferBytes()
			if afterH != beforeH || afterD != beforeD {
				t.Fatalf("intermediate transfers H2D %d D2H %d", afterH-beforeH, afterD-beforeD)
			}
			if !got.Ready() {
				t.Fatal("successful checked output not ready")
			}
			check := func(name string, x, y []float32) {
				t.Helper()
				if len(x) != len(y) {
					t.Fatal(name, "length")
				}
				for i := range x {
					if math.IsNaN(float64(x[i])) || math.IsInf(float64(x[i]), 0) || math.Float32bits(x[i]) != math.Float32bits(y[i]) {
						t.Fatalf("%s[%d] %08x != %08x", name, i, math.Float32bits(x[i]), math.Float32bits(y[i]))
					}
				}
			}
			check("output", c.Read(got), want)
			check("prior", c.Read(req.PriorHidden), ph)
			check("embedding", c.Read(req.CurrentEmbedding), eh)
			check("hidden norm", c.Read(req.HiddenNorm), hn)
			check("embedding norm", c.Read(req.EmbeddingNorm), en)
			check("FC", c.Read(req.FC), wh)
			// Pure Go geometry refusal must allocate and submit nothing.
			cudaMu.Lock()
			transients := len(c.transient)
			cudaMu.Unlock()
			malformed := req
			malformed.FC.Shape = []int{2 * h, h}
			if h == 1 {
				malformed.Epsilon = 0
			}
			if out, err := c.Qwen35MTPFuse(malformed); err == nil || out.Buf() != nil {
				t.Fatal("malformed request accepted")
			}
			cudaMu.Lock()
			after := len(c.transient)
			cudaMu.Unlock()
			if after != transients {
				t.Fatal("refused request allocated")
			}
		})
	}
}
