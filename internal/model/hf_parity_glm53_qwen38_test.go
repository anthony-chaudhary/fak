package model

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// SyntheticMLPReceipt records host operators compared with a local float64 formula.
// No HuggingFace runtime or downloaded model weights execute in this test.
type SyntheticMLPReceipt struct {
	Schema               string  `json:"schema"`
	Issue                int     `json:"issue"`
	Role                 string  `json:"role"`
	Engine               string  `json:"engine"`
	Fixture              string  `json:"fixture"`
	Reference            string  `json:"reference"`
	ReferenceScope       string  `json:"reference_scope"`
	HuggingFaceQualified bool    `json:"huggingface_qualified"`
	BitExactParityProven bool    `json:"bit_exact_parity_proven"`
	DenseCosineScore     float64 `json:"dense_cosine_similarity"`
	MoECosineScore       float64 `json:"moe_cosine_similarity"`
	MinCosineThreshold   float64 `json:"min_cosine_threshold"`
}

// fak-test:runtime fast est=10ms lane=default
func TestSyntheticGLMMLPSoftwareReceipt(t *testing.T) {
	const inDim = 16
	const interDim = 32
	const numExperts = 8

	x := make([]float32, inDim)
	for i := range x {
		x[i] = float32(i%5-2) * 0.25
	}

	// Synthetic GLM-style dense operator check
	glmDenseParams := GLM5NextDenseMLPParams{
		InDim:    inDim,
		InterDim: interDim,
		WAct:     make([]float32, interDim*inDim),
		WUp:      make([]float32, interDim*inDim),
		WDown:    make([]float32, inDim*interDim),
	}
	for i := range glmDenseParams.WAct {
		glmDenseParams.WAct[i] = float32(i%7-3) * 0.05
		glmDenseParams.WUp[i] = float32(i%11-5) * 0.05
	}
	for i := range glmDenseParams.WDown {
		glmDenseParams.WDown[i] = float32(i%13-6) * 0.05
	}
	glmDenseNative := ExecuteGLM5NextDenseMLP(x, glmDenseParams)

	// Independent local float64 formula (SwiGLU: down(silu(act(x)) * up(x)))
	formula := func(params GLM5NextDenseMLPParams) []float32 {
		result := make([]float32, inDim)
		act := make([]float64, interDim)
		up := make([]float64, interDim)
		for o := 0; o < interDim; o++ {
			for i := 0; i < inDim; i++ {
				act[o] += float64(params.WAct[o*inDim+i]) * float64(x[i])
				up[o] += float64(params.WUp[o*inDim+i]) * float64(x[i])
			}
			// silu
			act[o] = act[o] / (1.0 + math.Exp(-act[o]))
		}
		for o := 0; o < inDim; o++ {
			var sum float64
			for i := 0; i < interDim; i++ {
				sum += float64(params.WDown[o*interDim+i]) * (act[i] * up[i])
			}
			result[o] = float32(sum)
		}

		return result
	}
	glmDenseFormulaRef := formula(glmDenseParams)

	glmDenseCos := epCosine(glmDenseNative, glmDenseFormulaRef)
	if glmDenseCos < 0.9999 {
		t.Fatalf("GLM-5.3 Dense cosine similarity %g < 0.9999", glmDenseCos)
	}

	// Synthetic GLM-style MoE composition check
	routerW := make([]float32, numExperts*inDim)
	for i := range routerW {
		routerW[i] = float32(i%5-2) * 0.1
	}

	experts := make([]GLM5NextExpertWeight, numExperts)
	for e := 0; e < numExperts; e++ {
		g := make([]float32, interDim*inDim)
		u := make([]float32, interDim*inDim)
		d := make([]float32, inDim*interDim)
		for i := range g {
			g[i] = float32((e+i)%7-3) * 0.05
			u[i] = float32((e+i)%11-5) * 0.05
		}
		for i := range d {
			d[i] = float32((e+i)%13-6) * 0.05
		}
		experts[e] = GLM5NextExpertWeight{WAct: g, WUp: u, WDown: d}
	}

	glmMoEParams := GLM5NextMoEMLPParams{
		InDim:         inDim,
		MoEInterDim:   interDim,
		RoutedExperts: experts,
	}

	route := RouteGLM5NextMoE(x, routerW, numExperts, 2)
	glmMoENative := ExecuteGLM5NextSparseMoE(x, route, glmMoEParams)

	// Golden MoE reference calculation
	glmMoEFormulaRef := make([]float32, inDim)
	for idx, expID := range route.ExpertIndices {
		w := route.Weights[idx]
		ew := glmMoEParams.RoutedExperts[expID]
		expParams := GLM5NextDenseMLPParams{
			InDim: inDim, InterDim: interDim, WAct: ew.WAct, WUp: ew.WUp, WDown: ew.WDown,
		}
		expOut := formula(expParams)
		for i := 0; i < inDim; i++ {
			glmMoEFormulaRef[i] += w * expOut[i]
		}
	}
	glmMoECos := epCosine(glmMoENative, glmMoEFormulaRef)
	if glmMoECos < 0.9999 {
		t.Fatalf("GLM-5.3 MoE cosine similarity %g < 0.9999", glmMoECos)
	}

	// 3. Emit receipt
	receipt := SyntheticMLPReceipt{
		Schema:               "fak-synthetic-glm-mlp-software-check/v2",
		Issue:                10949,
		Role:                 "software-check",
		Engine:               "fak-native",
		Fixture:              "synthetic GLM-style dense and 8-expert MoE operators (16 input, 32 intermediate)",
		Reference:            "local float64 SwiGLU formula",
		ReferenceScope:       "MLP math only; MoE shares native routing; no HuggingFace runtime, model weights, full forward, or Qwen execution",
		HuggingFaceQualified: false,
		BitExactParityProven: false,
		DenseCosineScore:     float64(glmDenseCos),
		MoECosineScore:       float64(glmMoECos),
		MinCosineThreshold:   0.9999,
	}
	receiptDir := t.TempDir()

	data, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		t.Fatalf("failed to marshal receipt: %v", err)
	}
	t.Logf("software receipt: %s", data)
	receiptPath := filepath.Join(receiptDir, "receipt.json")
	if err := os.WriteFile(receiptPath, data, 0644); err != nil {
		t.Fatalf("failed to write receipt: %v", err)
	}
}
