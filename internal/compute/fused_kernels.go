package compute

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
)

// Optional fused kernels that have a composed fallback.
const (
	FusedOpQ4KRMSNormMatMul2  = "q4k_rmsnorm_matmul2"
	FusedOpQ4KSwiGLUMatMulAdd = "q4k_swiglu_matmul_add"
	FusedOpQ2KRMSNormMatMul2  = "q2k_rmsnorm_matmul2"
	FusedOpGDNQ8InProj        = "gdn_q8_in_proj"
)

var fusedOps = [...]string{FusedOpQ4KRMSNormMatMul2, FusedOpQ4KSwiGLUMatMulAdd, FusedOpQ2KRMSNormMatMul2, FusedOpGDNQ8InProj}

// FusedCount is one op's cumulative dispatch count in one mode ("fused" or "composed").
type FusedCount struct {
	Op    string
	Mode  string
	Count uint64
}

var fusedDispatches struct {
	declared atomic.Bool
	counts   [len(fusedOps)][2]atomic.Uint64
}

// DeclareFusedOps exposes every fused op at 0 once a backend that dispatches
// them is live, so "never fused" reads differently from "not instrumented".
func DeclareFusedOps() { fusedDispatches.declared.Store(true) }

// CountFusedDispatch records one dispatch of op through its fused kernel or its
// composed fallback. An op outside the closed set is ignored.
func CountFusedDispatch(op string, fused bool) {
	for i, known := range fusedOps {
		if known == op {
			mode := 1
			if fused {
				mode = 0
			}
			fusedDispatches.counts[i][mode].Add(1)
			fusedDispatches.declared.Store(true)
			return
		}
	}
}

// FusedCounts returns every op in both modes, or nil before any backend declared them.
func FusedCounts() []FusedCount {
	if !fusedDispatches.declared.Load() {
		return nil
	}
	out := make([]FusedCount, 0, 2*len(fusedOps))
	for i, op := range fusedOps {
		out = append(out,
			FusedCount{Op: op, Mode: "fused", Count: fusedDispatches.counts[i][0].Load()},
			FusedCount{Op: op, Mode: "composed", Count: fusedDispatches.counts[i][1].Load()})
	}
	return out
}

// ErrFusedShaderMissing is the sentinel under every FusedShaderMissingError.
var ErrFusedShaderMissing = errors.New("compute: optional fused shader is missing; the composed path runs instead")

// FusedShaderMissingError is the typed init warning for an absent fused .spv.
type FusedShaderMissingError struct {
	Op     string
	Shader string
	Cause  error
}

func (e *FusedShaderMissingError) Error() string {
	return fmt.Sprintf("%v: op=%s shader=%s: %v", ErrFusedShaderMissing, e.Op, e.Shader, e.Cause)
}

func (e *FusedShaderMissingError) Unwrap() []error { return []error{ErrFusedShaderMissing, e.Cause} }

var vulkanFusedShaders = []struct{ op, file string }{
	{FusedOpQ4KRMSNormMatMul2, "rmsnorm_q4k_matmul2.spv"},
	{FusedOpQ4KSwiGLUMatMulAdd, "swiglu_q4k_matmul_add.spv"},
}

// probeFusedShaders stats each optional fused shader under dir and returns the
// ops whose shader is present plus one typed warning per missing shader.
func probeFusedShaders(dir string, stat func(string) (os.FileInfo, error)) (map[string]bool, []*FusedShaderMissingError) {
	present := make(map[string]bool, len(vulkanFusedShaders))
	var missing []*FusedShaderMissingError
	for _, s := range vulkanFusedShaders {
		path := filepath.Join(dir, s.file)
		if _, err := stat(path); err != nil {
			missing = append(missing, &FusedShaderMissingError{Op: s.op, Shader: path, Cause: err})
			continue
		}
		present[s.op] = true
	}
	return present, missing
}

var fusedShaderWarnings struct {
	mu   sync.Mutex
	list []*FusedShaderMissingError
}

func recordFusedShaderWarnings(ws []*FusedShaderMissingError, logf func(string, ...any)) {
	fusedShaderWarnings.mu.Lock()
	fusedShaderWarnings.list = append(fusedShaderWarnings.list[:0], ws...)
	fusedShaderWarnings.mu.Unlock()
	for _, w := range ws {
		logf("compute: warning: %v", w)
	}
}

// FusedShaderWarnings returns the missing-shader warnings recorded at backend init.
func FusedShaderWarnings() []*FusedShaderMissingError {
	fusedShaderWarnings.mu.Lock()
	defer fusedShaderWarnings.mu.Unlock()
	return append([]*FusedShaderMissingError(nil), fusedShaderWarnings.list...)
}

// FusionArm is one fusion that stays off unless its env var opts in.
type FusionArm struct {
	Op      string `json:"op"`
	Env     string `json:"env"`
	Enabled bool   `json:"enabled"`
}

func fusionOptIn(value string) bool {
	switch strings.TrimSpace(strings.ToLower(value)) {
	case "candidate", "fusion", "fused", "1", "true", "on", "yes":
		return true
	}
	return false
}

func q4kFusionOptIn() bool {
	v := os.Getenv("FAK_VULKAN_Q4K_FUSION")
	if v == "" {
		v = os.Getenv("FAK_VULKAN_Q4K_ARM")
	}
	return fusionOptIn(v)
}

func q2kFusionOptIn() bool { return fusionOptIn(computeEnvValue("FAK_VULKAN_Q2K_FUSION")) }

// DefaultOffFusions reports each default-off fusion arm and whether this
// process opted in.
func DefaultOffFusions() []FusionArm {
	q4k := q4kFusionOptIn()
	return []FusionArm{
		{Op: FusedOpQ4KRMSNormMatMul2, Env: "FAK_VULKAN_Q4K_FUSION", Enabled: q4k},
		{Op: FusedOpQ4KSwiGLUMatMulAdd, Env: "FAK_VULKAN_Q4K_FUSION", Enabled: q4k},
		{Op: FusedOpQ2KRMSNormMatMul2, Env: "FAK_VULKAN_Q2K_FUSION", Enabled: q2kFusionOptIn()},
	}
}
