// q4k_smallp.go — platform-neutral policy for the short-prompt (2<=P<=20) K-quant prefill route
// (fak#13694). The scalar q4k_gemm kernel stages a fixed 64-token tile, so a 6-token prompt keeps
// only 2 of its 16 thread-columns busy and pays ~6.5x one decode step per projection. The small-P
// route instead encodes ceil(P/8) evenly split multi-token GEMV dispatches (q4k_gemv_multiN /
// q6k_gemv_multiN, N=2..8) that stream each weight row once per chunk. This file owns only the
// prompt-band predicate and the process opt-out, so the selector policy is testable on any host;
// the darwin selector (q4kGEMMModeForPrompt) and the native encoders consume it.

package metalgemm

import (
	"os"
	"strings"
	"sync/atomic"
)

const (
	// q4kSmallPMinPrompt is the smallest prompt the small-P route serves. P=1 is a decode GEMV
	// and keeps its own kernels.
	q4kSmallPMinPrompt = 2
	// q4kSmallPMaxPrompt is the largest prompt the small-P route serves: three <=8-token chunks.
	// Measured on M3 Pro (fak#13694, contended host): Q4_K small-P beats the scalar tile on every
	// Qwen 4B/7B projection shape through P=20 (4B ffn-up ties at 20) and loses from P=24 on the
	// 4B shapes, so above 20 the scalar tile stays the executed kernel. Q6_K wins through P=48
	// but shares this band so a graph never mixes routes at one P.
	q4kSmallPMaxPrompt = 20
)

// q4kUseSmallP gates the small-P route. Default ON; FAK_Q4K_SMALLP=0|off|false at process start,
// or SetGEMMUseSmallP(false), restores the scalar kernel for every prompt length.
var q4kUseSmallP atomic.Bool

func init() { q4kUseSmallP.Store(q4kSmallPEnvEnabled(os.Getenv("FAK_Q4K_SMALLP"))) }

// q4kSmallPEnvEnabled parses the FAK_Q4K_SMALLP opt-out. Unset or any other value keeps the
// default-on route; only an explicit 0/off/false disables it.
func q4kSmallPEnvEnabled(v string) bool {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "0", "off", "false":
		return false
	}
	return true
}

// q4kSmallPPromptInBand reports whether P lies inside the small-P route's prompt band,
// independent of the opt-out.
func q4kSmallPPromptInBand(P int) bool {
	return P >= q4kSmallPMinPrompt && P <= q4kSmallPMaxPrompt
}

// q4kSmallPEligible reports whether the production selector should request the small-P route
// for a prompt of P tokens: the opt-out is not set and P is inside the band.
func q4kSmallPEligible(P int) bool {
	return q4kUseSmallP.Load() && q4kSmallPPromptInBand(P)
}

// SetGEMMUseSmallP turns the 2<=P<=20 batched multi-token GEMV prefill route on (the default) or
// off. Off restores the scalar q4k_gemm / q6k_gemm kernels for every prompt length.
func SetGEMMUseSmallP(on bool) { q4kUseSmallP.Store(on) }

// GEMMUseSmallP reports whether the small-P prefill route is enabled.
func GEMMUseSmallP() bool { return q4kUseSmallP.Load() }
