package vllmcompile

import (
	"fmt"
	"regexp"
	"strings"
)

// CacheTuple is the full set of fields that a GLM-5.2 (or any compiled-engine)
// warmup is a deterministic function of. TorchInductor codegen, Triton kernel
// compile, and DeepGEMM JIT pre-compile all key on exactly this tuple, so a
// persisted compile cache is safe to reuse iff every field matches the boot that
// wrote it. Building the on-disk path FROM the tuple makes invalidation
// structural (a mismatched field resolves to a different, empty directory and
// forces a rebuild) rather than a check that can be forgotten — the persistence
// contract in docs/notes/GLM52-COMPILE-CACHE-PERSISTENCE-CONTRACT-3052-2026-07-06.md (#3052).
type CacheTuple struct {
	Model     string // checkpoint repo, e.g. "PhalaCloud/GLM-5.2-W4AFP8"
	Quant     string // "fp8" | "w4afp8" | "nvfp4" | "int4" | "bf16"
	Arch      string // GPU compute capability, any form: "9.0", "sm_90", "Hopper (sm_90)", "90"
	TP        int    // tensor-parallel size (GPU count)
	Ctx       int    // served context length
	Engine    string // "sglang" | "vllm"
	EngineVer string // engine version string, e.g. "0.5.14"
	TorchVer  string // torch version string, e.g. "2.11.0+cu130"
}

// archDigits pulls the compute-capability digits out of any of the forms the
// preflight, nvidia-smi, or a human might hand us: "9.0" -> "90",
// "Hopper (sm_90)" -> "90", "sm_100" -> "100", "10.0" -> "100", "90" -> "90".
var archDigits = regexp.MustCompile(`sm[_-]?(\d+)|(\d+)\.(\d+)|(\d+)`)

// NormalizeArch collapses any compute-capability spelling to its bare digit
// token so sm_80 and sm_90 (and sm_100) always key distinct cache dirs — the
// acceptance requirement that a GPU-server sm_80 cache can never mis-hit an
// H100 Mega sm_90 boot. Unknown/empty input yields "unknown" so a boot that could not read
// the arch never silently shares a cache with one that could.
func NormalizeArch(raw string) string {
	m := archDigits.FindStringSubmatch(strings.TrimSpace(raw))
	if m == nil {
		return "unknown"
	}
	switch {
	case m[1] != "": // sm_90 / sm-100
		return m[1]
	case m[2] != "": // 9.0 -> 90, 10.0 -> 100
		return m[2] + m[3]
	default: // bare digits already, e.g. "90"
		return m[4]
	}
}

var slugUnsafe = regexp.MustCompile(`[^a-z0-9]+`)

// slug lowercases and reduces a free-form field to a filesystem-safe token so a
// checkpoint repo ("PhalaCloud/GLM-5.2-W4AFP8") or a version with build metadata
// ("2.11.0+cu130") never breaks the path or introduces a stray separator.
func slug(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	s = slugUnsafe.ReplaceAllString(s, "-")
	s = strings.Trim(s, "-")
	if s == "" {
		return "unknown"
	}
	return s
}

// Key is the tuple-keyed cache path suffix, appended to a persistent root to
// form the compile-cache directory. Format (fixed by the #3052 contract, matched
// byte-for-byte by tools/glm52_sglang_vllm_serve.sh so the runtime and the
// bench/gate seam read one key):
//
//	<model-slug>/<quant>/sm<arch>/tp<TP>/ctx<CTX>/<engine>-<engine-ver>-torch<torch-ver>
func (t CacheTuple) Key() string {
	return fmt.Sprintf(
		"%s/%s/sm%s/tp%d/ctx%d/%s-%s-torch%s",
		slug(t.Model),
		slug(t.Quant),
		NormalizeArch(t.Arch),
		t.TP,
		t.Ctx,
		slug(t.Engine),
		slug(t.EngineVer),
		slug(t.TorchVer),
	)
}

// Readout is the one-line operator preflight signal telling whether this boot
// will reuse a warm compile cache (fast) or rebuild it (pays the JIT tax). The
// bash serve wrapper prints the identical line; `populated` is the wrapper's
// answer to "does the keyed dir already hold artifacts?".
func Readout(key, dir string, populated bool) string {
	state := "rebuild"
	if populated {
		state = "hit"
	}
	return fmt.Sprintf("COMPILE_CACHE %s dir=%s tuple=%s", state, dir, key)
}

// WithCacheTuple stamps a Block with the compile-cache key derived from the live
// serve tuple and marks the cache observed-enabled/disabled from whether the
// keyed dir was populated at boot. This is the L3 seam of the #3052 contract:
// the operator path (the serve wrapper's readout) and the bench-honesty gate
// (Classify/Gate) then read one source of truth for the cache key and state, so
// a cold rebuild boot can never be quoted as a tuned baseline.
func (b Block) WithCacheTuple(t CacheTuple, populated bool) Block {
	b.CompileCacheKey = t.Key()
	b.CompileCacheEnabled = Bool(populated)
	return b
}
