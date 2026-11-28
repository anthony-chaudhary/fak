package ggufload

import (
	"errors"
	"fmt"
	"log"
	"os"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/kvcache"
)

// ErrInkling2FP8ScaleOverflow is returned when an FP8 E4M3FN global KV write detects
// a non-finite value or scale overflow at a position where an in-place FP16 retry is
// no longer possible (mid-session decode or append after pos 0).
var ErrInkling2FP8ScaleOverflow = errors.New("inkling2 resident: fp8 global KV scale overflow")

// ErrInkling2NVFP4ScaleOverflow is returned when an NVFP4 global KV write detects
// a non-finite value or block-scale overflow at a position where an in-place retry
// to a higher-precision KV dtype is no longer possible.
var ErrInkling2NVFP4ScaleOverflow = errors.New("inkling2 resident: nvfp4 global KV scale overflow")

// inkling2KVHalfRequested is FAK_INKLING2_KV=f16, read once at process start for
// legacy callers/tests. newInkling2ResidentRing reads FAK_INKLING2_KV dynamically
// through Inkling2KVDTypeFrom(os.LookupEnv) so tests can switch dtypes in-process.
var inkling2KVHalfRequested, inkling2KVHalfErr = inkling2KVHalfFrom(os.LookupEnv)

// Inkling2KVDType is the resident global-cache dtype selected by FAK_INKLING2_KV.
type Inkling2KVDType = ServeKVDType

const (
	Inkling2KVF32   = ServeKVF32
	Inkling2KVF16   = ServeKVF16
	Inkling2KVFP8   = ServeKVFP8
	Inkling2KVNVFP4 = ServeKVNVFP4
	Inkling2KVAsym  = ServeKVAsym
)

// ToKVCachePrecision maps Inkling2KVDType to the shared kvcache.Precision.
func (d Inkling2KVDType) ToKVCachePrecision() kvcache.Precision {
	switch d {
	case Inkling2KVF16:
		return kvcache.PrecisionFP16
	case Inkling2KVFP8:
		return kvcache.PrecisionFP8
	case Inkling2KVNVFP4:
		return kvcache.PrecisionNVFP4
	default:
		return kvcache.PrecisionFP32
	}
}

// Inkling2KVDTypeFromPrecision converts a shared kvcache.Precision to Inkling2KVDType.
func Inkling2KVDTypeFromPrecision(p kvcache.Precision) (Inkling2KVDType, error) {
	switch p {
	case kvcache.PrecisionNVFP4:
		return Inkling2KVNVFP4, nil
	case kvcache.PrecisionFP8:
		return Inkling2KVFP8, nil
	case kvcache.PrecisionFP16:
		return Inkling2KVF16, nil
	case kvcache.PrecisionFP32, "":
		return Inkling2KVF32, nil
	default:
		return Inkling2KVF32, fmt.Errorf("inkling2 resident: unsupported kvcache precision %q", p)
	}
}

// Inkling2KVDTypeFrom parses the public resident-KV selector. The serving layer
// uses the same typed result to disable unsupported prefix reuse modes.
func Inkling2KVDTypeFrom(lookup func(string) (string, bool)) (Inkling2KVDType, error) {
	if lookup == nil {
		lookup = os.LookupEnv
	}
	v, ok := lookup("FAK_INKLING2_KV")
	if !ok {
		v = ""
	}
	dt, err := ParseServeKVDType(v)
	if err != nil {
		return Inkling2KVF32, fmt.Errorf("inkling2 resident: FAK_INKLING2_KV=%q is not f32, f16, fp8, nvfp4, or fp8k_nvfp4v", v)
	}
	return dt, nil
}

// inkling2KVHalfFrom parses FAK_INKLING2_KV: f16/fp16 selects f16 global-slot caches,
// unset/f32/fp32 keeps fp32, fp8/e4m3 selects fp8 (half=false), and anything else is refused.
func inkling2KVHalfFrom(lookup func(string) (string, bool)) (bool, error) {
	dtype, err := Inkling2KVDTypeFrom(lookup)
	return dtype == Inkling2KVF16, err
}

var (
	inkling2KVHalfFallbackOnce          sync.Once
	inkling2KVFP8FallbackOnce           sync.Once
	inkling2KVFP8OverflowFallbackOnce   sync.Once
	inkling2KVNVFP4FallbackOnce         sync.Once
	inkling2KVNVFP4OverflowFallbackOnce sync.Once
)

func inkling2BackendHalfKVErr(be compute.Backend) error {
	if _, ok := be.(compute.Inkling2HalfKVDevice); !ok {
		return fmt.Errorf("backend %s has no f16 decode KV ops", be.Name())
	}
	pd, ok := be.(compute.Inkling2PrefillHalfKV)
	if !ok {
		return fmt.Errorf("backend %s has no f16 prefill ops", be.Name())
	}
	return pd.Ink2PHalfKVReady()
}

func inkling2BackendFP8KVErr(be compute.Backend) error {
	fd, ok := be.(compute.Inkling2FP8KVDevice)
	if !ok {
		return fmt.Errorf("backend %s has no fp8 KV ops", be.Name())
	}
	return fd.Ink2PFP8KVReady()
}

func inkling2BackendNVFP4KVErr(be compute.Backend) error {
	nd, ok := be.(compute.Inkling2NVFP4KVDevice)
	if !ok {
		return fmt.Errorf("backend %s has no nvfp4 KV ops", be.Name())
	}
	return nd.Ink2PNVFP4KVReady()
}

func inkling2BackendAsymKVErr(be compute.Backend) error {
	ad, ok := be.(compute.Inkling2AsymKVDevice)
	if !ok {
		return fmt.Errorf("backend %s has no fp8k_nvfp4v KV ops", be.Name())
	}
	return ad.Ink2PAsymKVReady()
}

// inkling2ResidentKVDType resolves the effective global-slot KV dtype on be for a
// requested dtype. Unsupported nvfp4/fp8k_nvfp4v falls back to fp8 -> f16 -> fp32;
// unsupported fp8 falls back to f16 -> fp32; unsupported f16 falls back to fp32. Each
// fallback is logged once.
func inkling2ResidentKVDType(be compute.Backend, req Inkling2KVDType) Inkling2KVDType {
	switch req {
	case Inkling2KVAsym:
		if err := inkling2BackendAsymKVErr(be); err == nil {
			return Inkling2KVAsym
		} else if fp8Err := inkling2BackendFP8KVErr(be); fp8Err == nil {
			inkling2KVNVFP4Fallback("fp8", err.Error())
			return Inkling2KVFP8
		} else if halfErr := inkling2BackendHalfKVErr(be); halfErr == nil {
			inkling2KVNVFP4Fallback("f16", err.Error())
			return Inkling2KVF16
		} else {
			inkling2KVNVFP4Fallback("fp32", err.Error())
			return Inkling2KVF32
		}
	case Inkling2KVNVFP4:
		if err := inkling2BackendNVFP4KVErr(be); err == nil {
			return Inkling2KVNVFP4
		} else if fp8Err := inkling2BackendFP8KVErr(be); fp8Err == nil {
			inkling2KVNVFP4Fallback("fp8", err.Error())
			return Inkling2KVFP8
		} else if halfErr := inkling2BackendHalfKVErr(be); halfErr == nil {
			inkling2KVNVFP4Fallback("f16", err.Error())
			return Inkling2KVF16
		} else {
			inkling2KVNVFP4Fallback("fp32", err.Error())
			return Inkling2KVF32
		}
	case Inkling2KVFP8:
		if err := inkling2BackendFP8KVErr(be); err == nil {
			return Inkling2KVFP8
		} else if halfErr := inkling2BackendHalfKVErr(be); halfErr == nil {
			inkling2KVFP8Fallback("f16", err.Error())
			return Inkling2KVF16
		} else {
			inkling2KVFP8Fallback("fp32", err.Error())
			return Inkling2KVF32
		}
	case Inkling2KVF16:
		if err := inkling2BackendHalfKVErr(be); err == nil {
			return Inkling2KVF16
		} else {
			inkling2KVHalfFallback(err.Error())
			return Inkling2KVF32
		}
	default:
		return Inkling2KVF32
	}
}

// inkling2BackendSupportsHalfKV reports whether be exposes ready f16 decode and
// prefill KV ops. When warn is true, a missing capability is logged once.
func inkling2BackendSupportsHalfKV(be compute.Backend, warn bool) bool {
	if be == nil {
		return false
	}
	if err := inkling2BackendHalfKVErr(be); err != nil {
		if warn {
			inkling2KVHalfFallback(err.Error())
		}
		return false
	}
	return true
}

// inkling2ResidentHalfKV reports whether a session on be keeps its global-slot caches
// f16 when FAK_INKLING2_KV=f16 was explicitly requested. A request the backend cannot
// serve falls back to fp32, logged once.
func inkling2ResidentHalfKV(be compute.Backend) bool {
	if !inkling2KVHalfRequested {
		return false
	}
	return inkling2BackendSupportsHalfKV(be, true)
}

// inkling2ResidentKVDTypeFor decides the effective global-slot KV dtype for m on be.
// Explicit FAK_INKLING2_KV (fp8/e4m3/e4m3fn, f16/fp16, or f32/fp32) always wins (with
// capability fallbacks logged once). When unset or blank, f16 defaults on when the
// backend provides ready f16 decode and prefill ops and speculative decode is not
// active (verify/draft ops read fp32 caches).
func inkling2ResidentKVDTypeFor(m *Inkling2LoopModel, be compute.Backend, lookup func(string) (string, bool)) (Inkling2KVDType, error) {
	dtype, _, err := inkling2ResidentKVDTypeForMode(m, be, lookup, false)
	return dtype, err
}

// inkling2AutoFP8MinF16Ctx is the minimum f16 auto-fit context before unset
// FAK_INKLING2_KV automatically selects fp8 global KV caches (when supported
// and speculative decode is inactive) to preserve 24K+ contexts on 16 GiB GPUs.
const inkling2AutoFP8MinF16Ctx = 24576

func inkling2ShouldAutoFP8KV(m *Inkling2LoopModel, be compute.Backend, lookup func(string) (string, bool)) bool {
	if m == nil || be == nil || m.HP.Layers == 0 {
		return false
	}
	if err := inkling2BackendFP8KVErr(be); err != nil {
		return false
	}
	_, free, known := compute.DeviceMemoryInfo(be)
	if !known || free == compute.FreeUnknown || free <= 0 {
		return false
	}
	return inkling2AutoFitCtxFor(m, be, free, lookup, Inkling2KVF16) < inkling2AutoFP8MinF16Ctx
}

// inkling2ResidentKVDTypeForMode is inkling2ResidentKVDTypeFor with forSpec indicating
// that the caller will enable speculative decode on the session, and autoHalf reporting
// whether f16 was auto-selected (unset FAK_INKLING2_KV) rather than explicitly requested.
func inkling2ResidentKVDTypeForMode(m *Inkling2LoopModel, be compute.Backend, lookup func(string) (string, bool), forSpec bool) (Inkling2KVDType, bool, error) {
	if lookup == nil {
		lookup = os.LookupEnv
	}
	v, ok := lookup("FAK_INKLING2_KV")
	if !ok {
		v = ""
	}
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "fp8k_nvfp4v", "k8v4":
		return inkling2ResidentKVDType(be, Inkling2KVAsym), false, nil
	case "nvfp4", "fp4", "e2m1":
		return inkling2ResidentKVDType(be, Inkling2KVNVFP4), false, nil
	case "fp8", "e4m3", "e4m3fn":
		return inkling2ResidentKVDType(be, Inkling2KVFP8), false, nil
	case "f16", "fp16":
		return inkling2ResidentKVDType(be, Inkling2KVF16), false, nil
	case "f32", "fp32":
		return Inkling2KVF32, false, nil
	case "":
		if inkling2KVHalfRequested {
			return inkling2ResidentKVDType(be, Inkling2KVF16), false, nil
		}
		if forSpec || inkling2SpecDepthFor(m) > 0 {
			return Inkling2KVF32, false, nil
		}
		if inkling2BackendSupportsHalfKV(be, false) {
			if inkling2ShouldAutoFP8KV(m, be, lookup) {
				return Inkling2KVFP8, false, nil
			}
			return Inkling2KVF16, true, nil
		}
		return Inkling2KVF32, false, nil
	}
	return Inkling2KVF32, false, fmt.Errorf("inkling2 resident: FAK_INKLING2_KV=%q is not f32, f16, fp8, nvfp4, or fp8k_nvfp4v", v)
}

// inkling2ResidentHalfKVFor reports whether a resident session for m on be stores its
// global-slot K and V caches as f16.
func inkling2ResidentHalfKVFor(m *Inkling2LoopModel, be compute.Backend, lookup func(string) (string, bool)) (bool, error) {
	dtype, err := inkling2ResidentKVDTypeFor(m, be, lookup)
	return dtype == Inkling2KVF16, err
}

func inkling2KVHalfFallback(reason string) {
	inkling2KVHalfFallbackOnce.Do(func() {
		log.Printf("inkling2 resident: FAK_INKLING2_KV=f16 falls back to fp32 caches: %s", reason)
	})
}

func inkling2KVFP8Fallback(target, reason string) {
	inkling2KVFP8FallbackOnce.Do(func() {
		log.Printf("inkling2 resident: FAK_INKLING2_KV=fp8 falls back to %s caches: %s", target, reason)
	})
}

func inkling2KVFP8OverflowLog() {
	inkling2KVFP8OverflowFallbackOnce.Do(func() {
		log.Printf("inkling2 resident: fp8 global KV scale overflow detected; falling back to f16 caches")
	})
}

func inkling2KVNVFP4Fallback(target, reason string) {
	inkling2KVNVFP4FallbackOnce.Do(func() {
		log.Printf("inkling2 resident: FAK_INKLING2_KV=nvfp4 falls back to %s caches: %s", target, reason)
	})
}

func inkling2KVNVFP4OverflowLog(target string) {
	inkling2KVNVFP4OverflowFallbackOnce.Do(func() {
		log.Printf("inkling2 resident: nvfp4 global KV scale overflow detected; falling back to %s caches", target)
	})
}
