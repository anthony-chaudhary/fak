package ggufload

import (
	"errors"
	"fmt"
	"log"
	"os"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// ErrInkling2FP8ScaleOverflow is returned when an FP8 E4M3FN global KV write detects
// a non-finite value or scale overflow at a position where an in-place FP16 retry is
// no longer possible (mid-session decode or append after pos 0).
var ErrInkling2FP8ScaleOverflow = errors.New("inkling2 resident: fp8 global KV scale overflow")

// inkling2KVHalfRequested is FAK_INKLING2_KV=f16, read once at process start for
// legacy callers/tests. newInkling2ResidentRing reads FAK_INKLING2_KV dynamically
// through Inkling2KVDTypeFrom(os.LookupEnv) so tests can switch dtypes in-process.
var inkling2KVHalfRequested, inkling2KVHalfErr = inkling2KVHalfFrom(os.LookupEnv)

// Inkling2KVDType is the resident global-cache dtype selected by FAK_INKLING2_KV.
type Inkling2KVDType uint8

const (
	Inkling2KVF32 Inkling2KVDType = iota
	Inkling2KVF16
	Inkling2KVFP8
)

// String returns the canonical FAK_INKLING2_KV selector token for d.
func (d Inkling2KVDType) String() string {
	switch d {
	case Inkling2KVF16:
		return "f16"
	case Inkling2KVFP8:
		return "fp8"
	default:
		return "f32"
	}
}

// Inkling2KVDTypeFrom parses the public resident-KV selector. The serving layer
// uses the same typed result to disable unsupported prefix reuse modes.
func Inkling2KVDTypeFrom(lookup func(string) (string, bool)) (Inkling2KVDType, error) {
	v, ok := lookup("FAK_INKLING2_KV")
	if !ok {
		v = ""
	}
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "fp8", "e4m3", "e4m3fn":
		return Inkling2KVFP8, nil
	case "f16", "fp16":
		return Inkling2KVF16, nil
	case "", "f32", "fp32":
		return Inkling2KVF32, nil
	}
	return Inkling2KVF32, fmt.Errorf("inkling2 resident: FAK_INKLING2_KV=%q is not f32, f16, or fp8", v)
}

// inkling2KVHalfFrom parses FAK_INKLING2_KV: f16/fp16 selects f16 global-slot caches,
// unset/f32/fp32 keeps fp32, fp8/e4m3 selects fp8 (half=false), and anything else is refused.
func inkling2KVHalfFrom(lookup func(string) (string, bool)) (bool, error) {
	dtype, err := Inkling2KVDTypeFrom(lookup)
	return dtype == Inkling2KVF16, err
}

var (
	inkling2KVHalfFallbackOnce        sync.Once
	inkling2KVFP8FallbackOnce         sync.Once
	inkling2KVFP8OverflowFallbackOnce sync.Once
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

// inkling2ResidentKVDType resolves the effective global-slot KV dtype on be for a
// requested dtype. Unsupported fp8 falls back to f16 when available, else fp32;
// unsupported f16 falls back to fp32. Each fallback is logged once.
func inkling2ResidentKVDType(be compute.Backend, req Inkling2KVDType) Inkling2KVDType {
	switch req {
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

// inkling2ResidentHalfKV reports whether a session on be keeps its global-slot caches
// f16. A request the backend cannot serve falls back to fp32, logged once: the decode
// ops, the f16 prefill ops, and their readiness must all be present.
func inkling2ResidentHalfKV(be compute.Backend) bool {
	if !inkling2KVHalfRequested {
		return false
	}
	return inkling2ResidentKVDType(be, Inkling2KVF16) == Inkling2KVF16
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
