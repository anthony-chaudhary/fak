package ggufload

import (
	"fmt"
	"log"
	"os"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// inkling2KVHalfRequested is FAK_INKLING2_KV=f16, read once at process start. Unset
// keeps every resident cache float32.
var inkling2KVHalfRequested, inkling2KVHalfErr = inkling2KVHalfFrom(os.LookupEnv)

// inkling2KVHalfFrom parses FAK_INKLING2_KV: f16/fp16 selects f16 global-slot caches,
// unset/f32/fp32 keeps fp32, and anything else is refused.
func inkling2KVHalfFrom(lookup func(string) (string, bool)) (bool, error) {
	v, _ := lookup("FAK_INKLING2_KV")
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "f16", "fp16":
		return true, nil
	case "", "f32", "fp32":
		return false, nil
	}
	return false, fmt.Errorf("inkling2 resident: FAK_INKLING2_KV=%q is not f32 or f16", v)
}

var inkling2KVHalfFallbackOnce sync.Once

// inkling2ResidentHalfKV reports whether a session on be keeps its global-slot caches
// f16. A request the backend cannot serve falls back to fp32, logged once: the decode
// ops, the f16 prefill ops, and their readiness must all be present.
func inkling2ResidentHalfKV(be compute.Backend) bool {
	if !inkling2KVHalfRequested {
		return false
	}
	if _, ok := be.(compute.Inkling2HalfKVDevice); !ok {
		inkling2KVHalfFallback(fmt.Sprintf("backend %s has no f16 decode KV ops", be.Name()))
		return false
	}
	pd, ok := be.(compute.Inkling2PrefillHalfKV)
	if !ok {
		inkling2KVHalfFallback(fmt.Sprintf("backend %s has no f16 prefill ops", be.Name()))
		return false
	}
	if err := pd.Ink2PHalfKVReady(); err != nil {
		inkling2KVHalfFallback(err.Error())
		return false
	}
	return true
}

func inkling2KVHalfFallback(reason string) {
	inkling2KVHalfFallbackOnce.Do(func() {
		log.Printf("inkling2 resident: FAK_INKLING2_KV=f16 falls back to fp32 caches: %s", reason)
	})
}
