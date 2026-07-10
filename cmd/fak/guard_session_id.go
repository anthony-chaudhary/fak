package main

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/session"
)

// resolveGuardSessionID keeps the cheap, process-local "guard" trace for an ordinary
// launch, but gives every implicitly named durable launch its own trace. Durable state is
// restored from the cross-process session registry, so reusing the literal "guard" there
// can resurrect a prior run's STOPPED/TIME_BUDGET_EXHAUSTED state before the new child gets
// a turn. Operators who want a stable resumable identity already have --session-id; an
// omitted id means a fresh launch and therefore gets a launch nonce.
func resolveGuardSessionID(explicit string, durabilityWanted bool, meta session.DescriptorMeta, nonce string) string {
	if id := strings.TrimSpace(explicit); id != "" {
		return id
	}
	if !durabilityWanted {
		return "guard"
	}
	base := guardSessionIDBase(meta)
	if nonce = strings.TrimSpace(nonce); nonce != "" {
		return base + "-" + nonce
	}
	return base
}

func guardSessionIDBase(meta session.DescriptorMeta) string {
	key := strings.TrimSpace(meta.CacheKey)
	if key == "" {
		return "guard"
	}
	if len(key) > 16 {
		key = key[:16]
	}
	return "guard-" + key
}

// newGuardLaunchNonce returns a path/ref-safe per-process launch suffix. crypto/rand is
// the normal path; the pid+clock fallback still prevents a stale durable trace from being
// selected when the host entropy source is unavailable.
func newGuardLaunchNonce() string {
	var b [8]byte
	if _, err := rand.Read(b[:]); err == nil {
		return hex.EncodeToString(b[:])
	}
	return fmt.Sprintf("%x-%x", os.Getpid(), time.Now().UnixNano())
}
