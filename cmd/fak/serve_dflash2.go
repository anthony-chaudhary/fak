package main

import (
	"errors"
	"os"
	"strings"
)

// ErrServeDFlash2OwnerUnavailable prevents the ordinary loader from acquiring
// weights for a route whose authenticated owner cannot yet be built by serve.
// Gateway/library callers with an authenticated DFlash2Target remain supported.
var ErrServeDFlash2OwnerUnavailable = errors.New("DFlash2 serve startup requires an authenticated target owner; target binding, pre-load memory admission and drain-aware cleanup are not wired")

// validateServeDFlash2Startup runs before backend/model acquisition. Parser
// acceptance is not a serving capability: ordinary loaded weights cannot be
// promoted to an authenticated DFlash2Target. Check both ambient selectors because
// the gateway rejects their conflicts even when the CLI resolver chose another
// mode. The already-resolved mode preserves an override of the original flag.
func validateServeDFlash2Startup(mode string, getenv func(string) string) error {
	if getenv == nil {
		getenv = os.Getenv
	}
	for _, selected := range []string{mode, getenv("FAK_SPECULATIVE"), getenv("SPECULATIVE")} {
		if strings.ToLower(strings.TrimSpace(selected)) == serveSpeculativeDFlash2 {
			return ErrServeDFlash2OwnerUnavailable
		}
	}
	return nil
}
