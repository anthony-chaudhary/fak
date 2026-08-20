package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/session"
)

// beginGuardProviderBoundary is the guard-owned write behind the lifecycle IPC.
// The provider session id deterministically names the child trace, making a
// duplicate SessionStart hook delivery idempotent without exposing the raw id in
// gateway routing labels.
func beginGuardProviderBoundary(previousTrace, provider, source, providerSessionID string) (guardProviderBoundaryResult, error) {
	previousTrace = strings.TrimSpace(previousTrace)
	provider = strings.ToLower(strings.TrimSpace(provider))
	source = strings.ToLower(strings.TrimSpace(source))
	providerSessionID = strings.TrimSpace(providerSessionID)
	if previousTrace == "" {
		return guardProviderBoundaryResult{}, fmt.Errorf("provider session boundary has no active fak trace")
	}
	if !validProviderBoundaryToken(provider) {
		return guardProviderBoundaryResult{}, fmt.Errorf("invalid provider %q", provider)
	}
	if source != "clear" {
		return guardProviderBoundaryResult{}, fmt.Errorf("unsupported provider session source %q", source)
	}
	if providerSessionID == "" || len(providerSessionID) > 512 {
		return guardProviderBoundaryResult{}, fmt.Errorf("invalid provider session id")
	}
	childTrace := providerBoundaryTrace(provider, providerSessionID)
	boundary := session.ProviderSessionBoundary{
		Provider: provider, Source: source, ProviderSessionID: providerSessionID,
	}
	child, applied := serveSessions.BeginProviderSessionAt(previousTrace, childTrace, boundary, time.Now())
	if applied {
		persistServeSessionRevision(context.Background(), previousTrace, serveSessions.Get(previousTrace))
		if serveSessionDurability != nil {
			if err := serveSessionDurability.register(context.Background(), childTrace, child); err != nil && serveSessionDurability.warnf != nil {
				serveSessionDurability.warnf("fak: provider-boundary descriptor register failed for %s: %v", childTrace, err)
			}
		}
	}
	resultPrevious := previousTrace
	if child.ProviderBoundary.PreviousTrace != "" {
		resultPrevious = child.ProviderBoundary.PreviousTrace
	}
	return guardProviderBoundaryResult{
		Applied: applied, PreviousTrace: resultPrevious, NewTrace: child.TraceID,
		Provider: provider, Source: source,
	}, nil
}

func validProviderBoundaryToken(provider string) bool {
	if provider == "" || len(provider) > 32 {
		return false
	}
	for _, r := range provider {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '-' || r == '_' || r == '.' {
			continue
		}
		return false
	}
	return true
}

func providerBoundaryTrace(provider, providerSessionID string) string {
	sum := sha256.Sum256([]byte(provider + "\x00" + providerSessionID))
	return provider + "-clear-" + hex.EncodeToString(sum[:12])
}
