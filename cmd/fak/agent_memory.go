package main

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/memoryread"
	"github.com/anthony-chaudhary/fak/internal/memq"
)

// resolveAgentMemoryOption inspects the workspace root and flags to discover or resolve
// the memory store, render a verified fresh notes digest via memq.RenderNotesDigest,
// and return an agent.WithMemoryDigest RunOption (or nil if disabled / no store).
func resolveAgentMemoryOption(enabled bool, storeFlag, root string) (agent.RunOption, string) {
	if !enabled {
		return nil, ""
	}
	store := memoryread.ResolveStore(root, storeFlag)
	if store == "" {
		return nil, ""
	}
	digest := memq.RenderNotesDigest(store, false, 60000)
	if strings.TrimSpace(digest) == "" || strings.HasPrefix(digest, "(no committed memory mirror at ") {
		return nil, store
	}
	return agent.WithMemoryDigest(digest), store
}
