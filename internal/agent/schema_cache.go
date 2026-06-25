package agent

import (
	"crypto/sha256"
	"encoding/json"
	"sync"
)

const (
	schemaCacheKeyGemini = "gemini"
	schemaCacheKeyOpenAI = "openai"
)

type normalizedSchemaCacheKey struct {
	provider string
	root     bool
	sum      [sha256.Size]byte
}

var normalizedSchemaCache sync.Map

func loadNormalizedSchema(provider string, root bool, raw json.RawMessage) (json.RawMessage, bool) {
	v, ok := normalizedSchemaCache.Load(normalizedSchemaKey(provider, root, raw))
	if !ok {
		return nil, false
	}
	b, ok := v.([]byte)
	if !ok {
		return nil, false
	}
	out := make([]byte, len(b))
	copy(out, b)
	return json.RawMessage(out), true
}

func storeNormalizedSchema(provider string, root bool, raw, normalized json.RawMessage) {
	b := make([]byte, len(normalized))
	copy(b, normalized)
	normalizedSchemaCache.Store(normalizedSchemaKey(provider, root, raw), b)
}

func normalizedSchemaKey(provider string, root bool, raw json.RawMessage) normalizedSchemaCacheKey {
	return normalizedSchemaCacheKey{
		provider: provider,
		root:     root,
		sum:      sha256.Sum256(raw),
	}
}
