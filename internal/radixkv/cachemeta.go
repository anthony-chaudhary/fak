package radixkv

import "github.com/anthony-chaudhary/fak/internal/cachemeta"

// CacheEntry lowers a radix node into the common cache metadata contract. The
// digest is over the token prefix represented by the root-to-node path; no KV
// tensors are copied or exposed.
func (n *node) CacheEntry(modelID, tokenizerID string) cachemeta.Entry {
	tokens := n.tokens()
	return cachemeta.FromKVPrefix(cachemeta.KVPrefix{
		Tokens:      tokens,
		Length:      len(tokens),
		ModelID:     modelID,
		TokenizerID: tokenizerID,
		Owner:       "radixkv",
	})
}

func (n *node) tokens() []int {
	if n == nil {
		return nil
	}
	var chunks [][]int
	for p := n; p != nil && p.parent != nil; p = p.parent {
		chunks = append(chunks, p.key)
	}
	total := 0
	for _, c := range chunks {
		total += len(c)
	}
	out := make([]int, 0, total)
	for i := len(chunks) - 1; i >= 0; i-- {
		out = append(out, chunks[i]...)
	}
	return out
}
