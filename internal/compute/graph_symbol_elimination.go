package compute

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
)

// GraphSymbolKind is the closed symbol envelope understood by the native graph cleanup.
type GraphSymbolKind string

const (
	GraphSymbolFunction GraphSymbolKind = "function"
	GraphSymbolConstant GraphSymbolKind = "constant"
)

// GraphSymbol records the symbol edges needed to preserve callable and non-call references.
type GraphSymbol struct {
	Name                string          `json:"name"`
	Kind                GraphSymbolKind `json:"kind"`
	Exported            bool            `json:"exported,omitempty"`
	Calls               []string        `json:"calls,omitempty"`
	References          []string        `json:"references,omitempty"`
	AttributeReferences []string        `json:"attribute_references,omitempty"`
}

// GraphSymbolModule is the narrow symbol-table seam used before native graph lowering.
type GraphSymbolModule struct {
	Symbols []GraphSymbol `json:"symbols"`
}

// GraphSymbolSelectionReceipt makes selection and fallback observable without claiming timing.
type GraphSymbolSelectionReceipt struct {
	Selected          bool     `json:"selected"`
	FallbackReason    string   `json:"fallback_reason,omitempty"`
	Removed           []string `json:"removed,omitempty"`
	FinalModuleDigest string   `json:"final_module_digest,omitempty"`
}

// SelectReachableGraphSymbols removes symbols unreachable from exported roots. Calls, ordinary
// references, and attribute references are all strong edges. Unsupported or malformed modules
// fail closed to the unchanged input, preserving the pre-selector path as the fallback.
func SelectReachableGraphSymbols(input GraphSymbolModule) (GraphSymbolModule, GraphSymbolSelectionReceipt) {
	fallback := cloneGraphSymbolModule(input)
	index := make(map[string]int, len(input.Symbols))
	work := make([]string, 0, len(input.Symbols))
	reachable := make(map[string]bool, len(input.Symbols))

	for i, symbol := range input.Symbols {
		if symbol.Name == "" {
			return fallback, graphSymbolFallback("symbol name must not be empty")
		}
		if symbol.Kind != GraphSymbolFunction && symbol.Kind != GraphSymbolConstant {
			return fallback, graphSymbolFallback(fmt.Sprintf("symbol %q has unsupported kind %q", symbol.Name, symbol.Kind))
		}
		if _, exists := index[symbol.Name]; exists {
			return fallback, graphSymbolFallback(fmt.Sprintf("duplicate symbol %q", symbol.Name))
		}
		index[symbol.Name] = i
		if symbol.Exported {
			reachable[symbol.Name] = true
			work = append(work, symbol.Name)
		}
	}
	if len(work) == 0 {
		return fallback, graphSymbolFallback("module has no exported roots")
	}

	for head := 0; head < len(work); head++ {
		symbol := input.Symbols[index[work[head]]]
		for _, references := range [][]string{symbol.Calls, symbol.References, symbol.AttributeReferences} {
			for _, reference := range references {
				if _, exists := index[reference]; !exists {
					return fallback, graphSymbolFallback(fmt.Sprintf("symbol %q references unknown symbol %q", symbol.Name, reference))
				}
				if !reachable[reference] {
					reachable[reference] = true
					work = append(work, reference)
				}
			}
		}
	}

	selected := GraphSymbolModule{Symbols: make([]GraphSymbol, 0, len(reachable))}
	removed := make([]string, 0, len(input.Symbols)-len(reachable))
	for _, symbol := range input.Symbols {
		if reachable[symbol.Name] {
			selected.Symbols = append(selected.Symbols, cloneGraphSymbol(symbol))
		} else {
			removed = append(removed, symbol.Name)
		}
	}
	sort.Strings(removed)
	encoded, err := json.Marshal(selected)
	if err != nil {
		return fallback, graphSymbolFallback(fmt.Sprintf("serialize selected module: %v", err))
	}
	digest := sha256.Sum256(encoded)
	return selected, GraphSymbolSelectionReceipt{
		Selected:          true,
		Removed:           removed,
		FinalModuleDigest: hex.EncodeToString(digest[:]),
	}
}

func graphSymbolFallback(reason string) GraphSymbolSelectionReceipt {
	return GraphSymbolSelectionReceipt{FallbackReason: reason}
}

func cloneGraphSymbolModule(module GraphSymbolModule) GraphSymbolModule {
	clone := GraphSymbolModule{Symbols: make([]GraphSymbol, len(module.Symbols))}
	for i, symbol := range module.Symbols {
		clone.Symbols[i] = cloneGraphSymbol(symbol)
	}
	return clone
}

func cloneGraphSymbol(symbol GraphSymbol) GraphSymbol {
	symbol.Calls = append([]string(nil), symbol.Calls...)
	symbol.References = append([]string(nil), symbol.References...)
	symbol.AttributeReferences = append([]string(nil), symbol.AttributeReferences...)
	return symbol
}
