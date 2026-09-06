package selfquery

import (
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
)

// ToolBundle represents a coherent group of tools that are discovered
// and faulted into context together atomically.
type ToolBundle struct {
	ID          string   `json:"id"`
	Name        string   `json:"name"`
	Description string   `json:"description"`
	Tools       []string `json:"tools"`
	Keywords    []string `json:"keywords"`
	Domain      string   `json:"domain"`
}

// ContainsTool reports whether the bundle contains the specified tool name.
func (b ToolBundle) ContainsTool(toolName string) bool {
	for _, t := range b.Tools {
		if t == toolName {
			return true
		}
	}
	return false
}

var (
	bundleMu sync.RWMutex

	// Canonical registered bundles defined by the specification.
	canonicalBundles = []ToolBundle{
		{
			ID:          "memory_drivers",
			Name:        "Memory Drivers",
			Description: "Durable memory management, recall, and persistence tools",
			Domain:      "memory",
			Tools: []string{
				"fak_read",
				"fak_recall",
				"fak_save_memory",
				"fak_delete_memory",
			},
			Keywords: []string{
				"memory",
				"durable memory",
				"manage durable memory",
				"recall",
				"store memory",
			},
		},
		{
			ID:          "context_mmu",
			Name:        "Context MMU",
			Description: "Context compaction, paging, snapshots, and KV cache eviction",
			Domain:      "context",
			Tools: []string{
				"ctx_page_in",
				"ctx_page_out",
				"ctx_snapshot",
				"ctx_evict",
			},
			Keywords: []string{
				"context",
				"context compaction",
				"mmu",
				"paging",
				"kv cache",
			},
		},
		{
			ID:          "kernel_adjudication",
			Name:        "Kernel Adjudication",
			Description: "Kernel admission control and execution primitives",
			Domain:      "kernel",
			Tools: []string{
				"fak_adjudicate",
				"fak_syscall",
			},
			Keywords: []string{
				"adjudicate",
				"admission",
				"kernel primitive",
				"syscall",
			},
		},
	}

	bundleAliases = map[string]string{
		"adjudication": "kernel_adjudication",
	}

	bundleRegistry map[string]ToolBundle
)

func init() {
	ResetBundles()
}

// ResetBundles resets the bundle registry to the default canonical bundles.
func ResetBundles() {
	bundleMu.Lock()
	defer bundleMu.Unlock()
	bundleRegistry = make(map[string]ToolBundle, len(canonicalBundles))
	for _, b := range canonicalBundles {
		bundleRegistry[b.ID] = cloneBundle(b)
	}
}

// RegisterBundle registers a tool bundle into the active registry.
func RegisterBundle(b ToolBundle) error {
	id := strings.TrimSpace(b.ID)
	if id == "" {
		return errors.New("tool bundle ID must not be empty")
	}
	if len(b.Tools) == 0 {
		return errors.New("tool bundle Tools must not be empty")
	}
	bundleMu.Lock()
	defer bundleMu.Unlock()
	bundleRegistry[id] = cloneBundle(b)
	return nil
}

// GetBundle returns a registered tool bundle by ID or alias.
func GetBundle(bundleID string) (ToolBundle, error) {
	bundleMu.RLock()
	defer bundleMu.RUnlock()
	id := strings.TrimSpace(bundleID)
	if id == "" {
		return ToolBundle{}, errors.New("tool bundle ID must not be empty")
	}
	if canonical, ok := bundleAliases[strings.ToLower(id)]; ok {
		id = canonical
	}
	if b, ok := bundleRegistry[id]; ok {
		return cloneBundle(b), nil
	}
	for k, b := range bundleRegistry {
		if strings.EqualFold(k, id) {
			return cloneBundle(b), nil
		}
	}
	return ToolBundle{}, fmt.Errorf("unknown tool bundle: %q", bundleID)
}

// RegisteredBundles returns all currently registered tool bundles, sorted by ID.
func RegisteredBundles() []ToolBundle {
	bundleMu.RLock()
	defer bundleMu.RUnlock()
	out := make([]ToolBundle, 0, len(bundleRegistry))
	for _, b := range bundleRegistry {
		out = append(out, cloneBundle(b))
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].ID < out[j].ID
	})
	return out
}

// FaultBundle returns all tool names registered in the specified bundle,
// or an error if the bundle ID is unknown. Tools are returned together atomically.
func FaultBundle(bundleID string) ([]string, error) {
	b, err := GetBundle(bundleID)
	if err != nil {
		return nil, err
	}
	return append([]string(nil), b.Tools...), nil
}

// FindBundles performs lexical and keyword matching of query against bundle
// names, descriptions, domains, keywords, and tools, returning matching bundles
// ranked best-match first.
func FindBundles(query string) []ToolBundle {
	q := strings.TrimSpace(query)
	if q == "" {
		return nil
	}

	normQuery := strings.ToLower(q)
	qToks := tokens(normQuery)
	if len(qToks) == 0 {
		return nil
	}

	bundles := RegisteredBundles()

	type scoredBundle struct {
		bundle ToolBundle
		score  float64
	}

	var matches []scoredBundle
	for _, b := range bundles {
		score := scoreBundle(b, normQuery, qToks)
		if score > 0 {
			matches = append(matches, scoredBundle{
				bundle: b,
				score:  score,
			})
		}
	}

	sort.SliceStable(matches, func(i, j int) bool {
		if matches[i].score != matches[j].score {
			return matches[i].score > matches[j].score
		}
		return matches[i].bundle.ID < matches[j].bundle.ID
	})

	out := make([]ToolBundle, len(matches))
	for i, m := range matches {
		out[i] = m.bundle
	}
	return out
}

// SearchWithBundles finds tool bundles matching the query and returns both the
// bundles and the deduplicated tool names registered across those bundles.
func SearchWithBundles(query string) ([]ToolBundle, []string) {
	bundles := FindBundles(query)
	if len(bundles) == 0 {
		return nil, nil
	}
	var tools []string
	seen := make(map[string]bool)
	for _, b := range bundles {
		for _, t := range b.Tools {
			if !seen[t] {
				seen[t] = true
				tools = append(tools, t)
			}
		}
	}
	return bundles, tools
}

// FindBundles on Catalog searches bundles using FindBundles.
func (c *Catalog) FindBundles(query string) []ToolBundle {
	return FindBundles(query)
}

// FaultBundle on Catalog faults tools for the bundle ID using FaultBundle.
func (c *Catalog) FaultBundle(bundleID string) ([]string, error) {
	return FaultBundle(bundleID)
}

// SearchWithBundles on Catalog integrates bundle discovery with catalog tool search.
// Any tools from matching bundles are faulted in first (in bundle priority order),
// followed by any additional matching tools from the catalog.
func (c *Catalog) SearchWithBundles(query string) ([]ToolBundle, []string) {
	bundles, bundleTools := SearchWithBundles(query)
	if c == nil || len(c.tools) == 0 {
		return bundles, bundleTools
	}

	seen := make(map[string]bool, len(bundleTools))
	tools := make([]string, 0, len(bundleTools)+len(c.tools))
	for _, t := range bundleTools {
		seen[t] = true
		tools = append(tools, t)
	}

	q := strings.TrimSpace(query)
	if q != "" {
		ranked := rankCards(c.toolCards(), q)
		for _, card := range ranked {
			if !seen[card.Name] {
				seen[card.Name] = true
				tools = append(tools, card.Name)
			}
		}
	} else {
		for _, td := range c.tools {
			if !seen[td.Name] {
				seen[td.Name] = true
				tools = append(tools, td.Name)
			}
		}
	}
	return bundles, tools
}

func scoreBundle(b ToolBundle, normQuery string, qToks []string) float64 {
	var score float64
	cleanQuery := strings.Join(qToks, " ")
	matchedTokCount := 0
	matchedTokMap := make(map[string]bool)

	// 1. Exact phrase / keyword matching
	for _, kw := range b.Keywords {
		kwToks := tokens(kw)
		if len(kwToks) == 0 {
			continue
		}
		cleanKW := strings.Join(kwToks, " ")
		if cleanQuery == cleanKW {
			score += 100.0
		} else if strings.Contains(cleanQuery, cleanKW) {
			score += 60.0
		} else if len(cleanQuery) >= 3 && strings.Contains(cleanKW, cleanQuery) {
			score += 40.0
		}
	}

	// ID and Name phrase matches
	idToks := tokens(b.ID)
	cleanID := strings.Join(idToks, " ")
	if cleanQuery == cleanID {
		score += 90.0
	} else if strings.Contains(cleanQuery, cleanID) {
		score += 50.0
	} else if len(cleanQuery) >= 3 && strings.Contains(cleanID, cleanQuery) {
		score += 30.0
	}

	nameToksList := tokens(b.Name)
	cleanName := strings.Join(nameToksList, " ")
	if cleanQuery == cleanName {
		score += 80.0
	} else if strings.Contains(cleanQuery, cleanName) {
		score += 45.0
	} else if len(cleanQuery) >= 3 && strings.Contains(cleanName, cleanQuery) {
		score += 30.0
	}

	// Domain phrase match
	domainToksList := tokens(b.Domain)
	cleanDomain := strings.Join(domainToksList, " ")
	if cleanQuery == cleanDomain {
		score += 60.0
	} else if strings.Contains(cleanQuery, cleanDomain) {
		score += 30.0
	}

	// Tool name matches
	for _, tool := range b.Tools {
		toolToksList := tokens(tool)
		cleanTool := strings.Join(toolToksList, " ")
		toolNorm := strings.ToLower(tool)
		if normQuery == toolNorm || cleanQuery == cleanTool {
			score += 85.0
		} else if strings.Contains(normQuery, toolNorm) || strings.Contains(cleanQuery, cleanTool) {
			score += 40.0
		}
	}

	// 2. Token-level and Stem-level matching
	kwToksMap := make(map[string]bool)
	kwStemsMap := make(map[string]bool)
	for _, kw := range b.Keywords {
		for _, tk := range tokens(kw) {
			kwToksMap[tk] = true
			kwStemsMap[stemToken(tk)] = true
		}
	}

	nameToksMap := make(map[string]bool)
	nameStemsMap := make(map[string]bool)
	for _, tk := range nameToksList {
		nameToksMap[tk] = true
		nameStemsMap[stemToken(tk)] = true
	}

	descToksMap := make(map[string]bool)
	descStemsMap := make(map[string]bool)
	for _, tk := range tokens(b.Description) {
		descToksMap[tk] = true
		descStemsMap[stemToken(tk)] = true
	}

	domainToksMap := make(map[string]bool)
	for _, tk := range domainToksList {
		domainToksMap[tk] = true
	}

	toolToksMap := make(map[string]bool)
	for _, tool := range b.Tools {
		for _, tk := range tokens(tool) {
			toolToksMap[tk] = true
		}
	}

	for _, qTok := range qToks {
		qStem := stemToken(qTok)
		tokenMatched := false

		if kwToksMap[qTok] {
			score += 15.0
			tokenMatched = true
		} else if kwStemsMap[qStem] {
			score += 8.0
			tokenMatched = true
		}

		if nameToksMap[qTok] {
			score += 12.0
			tokenMatched = true
		} else if nameStemsMap[qStem] {
			score += 6.0
			tokenMatched = true
		}

		if domainToksMap[qTok] {
			score += 10.0
			tokenMatched = true
		}

		if toolToksMap[qTok] {
			score += 8.0
			tokenMatched = true
		}

		if descToksMap[qTok] {
			score += 5.0
			tokenMatched = true
		} else if descStemsMap[qStem] {
			score += 2.5
			tokenMatched = true
		}

		if tokenMatched && !matchedTokMap[qTok] {
			matchedTokMap[qTok] = true
			matchedTokCount++
		}
	}

	if matchedTokCount == 0 && score == 0 {
		return 0
	}

	// 3. Query coverage bonus
	if len(qToks) > 0 {
		coverage := float64(matchedTokCount) / float64(len(qToks))
		if coverage == 1.0 {
			score += 20.0
		} else if coverage >= 0.5 {
			score += 10.0
		}
	}

	return score
}

func cloneBundle(b ToolBundle) ToolBundle {
	c := b
	c.Tools = append([]string(nil), b.Tools...)
	c.Keywords = append([]string(nil), b.Keywords...)
	return c
}
