package agentopt

import (
	"math"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"unicode"
)

// Family 7: Retrieval & knowledge augmentation.
//
// Hybrid BM25 + dense/lexical keyword re-ranking system for repository file
// discovery. Combines keyword exact-match scores (BM25 term frequencies, doc lengths)
// with semantic similarity scores using Reciprocal Rank Fusion (RRF with k=60),
// prioritizing exact symbol and file path hits at top rank positions.

const (
	// DefaultRRFK is the classic reciprocal rank fusion constant (k=60).
	DefaultRRFK = 60.0

	// DefaultBM25K1 controls term frequency saturation.
	DefaultBM25K1 = 1.5

	// DefaultBM25B controls document length normalization.
	DefaultBM25B = 0.75

	// DefaultSymbolBoost is the score boost applied to exact symbol matches.
	DefaultSymbolBoost = 2.0

	// DefaultPathBoost is the score boost applied to exact file path or basename hits.
	DefaultPathBoost = 2.0

	// Field weights for lexical token frequency accumulation.
	fieldWeightPath     = 3.0
	fieldWeightBasename = 4.0
	fieldWeightSymbol   = 5.0
	fieldWeightContent  = 1.0
)

// Document represents a repository file or code artifact indexed for discovery.
type Document struct {
	Path     string            `json:"path"`
	Basename string            `json:"basename,omitempty"`
	Content  string            `json:"content,omitempty"`
	Symbols  []string          `json:"symbols,omitempty"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

// ScoredMatch represents the intermediate scoring breakdown for a document.
type ScoredMatch struct {
	Doc            Document `json:"doc"`
	BM25Score      float64  `json:"bm25_score"`
	DenseScore     float64  `json:"dense_score"`
	BM25Rank       int      `json:"bm25_rank"`
	DenseRank      int      `json:"dense_rank"`
	RRFScore       float64  `json:"rrf_score"`
	ExactPathHit   bool     `json:"exact_path_hit"`
	ExactNameHit   bool     `json:"exact_name_hit"`
	ExactSymbolHit bool     `json:"exact_symbol_hit"`
	Boost          float64  `json:"boost"`
	FinalScore     float64  `json:"final_score"`
}

// RankedItem represents an item ranked and sorted by the hybrid reranker.
type RankedItem struct {
	Doc       Document    `json:"doc"`
	Path      string      `json:"path"`
	Score     float64     `json:"score"`
	Rank      int         `json:"rank"`
	BM25Rank  int         `json:"bm25_rank"`
	DenseRank int         `json:"dense_rank"`
	Match     ScoredMatch `json:"match"`
}

// HybridRerankerConfig configures parameters for hybrid lexical and dense ranking.
type HybridRerankerConfig struct {
	RRFK        float64 `json:"rrf_k"`
	BM25K1      float64 `json:"bm25_k1"`
	BM25B       float64 `json:"bm25_b"`
	SymbolBoost float64 `json:"symbol_boost"`
	PathBoost   float64 `json:"path_boost"`
}

// DefaultHybridRerankerConfig returns default configuration parameters.
func DefaultHybridRerankerConfig() HybridRerankerConfig {
	return HybridRerankerConfig{
		RRFK:        DefaultRRFK,
		BM25K1:      DefaultBM25K1,
		BM25B:       DefaultBM25B,
		SymbolBoost: DefaultSymbolBoost,
		PathBoost:   DefaultPathBoost,
	}
}

type docTermBag struct {
	termWeights map[string]float64
	totalLength float64
	doc         Document
}

// HybridReranker implements BM25 + dense hybrid re-ranking for repository files.
type HybridReranker struct {
	mu          sync.RWMutex
	cfg         HybridRerankerConfig
	docs        []Document
	bags        []docTermBag
	docLens     []float64
	avgDocLen   float64
	docFreqs    map[string]int
	dirtyCorpus bool
}

// NewHybridReranker constructs a new hybrid reranker with optional configuration.
func NewHybridReranker(cfgs ...HybridRerankerConfig) *HybridReranker {
	cfg := DefaultHybridRerankerConfig()
	if len(cfgs) > 0 {
		c := cfgs[0]
		if c.RRFK > 0 {
			cfg.RRFK = c.RRFK
		}
		if c.BM25K1 > 0 {
			cfg.BM25K1 = c.BM25K1
		}
		if c.BM25B > 0 {
			cfg.BM25B = c.BM25B
		}
		if c.SymbolBoost > 0 {
			cfg.SymbolBoost = c.SymbolBoost
		}
		if c.PathBoost > 0 {
			cfg.PathBoost = c.PathBoost
		}
	}
	return &HybridReranker{
		cfg:      cfg,
		docFreqs: make(map[string]int),
	}
}

// Config returns the current configuration.
func (r *HybridReranker) Config() HybridRerankerConfig {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.cfg
}

// DocumentCount returns the number of indexed documents.
func (r *HybridReranker) DocumentCount() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.docs)
}

// Clear resets the indexed corpus.
func (r *HybridReranker) Clear() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.docs = nil
	r.bags = nil
	r.docLens = nil
	r.avgDocLen = 0
	r.docFreqs = make(map[string]int)
	r.dirtyCorpus = false
}

// GetDocument retrieves an indexed document by path.
func (r *HybridReranker) GetDocument(path string) (Document, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	for _, d := range r.docs {
		if d.Path == path {
			return d, true
		}
	}
	return Document{}, false
}

// IndexDocument indexes a single document into the corpus.
func (r *HybridReranker) IndexDocument(doc Document) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if doc.Basename == "" && doc.Path != "" {
		doc.Basename = cleanBasename(doc.Path)
	}
	if len(doc.Symbols) == 0 && doc.Content != "" {
		doc.Symbols = extractSymbols(doc.Content)
	}

	bag := r.buildDocBag(doc)
	r.docs = append(r.docs, doc)
	r.bags = append(r.bags, bag)
	r.docLens = append(r.docLens, bag.totalLength)

	for t := range bag.termWeights {
		r.docFreqs[t]++
	}
	r.dirtyCorpus = true
}

// IndexDocuments indexes multiple documents into the corpus.
func (r *HybridReranker) IndexDocuments(docs []Document) {
	for _, d := range docs {
		r.IndexDocument(d)
	}
}

func (r *HybridReranker) buildDocBag(doc Document) docTermBag {
	bag := make(map[string]float64)
	totalLen := 0.0

	addWeightedText := func(text string, weight float64) {
		if text == "" {
			return
		}
		toks := tokenize(text)
		for _, tk := range toks {
			st := stemToken(tk)
			if st != "" {
				bag[st] += weight
				totalLen += weight
			}
		}
	}

	addWeightedText(doc.Path, fieldWeightPath)
	addWeightedText(doc.Basename, fieldWeightBasename)
	for _, sym := range doc.Symbols {
		addWeightedText(sym, fieldWeightSymbol)
	}
	addWeightedText(doc.Content, fieldWeightContent)

	return docTermBag{
		termWeights: bag,
		totalLength: totalLen,
		doc:         doc,
	}
}

func (r *HybridReranker) refreshCorpusStatsLocked() {
	if !r.dirtyCorpus || len(r.docs) == 0 {
		return
	}
	totalLen := 0.0
	for _, l := range r.docLens {
		totalLen += l
	}
	r.avgDocLen = totalLen / float64(len(r.docs))
	r.dirtyCorpus = false
}

type intermediateScoredItem struct {
	doc              Document
	bm25Score        float64
	denseScore       float64
	bm25Rank         int
	denseRank        int
	hasBM25Hit       bool
	exactPathHit     bool
	exactNameHit     bool
	exactSymbolHit   bool
	partialSymbolHit bool
	boost            float64
	rrfScore         float64
	finalScore       float64
}

// Rerank re-ranks indexed documents by fusing BM25 lexical scores, dense similarity scores,
// and exact symbol/path prioritization via Reciprocal Rank Fusion (RRF).
func (r *HybridReranker) Rerank(query string, denseScores map[string]float64) []RankedItem {
	r.mu.Lock()
	r.refreshCorpusStatsLocked()
	r.mu.Unlock()

	r.mu.RLock()
	defer r.mu.RUnlock()

	nDocs := len(r.docs)
	if nDocs == 0 {
		return nil
	}

	queryTokens := tokenize(query)
	var queryStems []string
	stemSeen := make(map[string]bool)
	for _, tk := range queryTokens {
		st := stemToken(tk)
		if st != "" && !stemSeen[st] {
			stemSeen[st] = true
			queryStems = append(queryStems, st)
		}
	}

	idfMap := make(map[string]float64, len(queryStems))
	for _, qt := range queryStems {
		df := r.docFreqs[qt]
		idfMap[qt] = math.Log(1.0 + (float64(nDocs)-float64(df)+0.5)/(float64(df)+0.5))
	}

	items := make([]intermediateScoredItem, nDocs)
	avgdl := r.avgDocLen

	for i := 0; i < nDocs; i++ {
		doc := r.docs[i]
		bag := r.bags[i]

		pathHit, nameHit, symHit, partSymHit := checkExactMatches(doc, query)

		bm25 := 0.0
		if avgdl > 0 && len(queryStems) > 0 {
			norm := 1.0 - r.cfg.BM25B + r.cfg.BM25B*(bag.totalLength/avgdl)
			for _, qt := range queryStems {
				tf := bag.termWeights[qt]
				if tf <= 0 {
					continue
				}
				idfVal := idfMap[qt]
				if idfVal <= 0 {
					continue
				}
				bm25 += idfVal * (tf * (r.cfg.BM25K1 + 1.0)) / (tf + r.cfg.BM25K1*norm)
			}
		}

		if symHit {
			bm25 += r.cfg.SymbolBoost * 5.0
		} else if partSymHit {
			bm25 += r.cfg.SymbolBoost * 1.5
		}
		if pathHit {
			bm25 += r.cfg.PathBoost * 5.0
		} else if nameHit {
			bm25 += r.cfg.PathBoost * 4.0
		}

		dense := lookupDenseScore(doc, denseScores)

		boost := 0.0
		if pathHit {
			boost += r.cfg.PathBoost * 1.5
		}
		if nameHit {
			boost += r.cfg.PathBoost
		}
		if symHit {
			boost += r.cfg.SymbolBoost * 1.5
		} else if partSymHit {
			boost += r.cfg.SymbolBoost * 0.4
		}

		items[i] = intermediateScoredItem{
			doc:              doc,
			bm25Score:        bm25,
			denseScore:       dense,
			bm25Rank:         -1,
			denseRank:        -1,
			hasBM25Hit:       bm25 > 0,
			exactPathHit:     pathHit,
			exactNameHit:     nameHit,
			exactSymbolHit:   symHit,
			partialSymbolHit: partSymHit,
			boost:            boost,
		}
	}

	bm25Indices := make([]int, nDocs)
	for i := 0; i < nDocs; i++ {
		bm25Indices[i] = i
	}
	sort.SliceStable(bm25Indices, func(i, j int) bool {
		a, b := bm25Indices[i], bm25Indices[j]
		if items[a].bm25Score != items[b].bm25Score {
			return items[a].bm25Score > items[b].bm25Score
		}
		return items[a].doc.Path < items[b].doc.Path
	})
	for rank, idx := range bm25Indices {
		items[idx].bm25Rank = rank
	}

	var denseIndices []int
	for i := 0; i < nDocs; i++ {
		if items[i].denseScore > 0 {
			denseIndices = append(denseIndices, i)
		}
	}
	sort.SliceStable(denseIndices, func(i, j int) bool {
		a, b := denseIndices[i], denseIndices[j]
		if items[a].denseScore != items[b].denseScore {
			return items[a].denseScore > items[b].denseScore
		}
		return items[a].doc.Path < items[b].doc.Path
	})
	for rank, idx := range denseIndices {
		items[idx].denseRank = rank
	}

	k := r.cfg.RRFK
	for i := 0; i < nDocs; i++ {
		rrf := 0.0
		if items[i].hasBM25Hit && items[i].bm25Rank >= 0 {
			rrf += 1.0 / (k + float64(items[i].bm25Rank) + 1.0)
		}
		if items[i].denseRank >= 0 {
			rrf += 1.0 / (k + float64(items[i].denseRank) + 1.0)
		}
		items[i].rrfScore = rrf
		items[i].finalScore = rrf + items[i].boost
	}

	sort.SliceStable(items, func(i, j int) bool {
		if items[i].finalScore != items[j].finalScore {
			return items[i].finalScore > items[j].finalScore
		}
		if items[i].rrfScore != items[j].rrfScore {
			return items[i].rrfScore > items[j].rrfScore
		}
		if items[i].bm25Score != items[j].bm25Score {
			return items[i].bm25Score > items[j].bm25Score
		}
		if items[i].denseScore != items[j].denseScore {
			return items[i].denseScore > items[j].denseScore
		}
		return items[i].doc.Path < items[j].doc.Path
	})

	results := make([]RankedItem, nDocs)
	for i, it := range items {
		results[i] = RankedItem{
			Doc:       it.doc,
			Path:      it.doc.Path,
			Score:     it.finalScore,
			Rank:      i,
			BM25Rank:  it.bm25Rank,
			DenseRank: it.denseRank,
			Match: ScoredMatch{
				Doc:            it.doc,
				BM25Score:      it.bm25Score,
				DenseScore:     it.denseScore,
				BM25Rank:       it.bm25Rank,
				DenseRank:      it.denseRank,
				RRFScore:       it.rrfScore,
				ExactPathHit:   it.exactPathHit,
				ExactNameHit:   it.exactNameHit,
				ExactSymbolHit: it.exactSymbolHit,
				Boost:          it.boost,
				FinalScore:     it.finalScore,
			},
		}
	}

	return results
}

func lookupDenseScore(doc Document, denseScores map[string]float64) float64 {
	if denseScores == nil {
		return 0.0
	}
	if s, ok := denseScores[doc.Path]; ok {
		return s
	}
	if s, ok := denseScores[doc.Basename]; ok {
		return s
	}
	normPath := strings.ToLower(strings.ReplaceAll(doc.Path, "\\", "/"))
	if s, ok := denseScores[normPath]; ok {
		return s
	}
	return 0.0
}

func checkExactMatches(doc Document, query string) (pathHit, nameHit, symHit, partSymHit bool) {
	cleanQuery := strings.TrimSpace(query)
	cleanQuery = strings.Trim(cleanQuery, "()[]{};:\"'`")
	if cleanQuery == "" {
		return false, false, false, false
	}
	queryLower := strings.ToLower(cleanQuery)

	normDocPath := strings.ToLower(strings.ReplaceAll(doc.Path, "\\", "/"))
	normQueryPath := strings.ToLower(strings.ReplaceAll(cleanQuery, "\\", "/"))
	if normDocPath != "" && (normDocPath == normQueryPath || strings.HasSuffix(normDocPath, "/"+normQueryPath)) {
		pathHit = true
	}

	normBasename := strings.ToLower(doc.Basename)
	if normBasename != "" {
		if normBasename == queryLower {
			nameHit = true
		}
		ext := filepath.Ext(normBasename)
		if ext != "" && strings.TrimSuffix(normBasename, ext) == queryLower {
			nameHit = true
		}
	}

	for _, sym := range doc.Symbols {
		symClean := strings.TrimSpace(sym)
		symLower := strings.ToLower(symClean)
		if symClean == cleanQuery || symLower == queryLower {
			symHit = true
			break
		}
	}

	if !symHit && len(doc.Symbols) > 0 {
		queryWords := strings.FieldsFunc(cleanQuery, func(r rune) bool {
			return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '_'
		})
		for _, qw := range queryWords {
			if len(qw) < 3 {
				continue
			}
			for _, sym := range doc.Symbols {
				if strings.EqualFold(qw, sym) {
					partSymHit = true
					break
				}
			}
			if partSymHit {
				break
			}
		}
	}

	return pathHit, nameHit, symHit, partSymHit
}

func cleanBasename(p string) string {
	p = strings.ReplaceAll(p, "\\", "/")
	p = strings.TrimRight(p, "/")
	idx := strings.LastIndex(p, "/")
	if idx >= 0 {
		return p[idx+1:]
	}
	return p
}

func extractSymbols(content string) []string {
	var symbols []string
	seen := make(map[string]bool)
	lines := strings.Split(content, "\n")
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		var id string
		if strings.HasPrefix(trimmed, "func ") {
			rest := strings.TrimPrefix(trimmed, "func ")
			if strings.HasPrefix(rest, "(") {
				idx := strings.Index(rest, ")")
				if idx >= 0 && idx+1 < len(rest) {
					rest = strings.TrimSpace(rest[idx+1:])
				}
			}
			id = extractIdentifier(rest)
		} else if strings.HasPrefix(trimmed, "type ") {
			rest := strings.TrimPrefix(trimmed, "type ")
			id = extractIdentifier(rest)
		} else if strings.HasPrefix(trimmed, "const ") {
			rest := strings.TrimPrefix(trimmed, "const ")
			id = extractIdentifier(rest)
		} else if strings.HasPrefix(trimmed, "var ") {
			rest := strings.TrimPrefix(trimmed, "var ")
			id = extractIdentifier(rest)
		}
		if id != "" && !seen[id] && len(id) >= 2 {
			seen[id] = true
			symbols = append(symbols, id)
		}
	}
	return symbols
}

func extractIdentifier(s string) string {
	s = strings.TrimSpace(s)
	var sb strings.Builder
	for i, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || r == '_' || (i > 0 && r >= '0' && r <= '9') {
			sb.WriteRune(r)
		} else {
			break
		}
	}
	return sb.String()
}

func tokenize(text string) []string {
	var tokens []string
	fields := strings.FieldsFunc(text, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '_'
	})
	for _, f := range fields {
		f = strings.Trim(f, "_")
		if f == "" {
			continue
		}
		fLower := strings.ToLower(f)
		tokens = append(tokens, fLower)
		subParts := splitCamelAndSnake(f)
		for _, sp := range subParts {
			spLower := strings.ToLower(sp)
			if spLower != fLower && spLower != "" {
				tokens = append(tokens, spLower)
			}
		}
	}
	return tokens
}

func splitCamelAndSnake(s string) []string {
	var result []string
	parts := strings.Split(s, "_")
	for _, part := range parts {
		if part == "" {
			continue
		}
		sub := splitCamelCase(part)
		result = append(result, sub...)
	}
	return result
}

func splitCamelCase(s string) []string {
	var tokens []string
	var current strings.Builder
	runes := []rune(s)
	for i := 0; i < len(runes); i++ {
		r := runes[i]
		if i > 0 && unicode.IsUpper(r) {
			prevLower := unicode.IsLower(runes[i-1])
			nextLower := i+1 < len(runes) && unicode.IsLower(runes[i+1])
			if prevLower || (unicode.IsUpper(runes[i-1]) && nextLower) {
				if current.Len() > 0 {
					tokens = append(tokens, current.String())
					current.Reset()
				}
			}
		}
		current.WriteRune(r)
	}
	if current.Len() > 0 {
		tokens = append(tokens, current.String())
	}
	return tokens
}

var stemSuffixes = []string{"ing", "ies", "ion", "ers", "ise", "ize", "ed", "es", "er", "ly", "s"}

func stemToken(t string) string {
	for {
		next := stripOneSuffix(t)
		if next == t {
			return t
		}
		t = next
	}
}

func stripOneSuffix(t string) string {
	for _, suf := range stemSuffixes {
		if len(t) >= len(suf)+3 && strings.HasSuffix(t, suf) {
			return t[:len(t)-len(suf)]
		}
	}
	return t
}
