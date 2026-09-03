package agentopt

import (
	"errors"
	"fmt"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

// OutlineSlice represents a structural section of a file (function, type, or outline span).
type OutlineSlice struct {
	PageIndex int    `json:"page_index"`
	StartLine int    `json:"start_line"` // 1-based start line
	EndLine   int    `json:"end_line"`   // 1-based end line
	Signature string `json:"signature"`  // declaration signature or section heading
	Summary   string `json:"summary"`    // structural summary
	IsLoaded  bool   `json:"is_loaded"`  // true if full body is resident in context
	Content   string `json:"content,omitempty"`
}

// LineCount returns the number of lines in this outline slice.
func (s OutlineSlice) LineCount() int {
	if s.EndLine < s.StartLine {
		return 0
	}
	return s.EndLine - s.StartLine + 1
}

// PageFault records an event where an un-loaded page was requested.
type PageFault struct {
	PageIndex  int       `json:"page_index"`
	FilePath   string    `json:"file_path"`
	Timestamp  time.Time `json:"timestamp"`
	PrunedPage int       `json:"pruned_page"` // -1 if no pruneion occurred
}

// DemandPageReader specifies the interface for demand-paged context loading of large files.
type DemandPageReader interface {
	LoadOutline(filePath, content string) ([]OutlineSlice, error)
	RequestPage(pageIndex int) (string, error)
	PageFaultCount() int
	GetLoadedOutline() string
}

type pageItemRecord struct {
	pageIndex   int
	accessCount int
	lastTick    uint64
}

// PagedFileLoader implements demand-paged context loading for large file inspection.
// It serves structural outlines first and pages in full function/section bodies on demand,
// caching frequently referenced spans with a bounded page capacity.
type PagedFileLoader struct {
	mu           sync.RWMutex
	filePath     string
	capacity     int
	slices       []OutlineSlice
	bodies       []string
	loaded       map[int]*pageItemRecord
	tick         uint64
	faultCount   int
	faultHistory []PageFault
}

// Compile-time check that PagedFileLoader implements DemandPageReader.
var _ DemandPageReader = (*PagedFileLoader)(nil)

// NewPagedFileLoader initializes a PagedFileLoader with a bounded page capacity.
func NewPagedFileLoader(capacity int) *PagedFileLoader {
	if capacity <= 0 {
		capacity = 4
	}
	return &PagedFileLoader{
		capacity: capacity,
		loaded:   make(map[int]*pageItemRecord),
	}
}

// NewDemandPageReader initializes a DemandPageReader with a bounded page capacity.
func NewDemandPageReader(capacity int) *PagedFileLoader {
	return NewPagedFileLoader(capacity)
}

// LoadOutline parses the file into structural outline slices without loading full bodies.
func (p *PagedFileLoader) LoadOutline(filePath, content string) ([]OutlineSlice, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if content == "" && filePath != "" {
		data, err := os.ReadFile(filePath)
		if err == nil {
			content = string(data)
		}
	}
	if content == "" {
		return nil, errors.New("content cannot be empty")
	}

	slices, bodies := parseOutline(filePath, content)
	if len(slices) == 0 {
		return nil, errors.New("failed to parse structural outline")
	}

	p.filePath = filePath
	p.slices = slices
	p.bodies = bodies
	p.loaded = make(map[int]*pageItemRecord)
	p.tick = 0
	p.faultCount = 0
	p.faultHistory = nil

	out := make([]OutlineSlice, len(p.slices))
	copy(out, p.slices)
	return out, nil
}

// RequestPage retrieves the full content of a specified page, triggering a page fault
// if the page was not resident in buffer. Frequently referenced pages are retained.
func (p *PagedFileLoader) RequestPage(pageIndex int) (string, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if len(p.slices) == 0 {
		return "", errors.New("no outline loaded")
	}
	if pageIndex < 0 || pageIndex >= len(p.slices) {
		return "", fmt.Errorf("page index %d out of bounds (0-%d)", pageIndex, len(p.slices)-1)
	}

	p.tick++

	// Buffer Hit: already resident in bounded buffer
	if record, exists := p.loaded[pageIndex]; exists {
		record.accessCount++
		record.lastTick = p.tick
		return p.bodies[pageIndex], nil
	}

	// Page Fault: page is not resident
	p.faultCount++
	prunedPage := -1

	// Prune if capacity limit is reached (LFU with LRU tie-breaking)
	if len(p.loaded) >= p.capacity {
		victimIdx := -1
		minAccess := -1
		var oldestTick uint64

		for idx, rec := range p.loaded {
			if victimIdx == -1 ||
				rec.accessCount < minAccess ||
				(rec.accessCount == minAccess && rec.lastTick < oldestTick) {
				victimIdx = idx
				minAccess = rec.accessCount
				oldestTick = rec.lastTick
			}
		}

		if victimIdx != -1 {
			delete(p.loaded, victimIdx)
			p.slices[victimIdx].IsLoaded = false
			p.slices[victimIdx].Content = ""
			prunedPage = victimIdx
		}
	}

	// Page in the requested slice
	p.loaded[pageIndex] = &pageItemRecord{
		pageIndex:   pageIndex,
		accessCount: 1,
		lastTick:    p.tick,
	}
	p.slices[pageIndex].IsLoaded = true
	p.slices[pageIndex].Content = p.bodies[pageIndex]

	p.faultHistory = append(p.faultHistory, PageFault{
		PageIndex:  pageIndex,
		FilePath:   p.filePath,
		Timestamp:  time.Now(),
		PrunedPage: prunedPage,
	})

	return p.bodies[pageIndex], nil
}

// PageFaultCount returns the total number of page faults recorded.
func (p *PagedFileLoader) PageFaultCount() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.faultCount
}

// GetLoadedOutline assembles the current view of the file: loaded pages display their full
// body, while un-loaded pages display compact structural outlines.
func (p *PagedFileLoader) GetLoadedOutline() string {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if len(p.slices) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("// [Demand-Paged Context: %s | pages: %d | loaded: %d/%d | faults: %d]\n\n",
		p.filePath, len(p.slices), len(p.loaded), p.capacity, p.faultCount))

	for i, s := range p.slices {
		if s.IsLoaded {
			sb.WriteString(fmt.Sprintf("// --- Page %d (lines %d-%d): %s [LOADED] ---\n",
				i, s.StartLine, s.EndLine, s.Signature))
			sb.WriteString(p.bodies[i])
			if !strings.HasSuffix(p.bodies[i], "\n") {
				sb.WriteString("\n")
			}
			sb.WriteString("\n")
		} else {
			sb.WriteString(fmt.Sprintf("// --- Page %d (lines %d-%d): %s [PAGED OUT] ---\n",
				i, s.StartLine, s.EndLine, s.Signature))
			sb.WriteString(fmt.Sprintf("// ... [body paged out; call RequestPage(%d) to load lines %d-%d] ...\n\n",
				i, s.StartLine, s.EndLine))
		}
	}

	return strings.TrimSpace(sb.String())
}

// LoadedPages returns a sorted list of page indices currently resident in context.
func (p *PagedFileLoader) LoadedPages() []int {
	p.mu.RLock()
	defer p.mu.RUnlock()

	pages := make([]int, 0, len(p.loaded))
	for idx := range p.loaded {
		pages = append(pages, idx)
	}
	sort.Ints(pages)
	return pages
}

// Outline returns a snapshot copy of all structural outline slices.
func (p *PagedFileLoader) Outline() []OutlineSlice {
	p.mu.RLock()
	defer p.mu.RUnlock()

	out := make([]OutlineSlice, len(p.slices))
	copy(out, p.slices)
	return out
}

// Capacity returns the maximum number of pages that can be concurrently loaded.
func (p *PagedFileLoader) Capacity() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.capacity
}

// FaultHistory returns a snapshot of recorded page faults.
func (p *PagedFileLoader) FaultHistory() []PageFault {
	p.mu.RLock()
	defer p.mu.RUnlock()

	history := make([]PageFault, len(p.faultHistory))
	copy(history, p.faultHistory)
	return history
}

// FilePath returns the path of the file currently loaded.
func (p *PagedFileLoader) FilePath() string {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.filePath
}

// parseOutline parses the file content into structural slices based on file type.
func parseOutline(filePath, content string) ([]OutlineSlice, []string) {
	norm := strings.ReplaceAll(content, "\r\n", "\n")
	norm = strings.TrimSuffix(norm, "\n")
	lines := strings.Split(norm, "\n")
	if len(lines) == 0 {
		return nil, nil
	}

	if isGoFile(filePath, lines) {
		slices, bodies := parseGoOutline(lines)
		if len(slices) > 0 {
			return slices, bodies
		}
	} else if isPythonFile(filePath) {
		slices, bodies := parsePythonOutline(lines)
		if len(slices) > 0 {
			return slices, bodies
		}
	} else if isMarkdownFile(filePath) {
		slices, bodies := parseMarkdownOutline(lines)
		if len(slices) > 0 {
			return slices, bodies
		}
	}

	return parseChunkedOutline(lines, 40)
}

func isGoFile(path string, lines []string) bool {
	if strings.HasSuffix(path, ".go") {
		return true
	}
	for i := 0; i < len(lines) && i < 10; i++ {
		trimmed := strings.TrimSpace(lines[i])
		if strings.HasPrefix(trimmed, "package ") {
			return true
		}
	}
	return false
}

func parseGoOutline(lines []string) ([]OutlineSlice, []string) {
	type decl struct {
		startLine int // 0-based
		declLine  int // 0-based
		endLine   int // 0-based
		signature string
		summary   string
	}

	var decls []decl

	// 1. Locate all top-level func and type declarations
	for i := 0; i < len(lines); i++ {
		trimmed := strings.TrimSpace(lines[i])
		isFunc := strings.HasPrefix(trimmed, "func ") || strings.HasPrefix(trimmed, "func(")
		isType := strings.HasPrefix(trimmed, "type ")

		if !isFunc && !isType {
			continue
		}

		// Find leading comments immediately above
		commentStart := i
		for commentStart > 0 && strings.HasPrefix(strings.TrimSpace(lines[commentStart-1]), "//") {
			commentStart--
		}

		// Signature
		sig := trimmed
		if idx := strings.Index(sig, "{"); idx != -1 {
			sig = strings.TrimSpace(sig[:idx])
		}

		summary := sig
		if isFunc {
			parts := strings.Fields(sig)
			if len(parts) >= 2 {
				summary = parts[0] + " " + parts[1]
			}
		} else if isType {
			parts := strings.Fields(sig)
			if len(parts) >= 2 {
				summary = parts[0] + " " + parts[1]
			}
		}

		// Find block closing brace
		braceCount := 0
		hasBrace := false
		endIdx := i

		for j := i; j < len(lines); j++ {
			l := lines[j]
			for k := 0; k < len(l); k++ {
				if l[k] == '{' {
					braceCount++
					hasBrace = true
				} else if l[k] == '}' {
					braceCount--
				}
			}
			if hasBrace && braceCount <= 0 {
				endIdx = j
				break
			}
		}

		decls = append(decls, decl{
			startLine: commentStart,
			declLine:  i,
			endLine:   endIdx,
			signature: sig,
			summary:   summary,
		})

		i = endIdx
	}

	if len(decls) == 0 {
		return nil, nil
	}

	var slices []OutlineSlice
	var bodies []string
	pageIdx := 0

	// 2. Extract preamble (package & imports before first declaration)
	firstStart := decls[0].startLine
	if firstStart > 0 {
		preambleLines := lines[0:firstStart]
		preambleBody := strings.Join(preambleLines, "\n")
		pkgSig := "package"
		for _, pl := range preambleLines {
			pt := strings.TrimSpace(pl)
			if strings.HasPrefix(pt, "package ") {
				pkgSig = pt
				break
			}
		}

		slices = append(slices, OutlineSlice{
			PageIndex: pageIdx,
			StartLine: 1,
			EndLine:   firstStart,
			Signature: pkgSig,
			Summary:   "package declaration and imports",
			IsLoaded:  false,
		})
		bodies = append(bodies, preambleBody)
		pageIdx++
	}

	// 3. Add each declaration as a slice
	for _, d := range decls {
		bodyLines := lines[d.startLine : d.endLine+1]
		body := strings.Join(bodyLines, "\n")

		slices = append(slices, OutlineSlice{
			PageIndex: pageIdx,
			StartLine: d.startLine + 1,
			EndLine:   d.endLine + 1,
			Signature: d.signature,
			Summary:   d.summary,
			IsLoaded:  false,
		})
		bodies = append(bodies, body)
		pageIdx++
	}

	return slices, bodies
}

func isPythonFile(path string) bool {
	return strings.HasSuffix(path, ".py")
}

func parsePythonOutline(lines []string) ([]OutlineSlice, []string) {
	type pyDecl struct {
		startLine int
		declLine  int
		endLine   int
		signature string
		summary   string
	}

	var decls []pyDecl

	for i := 0; i < len(lines); i++ {
		trimmed := strings.TrimSpace(lines[i])
		isDef := strings.HasPrefix(trimmed, "def ") || strings.HasPrefix(trimmed, "async def ")
		isClass := strings.HasPrefix(trimmed, "class ")

		if !isDef && !isClass {
			continue
		}

		commentStart := i
		for commentStart > 0 && strings.HasPrefix(strings.TrimSpace(lines[commentStart-1]), "#") {
			commentStart--
		}

		sig := strings.TrimSuffix(trimmed, ":")
		summary := sig
		parts := strings.Fields(sig)
		if len(parts) >= 2 {
			summary = parts[0] + " " + parts[1]
		}

		// Calculate indentation of definition
		indent := len(lines[i]) - len(strings.TrimLeft(lines[i], " \t"))

		endIdx := i
		for j := i + 1; j < len(lines); j++ {
			lineTrimmed := strings.TrimSpace(lines[j])
			if lineTrimmed == "" {
				continue
			}
			lineIndent := len(lines[j]) - len(strings.TrimLeft(lines[j], " \t"))
			if lineIndent <= indent {
				break
			}
			endIdx = j
		}

		decls = append(decls, pyDecl{
			startLine: commentStart,
			declLine:  i,
			endLine:   endIdx,
			signature: sig,
			summary:   summary,
		})

		i = endIdx
	}

	if len(decls) == 0 {
		return nil, nil
	}

	var slices []OutlineSlice
	var bodies []string
	pageIdx := 0

	firstStart := decls[0].startLine
	if firstStart > 0 {
		preambleBody := strings.Join(lines[0:firstStart], "\n")
		slices = append(slices, OutlineSlice{
			PageIndex: pageIdx,
			StartLine: 1,
			EndLine:   firstStart,
			Signature: "module header and imports",
			Summary:   "module header",
			IsLoaded:  false,
		})
		bodies = append(bodies, preambleBody)
		pageIdx++
	}

	for _, d := range decls {
		body := strings.Join(lines[d.startLine:d.endLine+1], "\n")
		slices = append(slices, OutlineSlice{
			PageIndex: pageIdx,
			StartLine: d.startLine + 1,
			EndLine:   d.endLine + 1,
			Signature: d.signature,
			Summary:   d.summary,
			IsLoaded:  false,
		})
		bodies = append(bodies, body)
		pageIdx++
	}

	return slices, bodies
}

func isMarkdownFile(path string) bool {
	return strings.HasSuffix(path, ".md") || strings.HasSuffix(path, ".markdown")
}

func parseMarkdownOutline(lines []string) ([]OutlineSlice, []string) {
	type mdSection struct {
		startLine int
		signature string
	}

	var sections []mdSection
	for i := 0; i < len(lines); i++ {
		trimmed := strings.TrimSpace(lines[i])
		if strings.HasPrefix(trimmed, "#") {
			sections = append(sections, mdSection{
				startLine: i,
				signature: trimmed,
			})
		}
	}

	if len(sections) == 0 {
		return nil, nil
	}

	var slices []OutlineSlice
	var bodies []string
	pageIdx := 0

	if sections[0].startLine > 0 {
		preambleBody := strings.Join(lines[0:sections[0].startLine], "\n")
		slices = append(slices, OutlineSlice{
			PageIndex: pageIdx,
			StartLine: 1,
			EndLine:   sections[0].startLine,
			Signature: "preamble",
			Summary:   "document preamble",
			IsLoaded:  false,
		})
		bodies = append(bodies, preambleBody)
		pageIdx++
	}

	for idx, sec := range sections {
		endLine := len(lines) - 1
		if idx+1 < len(sections) {
			endLine = sections[idx+1].startLine - 1
		}

		body := strings.Join(lines[sec.startLine:endLine+1], "\n")
		slices = append(slices, OutlineSlice{
			PageIndex: pageIdx,
			StartLine: sec.startLine + 1,
			EndLine:   endLine + 1,
			Signature: sec.signature,
			Summary:   sec.signature,
			IsLoaded:  false,
		})
		bodies = append(bodies, body)
		pageIdx++
	}

	return slices, bodies
}

func parseChunkedOutline(lines []string, chunkSize int) ([]OutlineSlice, []string) {
	if chunkSize <= 0 {
		chunkSize = 40
	}

	var slices []OutlineSlice
	var bodies []string
	pageIdx := 0

	for i := 0; i < len(lines); i += chunkSize {
		end := i + chunkSize
		if end > len(lines) {
			end = len(lines)
		}

		chunkLines := lines[i:end]
		body := strings.Join(chunkLines, "\n")
		startLine := i + 1
		endLine := end

		sig := fmt.Sprintf("Lines %d-%d", startLine, endLine)
		for _, l := range chunkLines {
			trimmed := strings.TrimSpace(l)
			if trimmed != "" {
				if len(trimmed) > 60 {
					trimmed = trimmed[:57] + "..."
				}
				sig = trimmed
				break
			}
		}

		slices = append(slices, OutlineSlice{
			PageIndex: pageIdx,
			StartLine: startLine,
			EndLine:   endLine,
			Signature: sig,
			Summary:   fmt.Sprintf("lines %d-%d (%d lines)", startLine, endLine, endLine-startLine+1),
			IsLoaded:  false,
		})
		bodies = append(bodies, body)
		pageIdx++
	}

	return slices, bodies
}
