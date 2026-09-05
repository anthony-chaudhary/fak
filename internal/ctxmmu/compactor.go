package ctxmmu

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"sync"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// PageKind identifies the structural role of a token page in context.
type PageKind uint8

const (
	PageKindUnknown      PageKind = iota
	PageKindPrefixSystem          // System prompt instructions (pinned prefix)
	PageKindPrefixTools           // Tool schemas & catalog definitions (pinned prefix)
	PageKindUser                  // User interactive input
	PageKindAssistant             // Assistant conversational reasoning / response
	PageKindToolCall              // Assistant tool invocation
	PageKindToolResult            // Tool execution output (compactable in middle turns)
)

// String renders the page kind name.
func (k PageKind) String() string {
	switch k {
	case PageKindPrefixSystem:
		return "system_prefix"
	case PageKindPrefixTools:
		return "tools_prefix"
	case PageKindUser:
		return "user"
	case PageKindAssistant:
		return "assistant"
	case PageKindToolCall:
		return "tool_call"
	case PageKindToolResult:
		return "tool_result"
	default:
		return "unknown"
	}
}

// IsPrefix reports whether the page kind belongs to the immutable prefix.
func (k PageKind) IsPrefix() bool {
	return k == PageKindPrefixSystem || k == PageKindPrefixTools
}

// Tombstone holds metadata for a compacted tool execution payload.
type Tombstone struct {
	Active         bool     `json:"active"`
	Ref            string   `json:"ref,omitempty"`     // CAS content address (hex sha256)
	Digest         [32]byte `json:"-"`                 // raw sha256 sum (zero-alloc representation)
	OriginalBytes  int      `json:"original_bytes"`    // byte length before compaction
	OriginalTokens int      `json:"original_tokens"`   // token count before compaction
	Tool           string   `json:"tool,omitempty"`    // tool name
	Summary        string   `json:"summary,omitempty"` // compact summary digest
}

// HasRef reports whether the tombstone carries a valid CAS reference.
func (t *Tombstone) HasRef() bool {
	return t.Ref != "" || t.Digest != [32]byte{}
}

// RefString returns the hex CAS reference string.
func (t *Tombstone) RefString() string {
	if t.Ref != "" {
		return t.Ref
	}
	if t.Digest != [32]byte{} {
		return hex.EncodeToString(t.Digest[:])
	}
	return ""
}

// TokenPage represents one page or unit of context memory in the CtxMMU.
type TokenPage struct {
	ID             uint64    `json:"id"`
	TurnIndex      int       `json:"turn_index"` // <1 for prefix; 1..N for interactive turns
	Kind           PageKind  `json:"kind"`
	Role           string    `json:"role"`                      // "system", "user", "assistant", "tool"
	ToolName       string    `json:"tool_name,omitempty"`       // e.g. "bash", "read", "grep"
	Content        []byte    `json:"-"`                         // resident bytes / formatted tombstone
	Tokens         int       `json:"tokens"`                    // current token count
	Resident       bool      `json:"resident"`                  // true if resident in context
	Pinned         bool      `json:"pinned,omitempty"`          // if true, never evicted or tombstoned
	IsContinuation bool      `json:"is_continuation,omitempty"` // true if part of a multi-page tool result
	Tombstone      Tombstone `json:"tombstone,omitempty"`
}

// Reset clears page fields while preserving slice capacity.
func (p *TokenPage) Reset() {
	buf := p.Content[:0]
	*p = TokenPage{
		Content: buf,
	}
}

// Default compaction constants.
const (
	DefaultWindowSizeK            = 4
	DefaultVerboseThresholdBytes  = 512
	DefaultVerboseThresholdTokens = 128
)

// CompactorConfig configures sliding-window semantic compaction.
type CompactorConfig struct {
	// WindowSizeK is the number of recent interactive turns to retain verbatim.
	WindowSizeK int
	// VerboseThresholdBytes is the byte threshold above which middle-turn
	// tool execution payloads are compacted to tombstones.
	VerboseThresholdBytes int
	// VerboseThresholdTokens is the token threshold for compaction.
	VerboseThresholdTokens int
	// GenerateSummary enables compact summary digest generation for tombstones.
	GenerateSummary bool
	// CASPageOut enables writing compacted payloads into the CAS store.
	CASPageOut bool
}

// ScanReport summarizes context residency and compaction opportunities without allocating.
type ScanReport struct {
	TotalPages        int `json:"total_pages"`
	TotalTokens       int `json:"total_tokens"`
	TotalBytes        int `json:"total_bytes"`
	PrefixPages       int `json:"prefix_pages"`
	PrefixTokens      int `json:"prefix_tokens"`
	MiddlePages       int `json:"middle_pages"`
	MiddleTokens      int `json:"middle_tokens"`
	ActivePages       int `json:"active_pages"`
	ActiveTokens      int `json:"active_tokens"`
	ReclaimablePages  int `json:"reclaimable_pages"`
	ReclaimableTokens int `json:"reclaimable_tokens"`
	ReclaimableBytes  int `json:"reclaimable_bytes"`
	InteractiveTurns  int `json:"interactive_turns"`
	ActiveWindowTurns int `json:"active_window_turns"`
}

// CompactionReport records the observed effects of a compaction pass.
type CompactionReport struct {
	BeforeTokens      int  `json:"before_tokens"`
	AfterTokens       int  `json:"after_tokens"`
	TokensReclaimed   int  `json:"tokens_reclaimed"`
	BeforeBytes       int  `json:"before_bytes"`
	AfterBytes        int  `json:"after_bytes"`
	BytesReclaimed    int  `json:"bytes_reclaimed"`
	TombstonesCreated int  `json:"tombstones_created"`
	PagesReclaimed    int  `json:"pages_reclaimed"`
	PrefixWarm        bool `json:"prefix_warm"`
	NoOp              bool `json:"no_op"`
}

// TokenPageFreelist provides a zero-allocation freelist for TokenPages.
type TokenPageFreelist struct {
	mu        sync.Mutex
	freelist  []*TokenPage
	maxFree   int
	allocated int64
	reclaimed int64
}

// NewTokenPageFreelist creates a reusable page freelist with the specified capacity.
func NewTokenPageFreelist(capacity int) *TokenPageFreelist {
	if capacity <= 0 {
		capacity = 1024
	}
	return &TokenPageFreelist{
		freelist: make([]*TokenPage, 0, capacity),
		maxFree:  capacity,
	}
}

// Get acquires a page from the freelist or allocates a new one.
func (p *TokenPageFreelist) Get() *TokenPage {
	if p == nil {
		return &TokenPage{Content: make([]byte, 0, 4096), Resident: true}
	}
	p.mu.Lock()
	n := len(p.freelist)
	if n > 0 {
		page := p.freelist[n-1]
		p.freelist = p.freelist[:n-1]
		p.mu.Unlock()
		page.Reset()
		page.Resident = true
		return page
	}
	p.mu.Unlock()
	atomic.AddInt64(&p.allocated, 1)
	return &TokenPage{
		Content:  make([]byte, 0, 4096),
		Resident: true,
	}
}

// Put returns a page to the freelist for reuse.
func (p *TokenPageFreelist) Put(page *TokenPage) {
	if p == nil || page == nil {
		return
	}
	atomic.AddInt64(&p.reclaimed, 1)
	page.Reset()
	page.Resident = false
	p.mu.Lock()
	if len(p.freelist) < p.maxFree {
		p.freelist = append(p.freelist, page)
	}
	p.mu.Unlock()
}

// Stats returns freelist allocation and reclamation counts.
func (p *TokenPageFreelist) Stats() (allocated, reclaimed int64, freeCount int) {
	if p == nil {
		return 0, 0, 0
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return atomic.LoadInt64(&p.allocated), atomic.LoadInt64(&p.reclaimed), len(p.freelist)
}

// Compactor implements sliding-window semantic compaction with zero-alloc page reclamation.
type Compactor struct {
	mu       sync.RWMutex
	mmu      *MMU
	config   CompactorConfig
	freelist *TokenPageFreelist
}

// NewCompactor builds a compactor with the given configuration.
func NewCompactor(cfg CompactorConfig) *Compactor {
	if cfg.WindowSizeK <= 0 {
		cfg.WindowSizeK = DefaultWindowSizeK
	}
	if cfg.VerboseThresholdBytes <= 0 {
		cfg.VerboseThresholdBytes = DefaultVerboseThresholdBytes
	}
	if cfg.VerboseThresholdTokens <= 0 {
		cfg.VerboseThresholdTokens = DefaultVerboseThresholdTokens
	}
	return &Compactor{
		config:   cfg,
		freelist: NewTokenPageFreelist(1024),
	}
}

// NewCompactorWithMMU builds a compactor backed by the specified MMU.
func NewCompactorWithMMU(cfg CompactorConfig, m *MMU) *Compactor {
	c := NewCompactor(cfg)
	c.mmu = m
	return c
}

// Freelist returns the compactor's token page freelist.
func (c *Compactor) Freelist() *TokenPageFreelist {
	return c.freelist
}

// Scan inspects the context page slice and returns a ScanReport.
func (c *Compactor) Scan(pages []TokenPage) ScanReport {
	var report ScanReport
	c.ScanInto(pages, &report)
	return report
}

// ScanInto writes context analysis into out without any heap allocations.
func (c *Compactor) ScanInto(pages []TokenPage, out *ScanReport) {
	if out == nil {
		return
	}
	*out = ScanReport{}
	if len(pages) == 0 {
		return
	}

	maxTurn := 0
	for i := range pages {
		p := &pages[i]
		if !p.Kind.IsPrefix() && p.TurnIndex > maxTurn {
			maxTurn = p.TurnIndex
		}
	}

	windowK := c.config.WindowSizeK
	if windowK <= 0 {
		windowK = DefaultWindowSizeK
	}
	activeWindowStart := maxTurn - windowK + 1
	if activeWindowStart < 1 {
		activeWindowStart = 1
	}

	out.InteractiveTurns = maxTurn
	if maxTurn >= windowK {
		out.ActiveWindowTurns = windowK
	} else {
		out.ActiveWindowTurns = maxTurn
	}

	threshBytes := c.config.VerboseThresholdBytes
	if threshBytes <= 0 {
		threshBytes = DefaultVerboseThresholdBytes
	}
	threshTokens := c.config.VerboseThresholdTokens
	if threshTokens <= 0 {
		threshTokens = DefaultVerboseThresholdTokens
	}

	for i := range pages {
		p := &pages[i]
		tokens := p.Tokens
		if tokens <= 0 {
			tokens = EstimateTokens(p.Content)
		}
		bytesLen := len(p.Content)

		out.TotalPages++
		out.TotalTokens += tokens
		out.TotalBytes += bytesLen

		if p.Kind.IsPrefix() || p.TurnIndex < 1 || p.Pinned {
			out.PrefixPages++
			out.PrefixTokens += tokens
			continue
		}

		if p.TurnIndex >= activeWindowStart {
			out.ActivePages++
			out.ActiveTokens += tokens
			continue
		}

		// Middle turn
		out.MiddlePages++
		out.MiddleTokens += tokens

		if p.Kind == PageKindToolResult && !p.Tombstone.Active {
			if p.IsContinuation {
				out.ReclaimablePages++
				out.ReclaimableTokens += tokens
				out.ReclaimableBytes += bytesLen
			} else if bytesLen > threshBytes || tokens > threshTokens {
				out.ReclaimablePages++
				estTombTokens := 25
				if tokens > estTombTokens {
					out.ReclaimableTokens += (tokens - estTombTokens)
				}
				if bytesLen > 100 {
					out.ReclaimableBytes += (bytesLen - 100)
				}
			}
		}
	}
}

// CompactInPlace compacts pages in place with zero heap allocations during the scan and compaction pass.
func (c *Compactor) CompactInPlace(pages []TokenPage, out *CompactionReport) ([]TokenPage, error) {
	var report CompactionReport
	if out == nil {
		out = &report
	} else {
		*out = CompactionReport{}
	}

	if len(pages) == 0 {
		out.PrefixWarm = true
		out.NoOp = true
		return pages, nil
	}

	for i := range pages {
		tok := pages[i].Tokens
		if tok <= 0 {
			tok = EstimateTokens(pages[i].Content)
			pages[i].Tokens = tok
		}
		out.BeforeTokens += tok
		out.BeforeBytes += len(pages[i].Content)
	}

	maxTurn := 0
	for i := range pages {
		if !pages[i].Kind.IsPrefix() && pages[i].TurnIndex > maxTurn {
			maxTurn = pages[i].TurnIndex
		}
	}

	windowK := c.config.WindowSizeK
	if windowK <= 0 {
		windowK = DefaultWindowSizeK
	}
	activeWindowStart := maxTurn - windowK + 1
	if activeWindowStart < 1 {
		activeWindowStart = 1
	}

	threshBytes := c.config.VerboseThresholdBytes
	if threshBytes <= 0 {
		threshBytes = DefaultVerboseThresholdBytes
	}
	threshTokens := c.config.VerboseThresholdTokens
	if threshTokens <= 0 {
		threshTokens = DefaultVerboseThresholdTokens
	}

	// If no middle turns exist, everything is either prefix or active window
	if maxTurn < 1 || activeWindowStart <= 1 {
		out.AfterTokens = out.BeforeTokens
		out.AfterBytes = out.BeforeBytes
		out.PrefixWarm = true
		out.NoOp = true
		return pages, nil
	}

	w := 0
	for r := 0; r < len(pages); r++ {
		p := &pages[r]

		// Prefix, active window, or pinned -> preserve verbatim
		if p.Kind.IsPrefix() || p.TurnIndex < 1 || p.TurnIndex >= activeWindowStart || p.Pinned {
			if r != w {
				pages[w] = *p
			}
			w++
			continue
		}

		// Middle turn: only compact verbose tool results
		if p.Kind != PageKindToolResult || p.Tombstone.Active {
			if r != w {
				pages[w] = *p
			}
			w++
			continue
		}

		// If this is a continuation page of a multi-page tool result in middle turns, reclaim it
		if p.IsContinuation {
			out.PagesReclaimed++
			out.TokensReclaimed += p.Tokens
			out.BytesReclaimed += len(p.Content)
			if c.freelist != nil {
				c.freelist.Put(p)
			}
			continue
		}

		origBytes := len(p.Content)
		origTokens := p.Tokens
		isVerbose := origBytes > threshBytes || origTokens > threshTokens
		if !isVerbose {
			if r != w {
				pages[w] = *p
			}
			w++
			continue
		}

		// Verbose middle-turn tool result! Compact it.
		sum := sha256.Sum256(p.Content)
		var casRef string
		if c.config.CASPageOut {
			casRef, _ = c.pageOutToCAS(p.Content)
		}

		p.Tombstone = Tombstone{
			Active:         true,
			Digest:         sum,
			Ref:            casRef,
			OriginalBytes:  origBytes,
			OriginalTokens: origTokens,
			Tool:           p.ToolName,
		}

		if c.config.GenerateSummary {
			var sumBuf [96]byte
			sBytes := AppendFastSummary(sumBuf[:0], p.Content)
			p.Tombstone.Summary = string(sBytes)
		}

		// Format tombstone directly into existing p.Content slice buffer
		p.Content = FormatTombstone(p.Tombstone, p.Content[:0])
		tombTokens := EstimateTokens(p.Content)
		p.Tokens = tombTokens

		out.TombstonesCreated++
		out.TokensReclaimed += (origTokens - tombTokens)
		out.BytesReclaimed += (origBytes - len(p.Content))

		if r != w {
			pages[w] = *p
		}
		w++
	}

	compacted := pages[:w]

	for i := range compacted {
		out.AfterTokens += compacted[i].Tokens
		out.AfterBytes += len(compacted[i].Content)
	}

	out.PrefixWarm = true
	return compacted, nil
}

// Compact copies pages and compacts them safely under ctx.
func (c *Compactor) Compact(ctx context.Context, pages []TokenPage) ([]TokenPage, CompactionReport, error) {
	_ = ctx
	if len(pages) == 0 {
		return nil, CompactionReport{PrefixWarm: true, NoOp: true}, nil
	}
	// Copy pages slice for non-destructive compaction
	copied := make([]TokenPage, len(pages))
	for i := range pages {
		copied[i] = pages[i]
		if len(pages[i].Content) > 0 {
			contentCopy := make([]byte, len(pages[i].Content), max(len(pages[i].Content), 512))
			copy(contentCopy, pages[i].Content)
			copied[i].Content = contentCopy
		}
	}
	var report CompactionReport
	compacted, err := c.CompactInPlace(copied, &report)
	return compacted, report, err
}

func (c *Compactor) pageOutToCAS(body []byte) (string, error) {
	if c.mmu != nil {
		h := c.mmu.pageOut(context.Background(), body)
		if h.Digest != "" {
			abi.PinResolved(h)
			return h.Digest, nil
		}
	}
	if b, ok := abi.PageOut("blob"); ok {
		inline := abi.Ref{Kind: abi.RefInline, Inline: body, Len: int64(len(body))}
		if h, err := b.PageOut(context.Background(), inline); err == nil && h.Digest != "" {
			abi.PinResolved(h)
			return h.Digest, nil
		}
	}
	sum := sha256.Sum256(body)
	return hex.EncodeToString(sum[:]), nil
}

// ReFault reads back the original payload bytes for a tombstoned page from the CAS store.
func (c *Compactor) ReFault(ctx context.Context, page *TokenPage) ([]byte, error) {
	if page == nil {
		return nil, errors.New("ctxmmu: nil token page")
	}
	if !page.Tombstone.Active {
		return page.Content, nil
	}
	ref := page.Tombstone.RefString()
	if ref == "" {
		return nil, errors.New("ctxmmu: tombstone has no CAS ref")
	}
	if res := abi.ActiveResolver(); res != nil {
		handle := abi.Ref{Kind: abi.RefBlob, Digest: ref}
		b, err := res.Resolve(ctx, handle)
		if err == nil && len(b) > 0 {
			return b, nil
		}
	}
	if b, ok := abi.PageOut("blob"); ok {
		handle := abi.Ref{Kind: abi.RefBlob, Digest: ref}
		refRes, err := b.PageIn(ctx, handle)
		if err == nil && len(refRes.Inline) > 0 {
			return refRes.Inline, nil
		}
	}
	return nil, fmt.Errorf("ctxmmu: ref %s not found in CAS store", ref)
}

// VerifyPrefixWarmth confirms that all prefix pages (system prompt, tool definitions)
// remain 100% byte-identical between before and after context states.
func VerifyPrefixWarmth(before, after []TokenPage) bool {
	prefixCount := 0
	for i := range before {
		if before[i].Kind.IsPrefix() || before[i].TurnIndex <= 0 {
			prefixCount++
		} else {
			break
		}
	}
	if len(after) < prefixCount {
		return false
	}
	for i := 0; i < prefixCount; i++ {
		b := &before[i]
		a := &after[i]
		if b.Kind != a.Kind || b.TurnIndex != a.TurnIndex || b.Role != a.Role || b.Tokens != a.Tokens {
			return false
		}
		if !bytes.Equal(b.Content, a.Content) {
			return false
		}
	}
	return true
}

const hextable = "0123456789abcdef"

func appendHex(dst []byte, src []byte) []byte {
	for _, v := range src {
		dst = append(dst, hextable[v>>4], hextable[v&0x0f])
	}
	return dst
}

func appendEscapedJSONBytes(dst []byte, src []byte) []byte {
	for _, b := range src {
		switch b {
		case '"':
			dst = append(dst, '\\', '"')
		case '\\':
			dst = append(dst, '\\', '\\')
		case '\n':
			dst = append(dst, '\\', 'n')
		case '\r':
			dst = append(dst, '\\', 'r')
		case '\t':
			dst = append(dst, '\\', 't')
		default:
			if b < 0x20 {
				dst = append(dst, '\\', 'u', '0', '0', hextable[b>>4], hextable[b&0x0f])
			} else {
				dst = append(dst, b)
			}
		}
	}
	return dst
}

// FormatTombstone formats tombstone metadata into structured JSON bytes in dst without heap allocation.
func FormatTombstone(t Tombstone, dst []byte) []byte {
	dst = append(dst, `{"_tombstone":true,"ref":"`...)
	if t.Ref != "" {
		dst = append(dst, t.Ref...)
	} else if t.Digest != [32]byte{} {
		dst = appendHex(dst, t.Digest[:])
	}
	dst = append(dst, '"')
	if t.Tool != "" {
		dst = append(dst, `,"tool":"`...)
		dst = appendEscapedJSONBytes(dst, []byte(t.Tool))
		dst = append(dst, '"')
	}
	dst = append(dst, `,"len":`...)
	dst = strconv.AppendInt(dst, int64(t.OriginalBytes), 10)
	dst = append(dst, `,"tokens":`...)
	dst = strconv.AppendInt(dst, int64(t.OriginalTokens), 10)
	if t.Summary != "" {
		dst = append(dst, `,"summary":"`...)
		dst = appendEscapedJSONBytes(dst, []byte(t.Summary))
		dst = append(dst, '"')
	}
	dst = append(dst, '}')
	return dst
}

// AppendFastSummary appends a compact summary of b into dst with zero heap allocations.
func AppendFastSummary(dst []byte, b []byte) []byte {
	if len(b) == 0 {
		return dst
	}
	lines := 1
	var firstLine []byte
	for i := 0; i < len(b); i++ {
		if b[i] == '\n' {
			lines++
			if firstLine == nil && i > 0 {
				firstLine = b[:i]
			}
		}
	}
	if firstLine == nil {
		firstLine = b
	}
	firstLine = bytes.TrimSpace(firstLine)
	if len(firstLine) > 48 {
		firstLine = firstLine[:48]
	}
	dst = strconv.AppendInt(dst, int64(lines), 10)
	dst = append(dst, " lines, "...)
	dst = strconv.AppendInt(dst, int64(len(b)), 10)
	dst = append(dst, " bytes; head: "...)
	dst = appendEscapedJSONBytes(dst, firstLine)
	return dst
}

// FastSummary produces a compact summary string for b.
func FastSummary(b []byte) string {
	var buf [96]byte
	res := AppendFastSummary(buf[:0], b)
	return string(res)
}

// IsTombstone checks whether b represents a structured tombstone reference.
func IsTombstone(b []byte) bool {
	return bytes.Contains(b, []byte(`"_tombstone":true`)) || bytes.Contains(b, []byte(`"_tombstone": true`))
}

type tombstoneJSON struct {
	Tombstone bool   `json:"_tombstone"`
	Ref       string `json:"ref"`
	Tool      string `json:"tool"`
	Len       int    `json:"len"`
	Tokens    int    `json:"tokens"`
	Summary   string `json:"summary"`
}

// ParseTombstone parses a structured tombstone reference from JSON.
func ParseTombstone(b []byte) (Tombstone, bool) {
	if !IsTombstone(b) {
		return Tombstone{}, false
	}
	var raw tombstoneJSON
	if err := json.Unmarshal(b, &raw); err != nil || !raw.Tombstone {
		return Tombstone{}, false
	}
	t := Tombstone{
		Active:         true,
		Ref:            raw.Ref,
		OriginalBytes:  raw.Len,
		OriginalTokens: raw.Tokens,
		Tool:           raw.Tool,
		Summary:        raw.Summary,
	}
	if len(raw.Ref) == 64 {
		if decoded, err := hex.DecodeString(raw.Ref); err == nil && len(decoded) == 32 {
			copy(t.Digest[:], decoded)
		}
	}
	return t, true
}

// EstimateTokens returns an estimate of tokens in b (standard ~4 bytes/token).
func EstimateTokens(b []byte) int {
	if len(b) == 0 {
		return 0
	}
	n := (len(b) + 3) / 4
	if n < 1 {
		return 1
	}
	return n
}

// SlidingWindow coordinates context growth, page allocation, and compaction in a multi-turn agent session.
type SlidingWindow struct {
	mu        sync.RWMutex
	compactor *Compactor
	freelist  *TokenPageFreelist
	pages     []TokenPage
	nextID    uint64
}

// NewSlidingWindow creates an active sliding context window buffer.
func NewSlidingWindow(c *Compactor) *SlidingWindow {
	if c == nil {
		c = NewCompactor(CompactorConfig{})
	}
	return &SlidingWindow{
		compactor: c,
		freelist:  c.freelist,
		pages:     make([]TokenPage, 0, 64),
	}
}

// AddPrefixSystem adds a pinned system prompt page to the session prefix.
func (sw *SlidingWindow) AddPrefixSystem(content []byte) {
	sw.mu.Lock()
	defer sw.mu.Unlock()
	sw.nextID++
	p := TokenPage{
		ID:        sw.nextID,
		TurnIndex: 0,
		Kind:      PageKindPrefixSystem,
		Role:      "system",
		Content:   append([]byte(nil), content...),
		Tokens:    EstimateTokens(content),
		Resident:  true,
		Pinned:    true,
	}
	sw.pages = append(sw.pages, p)
}

// AddPrefixTools adds a pinned tool catalog / schema definition page to the session prefix.
func (sw *SlidingWindow) AddPrefixTools(content []byte) {
	sw.mu.Lock()
	defer sw.mu.Unlock()
	sw.nextID++
	p := TokenPage{
		ID:        sw.nextID,
		TurnIndex: 0,
		Kind:      PageKindPrefixTools,
		Role:      "system",
		Content:   append([]byte(nil), content...),
		Tokens:    EstimateTokens(content),
		Resident:  true,
		Pinned:    true,
	}
	sw.pages = append(sw.pages, p)
}

// AddUserTurn appends a user prompt turn.
func (sw *SlidingWindow) AddUserTurn(turn int, content []byte) {
	sw.mu.Lock()
	defer sw.mu.Unlock()
	sw.nextID++
	p := TokenPage{
		ID:        sw.nextID,
		TurnIndex: turn,
		Kind:      PageKindUser,
		Role:      "user",
		Content:   append([]byte(nil), content...),
		Tokens:    EstimateTokens(content),
		Resident:  true,
	}
	sw.pages = append(sw.pages, p)
}

// AddAssistantTurn appends an assistant reasoning / conversational response.
func (sw *SlidingWindow) AddAssistantTurn(turn int, content []byte) {
	sw.mu.Lock()
	defer sw.mu.Unlock()
	sw.nextID++
	p := TokenPage{
		ID:        sw.nextID,
		TurnIndex: turn,
		Kind:      PageKindAssistant,
		Role:      "assistant",
		Content:   append([]byte(nil), content...),
		Tokens:    EstimateTokens(content),
		Resident:  true,
	}
	sw.pages = append(sw.pages, p)
}

// AddToolCall appends an assistant tool invocation.
func (sw *SlidingWindow) AddToolCall(turn int, tool string, args []byte) {
	sw.mu.Lock()
	defer sw.mu.Unlock()
	sw.nextID++
	p := TokenPage{
		ID:        sw.nextID,
		TurnIndex: turn,
		Kind:      PageKindToolCall,
		Role:      "assistant",
		ToolName:  tool,
		Content:   append([]byte(nil), args...),
		Tokens:    EstimateTokens(args),
		Resident:  true,
	}
	sw.pages = append(sw.pages, p)
}

// AddToolResult appends a tool execution payload.
func (sw *SlidingWindow) AddToolResult(turn int, tool string, result []byte) {
	sw.mu.Lock()
	defer sw.mu.Unlock()
	sw.nextID++
	p := TokenPage{
		ID:        sw.nextID,
		TurnIndex: turn,
		Kind:      PageKindToolResult,
		Role:      "tool",
		ToolName:  tool,
		Content:   append([]byte(nil), result...),
		Tokens:    EstimateTokens(result),
		Resident:  true,
	}
	sw.pages = append(sw.pages, p)
}

// AddToolResultPages appends a large tool execution payload split across multiple token pages.
func (sw *SlidingWindow) AddToolResultPages(turn int, tool string, chunks [][]byte) {
	sw.mu.Lock()
	defer sw.mu.Unlock()
	for i, chunk := range chunks {
		sw.nextID++
		p := TokenPage{
			ID:             sw.nextID,
			TurnIndex:      turn,
			Kind:           PageKindToolResult,
			Role:           "tool",
			ToolName:       tool,
			Content:        append([]byte(nil), chunk...),
			Tokens:         EstimateTokens(chunk),
			Resident:       true,
			IsContinuation: i > 0,
		}
		sw.pages = append(sw.pages, p)
	}
}

// Compact runs the sliding-window compaction pass on the session window.
func (sw *SlidingWindow) Compact() (CompactionReport, error) {
	sw.mu.Lock()
	defer sw.mu.Unlock()
	var report CompactionReport
	compacted, err := sw.compactor.CompactInPlace(sw.pages, &report)
	if err == nil {
		sw.pages = compacted
	}
	return report, err
}

// Scan runs a non-mutating scan on the session window.
func (sw *SlidingWindow) Scan() ScanReport {
	sw.mu.RLock()
	defer sw.mu.RUnlock()
	return sw.compactor.Scan(sw.pages)
}

// Pages returns a snapshot of the current pages.
func (sw *SlidingWindow) Pages() []TokenPage {
	sw.mu.RLock()
	defer sw.mu.RUnlock()
	out := make([]TokenPage, len(sw.pages))
	copy(out, sw.pages)
	return out
}

// TotalTokens returns the sum of tokens across all current pages.
func (sw *SlidingWindow) TotalTokens() int {
	sw.mu.RLock()
	defer sw.mu.RUnlock()
	tot := 0
	for i := range sw.pages {
		tot += sw.pages[i].Tokens
	}
	return tot
}

// PageCount returns the current number of pages.
func (sw *SlidingWindow) PageCount() int {
	sw.mu.RLock()
	defer sw.mu.RUnlock()
	return len(sw.pages)
}

// CompactPositive executes positive-state history compaction on the given turn trajectory.
func (c *Compactor) CompactPositive(turns []TurnRecord, originalGoal string) *PositiveCompactedHistory {
	return CompactPositiveState(turns, originalGoal)
}

// EpisodeTracker returns a semantic episode tracker backed by the provided CASStore, or a new
// CASStore if omitted or nil.
func (c *Compactor) EpisodeTracker(store ...*CASStore) *EpisodeTracker {
	var s *CASStore
	if len(store) > 0 {
		s = store[0]
	}
	return NewEpisodeTracker(s)
}

// CompactEpisodes delegates context page compaction to the provided EpisodeTracker.
// If tracker is nil, a new EpisodeTracker is created.
func (c *Compactor) CompactEpisodes(pages []TokenPage, tracker ...*EpisodeTracker) ([]TokenPage, CompactionReport, error) {
	var t *EpisodeTracker
	if len(tracker) > 0 && tracker[0] != nil {
		t = tracker[0]
	} else {
		t = NewEpisodeTracker(nil)
	}
	return t.CompactPages(pages)
}
