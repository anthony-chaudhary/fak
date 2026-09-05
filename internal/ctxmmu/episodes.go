package ctxmmu

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
)

// EpisodeType is the closed enum of semantic episode phases.
type EpisodeType string

const (
	EpisodeExplore  EpisodeType = "explore"
	EpisodeMutate   EpisodeType = "mutate"
	EpisodeVerify   EpisodeType = "verify"
	EpisodeRecovery EpisodeType = "recovery"
)

// IsValid reports whether the episode type is a recognized member of the closed enum.
func (e EpisodeType) IsValid() bool {
	switch e {
	case EpisodeExplore, EpisodeMutate, EpisodeVerify, EpisodeRecovery:
		return true
	default:
		return false
	}
}

// String returns the string representation of the episode type.
func (e EpisodeType) String() string {
	return string(e)
}

// EpisodeDigest is an immutable record summarizing a completed or in-progress semantic episode.
type EpisodeDigest struct {
	Type            EpisodeType `json:"type"`
	EpisodeID       string      `json:"episode_id,omitempty"`
	DiscoveredFiles []string    `json:"discovered_files,omitempty"`
	TargetLines     []string    `json:"target_lines,omitempty"`
	KeyErrors       []string    `json:"key_errors,omitempty"`
	ToolCallCount   int         `json:"tool_call_count"`
	TokenCount      int         `json:"token_count"`
	CompactedAt     time.Time   `json:"compacted_at"`
	Summary         string      `json:"summary,omitempty"`
}

// Files returns a defensive copy of DiscoveredFiles.
func (d EpisodeDigest) Files() []string {
	if len(d.DiscoveredFiles) == 0 {
		return nil
	}
	return append([]string(nil), d.DiscoveredFiles...)
}

// Targets returns a defensive copy of TargetLines.
func (d EpisodeDigest) Targets() []string {
	if len(d.TargetLines) == 0 {
		return nil
	}
	return append([]string(nil), d.TargetLines...)
}

// Errors returns a defensive copy of KeyErrors.
func (d EpisodeDigest) Errors() []string {
	if len(d.KeyErrors) == 0 {
		return nil
	}
	return append([]string(nil), d.KeyErrors...)
}

// String returns a compact representation of the episode digest.
func (d EpisodeDigest) String() string {
	return fmt.Sprintf("[Episode:%s calls=%d tokens=%d files=%d errors=%d]",
		d.Type, d.ToolCallCount, d.TokenCount, len(d.DiscoveredFiles), len(d.KeyErrors))
}

// CASStore provides thread-safe, Content-Addressed Storage for paged-out tool output chatter
// with tamper-evident SHA-256 validation.
type CASStore struct {
	mu      sync.RWMutex
	entries map[string][]byte
}

// NewCASStore allocates a new in-memory CASStore.
func NewCASStore() *CASStore {
	return &CASStore{
		entries: make(map[string][]byte),
	}
}

// isHex64 checks whether s consists of exactly 64 hexadecimal characters.
func isHex64(s string) bool {
	if len(s) != 64 {
		return false
	}
	for i := 0; i < 64; i++ {
		c := s[i]
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
			return false
		}
	}
	return true
}

// extractSHA256 parses a 64-hex SHA-256 digest from a CAS reference or raw hash string.
func extractSHA256(ref string) string {
	ref = strings.TrimSpace(ref)
	lower := strings.ToLower(ref)
	if idx := strings.Index(lower, "sha256:"); idx != -1 {
		rest := lower[idx+len("sha256:"):]
		if len(rest) >= 64 && isHex64(rest[:64]) {
			return rest[:64]
		}
	}
	if len(lower) == 64 && isHex64(lower) {
		return lower
	}
	// Search for any 64-hex substring
	for i := 0; i+64 <= len(lower); i++ {
		candidate := lower[i : i+64]
		if isHex64(candidate) {
			return candidate
		}
	}
	return ""
}

// countLines returns the number of lines in data.
func countLines(data []byte) int {
	if len(data) == 0 {
		return 0
	}
	n := bytes.Count(data, []byte{'\n'})
	if !bytes.HasSuffix(data, []byte{'\n'}) {
		n++
	}
	return n
}

// Put stores data into the CAS store keyed by its SHA-256 digest and returns a formatted CAS stub string.
// Format: `[CAS:sha256:<hash> <lines> lines paged out; summary: <summary>]`.
func (s *CASStore) Put(data []byte, summary string) (string, error) {
	if s == nil {
		return "", errors.New("ctxmmu: nil CAS store")
	}

	sum := sha256.Sum256(data)
	hashHex := hex.EncodeToString(sum[:])
	lines := countLines(data)

	cleanSummary := strings.TrimSpace(summary)
	if cleanSummary == "" {
		cleanSummary = FastSummary(data)
		if cleanSummary == "" {
			cleanSummary = "tool output"
		}
	}
	cleanSummary = strings.ReplaceAll(cleanSummary, "\n", " ")
	cleanSummary = strings.ReplaceAll(cleanSummary, "]", ")")

	stub := fmt.Sprintf("[CAS:sha256:%s %d lines paged out; summary: %s]", hashHex, lines, cleanSummary)

	s.mu.Lock()
	if _, exists := s.entries[hashHex]; !exists {
		cp := make([]byte, len(data))
		copy(cp, data)
		s.entries[hashHex] = cp
	}
	s.mu.Unlock()

	return stub, nil
}

// Get retrieves the original body bytes from the CAS store by reference, validating that
// the stored payload's SHA-256 digest matches the expected hash.
func (s *CASStore) Get(casRef string) ([]byte, bool, error) {
	if s == nil {
		return nil, false, errors.New("ctxmmu: nil CAS store")
	}

	hashHex := extractSHA256(casRef)
	if hashHex == "" {
		return nil, false, fmt.Errorf("ctxmmu: invalid CAS ref format %q", casRef)
	}

	s.mu.RLock()
	data, ok := s.entries[hashHex]
	s.mu.RUnlock()
	if !ok {
		return nil, false, nil
	}

	// Tamper-evident hash validation
	actualSum := sha256.Sum256(data)
	actualHex := hex.EncodeToString(actualSum[:])
	if actualHex != hashHex {
		return nil, false, fmt.Errorf("ctxmmu: CAS tamper detected for %s: hash mismatch (expected %s, got %s)", hashHex, hashHex, actualHex)
	}

	res := make([]byte, len(data))
	copy(res, data)
	return res, true, nil
}

// ResolveCAS retrieves the original body bytes for a CAS reference or returns an error if not found.
func (s *CASStore) ResolveCAS(ref string) ([]byte, error) {
	if s == nil {
		return nil, errors.New("ctxmmu: nil CAS store")
	}
	data, ok, err := s.Get(ref)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, fmt.Errorf("ctxmmu: ref %s not found in CAS store", ref)
	}
	return data, nil
}

// TamperEntryForTest corrupts a CAS entry to test tamper-evident hash validation.
func (s *CASStore) TamperEntryForTest(casRef string, corrupted []byte) error {
	if s == nil {
		return errors.New("ctxmmu: nil CAS store")
	}
	hashHex := extractSHA256(casRef)
	if hashHex == "" {
		return fmt.Errorf("ctxmmu: invalid CAS ref %q", casRef)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.entries[hashHex]; !ok {
		return fmt.Errorf("ctxmmu: ref %s not found", hashHex)
	}
	cp := make([]byte, len(corrupted))
	copy(cp, corrupted)
	s.entries[hashHex] = cp
	return nil
}

// Size returns the number of distinct blobs stored.
func (s *CASStore) Size() int {
	if s == nil {
		return 0
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.entries)
}

var defaultCASStore = NewCASStore()

// DefaultCASStore returns the package-level singleton CASStore.
func DefaultCASStore() *CASStore {
	return defaultCASStore
}

// ResolveCAS resolves a CAS reference using the provided store, or DefaultCASStore if omitted.
func ResolveCAS(ref string, store ...*CASStore) ([]byte, error) {
	s := defaultCASStore
	if len(store) > 0 && store[0] != nil {
		s = store[0]
	}
	return s.ResolveCAS(ref)
}

var casStubPattern = regexp.MustCompile(`(?i)\[CAS:sha256:([0-9a-fA-F]{64})[^\]]*\]`)

// HydrateCAS scans text for CAS stubs and replaces each known stub with the original content.
func HydrateCAS(text string, store *CASStore) string {
	if store == nil || text == "" {
		return text
	}
	return casStubPattern.ReplaceAllStringFunc(text, func(stub string) string {
		data, ok, err := store.Get(stub)
		if err == nil && ok {
			return string(data)
		}
		return stub
	})
}

// EpisodeTurnRecord captures the execution output and metadata of one tool or interaction turn.
type EpisodeTurnRecord struct {
	TurnIndex   int      `json:"turn_index"`
	ToolName    string   `json:"tool_name"`
	Input       string   `json:"input,omitempty"`
	Output      []byte   `json:"-"`
	Error       string   `json:"error,omitempty"`
	Tokens      int      `json:"tokens"`
	Files       []string `json:"files,omitempty"`
	TargetLines []string `json:"target_lines,omitempty"`
	CASStub     string   `json:"cas_stub,omitempty"`
	PagedOut    bool     `json:"paged_out"`
}

// EpisodeTracker is a state machine that tracks semantic episodes, records turn outputs,
// compiles immutable digests upon episode transitions, and pages out verbose tool chatter to CAS.
type EpisodeTracker struct {
	mu                   sync.RWMutex
	store                *CASStore
	currentEpisode       EpisodeType
	episodeIndex         int
	digests              []EpisodeDigest
	currentTurns         []EpisodeTurnRecord
	allTurns             []EpisodeTurnRecord
	discoveredFiles      map[string]struct{}
	targetLines          map[string]struct{}
	keyErrors            []string
	toolCallCount        int
	tokenCount           int
	totalInitialTokens   int
	totalCompactedTokens int
}

// NewEpisodeTracker creates an EpisodeTracker with the given CASStore. If store is nil, a new CASStore is created.
func NewEpisodeTracker(store *CASStore) *EpisodeTracker {
	if store == nil {
		store = NewCASStore()
	}
	return &EpisodeTracker{
		store:           store,
		currentEpisode:  EpisodeExplore,
		episodeIndex:    1,
		discoveredFiles: make(map[string]struct{}),
		targetLines:     make(map[string]struct{}),
	}
}

// CASStore returns the underlying CASStore.
func (et *EpisodeTracker) CASStore() *CASStore {
	if et == nil {
		return nil
	}
	et.mu.RLock()
	defer et.mu.RUnlock()
	return et.store
}

// CurrentEpisode returns the active episode type.
func (et *EpisodeTracker) CurrentEpisode() EpisodeType {
	if et == nil {
		return EpisodeExplore
	}
	et.mu.RLock()
	defer et.mu.RUnlock()
	return et.currentEpisode
}

// EpisodeIndex returns the 1-based index of the active episode.
func (et *EpisodeTracker) EpisodeIndex() int {
	if et == nil {
		return 0
	}
	et.mu.RLock()
	defer et.mu.RUnlock()
	return et.episodeIndex
}

// RecordTurn adds an EpisodeTurnRecord to the current episode, tracking discovered files, target lines,
// errors, and token estimations.
func (et *EpisodeTracker) RecordTurn(rec EpisodeTurnRecord) (EpisodeTurnRecord, error) {
	if et == nil {
		return rec, errors.New("ctxmmu: nil EpisodeTracker")
	}
	et.mu.Lock()
	defer et.mu.Unlock()

	if rec.Tokens <= 0 && len(rec.Output) > 0 {
		rec.Tokens = EstimateTokens(rec.Output)
	}

	for _, f := range rec.Files {
		clean := strings.TrimSpace(f)
		if clean != "" {
			et.discoveredFiles[clean] = struct{}{}
		}
	}
	for _, l := range rec.TargetLines {
		clean := strings.TrimSpace(l)
		if clean != "" {
			et.targetLines[clean] = struct{}{}
		}
	}
	if rec.Error != "" {
		et.keyErrors = append(et.keyErrors, rec.Error)
	}

	et.toolCallCount++
	et.tokenCount += rec.Tokens
	et.totalInitialTokens += rec.Tokens

	et.currentTurns = append(et.currentTurns, rec)
	return rec, nil
}

// RecordTurnOutput is a convenience helper to record a tool call output.
func (et *EpisodeTracker) RecordTurnOutput(toolName string, output []byte, err error, tokens int) (EpisodeTurnRecord, error) {
	var errStr string
	if err != nil {
		errStr = err.Error()
	}
	return et.RecordTurn(EpisodeTurnRecord{
		ToolName: toolName,
		Output:   output,
		Error:    errStr,
		Tokens:   tokens,
	})
}

// RecordTurnWithMetadata records a tool call output with explicit file and target line metadata.
func (et *EpisodeTracker) RecordTurnWithMetadata(toolName string, output []byte, files []string, targets []string, err error, tokens int) (EpisodeTurnRecord, error) {
	var errStr string
	if err != nil {
		errStr = err.Error()
	}
	return et.RecordTurn(EpisodeTurnRecord{
		ToolName:    toolName,
		Output:      output,
		Error:       errStr,
		Tokens:      tokens,
		Files:       files,
		TargetLines: targets,
	})
}

// RecordPositiveTurn bridges TurnRecord from positive_compactor into EpisodeTracker.
func (et *EpisodeTracker) RecordPositiveTurn(t TurnRecord) (EpisodeTurnRecord, error) {
	var errStr string
	if t.IsFailure {
		errStr = "failure"
	}
	return et.RecordTurn(EpisodeTurnRecord{
		ToolName: t.ToolCallName,
		Input:    t.ToolCallArgs,
		Output:   []byte(t.Content),
		Error:    errStr,
	})
}

// Transition advances the state machine to next EpisodeType, compiling an immutable EpisodeDigest
// for the outgoing episode and paging out verbose tool outputs to CAS stubs.
func (et *EpisodeTracker) Transition(next EpisodeType) (EpisodeDigest, error) {
	if et == nil {
		return EpisodeDigest{}, errors.New("ctxmmu: nil EpisodeTracker")
	}
	if !next.IsValid() {
		return EpisodeDigest{}, fmt.Errorf("ctxmmu: invalid episode type %q", next)
	}

	et.mu.Lock()
	defer et.mu.Unlock()

	// Compile digest for the outgoing episode
	files := make([]string, 0, len(et.discoveredFiles))
	for f := range et.discoveredFiles {
		files = append(files, f)
	}
	sort.Strings(files)

	targets := make([]string, 0, len(et.targetLines))
	for t := range et.targetLines {
		targets = append(targets, t)
	}
	sort.Strings(targets)

	dedupErrors := dedupeStrings(et.keyErrors)

	// Page out verbose tool outputs to CAS stubs
	pagedOutTurnTokens := 0
	for i := range et.currentTurns {
		turn := &et.currentTurns[i]
		if len(turn.Output) > 0 && !turn.PagedOut {
			summary := turn.ToolName
			if summary == "" {
				summary = FastSummary(turn.Output)
			}
			stub, err := et.store.Put(turn.Output, summary)
			if err == nil {
				turn.CASStub = stub
				turn.PagedOut = true
				stubToks := EstimateTokens([]byte(stub))
				pagedOutTurnTokens += stubToks
			} else {
				pagedOutTurnTokens += turn.Tokens
			}
		} else {
			pagedOutTurnTokens += turn.Tokens
		}
	}
	et.totalCompactedTokens += pagedOutTurnTokens

	digest := EpisodeDigest{
		Type:            et.currentEpisode,
		EpisodeID:       fmt.Sprintf("ep-%d-%s", et.episodeIndex, et.currentEpisode),
		DiscoveredFiles: files,
		TargetLines:     targets,
		KeyErrors:       dedupErrors,
		ToolCallCount:   et.toolCallCount,
		TokenCount:      et.tokenCount,
		CompactedAt:     time.Now().UTC(),
		Summary: fmt.Sprintf("completed %s with %d calls, %d files, %d errors",
			et.currentEpisode, et.toolCallCount, len(files), len(dedupErrors)),
	}

	et.digests = append(et.digests, digest)
	et.allTurns = append(et.allTurns, et.currentTurns...)

	// Reset for new episode
	et.discoveredFiles = make(map[string]struct{})
	et.targetLines = make(map[string]struct{})
	et.keyErrors = nil
	et.toolCallCount = 0
	et.tokenCount = 0
	et.currentTurns = nil
	et.episodeIndex++
	et.currentEpisode = next

	return digest, nil
}

// CurrentDigest returns a snapshot digest of the active, in-flight episode.
func (et *EpisodeTracker) CurrentDigest() EpisodeDigest {
	if et == nil {
		return EpisodeDigest{}
	}
	et.mu.RLock()
	defer et.mu.RUnlock()

	files := make([]string, 0, len(et.discoveredFiles))
	for f := range et.discoveredFiles {
		files = append(files, f)
	}
	sort.Strings(files)

	targets := make([]string, 0, len(et.targetLines))
	for t := range et.targetLines {
		targets = append(targets, t)
	}
	sort.Strings(targets)

	dedupErrors := dedupeStrings(et.keyErrors)

	return EpisodeDigest{
		Type:            et.currentEpisode,
		EpisodeID:       fmt.Sprintf("ep-%d-%s", et.episodeIndex, et.currentEpisode),
		DiscoveredFiles: files,
		TargetLines:     targets,
		KeyErrors:       dedupErrors,
		ToolCallCount:   et.toolCallCount,
		TokenCount:      et.tokenCount,
		CompactedAt:     time.Now().UTC(),
		Summary: fmt.Sprintf("in-flight %s with %d calls",
			et.currentEpisode, et.toolCallCount),
	}
}

// Digests returns a defensive copy of all completed EpisodeDigests.
func (et *EpisodeTracker) Digests() []EpisodeDigest {
	if et == nil {
		return nil
	}
	et.mu.RLock()
	defer et.mu.RUnlock()
	if len(et.digests) == 0 {
		return nil
	}
	return append([]EpisodeDigest(nil), et.digests...)
}

// AllTurns returns a defensive copy of all historical turns recorded across completed episodes.
func (et *EpisodeTracker) AllTurns() []EpisodeTurnRecord {
	if et == nil {
		return nil
	}
	et.mu.RLock()
	defer et.mu.RUnlock()
	if len(et.allTurns) == 0 {
		return nil
	}
	return append([]EpisodeTurnRecord(nil), et.allTurns...)
}

// HydrateCAS replaces any CAS stubs in text with their original bodies from the tracker's CASStore.
func (et *EpisodeTracker) HydrateCAS(text string) string {
	if et == nil {
		return text
	}
	et.mu.RLock()
	store := et.store
	et.mu.RUnlock()
	return HydrateCAS(text, store)
}

// ResolveCAS resolves a CAS reference against the tracker's CASStore.
func (et *EpisodeTracker) ResolveCAS(ref string) ([]byte, error) {
	if et == nil {
		return nil, errors.New("ctxmmu: nil EpisodeTracker")
	}
	et.mu.RLock()
	store := et.store
	et.mu.RUnlock()
	return store.ResolveCAS(ref)
}

// CompactPages compacts a slice of TokenPage by paging out verbose middle-turn tool outputs
// into CAS stubs while strictly preserving prefix warm pages.
func (et *EpisodeTracker) CompactPages(pages []TokenPage) ([]TokenPage, CompactionReport, error) {
	if et == nil {
		return pages, CompactionReport{}, errors.New("ctxmmu: nil EpisodeTracker")
	}
	et.mu.Lock()
	defer et.mu.Unlock()

	var report CompactionReport
	if len(pages) == 0 {
		report.PrefixWarm = true
		report.NoOp = true
		return pages, report, nil
	}

	for i := range pages {
		tok := pages[i].Tokens
		if tok <= 0 {
			tok = EstimateTokens(pages[i].Content)
			pages[i].Tokens = tok
		}
		report.BeforeTokens += tok
		report.BeforeBytes += len(pages[i].Content)
	}

	result := make([]TokenPage, len(pages))
	for i := range pages {
		p := pages[i]
		// Prefix pages (system, tools) or turn 0 are strictly immutable
		if p.Kind.IsPrefix() || p.TurnIndex <= 0 {
			result[i] = p
			continue
		}

		// Verbose tool result pages are paged out to CAS
		if p.Kind == PageKindToolResult && len(p.Content) > 0 {
			origBytes := len(p.Content)
			origTokens := p.Tokens

			summary := p.ToolName
			if summary == "" {
				summary = FastSummary(p.Content)
			}

			stub, err := et.store.Put(p.Content, summary)
			if err == nil {
				sum := sha256.Sum256(p.Content)
				stubBytes := []byte(stub)
				stubTokens := EstimateTokens(stubBytes)

				p.Tombstone = Tombstone{
					Active:         true,
					Ref:            stub,
					Digest:         sum,
					OriginalBytes:  origBytes,
					OriginalTokens: origTokens,
					Tool:           p.ToolName,
					Summary:        summary,
				}
				p.Content = stubBytes
				p.Tokens = stubTokens

				report.TombstonesCreated++
				report.TokensReclaimed += (origTokens - stubTokens)
				report.BytesReclaimed += (origBytes - len(stubBytes))
			}
		}
		result[i] = p
	}

	for i := range result {
		report.AfterTokens += result[i].Tokens
		report.AfterBytes += len(result[i].Content)
	}
	report.PrefixWarm = VerifyPrefixWarmth(pages, result)

	return result, report, nil
}

// TokenStats returns the total initial and compacted token counts recorded across all episode transitions.
func (et *EpisodeTracker) TokenStats() (initial, compacted int, reduction float64) {
	if et == nil {
		return 0, 0, 0
	}
	et.mu.RLock()
	defer et.mu.RUnlock()

	initial = et.totalInitialTokens
	compacted = et.totalCompactedTokens
	if initial > 0 {
		reduction = float64(initial-compacted) / float64(initial)
	}
	return initial, compacted, reduction
}

func dedupeStrings(in []string) []string {
	if len(in) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(in))
	var out []string
	for _, s := range in {
		s = strings.TrimSpace(s)
		if s == "" {
			continue
		}
		if _, ok := seen[s]; !ok {
			seen[s] = struct{}{}
			out = append(out, s)
		}
	}
	sort.Strings(out)
	return out
}
