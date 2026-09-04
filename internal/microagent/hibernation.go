package microagent

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"
)

// HibernatedState captures the cold-storage serialized state of an inactive agent
// parked to disk or content-addressed storage (#11182).
type HibernatedState struct {
	ID       string            `json:"id"`
	Hash     string            `json:"hash"` // Content-addressed SHA-256 hex digest of Data
	Data     []byte            `json:"data"` // Serialized state bytes
	SavedAt  time.Time         `json:"saved_at"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

var (
	restoreMu    sync.RWMutex
	restoreStore = make(map[string][]byte)
)

// StoreRestoreContext caches dropped turn bytes under a content-addressed SHA-256 hash.
func StoreRestoreContext(id string, data []byte) {
	restoreMu.Lock()
	defer restoreMu.Unlock()
	restoreStore[id] = data
}

// RestoreContext retrieves previously compacted turn bytes by its content-addressed hash.
func RestoreContext(id string) ([]byte, bool) {
	restoreMu.RLock()
	defer restoreMu.RUnlock()
	data, ok := restoreStore[id]
	return data, ok
}

// recordColdState freezes an agent, calculates its content hash, and caches its HibernatedState.
func (b *WarmBand) recordColdState(id string, h Hibernable) (*HibernatedState, error) {
	if h == nil {
		return nil, ErrNilAgent
	}
	data, err := h.Freeze()
	if err != nil {
		return nil, fmt.Errorf("microagent: freeze %q for hibernation: %w", id, err)
	}
	hash := sha256.Sum256(data)
	hexHash := hex.EncodeToString(hash[:])
	st := &HibernatedState{
		ID:       id,
		Hash:     hexHash,
		Data:     data,
		SavedAt:  b.now(),
		Metadata: map[string]string{"source": "warmband_cold_storage"},
	}
	b.coldMu.Lock()
	if b.coldStates == nil {
		b.coldStates = make(map[string]*HibernatedState)
	}
	b.coldStates[id] = st
	b.coldMu.Unlock()
	return st, nil
}

// Hibernate explicitly moves an agent to cold storage, freeing its resident slot
// if currently held, and returning its HibernatedState (#11182).
func (b *WarmBand) Hibernate(id string) (*HibernatedState, error) {
	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		return nil, ErrWarmBandClosed
	}
	h, isHeld := b.held[id]
	if isHeld {
		delete(b.held, id)
	}
	b.mu.Unlock()

	if isHeld {
		st, err := b.recordColdState(id, h)
		if err != nil {
			b.releaseSlot()
			return nil, err
		}
		if _, parkErr := b.store.Park(id, h); parkErr != nil {
			b.releaseSlot()
			return nil, parkErr
		}
		b.mu.Lock()
		b.addParkedLocked(id)
		b.mu.Unlock()
		b.releaseSlot()
		b.kick()
		return st, nil
	}

	// Check if present in warm reserve
	if warmH, ok := b.reserve.Take(id); ok {
		st, err := b.recordColdState(id, warmH)
		if err != nil {
			return nil, err
		}
		if _, parkErr := b.store.Park(id, warmH); parkErr != nil {
			return nil, parkErr
		}
		b.mu.Lock()
		delete(b.warmAt, id)
		b.addParkedLocked(id)
		b.mu.Unlock()
		b.kick()
		return st, nil
	}

	// Check if already in cold storage cache
	b.coldMu.RLock()
	st, ok := b.coldStates[id]
	b.coldMu.RUnlock()
	if ok && st != nil {
		return st, nil
	}

	// Fallback: check on-disk store
	if b.store != nil && b.store.Parked(id) {
		p, err := b.store.path(id)
		if err == nil {
			data, err := os.ReadFile(p)
			if err == nil {
				hash := sha256.Sum256(data)
				hexHash := hex.EncodeToString(hash[:])
				st = &HibernatedState{
					ID:       id,
					Hash:     hexHash,
					Data:     data,
					SavedAt:  b.now(),
					Metadata: map[string]string{"source": "disk_store"},
				}
				b.coldMu.Lock()
				if b.coldStates == nil {
					b.coldStates = make(map[string]*HibernatedState)
				}
				b.coldStates[id] = st
				b.coldMu.Unlock()
				return st, nil
			}
		}
	}

	return nil, fmt.Errorf("microagent: hibernate %q: %w", id, ErrNotEnrolled)
}

// PreThaw pre-thaws a parked agent from cold storage into the warm reserve before
// its execution turn, ensuring its next Acquire is served at zero cold Thaw (#11182).
func (b *WarmBand) PreThaw(id string) error {
	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		return ErrWarmBandClosed
	}
	if _, held := b.held[id]; held {
		b.mu.Unlock()
		return nil
	}
	if _, warm := b.warmAt[id]; warm {
		b.mu.Unlock()
		return nil
	}
	if !b.onDisk[id] {
		b.mu.Unlock()
		return fmt.Errorf("microagent: pre-thaw %q: not parked", id)
	}
	blank := b.blanks[id]
	if blank == nil {
		b.mu.Unlock()
		return ErrNotEnrolled
	}
	b.warming[id] = true
	b.dropParkedLocked(id)
	b.mu.Unlock()

	h := blank()
	if h == nil {
		b.mu.Lock()
		delete(b.warming, id)
		b.addParkedLocked(id)
		b.mu.Unlock()
		return ErrNilBlank
	}

	err := b.store.Wake(id, h)
	b.mu.Lock()
	delete(b.warming, id)
	if err != nil {
		b.addParkedLocked(id)
		b.broadcastLocked()
		b.mu.Unlock()
		return err
	}

	if b.reserve.Reserve(id, h) {
		b.warmAt[id] = b.now()
		b.refills++
		b.broadcastLocked()
		b.mu.Unlock()
		return nil
	}

	// Reserve full; park back to store
	b.mu.Unlock()
	if _, parkErr := b.store.Park(id, h); parkErr != nil {
		return parkErr
	}
	b.mu.Lock()
	b.addParkedLocked(id)
	b.mu.Unlock()
	return nil
}

// HibernatedState returns the cold storage state for agent id.
func (b *WarmBand) HibernatedState(id string) (*HibernatedState, error) {
	b.coldMu.RLock()
	st, ok := b.coldStates[id]
	b.coldMu.RUnlock()
	if ok && st != nil {
		return st, nil
	}
	if b.store != nil && b.store.Parked(id) {
		p, err := b.store.path(id)
		if err == nil {
			data, err := os.ReadFile(p)
			if err == nil {
				hash := sha256.Sum256(data)
				hexHash := hex.EncodeToString(hash[:])
				st = &HibernatedState{
					ID:       id,
					Hash:     hexHash,
					Data:     data,
					SavedAt:  b.now(),
					Metadata: map[string]string{"source": "disk_store"},
				}
				b.coldMu.Lock()
				if b.coldStates == nil {
					b.coldStates = make(map[string]*HibernatedState)
				}
				b.coldStates[id] = st
				b.coldMu.Unlock()
				return st, nil
			}
		}
	}
	return nil, ErrNotHibernated
}

// ColdStates returns all HibernatedState records currently in cold storage.
func (b *WarmBand) ColdStates() []*HibernatedState {
	b.coldMu.RLock()
	defer b.coldMu.RUnlock()
	res := make([]*HibernatedState, 0, len(b.coldStates))
	for _, st := range b.coldStates {
		if st != nil {
			res = append(res, st)
		}
	}
	return res
}

// CheckWatermarks evaluates residency against High and Low watermarks:
// - At or above High: triggers hibernation of inactive agent state to cold storage (HibernatedState).
// - At or below Low: triggers pre-thawing of parked agents into warm reserve before execution turns.
func (b *WarmBand) CheckWatermarks() (hibernated int, prethawed int, err error) {
	b.mu.Lock()
	res := b.rc.Resident()
	high := b.rc.Limit()
	low := b.rc.LowWater()
	b.mu.Unlock()

	// High watermark check: hibernate inactive/warm reserve agents
	if high > 0 && res >= high {
		b.mu.Lock()
		warmIDs := make([]string, 0, len(b.warmAt))
		for wid := range b.warmAt {
			warmIDs = append(warmIDs, wid)
		}
		b.mu.Unlock()

		for _, wid := range warmIDs {
			if st, err := b.Hibernate(wid); err == nil && st != nil {
				hibernated++
			}
		}
	}

	// Low watermark check: pre-thaw parked agents
	if low > 0 && res <= low {
		b.mu.Lock()
		avail := len(b.parked)
		b.mu.Unlock()

		want := high - res
		if room := b.reserve.Cap() - b.reserve.Len(); want > room {
			want = room
		}
		if want > avail {
			want = avail
		}

		for i := 0; i < want; i++ {
			b.mu.Lock()
			var nextID string
			for _, pid := range b.parked {
				if !b.warming[pid] {
					nextID = pid
					break
				}
			}
			b.mu.Unlock()

			if nextID == "" {
				break
			}
			if err := b.PreThaw(nextID); err == nil {
				prethawed++
			}
		}
	}

	return hibernated, prethawed, nil
}

// WarmBandGovernor monitors a WarmBand and enforces two-watermark residency policies:
// hibernating inactive agent state to cold storage when residency hits the high-water mark,
// and pre-thawing parked agents before execution turns when residency hits the low-water mark.
type WarmBandGovernor struct {
	band *WarmBand
	low  int
	high int
}

// NewWarmBandGovernor creates a new governor over band.
func NewWarmBandGovernor(band *WarmBand) *WarmBandGovernor {
	low := 0
	high := 0
	if band != nil && band.rc != nil {
		low = band.rc.LowWater()
		high = band.rc.Limit()
	}
	return &WarmBandGovernor{
		band: band,
		low:  low,
		high: high,
	}
}

// CheckWatermarks evaluates residency watermarks on the governed band.
func (g *WarmBandGovernor) CheckWatermarks() (int, int, error) {
	if g.band == nil {
		return 0, 0, nil
	}
	return g.band.CheckWatermarks()
}

// Hibernate hibernates the specified agent on the governed band.
func (g *WarmBandGovernor) Hibernate(id string) (*HibernatedState, error) {
	if g.band == nil {
		return nil, ErrNotEnrolled
	}
	return g.band.Hibernate(id)
}

// PreThaw pre-thaws the specified agent on the governed band.
func (g *WarmBandGovernor) PreThaw(id string) error {
	if g.band == nil {
		return ErrNotEnrolled
	}
	return g.band.PreThaw(id)
}

// CompactAnthropicHistory preserves initial system/tool blocks with ephemeral cache control,
// pins active instructions tagged with [fak:goal], and compacts intermediate turns into restore
// stubs carrying content-addressed hashes (id=<sha256>) (#11182).
func CompactAnthropicHistory(raw []byte, budget int) []byte {
	if budget <= 0 || len(raw) == 0 {
		return raw
	}

	var doc map[string]json.RawMessage
	if err := json.Unmarshal(raw, &doc); err != nil {
		return raw
	}

	msgsRaw, ok := doc["messages"]
	if !ok {
		return raw
	}

	var msgs []json.RawMessage
	if err := json.Unmarshal(msgsRaw, &msgs); err != nil || len(msgs) < 3 {
		return raw
	}

	totalTokens := 0
	for _, m := range msgs {
		totalTokens += len(m) / 4
		if totalTokens < 1 {
			totalTokens = 1
		}
	}
	if totalTokens <= budget {
		return raw
	}

	// 1. Identify prefix: messages up to and including the first breakpoint,
	// or initial user-assistant turn pair.
	pfxEnd := 0
	for i, m := range msgs {
		if bytes.Contains(m, []byte("cache_control")) {
			pfxEnd = i
			break
		}
	}
	if pfxEnd == 0 && len(msgs) > 1 {
		pfxEnd = 1
	}

	// 2. Identify pinned instructions tagged with [fak:goal]
	pinnedIndices := make(map[int]bool)
	for i, m := range msgs {
		if bytes.Contains(m, []byte("[fak:goal]")) {
			pinnedIndices[i] = true
		}
	}

	// 3. Kept recent window (last message)
	keepStart := len(msgs) - 1
	if keepStart <= pfxEnd+1 {
		return raw
	}

	// 4. Intermediate messages to drop
	var droppedMsgs []json.RawMessage
	for i := pfxEnd + 1; i < keepStart; i++ {
		if pinnedIndices[i] {
			continue
		}
		droppedMsgs = append(droppedMsgs, msgs[i])
	}
	if len(droppedMsgs) == 0 {
		return raw
	}

	// 5. Content-addressed hash of dropped intermediate turns
	var droppedBuf bytes.Buffer
	for _, dm := range droppedMsgs {
		droppedBuf.Write(dm)
	}
	hash := sha256.Sum256(droppedBuf.Bytes())
	hashHex := hex.EncodeToString(hash[:])
	StoreRestoreContext(hashHex, droppedBuf.Bytes())

	// 6. Build restore stub message
	pfxLastRole := extractRole(msgs[pfxEnd])
	stubRole := "user"
	if pfxLastRole == "user" {
		stubRole = "assistant"
	}
	stubContent := fmt.Sprintf("[fak] compacted %d earlier turn(s). Detail is omitted to preserve context budget.\n[fak] originating task (compacted): id=%s [fak:restore id=%s]", len(droppedMsgs), hashHex, hashHex)
	stubJSON := fmt.Sprintf(`{"role":%q,"content":%q}`, stubRole, stubContent)

	// 7. Assemble compacted message list
	var newMsgs []json.RawMessage
	for i := 0; i <= pfxEnd; i++ {
		newMsgs = append(newMsgs, msgs[i])
	}
	newMsgs = append(newMsgs, json.RawMessage(stubJSON))
	lastRole := stubRole

	// Add pinned goals
	for i := pfxEnd + 1; i < keepStart; i++ {
		if pinnedIndices[i] {
			goalRole := extractRole(msgs[i])
			if goalRole == lastRole {
				bridgeRole := "assistant"
				if lastRole == "assistant" {
					bridgeRole = "user"
				}
				newMsgs = append(newMsgs, json.RawMessage(fmt.Sprintf(`{"role":%q,"content":"Acknowledged."}`, bridgeRole)))
				lastRole = bridgeRole
			}
			newMsgs = append(newMsgs, msgs[i])
			lastRole = goalRole
		}
	}

	// Add recent window
	for i := keepStart; i < len(msgs); i++ {
		rRole := extractRole(msgs[i])
		if rRole == lastRole {
			bridgeRole := "assistant"
			if lastRole == "assistant" {
				bridgeRole = "user"
			}
			newMsgs = append(newMsgs, json.RawMessage(fmt.Sprintf(`{"role":%q,"content":"Acknowledged."}`, bridgeRole)))
			lastRole = bridgeRole
		}
		newMsgs = append(newMsgs, msgs[i])
		lastRole = rRole
	}

	newMsgsRaw, err := json.Marshal(newMsgs)
	if err != nil {
		return raw
	}
	doc["messages"] = newMsgsRaw

	out, err := json.Marshal(doc)
	if err != nil {
		return raw
	}
	return out
}

func extractRole(msg json.RawMessage) string {
	var obj struct {
		Role string `json:"role"`
	}
	if err := json.Unmarshal(msg, &obj); err == nil && obj.Role != "" {
		return obj.Role
	}
	return "user"
}

// ExtractRestoreHandles parses a compacted Anthropic body and returns all content-addressed
// restore handles (id=<sha256>) found in restore stubs.
func ExtractRestoreHandles(compacted []byte) []string {
	var handles []string
	str := string(compacted)
	idx := 0
	for {
		pos := strings.Index(str[idx:], "id=")
		if pos == -1 {
			break
		}
		start := idx + pos + 3
		if start+64 <= len(str) {
			candidate := str[start : start+64]
			if isHex(candidate) {
				handles = append(handles, candidate)
				idx = start + 64
				continue
			}
		}
		idx = start
	}
	return handles
}

func isHex(s string) bool {
	if len(s) != 64 {
		return false
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
			return false
		}
	}
	return true
}
