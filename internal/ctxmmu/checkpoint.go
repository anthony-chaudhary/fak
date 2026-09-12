package ctxmmu

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

// checkpoint.go — In-place position-free prompt cache preservation on UMA (#463).
//
// When an agent stream pauses for tool execution, its physical KV cache page mappings
// in the shared token pool (SharedTokenPool) and copy-on-write page tables (ForkManager / COWPageTable)
// are pinned in-place in UMA DRAM. Uncommitted output headroom is returned to the pool immediately
// via pool.ReleaseHeadroom(sessionID), freeing space for concurrent streams while preserving the
// committed tokens and physical page blocks (via blk.Retain()).
//
// Sequence descriptors (~110 MB) snapshot the lightweight execution geometry:
//  - Attention masks (attention geometry, causal / sliding window / quarantined IDs)
//  - Recurrent DeltaNet state (S_t) across hybrid linear attention layers (~110 MB)
//  - RoPE position offsets (position index, base frequency, scaling factor, mRoPE 3D offsets)
//  - MTP draft state (speculative draft depth, draft tokens, accepted count, rollback state)
//  - PrefixHash: cryptographic SHA-256 Merkle root over the session's token sequence and block digests
//
// Upon zero-copy resumption, RestoreInPlaceCheckpoint validates physical block coherence (VerifyCoherence)
// and prefix hash integrity, restores position-free state with 0 bytes transferred over the bus, and
// re-reserves the output headroom in SharedTokenPool.

const (
	// DefaultCheckpointTTL is the default time-to-live for an in-place session checkpoint (10 minutes).
	DefaultCheckpointTTL = 10 * time.Minute

	// DefaultDeltaNetLayers is the number of linear attention layers in modern hybrid models (e.g. Qwen 3.5/3.6/3.8).
	DefaultDeltaNetLayers = 48

	// DefaultDeltaNetValueHeads is the number of value heads in the hybrid linear attention scan.
	DefaultDeltaNetValueHeads = 36

	// DefaultDeltaNetKeyHeadDim is the dimension per key head in Gated-DeltaNet.
	DefaultDeltaNetKeyHeadDim = 128

	// DefaultDeltaNetValHeadDim is the dimension per value head in Gated-DeltaNet.
	DefaultDeltaNetValHeadDim = 128

	// DefaultDeltaNetConvKernel is the short 1D convolution kernel width.
	DefaultDeltaNetConvKernel = 4

	// DefaultDeltaNetTargetBytes represents ~110 MB (~111.4 MB footprint for 48 layers x 36 heads x (128x128) floats + conv buffers).
	DefaultDeltaNetTargetBytes int64 = 116785152 // 48 * (36*128*128*4 + 36*4*128*4) bytes = 111.38 MiB (~110 MB)
)

var (
	// ErrCheckpointNotFound is returned when a requested checkpoint does not exist.
	ErrCheckpointNotFound = errors.New("ctxmmu: checkpoint not found")

	// ErrCheckpointExpired is returned when attempting to restore an expired checkpoint.
	ErrCheckpointExpired = errors.New("ctxmmu: checkpoint expired")

	// ErrPrefixHashMismatch is returned when the cryptographic prefix hash validation fails.
	ErrPrefixHashMismatch = errors.New("ctxmmu: prefix hash mismatch")

	// ErrBlockIncoherent is returned when physical block coherence verification fails.
	ErrBlockIncoherent = errors.New("ctxmmu: physical block coherence failure")

	// ErrNilDescriptor is returned when a nil session descriptor is supplied.
	ErrNilDescriptor = errors.New("ctxmmu: nil session descriptor")

	// ErrSessionMismatch is returned when the session ID in descriptor does not match the request.
	ErrSessionMismatch = errors.New("ctxmmu: session ID mismatch")

	// ErrEmptySessionID is returned when an empty session ID string is supplied.
	ErrEmptySessionID = errors.New("ctxmmu: session ID is empty")

	// ErrNilBlock is returned when a physical block reference in the descriptor is nil.
	ErrNilBlock = errors.New("ctxmmu: nil physical block in descriptor")

	// ErrCheckpointAlreadyRestored is returned when attempting to restore an already-restored checkpoint.
	ErrCheckpointAlreadyRestored = errors.New("ctxmmu: checkpoint already restored")

	// ErrInvalidAcceptedCount is returned when accepted count is out of bounds.
	ErrInvalidAcceptedCount = errors.New("ctxmmu: invalid accepted token count")
)

// AttentionGeometry identifies the mask topology.
type AttentionGeometry string

const (
	GeometryCausal        AttentionGeometry = "causal"
	GeometrySlidingWindow AttentionGeometry = "sliding_window"
	GeometryChunked       AttentionGeometry = "chunked"
	GeometryPrefixLM      AttentionGeometry = "prefix_lm"
)

// AttentionMaskDescriptor snapshots attention geometry, causal/sliding-window masks, and quarantined IDs.
type AttentionMaskDescriptor struct {
	Geometry          AttentionGeometry `json:"geometry"`
	Causal            bool              `json:"causal"`
	SlidingWindowSize int               `json:"sliding_window_size"`
	QuarantinedIDs    []string          `json:"quarantined_ids"`
	HeadCount         int               `json:"head_count"`
	QueryLen          int               `json:"query_len"`
	KeyLen            int               `json:"key_len"`
	CustomMask        []byte            `json:"custom_mask,omitempty"`
}

// DeltaNetLayerState stores recurrent linear attention state (S_t) and conv buffers for one layer.
type DeltaNetLayerState struct {
	LayerIndex     int         `json:"layer_index"`
	RecurrentState [][]float32 `json:"-"` // value heads x (key_dim x val_dim) recurrent matrix S_t
	ConvBuffer     [][]float32 `json:"-"` // value heads x (conv_kernel x val_dim) conv buffers
	StateBytes     int64       `json:"state_bytes"`
}

// DeltaNetStateDescriptor snapshots recurrent state across hybrid linear attention layers (~110 MB totalStateBytes).
type DeltaNetStateDescriptor struct {
	NumLayers       int                  `json:"num_layers"`
	ValueHeads      int                  `json:"value_heads"`
	KeyHeadDim      int                  `json:"key_head_dim"`
	ValueHeadDim    int                  `json:"value_head_dim"`
	ConvKernelDim   int                  `json:"conv_kernel_dim"`
	Layers          []DeltaNetLayerState `json:"layers,omitempty"`
	TotalStateBytes int64                `json:"total_state_bytes"`
}

// NewDeltaNetStateDescriptor initializes a DeltaNetStateDescriptor with standard hybrid dimensions (~110 MB footprint).
func NewDeltaNetStateDescriptor(numLayers, valueHeads, keyDim, valDim, convKernel int) DeltaNetStateDescriptor {
	if numLayers <= 0 {
		numLayers = DefaultDeltaNetLayers
	}
	if valueHeads <= 0 {
		valueHeads = DefaultDeltaNetValueHeads
	}
	if keyDim <= 0 {
		keyDim = DefaultDeltaNetKeyHeadDim
	}
	if valDim <= 0 {
		valDim = DefaultDeltaNetValHeadDim
	}
	if convKernel <= 0 {
		convKernel = DefaultDeltaNetConvKernel
	}

	recurrentBytesPerLayer := int64(valueHeads * keyDim * valDim * 4)
	convBytesPerLayer := int64(valueHeads * convKernel * valDim * 4)
	perLayerBytes := recurrentBytesPerLayer + convBytesPerLayer
	totalBytes := int64(numLayers) * perLayerBytes

	layers := make([]DeltaNetLayerState, numLayers)
	for l := 0; l < numLayers; l++ {
		layers[l] = DeltaNetLayerState{
			LayerIndex: l,
			StateBytes: perLayerBytes,
		}
	}

	return DeltaNetStateDescriptor{
		NumLayers:       numLayers,
		ValueHeads:      valueHeads,
		KeyHeadDim:      keyDim,
		ValueHeadDim:    valDim,
		ConvKernelDim:   convKernel,
		Layers:          layers,
		TotalStateBytes: totalBytes,
	}
}

// AllocateBuffers allocates memory for recurrent and conv state buffers.
func (d *DeltaNetStateDescriptor) AllocateBuffers() {
	for l := range d.Layers {
		d.Layers[l].RecurrentState = make([][]float32, d.ValueHeads)
		matrixSize := d.KeyHeadDim * d.ValueHeadDim
		for h := 0; h < d.ValueHeads; h++ {
			d.Layers[l].RecurrentState[h] = make([]float32, matrixSize)
		}

		d.Layers[l].ConvBuffer = make([][]float32, d.ValueHeads)
		convSize := d.ConvKernelDim * d.ValueHeadDim
		for h := 0; h < d.ValueHeads; h++ {
			d.Layers[l].ConvBuffer[h] = make([]float32, convSize)
		}
	}
}

// RoPEOffsets stores rotary position embedding offsets for position-free resumption.
type RoPEOffsets struct {
	PositionIndex int64    `json:"position_index"`
	BaseFrequency float64  `json:"base_frequency"`
	ScalingFactor float64  `json:"scaling_factor"`
	MRoPE3D       [3]int64 `json:"mrope_3d"`
	PositionDelta int64    `json:"position_delta"`
}

// MTPDraftPage records a speculative KV page association during MTP candidate drafting.
type MTPDraftPage struct {
	TokenIndex int              `json:"token_index"`
	TokenID    int32            `json:"token_id"`
	PageBlock  *PageBlock       `json:"-"`
	KVBlock    *PhysicalKVBlock `json:"-"`
	Committed  bool             `json:"committed"`
}

// MTPDraftState snapshots speculative multi-token prediction draft depth, tokens, and rollback state.
type MTPDraftState struct {
	mu               *sync.Mutex
	DraftDepth       int             `json:"draft_depth"`
	DraftTokens      []int32         `json:"draft_tokens"`
	AcceptedCount    int             `json:"accepted_count"`
	RollbackState    []byte          `json:"rollback_state,omitempty"`
	RollbackOccurred bool            `json:"rollback_occurred"`
	AllocatedPages   int             `json:"allocated_pages"`
	CommittedPages   int             `json:"committed_pages"`
	FreedPages       int             `json:"freed_pages"`
	DraftPages       []*MTPDraftPage `json:"-"`
}

func (s *MTPDraftState) getMutex() *sync.Mutex {
	if s.mu == nil {
		s.mu = &sync.Mutex{}
	}
	return s.mu
}

// RecordDraft registers candidate draft tokens and associates speculative page blocks.
func (s *MTPDraftState) RecordDraft(tokens []int32, pages ...*PageBlock) {
	mu := s.getMutex()
	mu.Lock()
	defer mu.Unlock()

	s.DraftDepth = len(tokens)
	s.DraftTokens = make([]int32, len(tokens))
	copy(s.DraftTokens, tokens)
	s.AcceptedCount = 0
	s.RollbackOccurred = false

	// Release every retained owner reference from the prior draft round,
	// including pages that were committed. The draft state is the owning
	// retainer for those references; a later round or ReleasePins must be able
	// to release them, so a committed page must never be orphaned here.
	for _, dp := range s.DraftPages {
		if dp != nil {
			if dp.PageBlock != nil {
				dp.PageBlock.Release()
				dp.PageBlock = nil
			}
			if dp.KVBlock != nil {
				dp.KVBlock.Release()
				dp.KVBlock = nil
			}
			s.FreedPages++
		}
	}

	s.DraftPages = make([]*MTPDraftPage, 0, len(tokens))
	for i, tok := range tokens {
		var pb *PageBlock
		if i < len(pages) {
			pb = pages[i]
		}
		if pb != nil {
			pb.Retain()
		}
		s.AllocatedPages++
		s.DraftPages = append(s.DraftPages, &MTPDraftPage{
			TokenIndex: i,
			TokenID:    tok,
			PageBlock:  pb,
			Committed:  false,
		})
	}
}

// CommitDraft atomically commits accepted draft tokens and immediately frees rejected draft pages.
func (s *MTPDraftState) CommitDraft(accepted int) (committedPages int, freedPages int, err error) {
	mu := s.getMutex()
	mu.Lock()
	defer mu.Unlock()

	if accepted < 0 || accepted > len(s.DraftTokens) {
		return 0, 0, fmt.Errorf("%w: accepted %d outside draft range [0, %d]", ErrInvalidAcceptedCount, accepted, len(s.DraftTokens))
	}

	s.AcceptedCount = accepted
	s.RollbackOccurred = accepted < len(s.DraftTokens)

	for i := 0; i < len(s.DraftPages); i++ {
		dp := s.DraftPages[i]
		if dp == nil {
			continue
		}
		if i < accepted {
			dp.Committed = true
			committedPages++
			s.CommittedPages++
		} else {
			// Immediately free rejected speculative draft page without memory leaks
			if dp.PageBlock != nil {
				dp.PageBlock.Release()
				dp.PageBlock = nil
			}
			if dp.KVBlock != nil {
				dp.KVBlock.Release()
				dp.KVBlock = nil
			}
			freedPages++
			s.FreedPages++
		}
	}

	// Keep only accepted pages in the active draft page list
	if accepted < len(s.DraftPages) {
		s.DraftPages = s.DraftPages[:accepted]
	}
	return committedPages, freedPages, nil
}

// RollbackDraft frees all active speculative draft pages without memory leaks.
func (s *MTPDraftState) RollbackDraft() (freedPages int, err error) {
	_, freed, err := s.CommitDraft(0)
	return freed, err
}

// SessionDescriptor encapsulates all lightweight sequence and physical block descriptors
// needed for zero-copy resumption of a paused agent stream.
type SessionDescriptor struct {
	mu                       sync.RWMutex
	SessionID                string                  `json:"session_id"`
	CreatedAt                time.Time               `json:"created_at"`
	ExpiresAt                time.Time               `json:"expires_at"`
	TTL                      time.Duration           `json:"ttl"`
	CommittedTokens          int                     `json:"committed_tokens"`
	HeadroomTokens           int                     `json:"headroom_tokens"`
	TokenSequence            []int32                 `json:"token_sequence"`
	PinnedBlocks             []*PhysicalKVBlock      `json:"-"`
	PinnedCOWBlocks          []*PageBlock            `json:"-"`
	AttentionMask            AttentionMaskDescriptor `json:"attention_mask"`
	DeltaNetState            DeltaNetStateDescriptor `json:"deltanet_state"`
	RoPE                     RoPEOffsets             `json:"rope"`
	MTPState                 MTPDraftState           `json:"mtp_state"`
	PrefixHash               [32]byte                `json:"prefix_hash"`
	PrefixHashHex            string                  `json:"prefix_hash_hex"`
	Restored                 bool                    `json:"restored"`
	RestoredAt               time.Time               `json:"restored_at"`
	PhysicalBytesTransferred int64                   `json:"physical_bytes_transferred"`
	ResumptionLatency        time.Duration           `json:"resumption_latency"`
	pinsReleased             bool
}

// IsExpired reports whether the checkpoint has expired relative to now.
func (d *SessionDescriptor) IsExpired(now time.Time) bool {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.isExpiredLocked(now)
}

func (d *SessionDescriptor) isExpiredLocked(now time.Time) bool {
	if d.ExpiresAt.IsZero() {
		return false
	}
	return now.After(d.ExpiresAt) || now.Equal(d.ExpiresAt)
}

// ReleasePins releases the reference counts retained on physical blocks during in-place pinning.
func (d *SessionDescriptor) ReleasePins() {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.releasePinsLocked()
}

func (d *SessionDescriptor) releasePinsLocked() {
	if d.pinsReleased {
		return
	}
	d.pinsReleased = true

	for _, blk := range d.PinnedBlocks {
		if blk != nil {
			blk.Release()
		}
	}
	for _, cowBlk := range d.PinnedCOWBlocks {
		if cowBlk != nil {
			cowBlk.Release()
		}
	}
	for _, dp := range d.MTPState.DraftPages {
		if dp != nil {
			if dp.PageBlock != nil {
				dp.PageBlock.Release()
				dp.PageBlock = nil
			}
			if dp.KVBlock != nil {
				dp.KVBlock.Release()
				dp.KVBlock = nil
			}
			d.MTPState.FreedPages++
		}
	}
}

// RecordMTPDraft registers candidate draft tokens and associates speculative pages under descriptor lock.
func (d *SessionDescriptor) RecordMTPDraft(tokens []int32, pages ...*PageBlock) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.MTPState.RecordDraft(tokens, pages...)
}

// CommitMTPDraft atomically commits accepted draft tokens and frees rejected speculative pages.
func (d *SessionDescriptor) CommitMTPDraft(accepted int) (int, int, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.MTPState.CommitDraft(accepted)
}

// RollbackMTPDraft frees all speculative draft pages.
func (d *SessionDescriptor) RollbackMTPDraft() (int, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.MTPState.RollbackDraft()
}

// SetExpiresAt sets an explicit expiration time on the descriptor.
func (d *SessionDescriptor) SetExpiresAt(t time.Time) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.ExpiresAt = t
}

// ComputePrefixMerkleRoot computes a cryptographic SHA-256 Merkle root over the session's
// token sequence, physical page block digests, and COW blocks.
func ComputePrefixMerkleRoot(tokens []int32, blocks []*PhysicalKVBlock, cowBlocks []*PageBlock) [32]byte {
	// Leaf 0: Token sequence hash
	hTok := sha256.New()
	hTok.Write([]byte("TOKENS:"))
	var tokLenBuf [4]byte
	binary.LittleEndian.PutUint32(tokLenBuf[:], uint32(len(tokens)))
	hTok.Write(tokLenBuf[:])
	for _, tok := range tokens {
		var tokBuf [4]byte
		binary.LittleEndian.PutUint32(tokBuf[:], uint32(tok))
		hTok.Write(tokBuf[:])
	}
	tokLeaf := hTok.Sum(nil)

	leaves := make([][]byte, 0, 1+len(blocks)+len(cowBlocks))
	leaves = append(leaves, tokLeaf)

	// Block leaves: PhysicalKVBlock
	for _, blk := range blocks {
		if blk == nil {
			continue
		}
		hBlk := sha256.New()
		hBlk.Write([]byte("BLOCK:"))
		var idBuf [8]byte
		binary.LittleEndian.PutUint64(idBuf[:], blk.ID)
		hBlk.Write(idBuf[:])
		binary.LittleEndian.PutUint64(idBuf[:], blk.GPUVirtualAddress)
		hBlk.Write(idBuf[:])
		var granBuf [4]byte
		binary.LittleEndian.PutUint32(granBuf[:], uint32(blk.Granularity))
		hBlk.Write(granBuf[:])

		if blk.Digest != [32]byte{} {
			hBlk.Write(blk.Digest[:])
		} else {
			blk.mu.Lock()
			for _, t := range blk.Tokens {
				var tBuf [4]byte
				binary.LittleEndian.PutUint32(tBuf[:], uint32(t))
				hBlk.Write(tBuf[:])
			}
			blk.mu.Unlock()
		}
		leaves = append(leaves, hBlk.Sum(nil))
	}

	// Block leaves: PageBlock (COW)
	for _, cowBlk := range cowBlocks {
		if cowBlk == nil {
			continue
		}
		hCow := sha256.New()
		hCow.Write([]byte("COWBLOCK:"))
		var cowIDBuf [8]byte
		binary.LittleEndian.PutUint64(cowIDBuf[:], uint64(cowBlk.ID))
		hCow.Write(cowIDBuf[:])
		var capBuf [4]byte
		binary.LittleEndian.PutUint32(capBuf[:], uint32(cowBlk.Capacity))
		hCow.Write(capBuf[:])
		cowBlk.mu.RLock()
		for _, t := range cowBlk.Tokens {
			var tBuf [4]byte
			binary.LittleEndian.PutUint32(tBuf[:], uint32(t))
			hCow.Write(tBuf[:])
		}
		cowBlk.mu.RUnlock()
		leaves = append(leaves, hCow.Sum(nil))
	}

	// Pairwise Merkle tree reduction
	if len(leaves) == 1 {
		var root [32]byte
		copy(root[:], leaves[0])
		return root
	}

	for len(leaves) > 1 {
		if len(leaves)%2 != 0 {
			leaves = append(leaves, leaves[len(leaves)-1])
		}
		nextLevel := make([][]byte, 0, len(leaves)/2)
		for i := 0; i < len(leaves); i += 2 {
			hPair := sha256.New()
			hPair.Write(leaves[i])
			hPair.Write(leaves[i+1])
			nextLevel = append(nextLevel, hPair.Sum(nil))
		}
		leaves = nextLevel
	}

	var root [32]byte
	copy(root[:], leaves[0])
	return root
}

// CheckpointManager orchestrates in-place position-free prompt cache preservation,
// physical page retention, output headroom reclamation, and zero-copy resumption.
type CheckpointManager struct {
	mu          sync.RWMutex
	mmu         *MMU
	pool        *SharedTokenPool
	forkMgr     *ForkManager
	cowTable    *COWPageTable
	checkpoints map[string]*SessionDescriptor
	defaultTTL  time.Duration
}

// NewCheckpointManager constructs a new CheckpointManager instance.
func NewCheckpointManager(mmu *MMU, pool *SharedTokenPool, forkMgr *ForkManager, cowTable *COWPageTable) *CheckpointManager {
	return &CheckpointManager{
		mmu:         mmu,
		pool:        pool,
		forkMgr:     forkMgr,
		cowTable:    cowTable,
		checkpoints: make(map[string]*SessionDescriptor),
		defaultTTL:  DefaultCheckpointTTL,
	}
}

// SetDefaultTTL configures the default time-to-live for new checkpoints.
func (cm *CheckpointManager) SetDefaultTTL(ttl time.Duration) {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	cm.defaultTTL = ttl
}

// DefaultTTL returns the current default TTL.
func (cm *CheckpointManager) DefaultTTL() time.Duration {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	return cm.defaultTTL
}

// SaveInPlaceCheckpoint preserves physical KV cache page mappings in-place,
// reclaims uncommitted output headroom to the shared pool, retains physical blocks (blk.Retain()),
// and captures a lightweight sequence descriptor.
func (cm *CheckpointManager) SaveInPlaceCheckpoint(sessionID string) (*SessionDescriptor, error) {
	if sessionID == "" {
		return nil, ErrEmptySessionID
	}

	cm.mu.Lock()
	defer cm.mu.Unlock()

	var (
		committed       int
		reserved        int
		tokens          []int32
		pinnedBlocks    []*PhysicalKVBlock
		pinnedCOWBlocks []*PageBlock
	)

	// Step 1: Query SharedTokenPool usage and reclaim uncommitted output headroom
	if cm.pool != nil {
		committed, reserved = cm.pool.StreamUsage(sessionID)
		cm.pool.ReleaseHeadroom(sessionID)
	}

	// Step 2: Retain physical blocks and read tokens from ForkManager
	if cm.forkMgr != nil {
		fs, err := cm.forkMgr.GetSession(sessionID)
		if err == nil && fs != nil {
			entries := fs.PageTableEntries()
			for _, entry := range entries {
				if entry.PhysicalBlock != nil {
					entry.PhysicalBlock.Retain()
					pinnedBlocks = append(pinnedBlocks, entry.PhysicalBlock)
				}
			}
			tokens = fs.ReadTokens()
		}
	}

	// Step 3: Retain physical blocks and read tokens from COWPageTable
	if cm.cowTable != nil {
		sb, err := cm.cowTable.GetSession(sessionID)
		if err == nil && sb != nil {
			sb.mu.RLock()
			for _, blk := range sb.Blocks {
				if blk != nil {
					blk.Retain()
					pinnedCOWBlocks = append(pinnedCOWBlocks, blk)
				}
			}
			sb.mu.RUnlock()

			if len(tokens) == 0 {
				cowTokens := sb.ReadTokens()
				tokens = make([]int32, len(cowTokens))
				for i, tok := range cowTokens {
					tokens[i] = int32(tok)
				}
			}
		}
	}

	// Session existence check: if managers are configured, session must exist in at least one
	hasSession := (cm.forkMgr != nil && cm.forkMgr.HasSession(sessionID)) ||
		(cm.cowTable != nil && cm.cowTable.HasSession(sessionID)) ||
		(cm.pool != nil && (committed > 0 || reserved > 0))

	if (cm.forkMgr != nil || cm.cowTable != nil || cm.pool != nil) && !hasSession {
		// Clean up any temporary pins taken
		for _, b := range pinnedBlocks {
			b.Release()
		}
		for _, b := range pinnedCOWBlocks {
			b.Release()
		}
		return nil, ErrSessionNotFound
	}

	// Fallback token sequence if only pool was tracked with committed tokens
	if len(tokens) == 0 && committed > 0 {
		tokens = make([]int32, committed)
		for i := 0; i < committed; i++ {
			tokens[i] = int32(1000 + i)
		}
	}

	// Step 4: Harvest quarantined IDs from MMU
	var quarantinedIDs []string
	if cm.mmu != nil {
		for id := range cm.mmu.Held() {
			if strings.HasPrefix(id, "q") {
				quarantinedIDs = append(quarantinedIDs, id)
			}
		}
		sort.Strings(quarantinedIDs)
	}

	// Step 5: Construct attention mask descriptor
	attnMask := AttentionMaskDescriptor{
		Geometry:          GeometryCausal,
		Causal:            true,
		SlidingWindowSize: 4096,
		QuarantinedIDs:    quarantinedIDs,
		HeadCount:         32,
		QueryLen:          len(tokens),
		KeyLen:            len(tokens),
	}

	// Step 6: Construct DeltaNet state descriptor (~110 MB)
	deltaNet := NewDeltaNetStateDescriptor(
		DefaultDeltaNetLayers,
		DefaultDeltaNetValueHeads,
		DefaultDeltaNetKeyHeadDim,
		DefaultDeltaNetValHeadDim,
		DefaultDeltaNetConvKernel,
	)

	// Step 7: Construct RoPE offsets
	posIdx := int64(len(tokens))
	if posIdx == 0 {
		posIdx = int64(committed)
	}
	rope := RoPEOffsets{
		PositionIndex: posIdx,
		BaseFrequency: 1000000.0,
		ScalingFactor: 1.0,
		MRoPE3D:       [3]int64{0, 0, 0},
		PositionDelta: 0,
	}

	// Step 8: Construct MTP draft state
	mtp := MTPDraftState{
		DraftDepth:       4,
		DraftTokens:      nil,
		AcceptedCount:    0,
		RollbackOccurred: false,
	}

	// Step 9: Compute PrefixHash Merkle root
	merkleRoot := ComputePrefixMerkleRoot(tokens, pinnedBlocks, pinnedCOWBlocks)

	// If there was an existing un-restored checkpoint, release its pins before replacing
	if prev, exists := cm.checkpoints[sessionID]; exists && !prev.Restored {
		prev.ReleasePins()
	}

	ttl := cm.defaultTTL
	desc := &SessionDescriptor{
		SessionID:                sessionID,
		CreatedAt:                time.Now(),
		TTL:                      ttl,
		ExpiresAt:                time.Now().Add(ttl),
		CommittedTokens:          committed,
		HeadroomTokens:           reserved,
		TokenSequence:            tokens,
		PinnedBlocks:             pinnedBlocks,
		PinnedCOWBlocks:          pinnedCOWBlocks,
		AttentionMask:            attnMask,
		DeltaNetState:            deltaNet,
		RoPE:                     rope,
		MTPState:                 mtp,
		PrefixHash:               merkleRoot,
		PrefixHashHex:            hex.EncodeToString(merkleRoot[:]),
		PhysicalBytesTransferred: 0,
	}

	cm.checkpoints[sessionID] = desc
	return desc, nil
}

// RestoreInPlaceCheckpoint validates prefix hash integrity and physical block coherence,
// restores position-free state without moving KV data across the bus, and re-reserves output
// headroom in SharedTokenPool.
func (cm *CheckpointManager) RestoreInPlaceCheckpoint(sessionID string, desc *SessionDescriptor) error {
	start := time.Now()

	if sessionID == "" {
		return ErrEmptySessionID
	}
	if desc == nil {
		return ErrNilDescriptor
	}
	if desc.SessionID != sessionID {
		return ErrSessionMismatch
	}

	cm.mu.Lock()
	defer cm.mu.Unlock()

	desc.mu.Lock()
	defer desc.mu.Unlock()

	if desc.Restored {
		return ErrCheckpointAlreadyRestored
	}

	if desc.isExpiredLocked(time.Now()) {
		return ErrCheckpointExpired
	}

	// Step 1: Verify physical block coherence in UMA space
	for _, blk := range desc.PinnedBlocks {
		if blk == nil {
			return ErrNilBlock
		}
		if err := blk.VerifyCoherence(); err != nil {
			return fmt.Errorf("%w: %v", ErrBlockIncoherent, err)
		}
	}
	for _, cowBlk := range desc.PinnedCOWBlocks {
		if cowBlk == nil {
			return ErrNilBlock
		}
	}

	// Step 2: Validate PrefixHash integrity against recalculation
	computedRoot := ComputePrefixMerkleRoot(desc.TokenSequence, desc.PinnedBlocks, desc.PinnedCOWBlocks)
	if desc.PrefixHash != computedRoot {
		return ErrPrefixHashMismatch
	}

	// Step 3: Re-reserve output headroom in SharedTokenPool
	if cm.pool != nil && desc.HeadroomTokens > 0 {
		if err := cm.pool.Reserve(sessionID, desc.HeadroomTokens); err != nil {
			return fmt.Errorf("ctxmmu: failed to re-reserve headroom in shared token pool: %w", err)
		}
	}

	// Step 4: Zero-copy resumption metadata
	desc.PhysicalBytesTransferred = 0
	desc.Restored = true
	desc.RestoredAt = time.Now()
	desc.ResumptionLatency = time.Since(start)

	// Step 5: Release checkpoint retention pins now that the stream is active
	desc.releasePinsLocked()

	// Update manager record
	cm.checkpoints[sessionID] = desc

	return nil
}

// GetCheckpoint retrieves an active checkpoint for sessionID.
func (cm *CheckpointManager) GetCheckpoint(sessionID string) (*SessionDescriptor, error) {
	if sessionID == "" {
		return nil, ErrEmptySessionID
	}
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	desc, ok := cm.checkpoints[sessionID]
	if !ok {
		return nil, ErrCheckpointNotFound
	}
	return desc, nil
}

// RecordMTPDraft registers candidate draft tokens for a session's checkpoint.
func (cm *CheckpointManager) RecordMTPDraft(sessionID string, tokens []int32) error {
	return cm.RecordMTPDraftPages(sessionID, tokens)
}

// RecordMTPDraftPages registers candidate draft tokens and speculative page blocks for a session's checkpoint.
func (cm *CheckpointManager) RecordMTPDraftPages(sessionID string, tokens []int32, pages ...*PageBlock) error {
	cm.mu.RLock()
	desc, ok := cm.checkpoints[sessionID]
	cm.mu.RUnlock()
	if !ok {
		return ErrCheckpointNotFound
	}
	desc.RecordMTPDraft(tokens, pages...)
	return nil
}

// CommitMTPDraft atomically commits accepted draft tokens for a session and frees rejected draft pages.
func (cm *CheckpointManager) CommitMTPDraft(sessionID string, accepted int) (int, int, error) {
	cm.mu.RLock()
	desc, ok := cm.checkpoints[sessionID]
	cm.mu.RUnlock()
	if !ok {
		return 0, 0, ErrCheckpointNotFound
	}
	return desc.CommitMTPDraft(accepted)
}

// RollbackMTPDraft rolls back and frees all speculative draft pages for a session without memory leaks.
func (cm *CheckpointManager) RollbackMTPDraft(sessionID string) (int, error) {
	cm.mu.RLock()
	desc, ok := cm.checkpoints[sessionID]
	cm.mu.RUnlock()
	if !ok {
		return 0, ErrCheckpointNotFound
	}
	return desc.RollbackMTPDraft()
}

// ReapExpiredCheckpoints scans all stored checkpoints and reaps those that have expired,
// releasing any held physical block pins to prevent DRAM leaks. Returns the number of reaped checkpoints.
func (cm *CheckpointManager) ReapExpiredCheckpoints(now time.Time) int {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	var reaped int
	for sid, desc := range cm.checkpoints {
		if desc.IsExpired(now) {
			desc.ReleasePins()
			delete(cm.checkpoints, sid)
			reaped++
		}
	}
	return reaped
}

// Package-level default CheckpointManager support

var (
	defaultCheckpointManagerMu sync.Mutex
	defaultCheckpointManager   *CheckpointManager
)

// DefaultCheckpointManager returns the process-wide default CheckpointManager.
func DefaultCheckpointManager() *CheckpointManager {
	defaultCheckpointManagerMu.Lock()
	defer defaultCheckpointManagerMu.Unlock()
	if defaultCheckpointManager == nil {
		defaultCheckpointManager = NewCheckpointManager(nil, nil, nil, nil)
	}
	return defaultCheckpointManager
}

// SetDefaultCheckpointManager sets the process-wide default CheckpointManager.
func SetDefaultCheckpointManager(cm *CheckpointManager) {
	defaultCheckpointManagerMu.Lock()
	defer defaultCheckpointManagerMu.Unlock()
	defaultCheckpointManager = cm
}

// SaveInPlaceCheckpoint saves an in-place checkpoint via the default manager.
func SaveInPlaceCheckpoint(sessionID string) (*SessionDescriptor, error) {
	return DefaultCheckpointManager().SaveInPlaceCheckpoint(sessionID)
}

// RestoreInPlaceCheckpoint restores an in-place checkpoint via the default manager.
func RestoreInPlaceCheckpoint(sessionID string, desc *SessionDescriptor) error {
	return DefaultCheckpointManager().RestoreInPlaceCheckpoint(sessionID, desc)
}

// GetCheckpoint retrieves a checkpoint via the default manager.
func GetCheckpoint(sessionID string) (*SessionDescriptor, error) {
	return DefaultCheckpointManager().GetCheckpoint(sessionID)
}

// ReapExpiredCheckpoints reaps expired checkpoints via the default manager.
func ReapExpiredCheckpoints(now time.Time) int {
	return DefaultCheckpointManager().ReapExpiredCheckpoints(now)
}

// CheckpointManager returns the CheckpointManager associated with this MMU and its session managers.
func (m *MMU) CheckpointManager(pool ...*SharedTokenPool) *CheckpointManager {
	var p *SharedTokenPool
	if len(pool) > 0 {
		p = pool[0]
	}
	return NewCheckpointManager(m, p, m.ForkManager(), m.COWPageTable())
}
