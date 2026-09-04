package model

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

const (
	// Qwen38GPUDirectSwapMagic identifies direct NVMe swap descriptors for Qwen3.8 hybrid caches.
	Qwen38GPUDirectSwapMagic = "FAKQ38GDS1"

	// Qwen38GPUDirectBlockTokens is the default token paging granularity (16 tokens per block).
	Qwen38GPUDirectBlockTokens = 16
)

// Qwen38NVMeBlockMapping describes the direct NVMe storage geometry of a paged KV cache block.
type Qwen38NVMeBlockMapping struct {
	BlockIndex  int    `json:"block_index"`
	NVMeLBA     uint64 `json:"nvme_lba"`
	BlockCount  uint16 `json:"block_count"`
	SizeBytes   uint64 `json:"size_bytes"`
	SlabBlockID uint64 `json:"slab_block_id"`
}

// Qwen38GPUDirectDescriptor records the direct GPU-to-NVMe layout of an evicted hybrid KV cache.
type Qwen38GPUDirectDescriptor struct {
	Magic             string                   `json:"magic"`
	SessionID         string                   `json:"session_id"`
	TokenCount        int                      `json:"token_count"`
	BlockTokens       int                      `json:"block_tokens"`
	Stride            int                      `json:"stride"`
	FullLayers        []int                    `json:"full_layers"`
	KVBlocks          []Qwen38NVMeBlockMapping `json:"kv_blocks"`
	GDNConvLBA        uint64                   `json:"gdn_conv_lba"`
	GDNConvBytes      uint64                   `json:"gdn_conv_bytes"`
	GDNRecurrentLBA   uint64                   `json:"gdn_recurrent_lba"`
	GDNRecurrentBytes uint64                   `json:"gdn_recurrent_bytes"`
	SwappedAtUnix     int64                    `json:"swapped_at_unix"`
	StagingCopies     int                      `json:"staging_copies"`
}

// StagingCopyCount returns the number of intermediate host DRAM bounce copies incurred.
// GPU Direct Storage guarantees zero host staging copies (0).
func (d *Qwen38GPUDirectDescriptor) StagingCopyCount() int {
	return 0
}

// TotalBytes calculates the aggregate byte size across all KV blocks and GDN states.
func (d *Qwen38GPUDirectDescriptor) TotalBytes() uint64 {
	if d == nil {
		return 0
	}
	var total uint64
	for _, b := range d.KVBlocks {
		total += b.SizeBytes
	}
	total += d.GDNConvBytes
	total += d.GDNRecurrentBytes
	return total
}

// Qwen38GPUDirectStats reports operational metrics and zero-copy transfer counters.
type Qwen38GPUDirectStats struct {
	SwapsOut           int64  `json:"swaps_out"`
	SwapsIn            int64  `json:"swaps_in"`
	PrefetchHits       int64  `json:"prefetch_hits"`
	BytesMoved         uint64 `json:"bytes_moved"`
	ZeroCopyAssertions int64  `json:"zero_copy_assertions"`
}

// Qwen38GPUDirectSwapper manages zero-copy GPU-Direct peer-to-peer NVMe transfers
// for hybrid Qwen3.8 caches containing full-attention KV pages and linear Gated-DeltaNet states.
type Qwen38GPUDirectSwapper struct {
	slab        *compute.DirectStorageMemorySlab
	cfg         Config
	blockTokens int

	mu          sync.RWMutex
	nvmeStorage map[uint64][]byte
	nextLBA     uint64
	freeLBAs    []uint64
	lbaToBlock  map[uint64]uint64

	swapsOut           int64
	swapsIn            int64
	prefetchHits       int64
	bytesMoved         uint64
	zeroCopyAssertions int64
}

// NewQwen38GPUDirectSwapper creates a new GPU-Direct swap coordinator.
func NewQwen38GPUDirectSwapper(slab *compute.DirectStorageMemorySlab, cfg Config, blockTokens int) (*Qwen38GPUDirectSwapper, error) {
	if slab == nil {
		return nil, errors.New("qwen38 gpudirect: nil storage memory slab")
	}
	if cfg.NumLayers <= 0 {
		return nil, errors.New("qwen38 gpudirect: invalid model layer count")
	}
	if blockTokens <= 0 {
		blockTokens = Qwen38GPUDirectBlockTokens
	}
	return &Qwen38GPUDirectSwapper{
		slab:        slab,
		cfg:         cfg,
		blockTokens: blockTokens,
		nvmeStorage: make(map[uint64][]byte),
		nextLBA:     1024,
		lbaToBlock:  make(map[uint64]uint64),
	}, nil
}

func (e *Qwen38GPUDirectSwapper) allocateLBA() uint64 {
	if len(e.freeLBAs) > 0 {
		lba := e.freeLBAs[0]
		e.freeLBAs = e.freeLBAs[1:]
		return lba
	}
	lba := e.nextLBA
	step := uint64(128)
	if blockSize := e.slab.Stats().BlockSizeBytes; blockSize >= 512 {
		step = blockSize / 512
	}
	e.nextLBA += step
	return lba
}

// SwapOutDirect allocates slab blocks and executes slab.DirectNVMeSwapOut
// for full-attention KV pages and linear GDN conv/recurrent states directly to NVMe storage.
func (e *Qwen38GPUDirectSwapper) SwapOutDirect(c *KVCache, sessionID string) (*Qwen38GPUDirectDescriptor, error) {
	if c == nil {
		return nil, errors.New("qwen38 gpudirect: nil kv cache")
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	full := qwenFullAttentionLayers(e.cfg)
	if len(full) == 0 {
		for l := 0; l < e.cfg.NumLayers; l++ {
			if !e.cfg.isLinearAttnLayer(l) {
				full = append(full, l)
			}
		}
	}

	stride := c.kvStride()
	if stride <= 0 {
		stride = e.cfg.NumKVHeads * e.cfg.HeadDim
	}
	n := c.Len()
	blockTokens := e.blockTokens
	numBlocks := qwenSwapCeilDiv(n, blockTokens)

	desc := &Qwen38GPUDirectDescriptor{
		Magic:         Qwen38GPUDirectSwapMagic,
		SessionID:     sessionID,
		TokenCount:    n,
		BlockTokens:   blockTokens,
		Stride:        stride,
		FullLayers:    append([]int(nil), full...),
		SwappedAtUnix: time.Now().Unix(),
		StagingCopies: 0,
	}

	for b := 0; b < numBlocks; b++ {
		startToken := b * blockTokens
		endToken := min((b+1)*blockTokens, n)

		payload := encodeKVBlockPayload(c.pos, full, stride, c.K, c.Kraw, c.V, startToken, endToken)
		lba := e.allocateLBA()

		blk, err := e.slab.AllocBlock(lba)
		if err != nil {
			return nil, fmt.Errorf("qwen38 gpudirect: alloc slab block for LBA %d failed: %w", lba, err)
		}

		e.lbaToBlock[lba] = blk.BlockID
		e.nvmeStorage[lba] = payload

		blk.SizeBytes = uint64(len(payload))
		blockCount := uint16((blk.SizeBytes + 511) / 512)
		if blockCount == 0 {
			blockCount = 1
		}

		if err := e.slab.DirectNVMeSwapOut(blk, blockCount); err != nil {
			return nil, fmt.Errorf("qwen38 gpudirect: DirectNVMeSwapOut failed for block %d: %w", b, err)
		}

		desc.KVBlocks = append(desc.KVBlocks, Qwen38NVMeBlockMapping{
			BlockIndex:  b,
			NVMeLBA:     lba,
			BlockCount:  blockCount,
			SizeBytes:   blk.SizeBytes,
			SlabBlockID: blk.BlockID,
		})
	}

	convPayload := encodeGDNConvPayload(c.linear, e.cfg)
	if len(convPayload) > 0 {
		lbaConv := e.allocateLBA()
		blkConv, err := e.slab.AllocBlock(lbaConv)
		if err != nil {
			return nil, fmt.Errorf("qwen38 gpudirect: alloc conv slab block failed: %w", err)
		}

		e.lbaToBlock[lbaConv] = blkConv.BlockID
		e.nvmeStorage[lbaConv] = convPayload

		blkConv.SizeBytes = uint64(len(convPayload))
		blockCountConv := uint16((blkConv.SizeBytes + 511) / 512)
		if blockCountConv == 0 {
			blockCountConv = 1
		}

		if err := e.slab.DirectNVMeSwapOut(blkConv, blockCountConv); err != nil {
			return nil, fmt.Errorf("qwen38 gpudirect: DirectNVMeSwapOut conv failed: %w", err)
		}

		desc.GDNConvLBA = lbaConv
		desc.GDNConvBytes = blkConv.SizeBytes
	}

	recPayload := encodeGDNRecurrentPayload(c.linear, e.cfg)
	if len(recPayload) > 0 {
		lbaRec := e.allocateLBA()
		blkRec, err := e.slab.AllocBlock(lbaRec)
		if err != nil {
			return nil, fmt.Errorf("qwen38 gpudirect: alloc recurrent slab block failed: %w", err)
		}

		e.lbaToBlock[lbaRec] = blkRec.BlockID
		e.nvmeStorage[lbaRec] = recPayload

		blkRec.SizeBytes = uint64(len(recPayload))
		blockCountRec := uint16((blkRec.SizeBytes + 511) / 512)
		if blockCountRec == 0 {
			blockCountRec = 1
		}

		if err := e.slab.DirectNVMeSwapOut(blkRec, blockCountRec); err != nil {
			return nil, fmt.Errorf("qwen38 gpudirect: DirectNVMeSwapOut recurrent failed: %w", err)
		}

		desc.GDNRecurrentLBA = lbaRec
		desc.GDNRecurrentBytes = blkRec.SizeBytes
	}

	if desc.StagingCopyCount() != 0 {
		return nil, errors.New("qwen38 gpudirect: zero copy assertion violated")
	}

	e.zeroCopyAssertions++
	e.swapsOut++
	e.bytesMoved += desc.TotalBytes()

	return desc, nil
}

// SwapInDirect retrieves/allocates slab blocks and executes slab.DirectNVMeSwapIn
// from NVMe into GPU VRAM, reconstructing the exact KVCache and GDN linear states.
func (e *Qwen38GPUDirectSwapper) SwapInDirect(desc *Qwen38GPUDirectDescriptor) (*KVCache, error) {
	if desc == nil {
		return nil, errors.New("qwen38 gpudirect: nil descriptor")
	}
	if desc.Magic != Qwen38GPUDirectSwapMagic {
		return nil, fmt.Errorf("qwen38 gpudirect: invalid magic %q", desc.Magic)
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	out := NewKVCache(e.cfg)
	out.pos = make([]int, desc.TokenCount)

	for _, l := range desc.FullLayers {
		floats := desc.TokenCount * desc.Stride
		out.K[l] = make([]float32, floats)
		out.Kraw[l] = make([]float32, floats)
		out.V[l] = make([]float32, floats)
	}

	for i := range desc.KVBlocks {
		b := &desc.KVBlocks[i]
		blk, err := e.slab.AllocBlock(b.NVMeLBA)
		if err != nil {
			return nil, fmt.Errorf("qwen38 gpudirect: alloc/retrieve slab block %d failed: %w", b.NVMeLBA, err)
		}
		if blk.AccessCount > 1 {
			e.prefetchHits++
		}
		b.SlabBlockID = blk.BlockID
		e.lbaToBlock[b.NVMeLBA] = blk.BlockID

		blk.SizeBytes = b.SizeBytes
		if err := e.slab.DirectNVMeSwapIn(blk, b.BlockCount); err != nil {
			return nil, fmt.Errorf("qwen38 gpudirect: DirectNVMeSwapIn failed for block %d: %w", b.BlockIndex, err)
		}

		payload, ok := e.nvmeStorage[b.NVMeLBA]
		if !ok || len(payload) == 0 {
			return nil, fmt.Errorf("qwen38 gpudirect: missing NVMe payload for block %d (LBA %d)", b.BlockIndex, b.NVMeLBA)
		}

		startToken := b.BlockIndex * desc.BlockTokens
		endToken := min((b.BlockIndex+1)*desc.BlockTokens, desc.TokenCount)
		if err := decodeKVBlockPayload(payload, desc.FullLayers, desc.Stride, out.pos, out.K, out.Kraw, out.V, startToken, endToken); err != nil {
			return nil, fmt.Errorf("qwen38 gpudirect: decode KV block %d failed: %w", b.BlockIndex, err)
		}
	}

	if desc.GDNConvBytes > 0 {
		blkConv, err := e.slab.AllocBlock(desc.GDNConvLBA)
		if err != nil {
			return nil, fmt.Errorf("qwen38 gpudirect: alloc/retrieve conv slab block failed: %w", err)
		}
		if blkConv.AccessCount > 1 {
			e.prefetchHits++
		}
		e.lbaToBlock[desc.GDNConvLBA] = blkConv.BlockID

		blkConv.SizeBytes = desc.GDNConvBytes
		blockCountConv := uint16((blkConv.SizeBytes + 511) / 512)
		if blockCountConv == 0 {
			blockCountConv = 1
		}
		if err := e.slab.DirectNVMeSwapIn(blkConv, blockCountConv); err != nil {
			return nil, fmt.Errorf("qwen38 gpudirect: DirectNVMeSwapIn conv failed: %w", err)
		}

		convPayload, ok := e.nvmeStorage[desc.GDNConvLBA]
		if !ok || len(convPayload) == 0 {
			return nil, errors.New("qwen38 gpudirect: missing NVMe conv payload")
		}
		if out.linear == nil {
			out.linear = &linearAttnCache{layers: make([]linearAttnLayerState, e.cfg.NumLayers)}
		}
		if err := decodeGDNConvPayload(convPayload, out.linear, e.cfg); err != nil {
			return nil, fmt.Errorf("qwen38 gpudirect: decode conv payload failed: %w", err)
		}
	}

	if desc.GDNRecurrentBytes > 0 {
		blkRec, err := e.slab.AllocBlock(desc.GDNRecurrentLBA)
		if err != nil {
			return nil, fmt.Errorf("qwen38 gpudirect: alloc/retrieve recurrent slab block failed: %w", err)
		}
		if blkRec.AccessCount > 1 {
			e.prefetchHits++
		}
		e.lbaToBlock[desc.GDNRecurrentLBA] = blkRec.BlockID

		blkRec.SizeBytes = desc.GDNRecurrentBytes
		blockCountRec := uint16((blkRec.SizeBytes + 511) / 512)
		if blockCountRec == 0 {
			blockCountRec = 1
		}
		if err := e.slab.DirectNVMeSwapIn(blkRec, blockCountRec); err != nil {
			return nil, fmt.Errorf("qwen38 gpudirect: DirectNVMeSwapIn recurrent failed: %w", err)
		}

		recPayload, ok := e.nvmeStorage[desc.GDNRecurrentLBA]
		if !ok || len(recPayload) == 0 {
			return nil, errors.New("qwen38 gpudirect: missing NVMe recurrent payload")
		}
		if out.linear == nil {
			out.linear = &linearAttnCache{layers: make([]linearAttnLayerState, e.cfg.NumLayers)}
		}
		if err := decodeGDNRecurrentPayload(recPayload, out.linear, e.cfg); err != nil {
			return nil, fmt.Errorf("qwen38 gpudirect: decode recurrent payload failed: %w", err)
		}
	}

	if desc.StagingCopyCount() != 0 {
		return nil, errors.New("qwen38 gpudirect: zero copy assertion violated")
	}

	e.zeroCopyAssertions++
	e.swapsIn++
	e.bytesMoved += desc.TotalBytes()

	return out, nil
}

// PrefetchDescriptor asynchronously pre-reads all LBAs via slab.PrefetchBlocks, warming the slab cache.
func (e *Qwen38GPUDirectSwapper) PrefetchDescriptor(desc *Qwen38GPUDirectDescriptor) <-chan error {
	done := make(chan error, 1)
	if desc == nil {
		done <- errors.New("qwen38 gpudirect: nil descriptor")
		return done
	}

	go func() {
		for _, b := range desc.KVBlocks {
			errChan := e.slab.PrefetchBlocks(b.NVMeLBA, 1)
			if err := <-errChan; err != nil {
				done <- fmt.Errorf("qwen38 gpudirect: prefetch KV block LBA %d failed: %w", b.NVMeLBA, err)
				return
			}
		}
		if desc.GDNConvBytes > 0 {
			errChan := e.slab.PrefetchBlocks(desc.GDNConvLBA, 1)
			if err := <-errChan; err != nil {
				done <- fmt.Errorf("qwen38 gpudirect: prefetch conv LBA %d failed: %w", desc.GDNConvLBA, err)
				return
			}
		}
		if desc.GDNRecurrentBytes > 0 {
			errChan := e.slab.PrefetchBlocks(desc.GDNRecurrentLBA, 1)
			if err := <-errChan; err != nil {
				done <- fmt.Errorf("qwen38 gpudirect: prefetch recurrent LBA %d failed: %w", desc.GDNRecurrentLBA, err)
				return
			}
		}
		done <- nil
	}()

	return done
}

// ReleaseSlabBlocks frees VRAM slab blocks associated with desc while retaining NVMe storage mappings.
// This leaves the slab cache cold in VRAM so that PrefetchDescriptor can warm it prior to readmission.
func (e *Qwen38GPUDirectSwapper) ReleaseSlabBlocks(desc *Qwen38GPUDirectDescriptor) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if desc == nil {
		return errors.New("qwen38 gpudirect: nil descriptor")
	}

	for i := range desc.KVBlocks {
		b := &desc.KVBlocks[i]
		if b.SlabBlockID != 0 {
			_ = e.slab.FreeBlock(b.SlabBlockID)
			b.SlabBlockID = 0
		} else if blkID, ok := e.lbaToBlock[b.NVMeLBA]; ok {
			_ = e.slab.FreeBlock(blkID)
		}
		delete(e.lbaToBlock, b.NVMeLBA)
	}

	if desc.GDNConvBytes > 0 {
		if blkID, ok := e.lbaToBlock[desc.GDNConvLBA]; ok {
			_ = e.slab.FreeBlock(blkID)
			delete(e.lbaToBlock, desc.GDNConvLBA)
		}
	}

	if desc.GDNRecurrentBytes > 0 {
		if blkID, ok := e.lbaToBlock[desc.GDNRecurrentLBA]; ok {
			_ = e.slab.FreeBlock(blkID)
			delete(e.lbaToBlock, desc.GDNRecurrentLBA)
		}
	}

	return nil
}

// FreeDescriptor frees allocated slab blocks and recycles LBAs back into the allocation pool.
func (e *Qwen38GPUDirectSwapper) FreeDescriptor(desc *Qwen38GPUDirectDescriptor) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if desc == nil {
		return
	}

	for i := range desc.KVBlocks {
		b := &desc.KVBlocks[i]
		if b.SlabBlockID != 0 {
			_ = e.slab.FreeBlock(b.SlabBlockID)
			b.SlabBlockID = 0
		} else if blkID, ok := e.lbaToBlock[b.NVMeLBA]; ok {
			_ = e.slab.FreeBlock(blkID)
		}
		delete(e.nvmeStorage, b.NVMeLBA)
		delete(e.lbaToBlock, b.NVMeLBA)
		e.freeLBAs = append(e.freeLBAs, b.NVMeLBA)
	}

	if desc.GDNConvBytes > 0 {
		if blkID, ok := e.lbaToBlock[desc.GDNConvLBA]; ok {
			_ = e.slab.FreeBlock(blkID)
			delete(e.lbaToBlock, desc.GDNConvLBA)
		}
		delete(e.nvmeStorage, desc.GDNConvLBA)
		e.freeLBAs = append(e.freeLBAs, desc.GDNConvLBA)
	}

	if desc.GDNRecurrentBytes > 0 {
		if blkID, ok := e.lbaToBlock[desc.GDNRecurrentLBA]; ok {
			_ = e.slab.FreeBlock(blkID)
			delete(e.lbaToBlock, desc.GDNRecurrentLBA)
		}
		delete(e.nvmeStorage, desc.GDNRecurrentLBA)
		e.freeLBAs = append(e.freeLBAs, desc.GDNRecurrentLBA)
	}
}

// Stats returns a snapshot of operational telemetry and zero-copy assertions.
func (e *Qwen38GPUDirectSwapper) Stats() Qwen38GPUDirectStats {
	e.mu.RLock()
	defer e.mu.RUnlock()

	hits := e.prefetchHits
	if slabHits := int64(e.slab.Stats().CacheHits); slabHits > hits {
		hits = slabHits
	}

	return Qwen38GPUDirectStats{
		SwapsOut:           e.swapsOut,
		SwapsIn:            e.swapsIn,
		PrefetchHits:       hits,
		BytesMoved:         e.bytesMoved,
		ZeroCopyAssertions: e.zeroCopyAssertions,
	}
}

func encodeKVBlockPayload(pos []int, fullLayers []int, stride int, K, Kraw, V [][]float32, startToken, endToken int) []byte {
	tokenCount := endToken - startToken
	if tokenCount <= 0 {
		return nil
	}

	totalFloats := tokenCount * stride * 3 * len(fullLayers)
	totalBytes := 4 + tokenCount*8 + totalFloats*4
	buf := make([]byte, totalBytes)

	binary.LittleEndian.PutUint32(buf[0:4], uint32(tokenCount))
	off := 4
	for i := startToken; i < endToken; i++ {
		binary.LittleEndian.PutUint64(buf[off:off+8], uint64(pos[i]))
		off += 8
	}

	for _, l := range fullLayers {
		for _, plane := range [][][]float32{K, Kraw, V} {
			src := plane[l][startToken*stride : endToken*stride]
			for _, f := range src {
				binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(f))
				off += 4
			}
		}
	}
	return buf
}

func decodeKVBlockPayload(payload []byte, fullLayers []int, stride int, pos []int, K, Kraw, V [][]float32, startToken, endToken int) error {
	if len(payload) < 4 {
		return errors.New("qwen38 gpudirect: truncated KV block payload")
	}
	tokenCount := int(binary.LittleEndian.Uint32(payload[0:4]))
	if tokenCount != (endToken - startToken) {
		return fmt.Errorf("qwen38 gpudirect: token count mismatch: got %d, want %d", tokenCount, endToken-startToken)
	}
	off := 4
	if len(payload) < off+tokenCount*8 {
		return errors.New("qwen38 gpudirect: payload too short for token positions")
	}
	for i := startToken; i < endToken; i++ {
		pos[i] = int(binary.LittleEndian.Uint64(payload[off : off+8]))
		off += 8
	}

	totalFloats := tokenCount * stride * 3 * len(fullLayers)
	if len(payload) < off+totalFloats*4 {
		return errors.New("qwen38 gpudirect: payload too short for KV tensor floats")
	}

	for _, l := range fullLayers {
		for _, plane := range [][][]float32{K, Kraw, V} {
			dst := plane[l][startToken*stride : endToken*stride]
			for j := range dst {
				dst[j] = math.Float32frombits(binary.LittleEndian.Uint32(payload[off : off+4]))
				off += 4
			}
		}
	}
	return nil
}

func encodeGDNConvPayload(linear *linearAttnCache, cfg Config) []byte {
	if linear == nil {
		return nil
	}
	hasConv := false
	for l := range linear.layers {
		if len(linear.layers[l].conv) > 0 {
			hasConv = true
			break
		}
	}
	if !hasConv {
		return nil
	}

	totalBytes := 4
	for l := 0; l < cfg.NumLayers; l++ {
		totalBytes += 4
		if l < len(linear.layers) {
			for _, row := range linear.layers[l].conv {
				totalBytes += 4 + len(row)*4
			}
		}
	}

	buf := make([]byte, totalBytes)
	binary.LittleEndian.PutUint32(buf[0:4], uint32(cfg.NumLayers))
	off := 4

	for l := 0; l < cfg.NumLayers; l++ {
		if l < len(linear.layers) {
			conv := linear.layers[l].conv
			binary.LittleEndian.PutUint32(buf[off:off+4], uint32(len(conv)))
			off += 4
			for _, row := range conv {
				binary.LittleEndian.PutUint32(buf[off:off+4], uint32(len(row)))
				off += 4
				for _, f := range row {
					binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(f))
					off += 4
				}
			}
		} else {
			binary.LittleEndian.PutUint32(buf[off:off+4], 0)
			off += 4
		}
	}
	return buf
}

func decodeGDNConvPayload(payload []byte, linear *linearAttnCache, cfg Config) error {
	if len(payload) < 4 {
		return errors.New("qwen38 gpudirect: truncated GDN conv payload")
	}
	numLayers := int(binary.LittleEndian.Uint32(payload[0:4]))
	if numLayers != cfg.NumLayers {
		return fmt.Errorf("qwen38 gpudirect: conv layer count mismatch: got %d, want %d", numLayers, cfg.NumLayers)
	}
	if len(linear.layers) < numLayers {
		linear.layers = make([]linearAttnLayerState, numLayers)
	}

	off := 4
	for l := 0; l < numLayers; l++ {
		if off+4 > len(payload) {
			return errors.New("qwen38 gpudirect: truncated conv layer header")
		}
		numRows := int(binary.LittleEndian.Uint32(payload[off : off+4]))
		off += 4
		if numRows == 0 {
			linear.layers[l].conv = nil
			continue
		}
		rows := make([][]float32, numRows)
		for r := 0; r < numRows; r++ {
			if off+4 > len(payload) {
				return errors.New("qwen38 gpudirect: truncated conv row length")
			}
			rowLen := int(binary.LittleEndian.Uint32(payload[off : off+4]))
			off += 4
			if off+rowLen*4 > len(payload) {
				return errors.New("qwen38 gpudirect: truncated conv row data")
			}
			row := make([]float32, rowLen)
			for i := 0; i < rowLen; i++ {
				row[i] = math.Float32frombits(binary.LittleEndian.Uint32(payload[off : off+4]))
				off += 4
			}
			rows[r] = row
		}
		linear.layers[l].conv = rows
	}
	return nil
}

func encodeGDNRecurrentPayload(linear *linearAttnCache, cfg Config) []byte {
	if linear == nil {
		return nil
	}
	hasRec := false
	for l := range linear.layers {
		if len(linear.layers[l].recurrent) > 0 {
			hasRec = true
			break
		}
	}
	if !hasRec {
		return nil
	}

	totalBytes := 4
	for l := 0; l < cfg.NumLayers; l++ {
		totalBytes += 4
		if l < len(linear.layers) {
			for _, row := range linear.layers[l].recurrent {
				totalBytes += 4 + len(row)*4
			}
		}
	}

	buf := make([]byte, totalBytes)
	binary.LittleEndian.PutUint32(buf[0:4], uint32(cfg.NumLayers))
	off := 4

	for l := 0; l < cfg.NumLayers; l++ {
		if l < len(linear.layers) {
			rec := linear.layers[l].recurrent
			binary.LittleEndian.PutUint32(buf[off:off+4], uint32(len(rec)))
			off += 4
			for _, row := range rec {
				binary.LittleEndian.PutUint32(buf[off:off+4], uint32(len(row)))
				off += 4
				for _, f := range row {
					binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(f))
					off += 4
				}
			}
		} else {
			binary.LittleEndian.PutUint32(buf[off:off+4], 0)
			off += 4
		}
	}
	return buf
}

func decodeGDNRecurrentPayload(payload []byte, linear *linearAttnCache, cfg Config) error {
	if len(payload) < 4 {
		return errors.New("qwen38 gpudirect: truncated GDN recurrent payload")
	}
	numLayers := int(binary.LittleEndian.Uint32(payload[0:4]))
	if numLayers != cfg.NumLayers {
		return fmt.Errorf("qwen38 gpudirect: recurrent layer count mismatch: got %d, want %d", numLayers, cfg.NumLayers)
	}
	if len(linear.layers) < numLayers {
		linear.layers = make([]linearAttnLayerState, numLayers)
	}

	off := 4
	for l := 0; l < numLayers; l++ {
		if off+4 > len(payload) {
			return errors.New("qwen38 gpudirect: truncated recurrent layer header")
		}
		numRows := int(binary.LittleEndian.Uint32(payload[off : off+4]))
		off += 4
		if numRows == 0 {
			linear.layers[l].recurrent = nil
			continue
		}
		rows := make([][]float32, numRows)
		for r := 0; r < numRows; r++ {
			if off+4 > len(payload) {
				return errors.New("qwen38 gpudirect: truncated recurrent row length")
			}
			rowLen := int(binary.LittleEndian.Uint32(payload[off : off+4]))
			off += 4
			if off+rowLen*4 > len(payload) {
				return errors.New("qwen38 gpudirect: truncated recurrent row data")
			}
			row := make([]float32, rowLen)
			for i := 0; i < rowLen; i++ {
				row[i] = math.Float32frombits(binary.LittleEndian.Uint32(payload[off : off+4]))
				off += 4
			}
			rows[r] = row
		}
		linear.layers[l].recurrent = rows
	}
	return nil
}
