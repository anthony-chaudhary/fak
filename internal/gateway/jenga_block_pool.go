package gateway

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sync"
)

// JengaLeafGeometry defines the physical memory footprint of one entry for a specific KV leaf type.
type JengaLeafGeometry struct {
	LeafType   string `json:"leaf_type"`   // distinct leaf identity e.g. "hybrid_a", "hybrid_b"
	EntryBytes int    `json:"entry_bytes"` // must be positive and <= HugeBlockBytes
}

// JengaHugeBlock is a uniform physical memory block partitioned into homogeneous entries for one leaf type.
type JengaHugeBlock struct {
	BlockID      int    `json:"block_id"`
	LeafType     string `json:"leaf_type"` // empty when block is free in the shared bank
	TotalEntries int    `json:"total_entries"`
	FreeEntries  int    `json:"free_entries"`
	EntryBytes   int    `json:"entry_bytes"`
	InUse        []bool `json:"-"`
}

// JengaBlockBank dynamically assigns uniform huge blocks across heterogeneous KV leaves,
// reclaiming physical capacity across leaf boundaries without cross-leaf aliasing.
type JengaBlockBank struct {
	mu             sync.Mutex
	HugeBlockBytes int
	Blocks         []*JengaHugeBlock
	TotalReclaims  int
}

// NewJengaBlockBank constructs a bank with numBlocks uniform huge blocks.
func NewJengaBlockBank(hugeBlockBytes int, numBlocks int) (*JengaBlockBank, error) {
	if hugeBlockBytes <= 0 || numBlocks <= 0 {
		return nil, fmt.Errorf("hugeBlockBytes and numBlocks must be positive")
	}

	blocks := make([]*JengaHugeBlock, numBlocks)
	for i := 0; i < numBlocks; i++ {
		blocks[i] = &JengaHugeBlock{
			BlockID:  i,
			LeafType: "",
		}
	}

	return &JengaBlockBank{
		HugeBlockBytes: hugeBlockBytes,
		Blocks:         blocks,
	}, nil
}

// AllocateEntry allocates an entry for geom from an existing compatible huge block or acquires a free huge block.
func (b *JengaBlockBank) AllocateEntry(geom JengaLeafGeometry, prefix string) (blockID, entryID int, entryDigest string, err error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if geom.LeafType == "" || geom.EntryBytes <= 0 || geom.EntryBytes > b.HugeBlockBytes {
		return -1, -1, "", fmt.Errorf("invalid geometry: leaf=%q entryBytes=%d", geom.LeafType, geom.EntryBytes)
	}

	entriesPerBlock := b.HugeBlockBytes / geom.EntryBytes

	// 1. Try to find an existing block with matching leaf type and free entries
	for _, block := range b.Blocks {
		if block.LeafType == geom.LeafType && block.FreeEntries > 0 {
			for eIdx, inUse := range block.InUse {
				if !inUse {
					block.InUse[eIdx] = true
					block.FreeEntries--
					digest := computeEntryDigest(geom.LeafType, prefix, block.BlockID, eIdx)
					return block.BlockID, eIdx, digest, nil
				}
			}
		}
	}

	// 2. Allocate an unassigned free huge block from the shared bank
	for _, block := range b.Blocks {
		if block.LeafType == "" {
			block.LeafType = geom.LeafType
			block.EntryBytes = geom.EntryBytes
			block.TotalEntries = entriesPerBlock
			block.FreeEntries = entriesPerBlock - 1
			block.InUse = make([]bool, entriesPerBlock)
			block.InUse[0] = true

			digest := computeEntryDigest(geom.LeafType, prefix, block.BlockID, 0)
			return block.BlockID, 0, digest, nil
		}
	}

	return -1, -1, "", fmt.Errorf("jenga bank exhausted: no free entries or huge blocks available")
}

// ReclaimEntry returns an entry to its huge block; if the huge block becomes completely empty,
// it is returned to the shared unassigned bank, making its capacity available to any leaf geometry.
func (b *JengaBlockBank) ReclaimEntry(blockID, entryID int) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if blockID < 0 || blockID >= len(b.Blocks) {
		return fmt.Errorf("blockID %d out of bounds", blockID)
	}
	block := b.Blocks[blockID]
	if block.LeafType == "" {
		return fmt.Errorf("block %d is already unassigned", blockID)
	}
	if entryID < 0 || entryID >= block.TotalEntries {
		return fmt.Errorf("entryID %d out of bounds [0, %d)", entryID, block.TotalEntries)
	}
	if !block.InUse[entryID] {
		return fmt.Errorf("entry %d in block %d is not in use", entryID, blockID)
	}

	block.InUse[entryID] = false
	block.FreeEntries++
	b.TotalReclaims++

	// If all entries in the huge block are now reclaimed, release block back to the shared bank
	if block.FreeEntries == block.TotalEntries {
		block.LeafType = ""
		block.TotalEntries = 0
		block.FreeEntries = 0
		block.EntryBytes = 0
		block.InUse = nil
	}

	return nil
}

func computeEntryDigest(leafType, prefix string, blockID, entryID int) string {
	h := sha256.New()
	fmt.Fprintf(h, "leaf=%s|prefix=%s|b=%d|e=%d", leafType, prefix, blockID, entryID)
	return hex.EncodeToString(h.Sum(nil))
}
