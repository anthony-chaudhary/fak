package strix

import (
	"fmt"
)

// Block layout constants for AMD Strix Halo GQA KV cache.
const (
	// BlockTokens is the standard sequence block granularity (64 tokens per block).
	BlockTokens = 64

	// DefaultGQAKVHeads is the number of Key-Value heads in the reference model (8 heads).
	DefaultGQAKVHeads = 8

	// DefaultGQAHeadDim is the dimension per attention head (128 channels).
	DefaultGQAHeadDim = 128

	// DefaultBytesPerFP16 is the byte size of FP16/BF16 weights/activations (2 bytes).
	DefaultBytesPerFP16 = 2

	// TokenFootprintBytes is the KV footprint per token across both Key and Value tensors:
	// 2 (Key + Value) * 8 heads * 128 dim * 2 bytes = 4,096 bytes (4 KiB/token).
	TokenFootprintBytes = 2 * DefaultGQAKVHeads * DefaultGQAHeadDim * DefaultBytesPerFP16

	// RawBlockSizeBytes is the byte size of one 64-token sequence block:
	// 64 tokens * 4,096 bytes/token = 262,144 bytes (256 KiB).
	RawBlockSizeBytes = BlockTokens * TokenFootprintBytes

	// LinesPerRawBlock is the number of 64B cache lines in a raw block:
	// 262,144 bytes / 64 bytes = 4,096 lines.
	LinesPerRawBlock = RawBlockSizeBytes / MALLCacheLineBytes

	// DefaultCoprimePaddingBytes is the coprime padding inserted between blocks:
	// 64 bytes (1 cache line). Stride becomes 4,097 lines.
	// gcd(4097, 32768) = 1, eliminating power-of-two set aliasing.
	DefaultCoprimePaddingBytes = 64

	// LinesPerPaddedBlock is the line stride of a padded block:
	// 4,096 + 1 = 4,097 lines.
	LinesPerPaddedBlock = LinesPerRawBlock + (DefaultCoprimePaddingBytes / MALLCacheLineBytes)
)

// BlockLayoutDescriptor describes a single physical or virtual KV cache block.
type BlockLayoutDescriptor struct {
	BlockID         int     // Unique sequence or global block identifier
	LayerID         int     // Transformer layer index
	HeadID          int     // Attention head index (0 if heads are packed in token)
	VirtualAddress  uintptr // Base virtual address (64-byte aligned)
	BaseSetIndex    uint32  // Permuted MALL set index of the block's base address
	RawSizeBytes    int     // Raw payload size without padding (e.g. 262,144 bytes)
	PaddedSizeBytes int     // Total allocated size including coprime padding (e.g. 262,208 bytes)
	CacheLineCount  int     // Number of valid 64B cache lines in payload
}

// BlockLayoutConfig specifies parameters for token block packing and padding.
type BlockLayoutConfig struct {
	TokensPerBlock       int            // Tokens per block (default 64)
	NumLayers            int            // Number of transformer layers (default 32)
	NumKVHeads           int            // Number of KV heads (default 8)
	HeadDim              int            // Head dimension (default 128)
	BytesPerElement      int            // Precision in bytes (default 2 for FP16)
	EnableCoprimePadding bool           // If true, insert coprime padding between blocks
	PaddingBytes         int            // Bytes of coprime padding per block (default 64)
	PermutationArm       PermutationArm // Set permutation strategy
}

// DefaultBlockLayoutConfig returns the standard 32-layer GQA FP16 configuration
// with 64-token blocks and prime-modulo permutation enabled.
func DefaultBlockLayoutConfig() BlockLayoutConfig {
	return BlockLayoutConfig{
		TokensPerBlock:       BlockTokens,
		NumLayers:            32,
		NumKVHeads:           DefaultGQAKVHeads,
		HeadDim:              DefaultGQAHeadDim,
		BytesPerElement:      DefaultBytesPerFP16,
		EnableCoprimePadding: true,
		PaddingBytes:         DefaultCoprimePaddingBytes,
		PermutationArm:       ArmPrimeModulo,
	}
}

// BlockLayoutPacker packs KV cache tokens into 64-token non-aliasing block layouts.
type BlockLayoutPacker struct {
	config             BlockLayoutConfig
	engine             *SetPermutationEngine
	tokenFootprint     int
	rawBlockSize       int
	paddedBlockSize    int
	cacheLinesPerBlock int
}

// NewBlockLayoutPacker validates configuration and initializes the block layout packer.
func NewBlockLayoutPacker(cfg BlockLayoutConfig, engine *SetPermutationEngine) (*BlockLayoutPacker, error) {
	if cfg.TokensPerBlock <= 0 {
		return nil, fmt.Errorf("TokensPerBlock must be > 0, got %d", cfg.TokensPerBlock)
	}
	if cfg.NumLayers <= 0 {
		return nil, fmt.Errorf("NumLayers must be > 0, got %d", cfg.NumLayers)
	}
	if cfg.NumKVHeads <= 0 {
		return nil, fmt.Errorf("NumKVHeads must be > 0, got %d", cfg.NumKVHeads)
	}
	if cfg.HeadDim <= 0 {
		return nil, fmt.Errorf("HeadDim must be > 0, got %d", cfg.HeadDim)
	}
	if cfg.BytesPerElement <= 0 {
		return nil, fmt.Errorf("BytesPerElement must be > 0, got %d", cfg.BytesPerElement)
	}

	if engine == nil {
		engine = NewSetPermutationEngine()
	}

	tokenFootprint := 2 * cfg.NumKVHeads * cfg.HeadDim * cfg.BytesPerElement
	rawBlockSize := cfg.TokensPerBlock * tokenFootprint
	padding := 0
	if cfg.EnableCoprimePadding {
		if cfg.PaddingBytes > 0 {
			if cfg.PaddingBytes%MALLCacheLineBytes != 0 {
				return nil, fmt.Errorf("PaddingBytes must be a multiple of %d bytes, got %d", MALLCacheLineBytes, cfg.PaddingBytes)
			}
			padding = cfg.PaddingBytes
		} else {
			padding = DefaultCoprimePaddingBytes
		}
	}
	paddedBlockSize := rawBlockSize + padding
	cacheLinesPerBlock := rawBlockSize / MALLCacheLineBytes

	return &BlockLayoutPacker{
		config:             cfg,
		engine:             engine,
		tokenFootprint:     tokenFootprint,
		rawBlockSize:       rawBlockSize,
		paddedBlockSize:    paddedBlockSize,
		cacheLinesPerBlock: cacheLinesPerBlock,
	}, nil
}

// AllocateBlock computes the descriptor for a specific block given basePtr, layer, head, and blockIdx.
// Ensures 64-byte alignment and translates base set index via the permutation engine.
func (p *BlockLayoutPacker) AllocateBlock(basePtr uintptr, layer, head, blockIdx int) BlockLayoutDescriptor {
	globalID := blockIdx
	if layer > 0 || head > 0 {
		blocksPerLayer := 4
		globalID = (layer*p.config.NumKVHeads+head)*blocksPerLayer + blockIdx
	}

	stride := p.paddedBlockSize
	vaddr := basePtr + uintptr(globalID)*uintptr(stride)
	baseSet := p.engine.AddressToSetIndex(vaddr, p.config.PermutationArm)

	return BlockLayoutDescriptor{
		BlockID:         globalID,
		LayerID:         layer,
		HeadID:          head,
		VirtualAddress:  vaddr,
		BaseSetIndex:    baseSet,
		RawSizeBytes:    p.rawBlockSize,
		PaddedSizeBytes: p.paddedBlockSize,
		CacheLineCount:  p.cacheLinesPerBlock,
	}
}

// GenerateContextLayout packs totalTokens into 64-token blocks across all model layers.
// Produces a full layout where each block maintains 64-byte alignment and coprime stride spacing.
func (p *BlockLayoutPacker) GenerateContextLayout(basePtr uintptr, totalTokens int) ([]BlockLayoutDescriptor, error) {
	if totalTokens <= 0 {
		return nil, fmt.Errorf("totalTokens must be > 0, got %d", totalTokens)
	}

	tokensPerLayer := (totalTokens + p.config.NumLayers - 1) / p.config.NumLayers
	blocksPerLayer := (tokensPerLayer + p.config.TokensPerBlock - 1) / p.config.TokensPerBlock
	totalBlocks := p.config.NumLayers * blocksPerLayer
	blocks := make([]BlockLayoutDescriptor, 0, totalBlocks)

	stride := p.paddedBlockSize

	for layer := 0; layer < p.config.NumLayers; layer++ {
		for b := 0; b < blocksPerLayer; b++ {
			globalID := layer*blocksPerLayer + b
			vaddr := basePtr + uintptr(globalID)*uintptr(stride)
			baseSet := p.engine.AddressToSetIndex(vaddr, p.config.PermutationArm)

			desc := BlockLayoutDescriptor{
				BlockID:         globalID,
				LayerID:         layer,
				HeadID:          0,
				VirtualAddress:  vaddr,
				BaseSetIndex:    baseSet,
				RawSizeBytes:    p.rawBlockSize,
				PaddedSizeBytes: p.paddedBlockSize,
				CacheLineCount:  p.cacheLinesPerBlock,
			}
			blocks = append(blocks, desc)
		}
	}

	return blocks, nil
}

// VerifyNonAliasing checks two critical layout invariants:
// 1. All block base addresses are 64-byte aligned (vaddr % 64 == 0).
// 2. Adjacent blocks never alias to the same base set index (BaseSetIndex[i] != BaseSetIndex[i-1]).
// 3. For sequences of blocks <= 32,768, no two blocks alias to the same base set index.
// Returns (true, counts) if clean; (false, counts) if any aliasing is detected.
func (p *BlockLayoutPacker) VerifyNonAliasing(blocks []BlockLayoutDescriptor) (bool, map[uint32]int) {
	counts := make(map[uint32]int, len(blocks))
	if len(blocks) == 0 {
		return true, counts
	}

	noAliasing := true

	for i, b := range blocks {
		// Verify 64-byte line alignment
		if b.VirtualAddress%uintptr(MALLCacheLineBytes) != 0 {
			noAliasing = false
		}

		// Verify adjacent blocks do not alias
		if i > 0 && b.BaseSetIndex == blocks[i-1].BaseSetIndex {
			noAliasing = false
		}

		counts[b.BaseSetIndex]++
	}

	// Verify blocks in sequence do not alias to the same base set index
	if len(blocks) <= MALLSets {
		for _, c := range counts {
			if c > 1 {
				noAliasing = false
				break
			}
		}
	}

	return noAliasing, counts
}

// Config returns the current configuration.
func (p *BlockLayoutPacker) Config() BlockLayoutConfig {
	return p.config
}

// TokenFootprint returns the byte footprint of a single token.
func (p *BlockLayoutPacker) TokenFootprint() int {
	return p.tokenFootprint
}

// RawBlockSize returns the unpadded byte size of a block.
func (p *BlockLayoutPacker) RawBlockSize() int {
	return p.rawBlockSize
}

// PaddedBlockSize returns the padded byte size of a block.
func (p *BlockLayoutPacker) PaddedBlockSize() int {
	return p.paddedBlockSize
}

// CacheLinesPerBlock returns the number of valid cache lines per block.
func (p *BlockLayoutPacker) CacheLinesPerBlock() int {
	return p.cacheLinesPerBlock
}
