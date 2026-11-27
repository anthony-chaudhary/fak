// Adapted from DeepGEMM (https://github.com/deepseek-ai/DeepGEMM),
// pinned at revision 5081f8513af57e61696de4f4f04b0e707ffb503e.
// Copyright (c) 2025 DeepSeek
// SPDX-License-Identifier: MIT
// See THIRD_PARTY_NOTICES.md ("DeepGEMM").

package compute

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
)

const (
	// sm90SmemCapacityBytes matches the Hopper sm_90 shared memory capacity budget.
	sm90SmemCapacityBytes = 232448
	sm90MaxPipelineStages = 16
	sm90TmaMapSizeBytes   = 64
	sm90WgmmaM            = 64
	sm90DefaultMKAlign    = 128
)

var (
	// ErrSM90HeuristicsDisabled is returned when the default-off SM90 heuristic
	// selector or precompile registry has not been explicitly enabled.
	ErrSM90HeuristicsDisabled = errors.New("compute: sm90 heuristics are disabled by default")
	// ErrSM90InvalidDesc reports an invalid GEMM descriptor or unsupported dtype/layout combination.
	ErrSM90InvalidDesc = errors.New("compute: invalid sm90 GEMM descriptor")
	// ErrSM90NoValidLayout reports that no tile/cluster/pipeline candidate satisfied SM90 constraints.
	ErrSM90NoValidLayout = errors.New("compute: no valid sm90 GEMM layout candidate")
)

// SM90GemmType classifies the GEMM grouping/layout family.
type SM90GemmType int

const (
	SM90GemmNormal SM90GemmType = iota
	SM90GemmMGroupedContiguous
	SM90GemmMGroupedMasked
	SM90GemmKGroupedContiguous
	SM90GemmBatched
	SM90GemmMGroupedContiguousWithPsumLayout
	SM90GemmKGroupedContiguousWithPsumLayout
)

func (g SM90GemmType) String() string {
	switch g {
	case SM90GemmNormal:
		return "normal"
	case SM90GemmMGroupedContiguous:
		return "m_grouped_contiguous"
	case SM90GemmMGroupedMasked:
		return "m_grouped_masked"
	case SM90GemmKGroupedContiguous:
		return "k_grouped_contiguous"
	case SM90GemmBatched:
		return "batched"
	case SM90GemmMGroupedContiguousWithPsumLayout:
		return "m_grouped_contiguous_psum"
	case SM90GemmKGroupedContiguousWithPsumLayout:
		return "k_grouped_contiguous_psum"
	default:
		return fmt.Sprintf("gemm(%d)", int(g))
	}
}

// SM90KernelType classifies scale-factor scheduling on SM90.
type SM90KernelType int

const (
	SM90Kernel1D1D SM90KernelType = iota
	SM90Kernel1D2D
	SM90KernelNoSF
)

func (k SM90KernelType) String() string {
	switch k {
	case SM90Kernel1D1D:
		return "1d1d"
	case SM90Kernel1D2D:
		return "1d2d"
	case SM90KernelNoSF:
		return "nosf"
	default:
		return fmt.Sprintf("kernel(%d)", int(k))
	}
}

// SM90DType identifies operand and output element types handled by SM90 heuristics.
type SM90DType int

const (
	SM90DTypeBF16 SM90DType = iota
	SM90DTypeFP8E4M3
	SM90DTypeFP32
)

func (d SM90DType) Bytes() int {
	switch d {
	case SM90DTypeBF16:
		return 2
	case SM90DTypeFP8E4M3:
		return 1
	case SM90DTypeFP32:
		return 4
	default:
		return 0
	}
}

func (d SM90DType) String() string {
	switch d {
	case SM90DTypeBF16:
		return "bf16"
	case SM90DTypeFP8E4M3:
		return "fp8_e4m3"
	case SM90DTypeFP32:
		return "fp32"
	default:
		return fmt.Sprintf("dtype(%d)", int(d))
	}
}

// SM90Major specifies K-major or MN-major tensor memory layout.
type SM90Major int

const (
	SM90MajorK SM90Major = iota
	SM90MajorMN
)

func (m SM90Major) String() string {
	switch m {
	case SM90MajorK:
		return "k"
	case SM90MajorMN:
		return "mn"
	default:
		return fmt.Sprintf("major(%d)", int(m))
	}
}

// SM90HeuristicsOptions configures the default-off SM90 heuristic selector and
// runtime alignment overrides. Zero value keeps Enabled=false.
type SM90HeuristicsOptions struct {
	Enabled                        bool
	MKAlignmentForContiguousLayout int
	BlockMMultipleOf               int
	BlockNMultipleOf               int
}

func (o SM90HeuristicsOptions) normalized() SM90HeuristicsOptions {
	if o.MKAlignmentForContiguousLayout <= 0 {
		o.MKAlignmentForContiguousLayout = sm90DefaultMKAlign
	}
	if o.BlockMMultipleOf <= 0 {
		o.BlockMMultipleOf = 1
	}
	if o.BlockNMultipleOf <= 0 {
		o.BlockNMultipleOf = 1
	}
	return o
}

// SM90GemmDesc describes a GEMM shape, types, majors, and SM count for SM90
// tile/cluster/pipeline heuristic selection.
type SM90GemmDesc struct {
	GemmType          SM90GemmType
	KernelType        SM90KernelType
	M                 int
	N                 int
	K                 int
	NumGroups         int
	ADType            SM90DType
	BDType            SM90DType
	CDDType           SM90DType
	MajorA            SM90Major
	MajorB            SM90Major
	WithAccumulation  bool
	NumSMs            int
	TCUtil            int
	CompiledDims      string
	ExpectedM         int
	ExpectedN         int
	ExpectedK         int
	ExpectedNumGroups int
}

func (d SM90GemmDesc) expectedM() int {
	if d.ExpectedM > 0 {
		return d.ExpectedM
	}
	return d.M
}

func (d SM90GemmDesc) expectedN() int {
	if d.ExpectedN > 0 {
		return d.ExpectedN
	}
	return d.N
}

func (d SM90GemmDesc) expectedK() int {
	if d.ExpectedK > 0 {
		return d.ExpectedK
	}
	return d.K
}

func (d SM90GemmDesc) expectedGroups() int {
	if d.ExpectedNumGroups > 0 {
		return d.ExpectedNumGroups
	}
	return d.NumGroups
}

func (d SM90GemmDesc) elementBits() int {
	if d.ADType == SM90DTypeBF16 {
		return 16
	}
	return 8
}

func (d SM90GemmDesc) validate() error {
	if d.M <= 0 || d.N <= 0 || d.K <= 0 || d.NumGroups <= 0 {
		return fmt.Errorf("%w: positive m/n/k/num_groups required (got %d,%d,%d,%d)",
			ErrSM90InvalidDesc, d.M, d.N, d.K, d.NumGroups)
	}
	if d.NumSMs <= 0 || d.NumSMs%2 != 0 {
		return fmt.Errorf("%w: num_sms must be positive and even (got %d)", ErrSM90InvalidDesc, d.NumSMs)
	}
	if d.ADType != d.BDType {
		return fmt.Errorf("%w: a_dtype %s != b_dtype %s", ErrSM90InvalidDesc, d.ADType, d.BDType)
	}
	switch d.ADType {
	case SM90DTypeBF16:
		if d.KernelType != SM90KernelNoSF {
			return fmt.Errorf("%w: bf16 operands require KernelNoSF (got %s)", ErrSM90InvalidDesc, d.KernelType)
		}
	case SM90DTypeFP8E4M3:
		if d.KernelType != SM90Kernel1D1D && d.KernelType != SM90Kernel1D2D {
			return fmt.Errorf("%w: fp8 operands require 1D1D or 1D2D kernel (got %s)", ErrSM90InvalidDesc, d.KernelType)
		}
	default:
		return fmt.Errorf("%w: unsupported operand dtype %s", ErrSM90InvalidDesc, d.ADType)
	}
	cdValid := d.CDDType == SM90DTypeBF16 || d.CDDType == SM90DTypeFP32 ||
		(d.CDDType == SM90DTypeFP8E4M3 && d.GemmType == SM90GemmBatched && !d.WithAccumulation)
	if !cdValid {
		return fmt.Errorf("%w: invalid output dtype %s for gemm=%s acc=%v",
			ErrSM90InvalidDesc, d.CDDType, d.GemmType, d.WithAccumulation)
	}
	if d.KernelType == SM90Kernel1D1D && d.CDDType == SM90DTypeFP32 &&
		(d.MajorA != SM90MajorK || d.MajorB != SM90MajorK) {
		return fmt.Errorf("%w: 1D1D fp32 output requires K-major A and B", ErrSM90InvalidDesc)
	}
	for i := 0; i < len(d.CompiledDims); i++ {
		switch d.CompiledDims[i] {
		case 'm', 'n', 'k':
		default:
			return fmt.Errorf("%w: invalid compiled_dims char %q", ErrSM90InvalidDesc, d.CompiledDims[i])
		}
	}
	return nil
}

// SM90Layout holds the tile and cluster geometry chosen for an SM90 GEMM.
type SM90Layout struct {
	SwapAB   int
	BlockM   int
	BlockN   int
	BlockK   int
	ClusterM int
	ClusterN int
}

// ClusterSize returns ClusterM * ClusterN.
func (l SM90Layout) ClusterSize() int {
	return l.ClusterM * l.ClusterN
}

// SM90StorageConfig holds TMA load/store tile extents and swizzle widths.
type SM90StorageConfig struct {
	LoadBlockM    int
	LoadBlockN    int
	StoreBlockM   int
	StoreBlockN   int
	SwizzleAMode  int
	SwizzleBMode  int
	SwizzleCDMode int
}

// SM90PipelineConfig holds SMEM bytes and pipeline stage counts.
type SM90PipelineConfig struct {
	SMEMSize          int
	NumStages         int
	NumTMAStoreStages int
}

// SM90LaunchConfig holds grid/cluster/thread counts for SM90 warp specialization.
type SM90LaunchConfig struct {
	NumSMs           int
	NumSMsPerCluster int
	NumThreads       int
	NumTMAThreads    int
	NumMathThreads   int
}

// SM90LayoutInfo summarizes wave efficiency and modeled L1/L2 cycles.
type SM90LayoutInfo struct {
	NumWaves     int
	LastWaveUtil int
	NumCycles    int64
	Layout       SM90Layout
}

// SM90GemmConfig is the complete heuristic selection result for one SM90 GEMM.
type SM90GemmConfig struct {
	Layout   SM90Layout
	Storage  SM90StorageConfig
	Pipeline SM90PipelineConfig
	Launch   SM90LaunchConfig
	Info     SM90LayoutInfo
}

func sm90CeilDiv(a, b int) int {
	return (a + b - 1) / b
}

func sm90Align(a, b int) int {
	return sm90CeilDiv(a, b) * b
}

func sm90GCD(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

func sm90LCM(a, b int) int {
	return (a / sm90GCD(a, b)) * b
}

func sm90SwizzleMode(blockSize, elemSize int) int {
	for _, mode := range []int{128, 64, 32, 16} {
		if (blockSize*elemSize)%mode == 0 {
			return mode
		}
	}
	return 0
}

func sm90StorageConfig(desc SM90GemmDesc, layout SM90Layout) SM90StorageConfig {
	loadM := layout.BlockM
	loadN := layout.BlockN
	storeM := layout.BlockM
	if desc.KernelType == SM90Kernel1D1D {
		storeM = sm90WgmmaM
	}
	storeN := layout.BlockN

	dimA := loadM
	if desc.MajorA == SM90MajorK {
		dimA = layout.BlockK
	}
	dimB := loadN
	if desc.MajorB == SM90MajorK {
		dimB = layout.BlockK
	}
	swizzleA := sm90SwizzleMode(dimA, desc.ADType.Bytes())
	swizzleB := sm90SwizzleMode(dimB, desc.BDType.Bytes())
	swizzleCD := 0
	if desc.CDDType != SM90DTypeFP32 {
		swizzleCD = sm90SwizzleMode(storeN, desc.CDDType.Bytes())
	}
	return SM90StorageConfig{
		LoadBlockM:    loadM,
		LoadBlockN:    loadN,
		StoreBlockM:   storeM,
		StoreBlockN:   storeN,
		SwizzleAMode:  swizzleA,
		SwizzleBMode:  swizzleB,
		SwizzleCDMode: swizzleCD,
	}
}

func sm90PipelineConfig(desc SM90GemmDesc, layout SM90Layout, storage SM90StorageConfig) SM90PipelineConfig {
	smemCD := sm90Align(layout.BlockM*layout.BlockN*desc.CDDType.Bytes(), 1024)
	smemBarriers := sm90MaxPipelineStages * 8 * 2

	smemAPerStage := storage.LoadBlockM * layout.BlockK * desc.ADType.Bytes()
	smemBPerStage := storage.LoadBlockN * layout.BlockK * desc.BDType.Bytes()

	smemSFAPerStage := 0
	if desc.KernelType != SM90KernelNoSF {
		smemSFAPerStage = sm90Align(layout.BlockM*4, 128)
	}
	smemSFBPerStage := 0
	if desc.KernelType == SM90Kernel1D1D {
		smemSFBPerStage = sm90Align(layout.BlockN*4, 128)
	}

	smemExtraSFB := 0
	if desc.KernelType == SM90Kernel1D2D {
		useUniformSFB := 2
		if layout.BlockK%layout.BlockN == 0 {
			useUniformSFB = 1
		}
		smemExtraSFB = sm90Align(sm90CeilDiv(desc.K, layout.BlockK)*4*useUniformSFB, 8)
	}

	smemTensormap := 0
	if desc.GemmType == SM90GemmKGroupedContiguous {
		smemTensormap = 4 * sm90TmaMapSizeBytes
	}

	smemExtra := smemCD + smemBarriers + smemExtraSFB + smemTensormap
	smemPerStage := smemAPerStage + smemBPerStage + smemSFAPerStage + smemSFBPerStage
	if smemPerStage <= 0 || smemCapacityRemaining(smemExtra) < 0 {
		return SM90PipelineConfig{}
	}
	numStages := smemCapacityRemaining(smemExtra) / smemPerStage
	if numStages > sm90MaxPipelineStages {
		numStages = sm90MaxPipelineStages
	}
	return SM90PipelineConfig{
		SMEMSize:          smemExtra + numStages*smemPerStage,
		NumStages:         numStages,
		NumTMAStoreStages: 2,
	}
}

func smemCapacityRemaining(smemExtra int) int {
	return sm90SmemCapacityBytes - smemExtra
}

func sm90LaunchConfig(desc SM90GemmDesc, layout SM90Layout) SM90LaunchConfig {
	const numTMAThreads = 128
	numMathThreads := 256
	if layout.BlockM <= 64 {
		numMathThreads = 128
	}
	return SM90LaunchConfig{
		NumSMs:           desc.NumSMs,
		NumSMsPerCluster: layout.ClusterSize(),
		NumThreads:       numTMAThreads + numMathThreads,
		NumTMAThreads:    numTMAThreads,
		NumMathThreads:   numMathThreads,
	}
}

func sm90LayoutInfo(desc SM90GemmDesc, layout SM90Layout) SM90LayoutInfo {
	numBlocks := sm90CeilDiv(desc.expectedM(), layout.BlockM) *
		sm90CeilDiv(desc.expectedN(), layout.BlockN) *
		desc.expectedGroups()
	numWaves := sm90CeilDiv(numBlocks, desc.NumSMs)
	lastBlocks := numBlocks % desc.NumSMs
	lastWaveUtil := lastBlocks
	if lastWaveUtil == 0 {
		lastWaveUtil = desc.NumSMs
	}

	l2BandwidthPerCycle := int(math.Min(64.0*float64(desc.NumSMs), 8e6/1.3e3))
	l1BandwidthPerCycle := 128 * desc.NumSMs

	elemAB := int64(desc.ADType.Bytes())
	elemCD := int64(desc.CDDType.Bytes())
	expectedK := int64(desc.expectedK())

	numBytesL2AB := expectedK * int64(layout.BlockM/layout.ClusterN+layout.BlockN/layout.ClusterM) * elemAB
	numBytesL1AB := expectedK * int64(layout.BlockM+layout.BlockN) * elemAB
	maxM := layout.BlockM
	if sm90WgmmaM > maxM {
		maxM = sm90WgmmaM
	}
	numBytesL1TC := expectedK*int64(maxM+layout.BlockN)*elemAB + int64(layout.BlockM*layout.BlockN)*elemCD
	accFactor := int64(1)
	if desc.WithAccumulation {
		accFactor = 2
	}
	numBytesL1L2CD := int64(layout.BlockM*layout.BlockN) * elemCD * accFactor

	numL2Cycles := (numBytesL2AB + numBytesL1L2CD) * int64(numBlocks) / int64(l2BandwidthPerCycle)
	numL1Cycles := (numBytesL1AB + numBytesL1TC + numBytesL1L2CD) * int64(numBlocks) / int64(l1BandwidthPerCycle)
	waveEfficiency := float32(numBlocks) / float32(numWaves*desc.NumSMs)
	maxCycles := numL1Cycles
	if numL2Cycles > maxCycles {
		maxCycles = numL2Cycles
	}
	numCycles := int64(float64(maxCycles) / float64(waveEfficiency))

	if layout.ClusterSize() > 1 && numWaves <= 1 {
		numCycles = math.MaxInt64
	}

	return SM90LayoutInfo{
		NumWaves:     numWaves,
		LastWaveUtil: lastWaveUtil,
		NumCycles:    numCycles,
		Layout:       layout,
	}
}

// SM90LayoutCandidates enumerates all valid SM90 tile/cluster candidates for desc.
func SM90LayoutCandidates(opts SM90HeuristicsOptions, desc SM90GemmDesc) ([]SM90Layout, error) {
	if !opts.Enabled {
		return nil, ErrSM90HeuristicsDisabled
	}
	if err := desc.validate(); err != nil {
		return nil, err
	}
	norm := opts.normalized()

	var blockMCandidates []int
	switch desc.GemmType {
	case SM90GemmNormal, SM90GemmBatched, SM90GemmKGroupedContiguous, SM90GemmKGroupedContiguousWithPsumLayout:
		blockMCandidates = []int{64, 128}
		if desc.M <= 16 {
			blockMCandidates = append(blockMCandidates, 16)
		}
		if desc.M <= 32 {
			blockMCandidates = append(blockMCandidates, 32)
		}
		if desc.CDDType != SM90DTypeFP32 {
			blockMCandidates = append(blockMCandidates, 256)
		}
	case SM90GemmMGroupedContiguous, SM90GemmMGroupedContiguousWithPsumLayout:
		blockMCandidates = []int{norm.MKAlignmentForContiguousLayout}
	case SM90GemmMGroupedMasked:
		blockMCandidates = []int{64, 128}
	default:
		return nil, fmt.Errorf("%w: unknown gemm type %s", ErrSM90InvalidDesc, desc.GemmType)
	}

	var blockNCandidates []int
	step := sm90LCM(16, norm.BlockNMultipleOf)
	start := step
	if desc.KernelType == SM90Kernel1D1D && desc.CDDType == SM90DTypeFP32 {
		start = 24
		blockNCandidates = append(blockNCandidates, 16)
	}
	end := 256
	if desc.KernelType == SM90Kernel1D2D {
		end = 192
	} else if desc.KernelType == SM90Kernel1D1D {
		end = 160
	}
	for i := start; i <= end; i += step {
		blockNCandidates = append(blockNCandidates, i)
	}

	blockK := 128 * 8 / desc.elementBits()
	disableMulticast := (desc.GemmType == SM90GemmKGroupedContiguous && desc.NumGroups > 4) ||
		desc.GemmType == SM90GemmBatched
	maxClusterDim := 2
	if disableMulticast {
		maxClusterDim = 1
	}

	var candidates []SM90Layout
	for clusterM := 1; clusterM <= maxClusterDim; clusterM++ {
		for clusterN := 1; clusterN <= maxClusterDim; clusterN++ {
			clusterSize := clusterM * clusterN
			if clusterSize > 2 || desc.NumSMs%clusterSize != 0 {
				continue
			}
			for _, blockM := range blockMCandidates {
				for _, blockN := range blockNCandidates {
					if desc.KernelType == SM90Kernel1D2D && blockN > blockK &&
						blockN%(blockN-blockK) != 0 && blockK%(blockN-blockK) != 0 {
						continue
					}
					if (desc.GemmType == SM90GemmMGroupedMasked ||
						desc.GemmType == SM90GemmMGroupedContiguousWithPsumLayout) &&
						sm90CeilDiv(desc.N, blockN)%clusterSize != 0 {
						continue
					}
					if blockM > 128 && blockN > 128 {
						continue
					}
					layout := SM90Layout{
						BlockM:   blockM,
						BlockN:   blockN,
						BlockK:   blockK,
						ClusterM: clusterM,
						ClusterN: clusterN,
					}
					storage := sm90StorageConfig(desc, layout)
					if storage.SwizzleAMode == 0 || storage.SwizzleBMode == 0 ||
						storage.SwizzleAMode%64 != 0 || storage.SwizzleBMode%64 != 0 {
						continue
					}
					pipeline := sm90PipelineConfig(desc, layout, storage)
					if pipeline.NumStages < 3 || (blockM*blockN < 128*192 && pipeline.NumStages < 4) {
						continue
					}
					candidates = append(candidates, layout)
				}
			}
		}
	}

	if len(candidates) == 0 {
		return nil, fmt.Errorf("%w: m=%d n=%d k=%d kernel=%s gemm=%s",
			ErrSM90NoValidLayout, desc.M, desc.N, desc.K, desc.KernelType, desc.GemmType)
	}
	return candidates, nil
}

// SelectSM90GemmConfig evaluates all SM90 layout candidates and returns the
// lowest-cycle configuration deterministically. Refuses when opts.Enabled is false.
func SelectSM90GemmConfig(opts SM90HeuristicsOptions, desc SM90GemmDesc) (SM90GemmConfig, error) {
	candidates, err := SM90LayoutCandidates(opts, desc)
	if err != nil {
		return SM90GemmConfig{}, err
	}
	bestLayout := candidates[0]
	bestInfo := sm90LayoutInfo(desc, bestLayout)
	for i := 1; i < len(candidates); i++ {
		info := sm90LayoutInfo(desc, candidates[i])
		if info.NumCycles < bestInfo.NumCycles {
			bestLayout = candidates[i]
			bestInfo = info
		}
	}
	storage := sm90StorageConfig(desc, bestLayout)
	pipeline := sm90PipelineConfig(desc, bestLayout, storage)
	launch := sm90LaunchConfig(desc, bestLayout)
	return SM90GemmConfig{
		Layout:   bestLayout,
		Storage:  storage,
		Pipeline: pipeline,
		Launch:   launch,
		Info:     bestInfo,
	}, nil
}

func compiledDim(dim int, axis byte, compiledDims string) int {
	if strings.IndexByte(compiledDims, axis) >= 0 {
		return dim
	}
	return 0
}

// SM90KernelSignature is the canonical specialization key for an SM90 compiled kernel.
// Uncompiled dimensions collapse to 0 so compatible dynamic shapes share one artifact.
type SM90KernelSignature struct {
	KernelTag         string
	GemmType          SM90GemmType
	KernelType        SM90KernelType
	CompiledM         int
	CompiledN         int
	CompiledK         int
	NumGroups         int
	BlockM            int
	BlockN            int
	BlockK            int
	SwizzleAMode      int
	SwizzleBMode      int
	SwizzleCDMode     int
	NumStages         int
	NumTMAThreads     int
	NumMathThreads    int
	ClusterSize       int
	ClusterNMulticast bool
	NumSMs            int
	ADType            SM90DType
	BDType            SM90DType
	CDDType           SM90DType
	MajorA            SM90Major
	MajorB            SM90Major
	WithAccumulation  bool
}

// Key returns the canonical deduplication key for manifest lookup.
func (s SM90KernelSignature) Key() string {
	return fmt.Sprintf(
		"%s|gt=%s|kt=%s|cm=%d|cn=%d|ck=%d|g=%d|bm=%d|bn=%d|bk=%d|sa=%d|sb=%d|scd=%d|st=%d|tt=%d|mt=%d|cs=%d|mcast=%t|sms=%d|a=%s|b=%s|cd=%s|ma=%s|mb=%s|acc=%t",
		s.KernelTag, s.GemmType, s.KernelType,
		s.CompiledM, s.CompiledN, s.CompiledK, s.NumGroups,
		s.BlockM, s.BlockN, s.BlockK,
		s.SwizzleAMode, s.SwizzleBMode, s.SwizzleCDMode,
		s.NumStages, s.NumTMAThreads, s.NumMathThreads,
		s.ClusterSize, s.ClusterNMulticast, s.NumSMs,
		s.ADType, s.BDType, s.CDDType, s.MajorA, s.MajorB, s.WithAccumulation,
	)
}

// BuildSM90KernelSignature derives the specialization key for desc + cfg.
func BuildSM90KernelSignature(kernelTag string, desc SM90GemmDesc, cfg SM90GemmConfig) SM90KernelSignature {
	return SM90KernelSignature{
		KernelTag:         kernelTag,
		GemmType:          desc.GemmType,
		KernelType:        desc.KernelType,
		CompiledM:         compiledDim(desc.M, 'm', desc.CompiledDims),
		CompiledN:         compiledDim(desc.N, 'n', desc.CompiledDims),
		CompiledK:         compiledDim(desc.K, 'k', desc.CompiledDims),
		NumGroups:         desc.NumGroups,
		BlockM:            cfg.Layout.BlockM,
		BlockN:            cfg.Layout.BlockN,
		BlockK:            cfg.Layout.BlockK,
		SwizzleAMode:      cfg.Storage.SwizzleAMode,
		SwizzleBMode:      cfg.Storage.SwizzleBMode,
		SwizzleCDMode:     cfg.Storage.SwizzleCDMode,
		NumStages:         cfg.Pipeline.NumStages,
		NumTMAThreads:     cfg.Launch.NumTMAThreads,
		NumMathThreads:    cfg.Launch.NumMathThreads,
		ClusterSize:       cfg.Layout.ClusterSize(),
		ClusterNMulticast: cfg.Layout.ClusterN > 1,
		NumSMs:            cfg.Launch.NumSMs,
		ADType:            desc.ADType,
		BDType:            desc.BDType,
		CDDType:           desc.CDDType,
		MajorA:            desc.MajorA,
		MajorB:            desc.MajorB,
		WithAccumulation:  desc.WithAccumulation,
	}
}

// SM90PrecompileEntry records one compiled specialization in the manifest.
type SM90PrecompileEntry struct {
	Signature SM90KernelSignature
	Config    SM90GemmConfig
}

// SM90PrecompileStats summarizes compile/hit/dedup counts for the registry.
type SM90PrecompileStats struct {
	CompileCalls    int
	CacheHits       int
	ManifestEntries int
}

// SM90PrecompileRegistry implements ahead-of-serve kernel specialization
// manifest accounting so runtime lookup short-circuits compile calls on hits.
// Default-off: opts.Enabled must be true.
type SM90PrecompileRegistry struct {
	mu           sync.Mutex
	opts         SM90HeuristicsOptions
	entries      map[string]SM90PrecompileEntry
	compileCalls int
	cacheHits    int
}

// NewSM90PrecompileRegistry creates a precompile manifest registry bound to opts.
func NewSM90PrecompileRegistry(opts SM90HeuristicsOptions) *SM90PrecompileRegistry {
	return &SM90PrecompileRegistry{
		opts:    opts,
		entries: make(map[string]SM90PrecompileEntry),
	}
}

// ResolveOrCompile selects the SM90 config for desc, checks the manifest by
// canonical kernel signature, and short-circuits compileFn on a registry hit.
func (r *SM90PrecompileRegistry) ResolveOrCompile(
	kernelTag string,
	desc SM90GemmDesc,
	compileFn func(SM90KernelSignature) error,
) (SM90PrecompileEntry, bool, error) {
	if r == nil {
		return SM90PrecompileEntry{}, false, ErrSM90HeuristicsDisabled
	}
	if !r.opts.Enabled {
		return SM90PrecompileEntry{}, false, ErrSM90HeuristicsDisabled
	}
	if kernelTag == "" {
		return SM90PrecompileEntry{}, false, fmt.Errorf("%w: empty kernel tag", ErrSM90InvalidDesc)
	}
	cfg, err := SelectSM90GemmConfig(r.opts, desc)
	if err != nil {
		return SM90PrecompileEntry{}, false, err
	}
	sig := BuildSM90KernelSignature(kernelTag, desc, cfg)
	key := sig.Key()

	r.mu.Lock()
	defer r.mu.Unlock()
	if entry, ok := r.entries[key]; ok {
		r.cacheHits++
		return entry, true, nil
	}
	if compileFn != nil {
		if err := compileFn(sig); err != nil {
			return SM90PrecompileEntry{}, false, err
		}
	}
	entry := SM90PrecompileEntry{
		Signature: sig,
		Config:    cfg,
	}
	r.entries[key] = entry
	r.compileCalls++
	return entry, false, nil
}

// Precompile populates the manifest over a batch of serving shapes, deduplicating
// identical specializations before serve admission.
func (r *SM90PrecompileRegistry) Precompile(
	kernelTag string,
	descs []SM90GemmDesc,
	compileFn func(SM90KernelSignature) error,
) (SM90PrecompileStats, error) {
	if r == nil || !r.opts.Enabled {
		return SM90PrecompileStats{}, ErrSM90HeuristicsDisabled
	}
	for i, d := range descs {
		if _, _, err := r.ResolveOrCompile(kernelTag, d, compileFn); err != nil {
			return r.Stats(), fmt.Errorf("precompile shape[%d]: %w", i, err)
		}
	}
	return r.Stats(), nil
}

// Manifest returns a deterministically ordered snapshot of all compiled entries.
func (r *SM90PrecompileRegistry) Manifest() []SM90PrecompileEntry {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]SM90PrecompileEntry, 0, len(r.entries))
	for _, e := range r.entries {
		out = append(out, e)
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].Signature.Key() < out[j].Signature.Key()
	})
	return out
}

// Stats returns the current compile, cache-hit, and manifest entry counts.
func (r *SM90PrecompileRegistry) Stats() SM90PrecompileStats {
	if r == nil {
		return SM90PrecompileStats{}
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return SM90PrecompileStats{
		CompileCalls:    r.compileCalls,
		CacheHits:       r.cacheHits,
		ManifestEntries: len(r.entries),
	}
}
