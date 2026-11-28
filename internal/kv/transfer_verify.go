// Ported from github.com/SemiAnalysisAI/InferenceX
// (collectivex/bench/kv_workload.py and collectivex/bench/run_kv.py

package kv

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
)

// TransferSentinel is the byte a destination pool is wiped to before the
// verified transfer, so an entry the transfer never wrote cannot pass.
const TransferSentinel byte = 0x5A

const transferSaltMix uint64 = 0x5851F42D4C957F2D

// ErrTransferVerify is wrapped by every transfer-verification failure.
var ErrTransferVerify = errors.New("kv: transfer verification failed")

// TransferSaltMix spreads a rank salt across the word so pools painted with
// different salts never agree on any word index.
func TransferSaltMix(salt int64) uint64 {
	return (uint64(salt+1) * transferSaltMix) & 0x7FFFFFFFFFFFFFFF
}

// TransferPatternWord is word i of a pool painted with salt.
func TransferPatternWord(i uint64, salt int64) uint64 {
	return i ^ TransferSaltMix(salt)
}

// FillTransferPattern paints word i of words with i XOR TransferSaltMix(salt),
// so no two words of one pool are equal and a misplaced word is detectable.
func FillTransferPattern(words []uint64, salt int64) {
	mix := TransferSaltMix(salt)
	for i := range words {
		words[i] = uint64(i) ^ mix
	}
}

// WipeTransferSentinel sets every byte of words to TransferSentinel.
func WipeTransferSentinel(words []uint64) {
	const w = uint64(TransferSentinel) * 0x0101010101010101
	for i := range words {
		words[i] = w
	}
}

// TransferGeometry is the shape both sides of a transfer derive independently:
// a pool of PoolRows rows of RowBytes, requests of Descs rows each, and
// TableSets disjoint sets of BatchMax requests.
type TransferGeometry struct {
	Preset     string
	ISL        int
	PageTokens int
	RowBytes   int
	Descs      int
	BatchMax   int
	TableSets  int
	PoolRows   int
}

// NewTransferGeometry sizes the pool for tableSets disjoint sets of batchMax
// requests plus head-room, as the upstream planner does. descs is the number
// of row descriptors one request moves.
func NewTransferGeometry(preset string, isl, pageTokens, rowBytes, descs, batchMax, tableSets int, poolSlack float64) (TransferGeometry, error) {
	if rowBytes <= 0 || rowBytes%8 != 0 || descs <= 0 || batchMax <= 0 || tableSets <= 0 {
		return TransferGeometry{}, fmt.Errorf("%w: transfer plan rowBytes=%d descs=%d batchMax=%d tableSets=%d",
			ErrInvalidConfig, rowBytes, descs, batchMax, tableSets)
	}
	perRequest := math.Max(poolSlack, float64(batchMax)*1.25)
	return TransferGeometry{
		Preset:     preset,
		ISL:        isl,
		PageTokens: pageTokens,
		RowBytes:   rowBytes,
		Descs:      descs,
		BatchMax:   batchMax,
		TableSets:  tableSets,
		PoolRows:   int(float64(descs*tableSets)*perRequest) + 8,
	}, nil
}

// PoolWords is the pool length in 8-byte words.
func (p TransferGeometry) PoolWords() int { return p.PoolRows * p.RowBytes / 8 }

// TransferSide names whose block table a seed derives.
type TransferSide string

const (
	TransferSideLocal  TransferSide = "local"
	TransferSideRemote TransferSide = "remote"
)

// TransferTableSeed lets both ranks derive both sides' tables from the plan
// and the sweep seed, with no metadata exchange.
func TransferTableSeed(p TransferGeometry, side TransferSide, seed int64) int64 {
	base := int64(p.ISL)*31 + int64(p.PageTokens) + int64(len(p.Preset))*7 + seed*7919
	if side == TransferSideLocal {
		return base + 1000
	}
	return base
}

// TransferBlockTable is one request's row ids: a slice of one seeded
// permutation of the pool's rows. Every (tableSet, request) slices a disjoint
// range, as a real allocator's live requests never alias blocks.
func TransferBlockTable(p TransferGeometry, seed int64, request, tableSet int) ([]int64, error) {
	low := (tableSet*p.BatchMax + request) * p.Descs
	if request < 0 || tableSet < 0 || low+p.Descs > p.PoolRows {
		return nil, fmt.Errorf("%w: pool too small for table set %d request %d", ErrInvalidConfig, tableSet, request)
	}
	rng := rand.New(rand.NewPCG(uint64(seed), transferSaltMix))
	perm := rng.Perm(p.PoolRows)
	rows := make([]int64, p.Descs)
	for i := range rows {
		rows[i] = int64(perm[low+i])
	}
	return rows, nil
}

// TransferEntries are byte offsets and sizes, one per descriptor.
type TransferEntries struct {
	Local, Remote, Sizes []uint64
}

// NixlTransferEntries posts one whole-row descriptor per block id.
func NixlTransferEntries(p TransferGeometry, localRows, remoteRows []int64) (TransferEntries, error) {
	if len(localRows) != p.Descs || len(remoteRows) != p.Descs {
		return TransferEntries{}, fmt.Errorf("%w: tables have %d/%d rows, plan wants %d",
			ErrInvalidConfig, len(localRows), len(remoteRows), p.Descs)
	}
	row := uint64(p.RowBytes)
	e := TransferEntries{
		Local:  make([]uint64, p.Descs),
		Remote: make([]uint64, p.Descs),
		Sizes:  make([]uint64, p.Descs),
	}
	for i := range localRows {
		e.Local[i] = uint64(localRows[i]) * row
		e.Remote[i] = uint64(remoteRows[i]) * row
		e.Sizes[i] = row
	}
	return e, nil
}

var transferProbeNames = [3]string{"first", "interior", "last"}

// VerifyTransferEntries checks that each destination entry holds its source
// entry: every entry is probed at its first, last and one seeded interior
// word, each expected to equal the source word index XOR the source salt mix.
// Offsets and sizes are bytes. A probe outside dst fails closed.
func VerifyTransferEntries(dst []uint64, dstOffsets, srcOffsets, sizes []uint64, srcSalt int64, seed int64) error {
	n := len(sizes)
	if len(dstOffsets) != n || len(srcOffsets) != n {
		return fmt.Errorf("%w: %d dst / %d src offsets for %d sizes", ErrTransferVerify, len(dstOffsets), len(srcOffsets), n)
	}
	rng := rand.New(rand.NewPCG(uint64(seed), transferSaltMix))
	mix := TransferSaltMix(srcSalt)
	var first error
	mismatched := 0
	for e := 0; e < n; e++ {
		d, s, words := dstOffsets[e]/8, srcOffsets[e]/8, sizes[e]/8
		u := rng.Float64()
		if words == 0 {
			return fmt.Errorf("%w: entry=%d has no whole word (size=%d)", ErrTransferVerify, e, sizes[e])
		}
		interior := uint64(u*float64(max(int64(words)-2, 1))) + 1
		within := [3]uint64{0, min(interior, words-1), words - 1}
		for k, w := range within {
			idx := d + w
			if idx >= uint64(len(dst)) {
				return fmt.Errorf("%w: entry=%d probe=%s dst_offset=%d beyond pool of %d words",
					ErrTransferVerify, e, transferProbeNames[k], d*8, len(dst))
			}
			expected := (s + w) ^ mix
			if got := dst[idx]; got != expected {
				mismatched++
				if first == nil {
					first = fmt.Errorf("%w: entry=%d probe=%s dst_offset=%d src_offset=%d expected=%d got=%d",
						ErrTransferVerify, e, transferProbeNames[k], d*8, s*8, expected, got)
				}
			}
		}
	}
	if first != nil {
		return fmt.Errorf("%w mismatched=%d/%d", first, mismatched, 3*n)
	}
	return nil
}

// VerifyTransferBulk checks a contiguous range at the same offsets on both
// sides: its first, last and samples seeded words must hold the source pattern.
func VerifyTransferBulk(dst []uint64, nbytes uint64, srcSalt int64, seed int64, samples int) error {
	words := nbytes / 8
	if words == 0 || words > uint64(len(dst)) {
		return fmt.Errorf("%w: bulk of %d words over pool of %d", ErrTransferVerify, words, len(dst))
	}
	rng := rand.New(rand.NewPCG(uint64(seed), transferSaltMix))
	idx := []uint64{0, words - 1}
	for range samples {
		idx = append(idx, rng.Uint64N(words))
	}
	slices.Sort(idx)
	idx = slices.Compact(idx)
	mix := TransferSaltMix(srcSalt)
	var first error
	mismatched := 0
	for _, i := range idx {
		if expected, got := i^mix, dst[i]; got != expected {
			mismatched++
			if first == nil {
				first = fmt.Errorf("%w: word=%d expected=%d got=%d", ErrTransferVerify, i, expected, got)
			}
		}
	}
	if first != nil {
		return fmt.Errorf("%w mismatched=%d/%d", first, mismatched, len(idx))
	}
	return nil
}

// TransferOp is the direction of a verified transfer: pull lands on the
// local pool, push on the remote one.
type TransferOp string

const (
	TransferPull TransferOp = "pull"
	TransferPush TransferOp = "push"
)

// VerifyTransferRequests checks every request of a burst against its own
// tables. localTables[r] and remoteTables[r] are request r's row ids.
func VerifyTransferRequests(p TransferGeometry, op TransferOp, dst []uint64, localTables, remoteTables [][]int64, srcSalt, seed int64) error {
	if len(localTables) != len(remoteTables) {
		return fmt.Errorf("%w: %d local vs %d remote tables", ErrTransferVerify, len(localTables), len(remoteTables))
	}
	for r := range localTables {
		e, err := NixlTransferEntries(p, localTables[r], remoteTables[r])
		if err != nil {
			return fmt.Errorf("%w: request=%d %v", ErrTransferVerify, r, err)
		}
		dstOff, srcOff := e.Remote, e.Local
		switch op {
		case TransferPull:
			dstOff, srcOff = e.Local, e.Remote
		case TransferPush:
		default:
			return fmt.Errorf("%w: unknown op %q", ErrTransferVerify, op)
		}
		if err := VerifyTransferEntries(dst, dstOff, srcOff, e.Sizes, srcSalt, seed*64+int64(r)); err != nil {
			return fmt.Errorf("request=%d %w", r, err)
		}
	}
	return nil
}

// RunVerifiedTransfer wipes dst to TransferSentinel, runs transfer, then
// verify; a transfer error fails the case before any verification, and a
// panicking verify fails it rather than escaping.
func RunVerifiedTransfer(dst []uint64, transfer func() error, verify func() error) (err error) {
	WipeTransferSentinel(dst)
	if terr := transfer(); terr != nil {
		return fmt.Errorf("%w: transfer raised %v", ErrTransferVerify, terr)
	}
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("%w: verify raised %v", ErrTransferVerify, r)
		}
	}()
	return verify()
}
