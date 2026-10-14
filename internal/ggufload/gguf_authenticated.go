package ggufload

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"path/filepath"
	"strings"
	"unicode/utf8"
)

// ShardIdentity describes one complete GGUF file in an ordered shard manifest.
// Its JSON field order and names match mtpbench's artifact_shards identity.
type ShardIdentity struct {
	Path          string `json:"path"`
	Size          int64  `json:"size"`
	RawFileSHA256 string `json:"raw_file_sha256"`
}

// OpenWeightsAuthenticated authenticates an ordered GGUF manifest and assembles
// tensor routing from the exact readers used to parse and hash its shards. The
// caller must supply a trusted SHA-256 of json.Marshal(shards), independently of
// the paths being opened. A digest computed from untrusted current files supplies
// no prior identity. Paths must be absolute, clean, and ordered from shard one.
//
// The source owns all readers on success; every failure closes readers opened so
// far. Path replacement cannot substitute later tensor reads. Checkpoint files
// must remain immutable until Close: retained file descriptors and read-only
// mappings do not protect against concurrent in-place writes to the same file.
// Reader selection follows OpenWeights, including its mmap opt-in.
func OpenWeightsAuthenticated(ctx context.Context, shards []ShardIdentity, expectedSetSHA256 string) (*WeightSource, error) {
	return openWeightsAuthenticated(ctx, shards, expectedSetSHA256, ggufMmapEnabled(), openShardReader)
}

func openWeightsAuthenticated(ctx context.Context, shards []ShardIdentity, expectedSetSHA256 string, mmap bool,
	open func(string, bool) (io.ReaderAt, int64, io.Closer, []byte, error)) (*WeightSource, error) {
	result := openWeightsAuthenticatedWithResult(ctx, shards, expectedSetSHA256, mmap, open)
	return result.Source, result.Err
}

// This observation covers only ordinary returns and retained-reader releases.
// Opaque opener failures cannot establish retirement of unreturned resources.
// A visible Close error establishes failure even if other retirement is unknown.
type authenticatedWeightRetirement struct {
	Available bool
	Completed bool
	Err       error
}

// Successful transfer is not retirement. Panic produces no result and retains
// the legacy unwind limitations, including interruption by a closer panic.
type authenticatedWeightOpenResult struct {
	Source               *WeightSource
	Err                  error
	OwnershipTransferred bool
	Retirement           authenticatedWeightRetirement
}

const authenticatedWeightCloseErrorLimit = 8

var errAuthenticatedWeightCloseOverflow = errors.New("gguf: additional authenticated reader Close failures omitted")

func openWeightsAuthenticatedWithResult(ctx context.Context, shards []ShardIdentity, expectedSetSHA256 string, mmap bool,
	open func(string, bool) (io.ReaderAt, int64, io.Closer, []byte, error)) (result authenticatedWeightOpenResult) {
	var closers []io.Closer
	opaqueIncomplete := false
	defer func() {
		if result.OwnershipTransferred {
			return
		}
		result.Retirement.Available, result.Retirement.Completed = !opaqueIncomplete, !opaqueIncomplete
		var failures [authenticatedWeightCloseErrorLimit + 1]error
		count, overflow := 0, false
		for _, closer := range closers {
			if err := closer.Close(); err != nil {
				if count < authenticatedWeightCloseErrorLimit {
					failures[count] = err
					count++
				} else {
					overflow = true
				}
			}
		}
		if overflow {
			failures[count] = errAuthenticatedWeightCloseOverflow
			count++
		}
		if count != 0 {
			result.Retirement.Available, result.Retirement.Completed = true, false
			result.Retirement.Err = errors.Join(failures[:count]...)
			if result.Err == nil {
				result.Err = result.Retirement.Err
			} else {
				result.Err = errors.Join(result.Err, result.Retirement.Err)
			}
		}
	}()
	if err := ctx.Err(); err != nil {
		return authenticatedWeightOpenResult{Err: err}
	}
	if err := validateShardManifest(shards, expectedSetSHA256); err != nil {
		return authenticatedWeightOpenResult{Err: err}
	}
	var first *File
	var readers []io.ReaderAt
	var sizes []int64
	var mapped [][]byte
	var tensors []TensorInfo
	var firstReader io.ReaderAt
	var firstData []byte
	var indexBase uint64
	for i, shard := range shards {
		if err := ctx.Err(); err != nil {
			return authenticatedWeightOpenResult{Err: err}
		}
		opaqueIncomplete = true
		r, size, closer, data, err := open(shard.Path, mmap)
		if err != nil {
			return authenticatedWeightOpenResult{Err: fmt.Errorf("gguf: open authenticated shard %d: %w", i+1, err)}
		}
		opaqueIncomplete = false
		closers = append(closers, closer)
		if size != shard.Size {
			return authenticatedWeightOpenResult{Err: fmt.Errorf("gguf: shard %d size mismatch: expected %d, observed %d", i+1, shard.Size, size)}
		}
		stream := io.NewSectionReader(authenticatedReaderAt{ctx, r}, 0, size)
		hash := sha256.New()
		parsed, err := ReadSize(io.TeeReader(stream, hash), size)
		if err != nil {
			return authenticatedWeightOpenResult{Err: fmt.Errorf("gguf: parse authenticated shard %d: %w", i+1, err)}
		}
		if _, err := io.Copy(hash, stream); err != nil {
			return authenticatedWeightOpenResult{Err: fmt.Errorf("gguf: hash authenticated shard %d: %w", i+1, err)}
		}
		// Include parser prefetch and require the complete declared byte extent.
		consumed, _ := stream.Seek(0, io.SeekCurrent)
		if consumed != size {
			return authenticatedWeightOpenResult{Err: fmt.Errorf("gguf: shard %d truncated: expected %d bytes, read %d", i+1, size, consumed)}
		}
		if !strings.EqualFold(hex.EncodeToString(hash.Sum(nil)), shard.RawFileSHA256) {
			return authenticatedWeightOpenResult{Err: fmt.Errorf("gguf: shard %d raw file SHA-256 mismatch", i+1)}
		}
		if i == 0 {
			first, firstReader, firstData = parsed, r, data
			if arch, ok := parsed.String("general.architecture"); !ok || arch == "" {
				return authenticatedWeightOpenResult{Err: fmt.Errorf("gguf: authenticated shard one has no general.architecture")}
			}
			indexBase, _ = parsed.Uint64("split.no")
		}
		count, countOK := parsed.Uint64("split.count")
		_, countPresent := parsed.Metadata["split.count"]
		if (len(shards) > 1 || countPresent) && (!countOK || count != uint64(len(shards))) {
			return authenticatedWeightOpenResult{Err: fmt.Errorf("gguf: shard %d split.count disagrees with authenticated manifest", i+1)}
		}
		if len(shards) > 1 {
			no, ok := parsed.Uint64("split.no")
			if !ok || indexBase > 1 || no != indexBase+uint64(i) {
				return authenticatedWeightOpenResult{Err: fmt.Errorf("gguf: shard %d split.no disagrees with authenticated order", i+1)}
			}
		}
		for _, tensor := range parsed.Tensors {
			tensors = append(tensors, tensor)
			readers = append(readers, r)
			sizes = append(sizes, size)
			mapped = append(mapped, data)
		}
	}
	if err := ctx.Err(); err != nil {
		return authenticatedWeightOpenResult{Err: err}
	}
	merged := *first
	merged.Tensors = tensors
	ws, err := NewWeightSource(&merged, firstReader, shards[0].Size)
	if err != nil {
		return authenticatedWeightOpenResult{Err: err}
	}
	ws.readerFor, ws.sizeFor, ws.dataFor = readers, sizes, mapped
	// Split tensors use only their own mapping. A portable fallback on a later
	// shard must not inherit the first shard's mapped bytes.
	if len(shards) == 1 {
		ws.data = firstData
	}
	ws.closers = closers
	return authenticatedWeightOpenResult{Source: ws, OwnershipTransferred: true}
}

func validateShardManifest(shards []ShardIdentity, expectedSetSHA256 string) error {
	if len(shards) == 0 || !shardSHA256Valid(expectedSetSHA256) {
		return fmt.Errorf("gguf: authenticated shard manifest requires entries and a set SHA-256")
	}
	raw, err := json.Marshal(shards)
	if err != nil {
		return fmt.Errorf("gguf: encode authenticated shard manifest: %w", err)
	}
	sum := sha256.Sum256(raw)
	if !strings.EqualFold(hex.EncodeToString(sum[:]), expectedSetSHA256) {
		return fmt.Errorf("gguf: authenticated shard manifest SHA-256 mismatch")
	}
	var paths []string
	if count, ok := shardPathCount(shards[0].Path); ok && count != len(shards) {
		return fmt.Errorf("gguf: authenticated shard manifest filename count mismatch")
	}
	if len(shards) > 1 {
		count, ok := shardPathCount(shards[0].Path)
		if !ok || count != len(shards) {
			return fmt.Errorf("gguf: authenticated shard manifest filename count mismatch")
		}
		paths, err = shardPaths(shards[0].Path, len(shards))
		if err != nil {
			return err
		}
	}
	for i, shard := range shards {
		// JSON replaces invalid UTF-8. Reject it instead of letting distinct
		// filesystem path bytes share the same serialized manifest identity.
		if !utf8.ValidString(shard.Path) || !filepath.IsAbs(shard.Path) || filepath.Clean(shard.Path) != shard.Path || shard.Size <= 0 || !shardSHA256Valid(shard.RawFileSHA256) {
			return fmt.Errorf("gguf: invalid authenticated shard %d identity", i+1)
		}
		if len(paths) > 0 && shard.Path != paths[i] {
			return fmt.Errorf("gguf: authenticated shard %d path disagrees with ordered set", i+1)
		}
	}
	return nil
}

func shardSHA256Valid(value string) bool {
	if len(value) != sha256.Size*2 {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

type authenticatedReaderAt struct {
	ctx context.Context
	r   io.ReaderAt
}

func (r authenticatedReaderAt) ReadAt(p []byte, off int64) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	return r.r.ReadAt(p, off)
}
