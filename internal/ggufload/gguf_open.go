package ggufload

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
)

// Open parses just the GGUF header/metadata at path and returns the parsed File,
// closing the underlying file before returning (it does not retain a tensor reader).
func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return Read(f)
}

// OpenWeights opens a GGUF checkpoint for weight reads, transparently assembling a
// merged WeightSource across every shard of a split checkpoint. It identifies the
// config-carrying shard by general.architecture (not the split.no index, whose base
// differs between HuggingFace and llama.cpp), so a caller may hand it any shard path.
func OpenWeights(path string) (*WeightSource, error) {
	f, gg, size, err := openAndRead(path)
	if err != nil {
		return nil, err
	}

	splitCount, hasSplit := gg.Uint64("split.count")
	if !hasSplit || splitCount <= 1 {
		// Single-file checkpoint.
		ws, err := NewWeightSource(gg, f, size)
		if err != nil {
			_ = f.Close()
			return nil, err
		}
		ws.closers = []io.Closer{f}
		return ws, nil
	}

	// Split checkpoint: shard 1 carries the model config (general.architecture
	// and friends); later shards carry only split.* metadata plus their tensor
	// subset. The config-carrying shard is identified by general.architecture
	// presence — NOT by split.no, which is 0-indexed in HuggingFace's split
	// writer and 1-indexed in llama.cpp's. If the caller handed us a later shard
	// (no architecture), close it and rebuild from shard 1's path.
	if _, hasArch := gg.String("general.architecture"); !hasArch {
		_ = f.Close()
		shard1, err := firstShardPath(path)
		if err != nil {
			return nil, fmt.Errorf("gguf: %s is a split shard but its name is not a shard path: %w", filepath.Base(path), err)
		}
		return openWeightsSplit(shard1, int(splitCount))
	}
	return openWeightsSplitFromFirst(path, int(splitCount), f, gg, size)
}

// openAndRead opens a GGUF file, parses its header, and returns the still-open
// file, the parsed File, and the file size. The caller owns the returned file
// and must Close it.
func openAndRead(path string) (*os.File, *File, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, 0, err
	}
	st, err := f.Stat()
	if err != nil {
		_ = f.Close()
		return nil, nil, 0, err
	}
	gg, err := Read(f)
	if err != nil {
		_ = f.Close()
		return nil, nil, 0, err
	}
	return f, gg, st.Size(), nil
}

// shardSuffixRe matches the "-NNNNN-of-MMMMM.gguf" shard suffix that the
// HuggingFace GGUF split writer produces (zero-padded to a fixed width). The
// two capture groups are the shard number and the total count, used to derive
// sibling shard paths while preserving the original padding width.
var shardSuffixRe = regexp.MustCompile(`-(\d+)-of-(\d+)\.gguf$`)

// firstShardPath rewrites a shard path's -N-of-M.gguf suffix to -1-of-M.gguf so
// the caller can open the config-carrying first shard.
func firstShardPath(path string) (string, error) {
	loc := shardSuffixRe.FindStringSubmatchIndex(path)
	if loc == nil {
		return "", fmt.Errorf("no -N-of-M.gguf shard suffix in %q", path)
	}
	// Rebuild the suffix with shard number 1, preserving the original width.
	numStr := path[loc[2]:loc[3]]
	countStr := path[loc[4]:loc[5]]
	width := len(numStr)
	prefix := path[:loc[2]]
	return fmt.Sprintf("%s%0*d-of-%s.gguf", prefix, width, 1, countStr), nil
}

// shardPaths expands shard 1's path into the full ordered shard-1..shard-N list,
// preserving the zero-padding width and total count encoded in the filename.
func shardPaths(shard1Path string, count int) ([]string, error) {
	loc := shardSuffixRe.FindStringSubmatchIndex(shard1Path)
	if loc == nil {
		return nil, fmt.Errorf("no -N-of-M.gguf shard suffix in %q", shard1Path)
	}
	numStr := shard1Path[loc[2]:loc[3]]
	countStr := shard1Path[loc[4]:loc[5]]
	width := len(numStr)
	prefix := shard1Path[:loc[2]]
	paths := make([]string, count)
	for i := 0; i < count; i++ {
		paths[i] = fmt.Sprintf("%s%0*d-of-%s.gguf", prefix, width, i+1, countStr)
	}
	return paths, nil
}

// openWeightsSplit opens shard 1 fresh and assembles the merged WeightSource.
func openWeightsSplit(shard1Path string, count int) (*WeightSource, error) {
	f, gg, size, err := openAndRead(shard1Path)
	if err != nil {
		return nil, err
	}
	// The config carrier is identified by general.architecture, not split.no
	// (whose index base differs between HuggingFace and llama.cpp writers).
	if _, ok := gg.String("general.architecture"); !ok {
		_ = f.Close()
		return nil, fmt.Errorf("gguf: %s expected to be the config-carrying shard 1, but general.architecture is absent", shard1Path)
	}
	return openWeightsSplitFromFirst(shard1Path, count, f, gg, size)
}

// validShardNo reports whether declared split.no is consistent with shard i
// (1-indexed by filename) under either convention: HuggingFace writes a 0-indexed
// split.no (so shard i => i-1) and llama.cpp writes a 1-indexed one (shard i => i).
// An absent split.no is treated as consistent (older/unknown writers).
func validShardNo(declared uint64, present bool, i int) bool {
	if !present {
		return true
	}
	return int(declared) == i-1 || int(declared) == i
}

// openWeightsSplitFromFirst assembles the merged WeightSource given an
// already-open shard 1. It opens shards 2..N, merges their tensor directories,
// and records which shard reader serves each tensor.
func openWeightsSplitFromFirst(shard1Path string, count int, shard1File *os.File, shard1GG *File, shard1Size int64) (*WeightSource, error) {
	paths, err := shardPaths(shard1Path, count)
	if err != nil {
		_ = shard1File.Close()
		return nil, err
	}
	if len(paths) == 0 || paths[0] != shard1Path {
		_ = shard1File.Close()
		return nil, fmt.Errorf("gguf: shard path derivation mismatch (%s vs %s)", paths[0], shard1Path)
	}

	// Merge view: shard 1's metadata/config + tensors from every shard, in order.
	tensors := make([]TensorInfo, 0, len(shard1GG.Tensors))
	readerFor := make([]io.ReaderAt, 0, len(shard1GG.Tensors))
	sizeFor := make([]int64, 0, len(shard1GG.Tensors))
	seen := make(map[string]bool, len(shard1GG.Tensors))
	for _, t := range shard1GG.Tensors {
		if seen[t.Name] {
			_ = shard1File.Close()
			return nil, fmt.Errorf("gguf: duplicate tensor %s within shard 1", t.Name)
		}
		seen[t.Name] = true
		tensors = append(tensors, t)
		readerFor = append(readerFor, shard1File)
		sizeFor = append(sizeFor, shard1Size)
	}
	closers := []io.Closer{shard1File}

	for i := 2; i <= count; i++ {
		p := paths[i-1]
		f, gg, sz, err := openAndRead(p)
		if err != nil {
			closeAll(closers)
			return nil, fmt.Errorf("gguf: open shard %d (%s): %w", i, p, err)
		}
		closers = append(closers, f)
		if no, ok := gg.Uint64("split.no"); !validShardNo(no, ok, i) {
			closeAll(closers)
			return nil, fmt.Errorf("gguf: shard %s declares split.no=%d, want %d or %d", p, no, i-1, i)
		}
		for _, t := range gg.Tensors {
			if seen[t.Name] {
				closeAll(closers)
				return nil, fmt.Errorf("gguf: duplicate tensor %s across shards", t.Name)
			}
			seen[t.Name] = true
			tensors = append(tensors, t)
			readerFor = append(readerFor, f)
			sizeFor = append(sizeFor, sz)
		}
	}

	merged := *shard1GG
	merged.Tensors = tensors
	ws, err := NewWeightSource(&merged, shard1File, shard1Size)
	if err != nil {
		closeAll(closers)
		return nil, err
	}
	ws.readerFor = readerFor
	ws.sizeFor = sizeFor
	ws.closers = closers
	return ws, nil
}

func closeAll(closers []io.Closer) {
	for _, c := range closers {
		_ = c.Close()
	}
}
