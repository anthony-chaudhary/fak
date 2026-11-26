package inklingmetal

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"math/bits"
	"os"
	"path/filepath"
)

// PrefixKey is everything besides the token ids that decides the state a
// session holds after a prompt: the files, what the kernels fix (layer passes,
// stored formats, the rows a prefill chunk rounds together), and the caller's
// own scope. Sessions repeat one another only under equal keys.
type PrefixKey struct {
	Model, Draft                  string
	ModelBytes, ModelModified     int64
	DraftModified                 int64
	Layers, KVFormat, StateFormat uint32
	ChunkRows, DraftWindow        uint32
	Scope                         string
}

func (k PrefixKey) String() string {
	data, err := json.Marshal(k)
	if err != nil {
		panic(err)
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// prefixFiles identifies the checkpoint files a model was opened from.
func prefixFiles(model, draft string) (PrefixKey, error) {
	var k PrefixKey
	path, err := filepath.EvalSymlinks(model)
	if err != nil {
		return k, err
	}
	if k.Model, err = filepath.Abs(path); err != nil {
		return k, err
	}
	info, err := os.Stat(k.Model)
	if err != nil {
		return k, err
	}
	k.ModelBytes, k.ModelModified = info.Size(), info.ModTime().UnixNano()
	if draft == "" {
		return k, nil
	}
	if k.Draft, err = filepath.Abs(draft); err != nil {
		return k, err
	}
	if info, err = os.Stat(filepath.Join(k.Draft, "config.json")); err != nil {
		return k, err
	}
	k.DraftModified = info.ModTime().UnixNano()
	return k, nil
}

// reuseChunks is how many leading prefill chunks a retained session keeps for
// a prompt of n tokens that shares its first shared tokens with the session.
// The session holds chunks full leading chunks and can restart after no fewer
// than oldest of them. Zero means a full prefill.
//
// Only whole chunks count: a chunk's kernels round its rows together, so rows
// prefilled in any other grouping differ from a new session's. At least one
// token stays to prefill, since its scores start the request. A draft keeps
// only its window of rows, which the retained session's later rows have
// overwritten, so the rows prefilled again must cover that window.
func reuseChunks(shared, n, oldest, chunks, chunkRows, draftWindow int) int {
	if chunkRows <= 0 || shared <= 0 || shared > n || oldest < 1 || draftWindow < 0 {
		return 0
	}
	keep := min(shared/chunkRows, chunks, (n-1)/chunkRows)
	if draftWindow > 0 {
		if n < draftWindow {
			return 0
		}
		keep = min(keep, (n-draftWindow)/chunkRows)
	}
	if keep < oldest {
		return 0
	}
	return keep
}

// prefixReserve sizes a new session past its first request, so a longer next
// prompt still fits the pages the session keeps. KV extents are allocated only
// as positions are reached.
func prefixReserve(capacity, limit int) int {
	if capacity <= 0 || capacity >= limit {
		return capacity
	}
	return min(1<<bits.Len(uint(capacity-1)), limit)
}
