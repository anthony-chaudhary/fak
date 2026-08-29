package model

import (
	"bytes"
	"hash/fnv"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"testing"
)

const (
	qwen38PagedSwapDogfoodEnv        = "FAK_QWEN38_SWAP_DOGFOOD"
	qwen38PagedSwapDogfoodTargetSize = 158488532
)

type qwen38PagedSwapDogfoodFile struct {
	path string
	size int
}

// TestQwen38PagedSwapDogfoodRepoRoundTrip is opt-in because it intentionally
// materializes the production-sized Qwen3.8 swap envelope. Its state is derived
// from the current tracked repository rather than a fixed synthetic tensor.
func TestQwen38PagedSwapDogfoodRepoRoundTrip(t *testing.T) {
	if os.Getenv(qwen38PagedSwapDogfoodEnv) != "1" {
		t.Skip("set " + qwen38PagedSwapDogfoodEnv + "=1 to run the production-sized repository dogfood witness")
	}

	repo := qwen38PagedSwapDogfoodRepoRoot(t)
	files, sourceBytes, sourceDigest := qwen38PagedSwapDogfoodManifest(t, repo)
	cfg := qwen38PagedSwapBenchConfig()
	cache := qwen38PagedSwapBenchCache(cfg, 8)
	qwen38PagedSwapDogfoodFill(cache, sourceDigest)

	blob, err := QwenHybridKVCacheToHost(cache, 4)
	if err != nil {
		t.Fatalf("encode repository-derived cache: %v", err)
	}
	if got := len(blob); got != qwen38PagedSwapDogfoodTargetSize {
		t.Fatalf("payload bytes = %d, want %d", got, qwen38PagedSwapDogfoodTargetSize)
	}
	restored, err := QwenHybridKVCacheFromHost(cfg, blob)
	if err != nil {
		t.Fatalf("decode repository-derived cache: %v", err)
	}
	qwen38PagedSwapDogfoodAssertEqual(t, cache, restored)

	t.Logf("repo_files=%d repo_bytes=%d repo_digest=%016x payload_bytes=%d round_trip_exact=true", len(files), sourceBytes, sourceDigest, len(blob))
}

func qwen38PagedSwapDogfoodRepoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatalf("repository root not found from %q", dir)
		}
		dir = parent
	}
}

func qwen38PagedSwapDogfoodManifest(t *testing.T, repo string) ([]qwen38PagedSwapDogfoodFile, int64, uint64) {
	t.Helper()
	gitDir := filepath.Join(repo, ".git")
	if data, err := os.ReadFile(gitDir); err == nil && bytes.HasPrefix(data, []byte("gitdir: ")) {
		gitDir = string(bytes.TrimSpace(bytes.TrimPrefix(data, []byte("gitdir: "))))
		if len(gitDir) >= 3 && gitDir[1] == ':' && (gitDir[2] == '/' || gitDir[2] == '\\') {
			drive := gitDir[0]
			if drive >= 'A' && drive <= 'Z' {
				drive += 'a' - 'A'
			}
			gitDir = "/mnt/" + string(drive) + "/" + filepath.ToSlash(gitDir[3:])
		} else if !filepath.IsAbs(gitDir) {
			gitDir = filepath.Join(repo, gitDir)
		}
	}
	gitDir = filepath.Clean(filepath.FromSlash(gitDir))
	cmd := exec.Command("git", "--git-dir="+gitDir, "--work-tree="+repo, "ls-files", "-z")
	cmd.Dir = repo
	listed, err := cmd.Output()
	if err != nil {
		t.Fatalf("list tracked repository files (git_dir=%q): %v", gitDir, err)
	}
	paths := bytes.Split(bytes.TrimSuffix(listed, []byte{0}), []byte{0})
	files := make([]qwen38PagedSwapDogfoodFile, 0, len(paths))
	var total int64
	h := fnv.New64a()
	for _, rawPath := range paths {
		rel := string(rawPath)
		path := filepath.Join(repo, filepath.FromSlash(rel))
		info, err := os.Stat(path)
		if err != nil {
			t.Fatalf("stat tracked file %q: %v", rel, err)
		}
		if !info.Mode().IsRegular() || info.Size() > 128<<10 {
			continue
		}
		file, err := os.Open(path)
		if err != nil {
			t.Fatalf("open tracked file %q: %v", rel, err)
		}
		_, copyErr := io.Copy(h, file)
		closeErr := file.Close()
		if copyErr != nil {
			t.Fatalf("hash tracked file %q: %v", rel, copyErr)
		}
		if closeErr != nil {
			t.Fatalf("close tracked file %q: %v", rel, closeErr)
		}
		files = append(files, qwen38PagedSwapDogfoodFile{path: rel, size: int(info.Size())})
		total += info.Size()
	}
	sort.Slice(files, func(i, j int) bool { return files[i].path < files[j].path })
	for _, file := range files {
		_, _ = h.Write([]byte(file.path))
		_, _ = h.Write([]byte{0})
		_, _ = h.Write([]byte(strconv.Itoa(file.size)))
		_, _ = h.Write([]byte{0})
	}
	if len(files) == 0 || total == 0 {
		t.Fatal("repository inventory was empty")
	}
	return files, total, h.Sum64()
}

func qwen38PagedSwapDogfoodFill(cache *KVCache, seed uint64) {
	state := seed
	fill := func(values []float32) {
		for i := range values {
			state ^= state << 13
			state ^= state >> 7
			state ^= state << 17
			values[i] = math.Float32frombits(uint32(state) | 0x00800000)
		}
	}
	for layer := range cache.K {
		fill(cache.K[layer])
		fill(cache.Kraw[layer])
		fill(cache.V[layer])
	}
	for layer := range cache.linear.layers {
		for row := range cache.linear.layers[layer].conv {
			fill(cache.linear.layers[layer].conv[row])
		}
		for row := range cache.linear.layers[layer].recurrent {
			fill(cache.linear.layers[layer].recurrent[row])
		}
	}
}

func qwen38PagedSwapDogfoodAssertEqual(t *testing.T, want, got *KVCache) {
	t.Helper()
	if len(want.pos) != len(got.pos) {
		t.Fatalf("position count = %d, want %d", len(got.pos), len(want.pos))
	}
	for i := range want.pos {
		if got.pos[i] != want.pos[i] {
			t.Fatalf("position[%d] = %d, want %d", i, got.pos[i], want.pos[i])
		}
	}
	assertRows := func(label string, wantRows, gotRows [][]float32) {
		if len(wantRows) != len(gotRows) {
			t.Fatalf("%s row count = %d, want %d", label, len(gotRows), len(wantRows))
		}
		for row := range wantRows {
			if len(wantRows[row]) != len(gotRows[row]) {
				t.Fatalf("%s[%d] length = %d, want %d", label, row, len(gotRows[row]), len(wantRows[row]))
			}
			for i := range wantRows[row] {
				if math.Float32bits(gotRows[row][i]) != math.Float32bits(wantRows[row][i]) {
					t.Fatalf("%s[%d][%d] bits = %08x, want %08x", label, row, i, math.Float32bits(gotRows[row][i]), math.Float32bits(wantRows[row][i]))
				}
			}
		}
	}
	assertRows("K", want.K, got.K)
	assertRows("Kraw", want.Kraw, got.Kraw)
	assertRows("V", want.V, got.V)
	for layer := range want.linear.layers {
		assertRows("conv["+strconv.Itoa(layer)+"]", want.linear.layers[layer].conv, got.linear.layers[layer].conv)
		assertRows("recurrent["+strconv.Itoa(layer)+"]", want.linear.layers[layer].recurrent, got.linear.layers[layer].recurrent)
	}
}
