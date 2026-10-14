package ggufload

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// fak-test:runtime fast est=100ms lane=default
func TestOpenWeightsAuthenticated(t *testing.T) {
	t.Parallel()
	fixture := func(t *testing.T, indexBase uint32) ([]ShardIdentity, [][]byte) {
		t.Helper()
		dir := t.TempDir()
		var shards []ShardIdentity
		var contents [][]byte
		for i := range 2 {
			data := writeSplitShard(t, uint32(i)+indexBase, 2, 2, i == 0, []splitTensor{{
				name: fmt.Sprintf("shard%d.weight", i), dims: []uint64{2}, typ: TensorF32, data: f32Payload(float32(i+1), float32(i+2)),
			}})
			// Exercise hashing beyond the parser's prefetched header buffer.
			data = append(data, bytes.Repeat([]byte{byte(i)}, 128<<10)...)
			path := filepath.Join(dir, fmt.Sprintf("tiny-%05d-of-00002.gguf", i+1))
			if err := os.WriteFile(path, data, 0o600); err != nil {
				t.Fatal(err)
			}
			sum := sha256.Sum256(data)
			shards = append(shards, ShardIdentity{path, int64(len(data)), hex.EncodeToString(sum[:])})
			contents = append(contents, data)
		}
		return shards, contents
	}
	digest := func(shards []ShardIdentity) string {
		raw, err := json.Marshal(shards)
		if err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(raw)
		return hex.EncodeToString(sum[:])
	}
	t.Run("public opener", func(t *testing.T) {
		shards, _ := fixture(t, 0)
		ws, err := OpenWeightsAuthenticated(context.Background(), shards, digest(shards))
		if err != nil {
			t.Fatal(err)
		}
		defer ws.Close()
		got, _, err := ws.TensorF32("shard1.weight")
		if err != nil || len(got) != 2 || got[0] != 2 || got[1] != 3 {
			t.Fatalf("second shard payload: %v, %v", got, err)
		}
	})
	for _, replace := range []bool{false, true} {
		for _, mmap := range []bool{false, true} {
			for _, base := range []uint32{0, 1} {
				t.Run(fmt.Sprintf("retained routing mmap=%v base=%d replace=%v", mmap, base, replace), func(t *testing.T) {
					if replace && runtime.GOOS == "windows" {
						t.Skip("Windows open file sharing forbids pathname replacement; ordinary routing remains covered")
					}
					shards, _ := fixture(t, base)
					var owned []*authenticatedTestCloser
					opener := func(path string, mapped bool) (io.ReaderAt, int64, io.Closer, []byte, error) {
						r, size, closer, data, err := openShardReader(path, mapped)
						if err != nil {
							return nil, 0, nil, nil, err
						}
						tracked := &authenticatedTestCloser{Closer: closer}
						owned = append(owned, tracked)
						// Replace the pathname after retaining its reader, before parse
						// and hash. Reopening it would observe invalid replacement bytes.
						if replace {
							if err := os.Rename(path, path+".original"); err != nil {
								t.Fatal(err)
							}
							if err := os.WriteFile(path, bytes.Repeat([]byte{'x'}, int(size)), 0o600); err != nil {
								t.Fatal(err)
							}
						}
						return r, size, tracked, data, nil
					}
					ws, err := openWeightsAuthenticated(context.Background(), shards, digest(shards), mmap, opener)
					if err != nil {
						t.Fatal(err)
					}
					t.Cleanup(func() { _ = ws.Close() })
					if len(owned) != 2 || len(ws.File.Tensors) != 2 {
						t.Fatalf("opened %d shards, routed %d tensors", len(owned), len(ws.File.Tensors))
					}
					for i := range 2 {
						got, _, err := ws.TensorF32(fmt.Sprintf("shard%d.weight", i))
						if err != nil || len(got) != 2 || got[0] != float32(i+1) || got[1] != float32(i+2) {
							t.Fatalf("shard %d retained payload: %v, %v", i, got, err)
						}
						if owned[i].closed != 0 {
							t.Fatal("reader closed before source ownership ended")
						}
					}
					if err := ws.Close(); err != nil {
						t.Fatal(err)
					}
					if err := ws.Close(); err != nil {
						t.Fatal(err)
					}
					for _, reader := range owned {
						if reader.closed != 1 {
							t.Fatalf("reader closed %d times, want 1", reader.closed)
						}
					}
				})
			}
		}
	}
	t.Run("manifest admission precedes opening", func(t *testing.T) {
		shards, _ := fixture(t, 0)
		pinned := digest(shards)
		for _, kind := range []string{"changed digest", "reordered", "missing", "duplicate", "invalid raw hash", "invalid path UTF-8"} {
			t.Run(kind, func(t *testing.T) {
				input := append([]ShardIdentity(nil), shards...)
				want := "manifest"
				switch kind {
				case "changed digest":
					input[1].RawFileSHA256 = strings.Repeat("0", 64)
				case "reordered":
					input[0], input[1] = input[1], input[0]
				case "missing":
					input = input[:1]
				case "duplicate":
					input[1] = input[0]
				case "invalid raw hash":
					input[1].RawFileSHA256 = "invalid"
					want = "identity"
				case "invalid path UTF-8":
					input[1].Path = strings.Replace(input[1].Path, "tiny-", "tiny-\xff", 1)
					want = "identity"
				}
				setHash := digest(input)
				if kind == "changed digest" {
					setHash = pinned
				}
				_, err := openWeightsAuthenticated(context.Background(), input, setHash, false,
					func(string, bool) (io.ReaderAt, int64, io.Closer, []byte, error) {
						t.Fatal("opened a shard before admitting the manifest")
						return nil, 0, nil, nil, nil
					})
				if err == nil || !strings.Contains(err.Error(), want) && !strings.Contains(err.Error(), "ordered set") {
					t.Fatalf("invalid manifest error = %v", err)
				}
			})
		}
	})
	t.Run("failure closes every retained reader", func(t *testing.T) {
		for _, kind := range []string{"size", "payload", "truncated", "parse", "split count", "split index", "duplicate tensor", "open", "cancel"} {
			t.Run(kind, func(t *testing.T) {
				shards, contents := fixture(t, 0)
				switch kind {
				case "payload":
					contents[1][len(contents[1])-1] ^= 1
				case "truncated":
					contents[1] = contents[1][:len(contents[1])-1]
				case "parse":
					contents[1][0] = 'x'
				case "split count", "split index", "duplicate tensor":
					count, index, name := uint32(2), uint32(1), "shard1.weight"
					if kind == "split count" {
						count = 3
					} else if kind == "split index" {
						index = 2
					} else {
						name = "shard0.weight"
					}
					contents[1] = writeSplitShard(t, index, count, 2, false, []splitTensor{{
						name: name, dims: []uint64{2}, typ: TensorF32, data: f32Payload(3, 4),
					}})
					shards[1].Size = int64(len(contents[1]))
					sum := sha256.Sum256(contents[1])
					shards[1].RawFileSHA256 = hex.EncodeToString(sum[:])
				}
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				var owned []*authenticatedTestCloser
				opened := 0
				ws, err := openWeightsAuthenticated(ctx, shards, digest(shards), false,
					func(string, bool) (io.ReaderAt, int64, io.Closer, []byte, error) {
						i := opened
						opened++
						if kind == "open" && i == 1 {
							return nil, 0, nil, nil, errors.New("synthetic open failure")
						}
						closer := &authenticatedTestCloser{}
						owned = append(owned, closer)
						size := shards[i].Size
						if kind == "size" && i == 1 {
							size++
						}
						if kind == "cancel" && i == 1 {
							cancel()
						}
						return bytes.NewReader(contents[i]), size, closer, nil, nil
					})
				if ws != nil || err == nil || opened != 2 {
					t.Fatalf("failure: source=%v, error=%v, opens=%d", ws, err, opened)
				}
				wantError := map[string]string{
					"size": "size mismatch", "payload": "SHA-256 mismatch", "truncated": "truncated",
					"parse": "bad magic", "split count": "split.count", "split index": "split.no",
					"duplicate tensor": "duplicate tensor", "open": "synthetic open failure", "cancel": "context canceled",
				}[kind]
				if !strings.Contains(err.Error(), wantError) {
					t.Fatalf("error %v does not identify %s", err, kind)
				}
				if kind == "cancel" && !errors.Is(err, context.Canceled) {
					t.Fatalf("cancellation lost: %v", err)
				}
				for _, reader := range owned {
					if reader.closed != 1 {
						t.Fatalf("reader closed %d times, want 1", reader.closed)
					}
				}
			})
		}
	})
}

type authenticatedTestCloser struct {
	io.Closer
	closed int
}

func (c *authenticatedTestCloser) Close() error {
	c.closed++
	if c.Closer != nil {
		return c.Closer.Close()
	}
	return nil
}
