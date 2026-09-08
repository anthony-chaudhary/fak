package amdgpu

import (
	"bytes"
	"compress/zlib"
	"context"
	"crypto/sha1"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"hash/crc32"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
)

func TestStrixGitObjectSnapshotFreezesOpenedHandlesWithoutLivePathFallback(t *testing.T) {
	const packID = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
	makeLoose := func(t *testing.T, body []byte) (string, []byte) {
		t.Helper()
		canonical := []byte("blob " + strconv.Itoa(len(body)) + "\x00" + string(body))
		oid := sha1.Sum(canonical)
		var compressed bytes.Buffer
		zw := zlib.NewWriter(&compressed)
		if _, err := zw.Write(canonical); err != nil {
			t.Fatal(err)
		}
		if err := zw.Close(); err != nil {
			t.Fatal(err)
		}
		hexOID := hex.EncodeToString(oid[:])
		return filepath.Join("objects", hexOID[:2], hexOID[2:]), compressed.Bytes()
	}
	makePackPair := func(t *testing.T) ([]byte, []byte) {
		t.Helper()
		body := []byte("pack-view")
		canonical := []byte("blob 9\x00pack-view")
		oid := sha1.Sum(canonical)
		var compressed bytes.Buffer
		zw := zlib.NewWriter(&compressed)
		if _, err := zw.Write(body); err != nil {
			t.Fatal(err)
		}
		if err := zw.Close(); err != nil {
			t.Fatal(err)
		}
		entry := append([]byte{0x39}, compressed.Bytes()...)
		pack := append([]byte("PACK"), make([]byte, 8)...)
		binary.BigEndian.PutUint32(pack[4:8], 2)
		binary.BigEndian.PutUint32(pack[8:12], 1)
		pack = append(pack, entry...)
		packChecksum := sha1.Sum(pack)
		pack = append(pack, packChecksum[:]...)

		idx := append([]byte{0xff, 0x74, 0x4f, 0x63}, make([]byte, 4+256*4)...)
		binary.BigEndian.PutUint32(idx[4:8], 2)
		for i := int(oid[0]); i < 256; i++ {
			binary.BigEndian.PutUint32(idx[8+i*4:12+i*4], 1)
		}
		idx = append(idx, oid[:]...)
		var word [4]byte
		binary.BigEndian.PutUint32(word[:], crc32.ChecksumIEEE(entry))
		idx = append(idx, word[:]...)
		binary.BigEndian.PutUint32(word[:], 12)
		idx = append(idx, word[:]...)
		idx = append(idx, packChecksum[:]...)
		idxChecksum := sha1.Sum(idx)
		idx = append(idx, idxChecksum[:]...)
		return pack, idx
	}
	looseRel, looseObject := makeLoose(t, []byte("loose-opened-view"))
	looseDir := filepath.Base(filepath.Dir(looseRel))
	packRel := filepath.Join("objects", "pack", "pack-"+packID+".pack")
	idxRel := filepath.Join("objects", "pack", "pack-"+packID+".idx")

	writeFile := func(t *testing.T, name string, body []byte) {
		t.Helper()
		if err := os.MkdirAll(filepath.Dir(name), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(name, body, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	newRepo := func(t *testing.T) string {
		t.Helper()
		root := filepath.Join(t.TempDir(), "repo")
		if err := os.MkdirAll(filepath.Join(root, ".git", "objects", "info"), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.MkdirAll(filepath.Join(root, ".git", "objects", "pack"), 0o700); err != nil {
			t.Fatal(err)
		}
		return root
	}
	seedAllowlisted := func(t *testing.T, root string, loose, pack, idx []byte) {
		t.Helper()
		writeFile(t, filepath.Join(root, ".git", looseRel), loose)
		writeFile(t, filepath.Join(root, ".git", packRel), pack)
		writeFile(t, filepath.Join(root, ".git", idxRel), idx)
	}
	readSnapshot := func(t *testing.T, snapshot *strixGitObjectSnapshot, rel string) []byte {
		t.Helper()
		body, err := os.ReadFile(filepath.Join(snapshot.gitDir(), rel))
		if err != nil {
			t.Fatal(err)
		}
		return body
	}
	assertRefusal := func(t *testing.T, err error, secrets ...string) {
		t.Helper()
		if err == nil {
			t.Fatal("snapshot unexpectedly admitted")
		}
		var refusal *strixGitSnapshotRefusal
		if !errors.As(err, &refusal) {
			t.Fatalf("error type = %T, want *strixGitSnapshotRefusal", err)
		}
		if refusal.token != strixGitSnapshotUnattestedToken {
			t.Fatalf("refusal token = %q, want %q", refusal.token, strixGitSnapshotUnattestedToken)
		}
		for _, secret := range secrets {
			if secret != "" && strings.Contains(err.Error(), secret) {
				t.Fatalf("refusal leaked %q: %v", secret, err)
			}
		}
	}
	assertNoSnapshot := func(t *testing.T, parent string) {
		t.Helper()
		entries, err := os.ReadDir(parent)
		if err != nil {
			t.Fatal(err)
		}
		if len(entries) != 0 {
			t.Fatalf("private snapshot was not cleaned: %v", entries)
		}
	}

	t.Run("loose paired pack index exact allowlist and publication freeze", func(t *testing.T) {
		root := newRepo(t)
		loose := append([]byte(nil), looseObject...)
		pack, idx := makePackPair(t)
		seedAllowlisted(t, root, loose, pack, idx)

		for rel, body := range map[string][]byte{
			filepath.Join(".git", "config"):                                      []byte("[include]\npath=/hostile\n"),
			filepath.Join(".git", "HEAD"):                                        []byte("ref: refs/heads/main\n"),
			filepath.Join(".git", "packed-refs"):                                 []byte("forbidden"),
			filepath.Join(".git", "info", "grafts"):                              []byte("forbidden"),
			filepath.Join(".git", "shallow"):                                     []byte("forbidden"),
			filepath.Join(".git", "refs", "replace", strings.Repeat("c", 40)):    []byte("forbidden"),
			filepath.Join(".git", "objects", "info", "alternates"):               []byte("forbidden"),
			filepath.Join(".git", "objects", "info", "http-alternates"):          []byte("forbidden"),
			filepath.Join(".git", "objects", "info", "commit-graph"):             []byte("forbidden"),
			filepath.Join(".git", "objects", "info", "commit-graphs", "graph"):   []byte("forbidden"),
			filepath.Join(".git", "objects", "pack", "multi-pack-index"):         []byte("forbidden"),
			filepath.Join(".git", "objects", "pack", "pack-"+packID+".bitmap"):   []byte("forbidden"),
			filepath.Join(".git", "objects", "pack", "pack-"+packID+".rev"):      []byte("forbidden"),
			filepath.Join(".git", "objects", "pack", "pack-"+packID+".promisor"): []byte("forbidden"),
			filepath.Join(".git", "objects", looseDir, "not-an-object"):          []byte("forbidden"),
		} {
			writeFile(t, filepath.Join(root, rel), body)
		}

		snapshot, err := newStrixGitObjectSnapshot(context.Background(), root)
		if err != nil {
			t.Fatal(err)
		}
		snapshotPath := snapshot.gitDir()
		t.Cleanup(func() { _ = snapshot.close() })

		rootInfo, err := os.Stat(snapshotPath)
		if err != nil {
			t.Fatal(err)
		}
		if got := rootInfo.Mode().Perm(); runtime.GOOS != "windows" && got != 0o700 {
			t.Fatalf("snapshot root mode = %o, want 700", got)
		}
		var files []string
		err = filepath.WalkDir(snapshotPath, func(name string, entry os.DirEntry, walkErr error) error {
			if walkErr != nil {
				return walkErr
			}
			if entry.IsDir() {
				return nil
			}
			rel, relErr := filepath.Rel(snapshotPath, name)
			if relErr != nil {
				return relErr
			}
			files = append(files, filepath.ToSlash(rel))
			info, infoErr := entry.Info()
			if infoErr != nil {
				return infoErr
			}
			if !info.Mode().IsRegular() || (runtime.GOOS != "windows" && info.Mode().Perm() != 0o600) {
				t.Fatalf("snapshot file %q mode = %s, want regular 0600", rel, info.Mode())
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		sort.Strings(files)
		wantFiles := []string{
			"config",
			filepath.ToSlash(looseRel),
			filepath.ToSlash(idxRel),
			filepath.ToSlash(packRel),
		}
		sort.Strings(wantFiles)
		if strings.Join(files, "\n") != strings.Join(wantFiles, "\n") {
			t.Fatalf("snapshot files = %q, want exact allowlist %q", files, wantFiles)
		}
		if got := string(readSnapshot(t, snapshot, "config")); got != strixGitSnapshotConfig {
			t.Fatalf("config = %q, want literal %q", got, strixGitSnapshotConfig)
		}
		if !bytes.Equal(readSnapshot(t, snapshot, looseRel), loose) ||
			!bytes.Equal(readSnapshot(t, snapshot, packRel), pack) ||
			!bytes.Equal(readSnapshot(t, snapshot, idxRel), idx) {
			t.Fatal("snapshot did not preserve allowlisted source bytes")
		}

		moved := root + "-moved"
		if err := os.Rename(root, moved); err != nil {
			t.Fatal(err)
		}
		if err := os.RemoveAll(moved); err != nil {
			t.Fatal(err)
		}
		if got := readSnapshot(t, snapshot, looseRel); !bytes.Equal(got, loose) {
			t.Fatalf("published snapshot changed after live-root removal: %q", got)
		}
		if err := snapshot.close(); err != nil {
			t.Fatal(err)
		}
		if _, err := os.Stat(snapshotPath); !os.IsNotExist(err) {
			t.Fatalf("close left snapshot behind: %v", err)
		}
	})

	t.Run("live root rename after all source handles open", func(t *testing.T) {
		if runtime.GOOS == "windows" {
			t.Skip("Windows does not permit renaming this open directory handle")
		}
		root := newRepo(t)
		loose := append([]byte(nil), looseObject...)
		writeFile(t, filepath.Join(root, ".git", looseRel), loose)
		moved := root + "-moved"
		snapshot, err := snapshotStrixGitObjects(context.Background(), root, strixGitSnapshotOptions{
			afterOpen: func() error {
				if err := os.Rename(root, moved); err != nil {
					return err
				}
				return os.RemoveAll(moved)
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		defer snapshot.close()
		if got := readSnapshot(t, snapshot, looseRel); !bytes.Equal(got, loose) {
			t.Fatalf("snapshot bytes = %q, want opened view %q", got, loose)
		}
	})

	t.Run("source pathname A to B to A cannot replace opened file", func(t *testing.T) {
		if runtime.GOOS == "windows" {
			t.Skip("Windows does not permit renaming this open file handle")
		}
		root := newRepo(t)
		original := append([]byte(nil), looseObject...)
		pathA := filepath.Join(root, ".git", looseRel)
		pathB := pathA + ".held"
		writeFile(t, pathA, original)
		snapshot, err := snapshotStrixGitObjects(context.Background(), root, strixGitSnapshotOptions{
			afterOpen: func() error {
				if err := os.Rename(pathA, pathB); err != nil {
					return err
				}
				if err := os.WriteFile(pathA, []byte("attacker-path-A"), 0o600); err != nil {
					return err
				}
				if err := os.Remove(pathA); err != nil {
					return err
				}
				return os.Rename(pathB, pathA)
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		defer snapshot.close()
		if got := readSnapshot(t, snapshot, looseRel); !bytes.Equal(got, original) {
			t.Fatalf("snapshot followed replaced pathname: %q", got)
		}
	})

	t.Run("same inode mutation during copy accepts one view or refuses", func(t *testing.T) {
		if runtime.GOOS == "windows" {
			t.Skip("Windows does not permit rewriting this open file handle")
		}
		root := newRepo(t)
		before := append([]byte(nil), looseObject...)
		after := bytes.Repeat([]byte("B"), 17)
		pathA := filepath.Join(root, ".git", looseRel)
		writeFile(t, pathA, before)
		var once sync.Once
		snapshot, err := snapshotStrixGitObjects(context.Background(), root, strixGitSnapshotOptions{
			copyBufferBytes: 4,
			afterCopyChunk: func(rel string) error {
				if filepath.Clean(rel) == filepath.Clean(looseRel) {
					var hookErr error
					once.Do(func() { hookErr = os.WriteFile(pathA, after, 0o600) })
					return hookErr
				}
				return nil
			},
		})
		if err != nil {
			assertRefusal(t, err, root)
			return
		}
		defer snapshot.close()
		got := readSnapshot(t, snapshot, looseRel)
		if !bytes.Equal(got, before) && !bytes.Equal(got, after) {
			t.Fatalf("accepted torn same-inode view: %q", got)
		}
	})

	t.Run("equal length same inode overwrite with restored mtime never publishes torn bytes", func(t *testing.T) {
		if runtime.GOOS == "windows" {
			t.Skip("Windows does not permit rewriting this open file handle")
		}
		root := newRepo(t)
		before := append([]byte(nil), looseObject...)
		after := bytes.Repeat([]byte("B"), len(before))
		pathA := filepath.Join(root, ".git", looseRel)
		writeFile(t, pathA, before)
		originalInfo, err := os.Stat(pathA)
		if err != nil {
			t.Fatal(err)
		}
		var once sync.Once
		snapshot, err := snapshotStrixGitObjects(context.Background(), root, strixGitSnapshotOptions{
			copyBufferBytes: 4,
			afterCopyChunk: func(rel string) error {
				if filepath.Clean(rel) != filepath.Clean(looseRel) {
					return nil
				}
				var hookErr error
				once.Do(func() {
					if hookErr = os.WriteFile(pathA, after, 0o600); hookErr == nil {
						hookErr = os.Chtimes(pathA, originalInfo.ModTime(), originalInfo.ModTime())
					}
				})
				return hookErr
			},
		})
		if err != nil {
			assertRefusal(t, err, root)
			return
		}
		defer snapshot.close()
		got := readSnapshot(t, snapshot, looseRel)
		if !bytes.Equal(got, before) && !bytes.Equal(got, after) {
			t.Fatalf("accepted torn equal-length view: %q", got)
		}
	})

	for _, tc := range []struct {
		name string
		seed func(*testing.T, string)
	}{
		{
			name: "symlink loose object",
			seed: func(t *testing.T, root string) {
				target := filepath.Join(root, "target")
				writeFile(t, target, []byte("target"))
				link := filepath.Join(root, ".git", looseRel)
				if err := os.MkdirAll(filepath.Dir(link), 0o700); err != nil {
					t.Fatal(err)
				}
				if err := os.Symlink(target, link); err != nil {
					t.Skipf("symlink unavailable: %v", err)
				}
			},
		},
		{
			name: "special nonregular loose object",
			seed: func(t *testing.T, root string) {
				if err := os.MkdirAll(filepath.Join(root, ".git", looseRel), 0o700); err != nil {
					t.Fatal(err)
				}
			},
		},
		{
			name: "unpaired pack",
			seed: func(t *testing.T, root string) {
				writeFile(t, filepath.Join(root, ".git", packRel), []byte("unpaired"))
			},
		},
		{
			name: "loose object intrinsic identity mismatch",
			seed: func(t *testing.T, root string) {
				writeFile(t, filepath.Join(root, ".git", "objects", "aa", strings.Repeat("a", 38)), looseObject)
			},
		},
		{
			name: "corrupt paired pack",
			seed: func(t *testing.T, root string) {
				pack, idx := makePackPair(t)
				pack[12] ^= 0xff
				writeFile(t, filepath.Join(root, ".git", packRel), pack)
				writeFile(t, filepath.Join(root, ".git", idxRel), idx)
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := newRepo(t)
			tc.seed(t, root)
			parent := t.TempDir()
			snapshot, err := snapshotStrixGitObjects(context.Background(), root, strixGitSnapshotOptions{tempParent: parent})
			if snapshot != nil {
				defer snapshot.close()
			}
			assertRefusal(t, err, root, parent)
			assertNoSnapshot(t, parent)
		})
	}

	for _, tc := range []struct {
		name string
		opts strixGitSnapshotOptions
	}{
		{name: "file count cap", opts: strixGitSnapshotOptions{maxFiles: 1}},
		{name: "per file cap", opts: strixGitSnapshotOptions{maxFileBytes: 3}},
		{name: "total byte cap", opts: strixGitSnapshotOptions{maxTotalBytes: 7}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := newRepo(t)
			writeFile(t, filepath.Join(root, ".git", looseRel), []byte("four"))
			writeFile(t, filepath.Join(root, ".git", "objects", "cc", strings.Repeat("c", 38)), []byte("four"))
			parent := t.TempDir()
			tc.opts.tempParent = parent
			snapshot, err := snapshotStrixGitObjects(context.Background(), root, tc.opts)
			if snapshot != nil {
				defer snapshot.close()
			}
			assertRefusal(t, err, root, parent)
			assertNoSnapshot(t, parent)
		})
	}

	t.Run("cancellation and refusal cleanup preserve source", func(t *testing.T) {
		root := newRepo(t)
		marker := filepath.Join(root, ".git", looseRel)
		writeFile(t, marker, []byte("source-must-survive"))
		parent := t.TempDir()
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		snapshot, err := snapshotStrixGitObjects(ctx, root, strixGitSnapshotOptions{tempParent: parent})
		if snapshot != nil {
			defer snapshot.close()
		}
		assertRefusal(t, err, root, parent)
		assertNoSnapshot(t, parent)
		if got, readErr := os.ReadFile(marker); readErr != nil || string(got) != "source-must-survive" {
			t.Fatalf("refusal changed source: body=%q err=%v", got, readErr)
		}
	})

	t.Run("mid-copy cancellation cleans unpublished snapshot", func(t *testing.T) {
		root := newRepo(t)
		writeFile(t, filepath.Join(root, ".git", looseRel), bytes.Repeat([]byte("c"), 64))
		parent := t.TempDir()
		ctx, cancel := context.WithCancel(context.Background())
		var once sync.Once
		snapshot, err := snapshotStrixGitObjects(ctx, root, strixGitSnapshotOptions{
			tempParent:      parent,
			copyBufferBytes: 4,
			afterCopyChunk: func(string) error {
				once.Do(cancel)
				return nil
			},
		})
		if snapshot != nil {
			defer snapshot.close()
		}
		assertRefusal(t, err, root, parent)
		assertNoSnapshot(t, parent)
	})

	t.Run("production seam rejects noncanonical repository roots", func(t *testing.T) {
		root := newRepo(t)
		for _, badRoot := range []string{"repo", root + string(filepath.Separator) + "."} {
			snapshot, err := newStrixGitObjectSnapshot(context.Background(), badRoot)
			if snapshot != nil {
				defer snapshot.close()
			}
			assertRefusal(t, err, root)
		}
		link := root + "-link"
		if err := os.Symlink(root, link); err != nil {
			t.Skipf("symlink unavailable: %v", err)
		}
		snapshot, err := newStrixGitObjectSnapshot(context.Background(), link)
		if snapshot != nil {
			defer snapshot.close()
		}
		assertRefusal(t, err, root, link)
	})

	t.Run("ignored-name flood consumes bounded enumeration budget", func(t *testing.T) {
		root := newRepo(t)
		for i := 0; i < 12; i++ {
			writeFile(t, filepath.Join(root, ".git", "objects", "ignored-"+strconv.Itoa(i)), []byte("ignored"))
		}
		parent := t.TempDir()
		snapshot, err := snapshotStrixGitObjects(context.Background(), root, strixGitSnapshotOptions{
			tempParent: parent,
			maxEntries: 4,
		})
		if snapshot != nil {
			defer snapshot.close()
		}
		assertRefusal(t, err, root, parent)
		assertNoSnapshot(t, parent)
	})

	t.Run("cancellation interrupts batched ignored-name enumeration", func(t *testing.T) {
		root := newRepo(t)
		for i := 0; i < 12; i++ {
			writeFile(t, filepath.Join(root, ".git", "objects", "ignored-"+strconv.Itoa(i)), []byte("ignored"))
		}
		parent := t.TempDir()
		ctx, cancel := context.WithCancel(context.Background())
		var once sync.Once
		snapshot, err := snapshotStrixGitObjects(ctx, root, strixGitSnapshotOptions{
			tempParent:          parent,
			readDirBatchEntries: 2,
			afterReadDirBatch: func() {
				once.Do(cancel)
			},
		})
		if snapshot != nil {
			defer snapshot.close()
		}
		assertRefusal(t, err, root, parent)
		assertNoSnapshot(t, parent)
	})

	t.Run("cleanup failure retains exact ownership for retry", func(t *testing.T) {
		root := newRepo(t)
		writeFile(t, filepath.Join(root, ".git", looseRel), looseObject)
		outside := filepath.Join(t.TempDir(), "outside-marker")
		writeFile(t, outside, []byte("keep"))
		calls := 0
		var owned string
		snapshot, err := snapshotStrixGitObjects(context.Background(), root, strixGitSnapshotOptions{
			removeAll: func(name string) error {
				calls++
				if name != owned {
					return errors.New("attempted cleanup outside owned root")
				}
				if calls == 1 {
					return errors.New("injected cleanup failure")
				}
				return os.RemoveAll(name)
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		owned = snapshot.gitDir()
		if err := snapshot.close(); err == nil {
			t.Fatal("first cleanup unexpectedly succeeded")
		}
		if snapshot.gitDir() != owned {
			t.Fatal("failed cleanup discarded owned root")
		}
		if _, err := os.Stat(owned); err != nil {
			t.Fatalf("failed cleanup removed owned root: %v", err)
		}
		if err := snapshot.close(); err != nil {
			t.Fatalf("cleanup retry: %v", err)
		}
		if snapshot.gitDir() != "" {
			t.Fatal("successful cleanup retained ownership")
		}
		if got, err := os.ReadFile(outside); err != nil || string(got) != "keep" {
			t.Fatalf("cleanup changed outside marker: body=%q err=%v", got, err)
		}
	})

	t.Run("unpublished cleanup failure is reported and retryable", func(t *testing.T) {
		root := newRepo(t)
		writeFile(t, filepath.Join(root, ".git", looseRel), looseObject)
		parent := t.TempDir()
		outside := filepath.Join(t.TempDir(), "outside-marker")
		writeFile(t, outside, []byte("keep"))
		calls := 0
		var attempted string
		snapshot, err := snapshotStrixGitObjects(context.Background(), root, strixGitSnapshotOptions{
			tempParent: parent,
			afterCopyChunk: func(rel string) error {
				if filepath.Clean(rel) == filepath.Clean(looseRel) {
					return errors.New("injected copy refusal")
				}
				return nil
			},
			removeAll: func(name string) error {
				calls++
				if calls == 1 {
					attempted = name
					return errors.New("injected cleanup failure")
				}
				if name != attempted {
					return errors.New("cleanup ownership changed")
				}
				return os.RemoveAll(name)
			},
		})
		assertRefusal(t, err, root, parent)
		var cleanupErr *strixGitSnapshotCleanupError
		if !errors.As(err, &cleanupErr) {
			t.Fatalf("cleanup failure was discarded: %T %v", err, err)
		}
		if snapshot == nil || snapshot.gitDir() == "" || snapshot.gitDir() != attempted {
			t.Fatal("unpublished cleanup failure did not return exact retry ownership")
		}
		if rel, relErr := filepath.Rel(parent, attempted); relErr != nil || rel == "." || !filepath.IsLocal(rel) {
			t.Fatalf("cleanup target %q is outside private parent %q", attempted, parent)
		}
		if err := snapshot.close(); err != nil {
			t.Fatalf("unpublished cleanup retry: %v", err)
		}
		assertNoSnapshot(t, parent)
		if got, err := os.ReadFile(outside); err != nil || string(got) != "keep" {
			t.Fatalf("cleanup changed outside marker: body=%q err=%v", got, err)
		}
	})
}
