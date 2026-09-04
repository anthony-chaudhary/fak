package snapshot

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"
)

// Manifest describes a snapshot directory.
type Manifest struct {
	Version       int      `json:"version"`
	ServerVersion string   `json:"server_version"`
	CreatedAt     string   `json:"created_at"`
	ShardCount    int      `json:"shard_count"`
	AllocatorMode string   `json:"allocator_mode"`
	TotalKeys     int64    `json:"total_keys"`
	TotalBytes    int64    `json:"total_bytes"`
	Files         []string `json:"files"`
}

// KVEntry is a single key-value pair with optional TTL.
type KVEntry struct {
	Key   []byte
	Value []byte
	TTLMs int64 // remaining ms from snapshot time; 0 = no expiry
}

// Writer writes snapshot data to a directory.
type Writer struct {
	dir string
}

// NewWriter creates a snapshot writer for the given directory.
func NewWriter(dir string) *Writer {
	return &Writer{dir: dir}
}

// WriteManifest writes the manifest file.
func (w *Writer) WriteManifest(m Manifest) error {
	if err := os.MkdirAll(w.dir, 0755); err != nil {
		return fmt.Errorf("create snapshot dir: %w", err)
	}
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(w.dir, "snapshot.json"), data, 0644)
}

// WriteShard writes entries for a single shard.
// Format: [keyLen:2][key][valueLen:4][value][ttlMs:8]... [0x0000] (sentinel)
func (w *Writer) WriteShard(shardID int, entries []KVEntry) error {
	if err := os.MkdirAll(w.dir, 0755); err != nil {
		return fmt.Errorf("create snapshot dir: %w", err)
	}
	path := filepath.Join(w.dir, fmt.Sprintf("shard-%d.dat", shardID))
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	var buf [8]byte
	for _, e := range entries {
		// keyLen (2 bytes)
		binary.LittleEndian.PutUint16(buf[:2], uint16(len(e.Key)))
		if _, err := f.Write(buf[:2]); err != nil {
			return err
		}
		// key
		if _, err := f.Write(e.Key); err != nil {
			return err
		}
		// valueLen (4 bytes)
		binary.LittleEndian.PutUint32(buf[:4], uint32(len(e.Value)))
		if _, err := f.Write(buf[:4]); err != nil {
			return err
		}
		// value
		if _, err := f.Write(e.Value); err != nil {
			return err
		}
		// ttlMs (8 bytes)
		binary.LittleEndian.PutUint64(buf[:8], uint64(e.TTLMs))
		if _, err := f.Write(buf[:8]); err != nil {
			return err
		}
	}
	// Sentinel: keyLen=0
	binary.LittleEndian.PutUint16(buf[:2], 0)
	_, err = f.Write(buf[:2])
	return err
}

// Reader reads snapshot data from a directory.
type Reader struct {
	dir string
}

// NewReader creates a snapshot reader.
func NewReader(dir string) *Reader {
	return &Reader{dir: dir}
}

// ReadManifest reads the manifest file.
func (r *Reader) ReadManifest() (Manifest, error) {
	data, err := os.ReadFile(filepath.Join(r.dir, "snapshot.json"))
	if err != nil {
		return Manifest{}, err
	}
	var m Manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return Manifest{}, err
	}
	return m, nil
}

// ReadShard reads all entries from a shard data file.
func (r *Reader) ReadShard(shardID int) ([]KVEntry, error) {
	path := filepath.Join(r.dir, fmt.Sprintf("shard-%d.dat", shardID))
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	return readEntries(f)
}

// ReadAllEntries reads entries from all shard files. Used when shard count differs.
func (r *Reader) ReadAllEntries() ([]KVEntry, error) {
	m, err := r.ReadManifest()
	if err != nil {
		return nil, err
	}
	var all []KVEntry
	for i := 0; i < m.ShardCount; i++ {
		entries, err := r.ReadShard(i)
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return nil, fmt.Errorf("read shard %d: %w", i, err)
		}
		all = append(all, entries...)
	}
	return all, nil
}

func readEntries(rd io.Reader) ([]KVEntry, error) {
	var entries []KVEntry
	var buf [8]byte

	for {
		// keyLen
		if _, err := io.ReadFull(rd, buf[:2]); err != nil {
			if err == io.EOF {
				break
			}
			return entries, err
		}
		keyLen := binary.LittleEndian.Uint16(buf[:2])
		if keyLen == 0 {
			break // sentinel
		}

		// key
		key := make([]byte, keyLen)
		if _, err := io.ReadFull(rd, key); err != nil {
			return entries, err
		}

		// valueLen
		if _, err := io.ReadFull(rd, buf[:4]); err != nil {
			return entries, err
		}
		valueLen := binary.LittleEndian.Uint32(buf[:4])

		// value
		value := make([]byte, valueLen)
		if _, err := io.ReadFull(rd, value); err != nil {
			return entries, err
		}

		// ttlMs
		if _, err := io.ReadFull(rd, buf[:8]); err != nil {
			return entries, err
		}
		ttlMs := int64(binary.LittleEndian.Uint64(buf[:8]))

		entries = append(entries, KVEntry{Key: key, Value: value, TTLMs: ttlMs})
	}
	return entries, nil
}

// Save writes a complete snapshot (manifest + per-shard data files).
// shardEntries maps shard ID → entries for that shard.
func Save(dir string, serverVersion string, allocatorMode string, shardEntries map[int][]KVEntry) error {
	w := NewWriter(dir)

	var totalKeys int64
	var totalBytes int64
	files := make([]string, 0, len(shardEntries))

	for shardID, entries := range shardEntries {
		if err := w.WriteShard(shardID, entries); err != nil {
			return fmt.Errorf("write shard %d: %w", shardID, err)
		}
		for _, e := range entries {
			totalKeys++
			totalBytes += int64(len(e.Key)) + int64(len(e.Value))
		}
		files = append(files, fmt.Sprintf("shard-%d.dat", shardID))
	}

	return w.WriteManifest(Manifest{
		Version:       1,
		ServerVersion: serverVersion,
		CreatedAt:     time.Now().Format(time.RFC3339),
		ShardCount:    len(shardEntries),
		AllocatorMode: allocatorMode,
		TotalKeys:     totalKeys,
		TotalBytes:    totalBytes,
		Files:         files,
	})
}

// Load reads a snapshot and returns the manifest and all entries.
// If the snapshot has a different shard count than currentShards,
// all entries are returned for re-hashing.
func Load(dir string, currentShards int) (Manifest, []KVEntry, error) {
	r := NewReader(dir)
	m, err := r.ReadManifest()
	if err != nil {
		return Manifest{}, nil, err
	}

	// Always read all entries — caller re-hashes into current shard layout
	entries, err := r.ReadAllEntries()
	if err != nil {
		return m, nil, err
	}
	return m, entries, nil
}
