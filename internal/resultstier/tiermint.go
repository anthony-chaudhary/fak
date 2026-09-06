package resultstier

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"
)

// PayloadIndexSchema is the canonical schema identifier for results payload index documents.
const PayloadIndexSchema = "fak/results-payload-index/v1"

// PayloadEntry captures provenance-locked metadata for a single externalized payload artifact.
type PayloadEntry struct {
	Path   string `json:"path"`
	Bytes  int64  `json:"bytes"`
	SHA256 string `json:"sha256"`
}

// VolumeKey returns the content-addressed storage key: results-payload/<sha[:2]>/<sha>.
func (e PayloadEntry) VolumeKey() (string, error) {
	if err := e.Complete(); err != nil {
		return "", fmt.Errorf("cannot derive volume key: %w", err)
	}
	return fmt.Sprintf("results-payload/%s/%s", e.SHA256[:2], e.SHA256), nil
}

// Complete returns an error if any required field is missing or invalid.
func (e PayloadEntry) Complete() error {
	p := strings.TrimSpace(e.Path)
	if p == "" {
		return errors.New("payload entry path is empty")
	}
	clean := path.Clean(filepath.ToSlash(p))
	if strings.HasPrefix(clean, "../") || clean == ".." || path.IsAbs(clean) || strings.Contains(p, ":") {
		return fmt.Errorf("payload entry %q contains invalid path traversal or absolute path", e.Path)
	}
	if e.Bytes <= 0 {
		return fmt.Errorf("payload entry %q byte count must be > 0, got %d", e.Path, e.Bytes)
	}
	if len(e.SHA256) != 64 {
		return fmt.Errorf("payload entry %q sha256 must be 64 characters, got %d", e.Path, len(e.SHA256))
	}
	for _, ch := range e.SHA256 {
		if !((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f')) {
			return fmt.Errorf("payload entry %q sha256 contains invalid character %q (must be lowercase hex)", e.Path, ch)
		}
	}
	return nil
}

// PayloadIndex models the index of externalized payload objects.
type PayloadIndex struct {
	Schema   string         `json:"schema"`
	StoreURI string         `json:"store_uri"`
	Entries  []PayloadEntry `json:"entries"`
}

// Lookup finds an entry by path, matching both exact and slash-normalized forms.
func (idx PayloadIndex) Lookup(p string) (PayloadEntry, bool) {
	clean := filepath.ToSlash(strings.TrimSpace(p))
	clean = path.Clean(clean)
	clean = strings.TrimPrefix(clean, "./")
	clean = strings.TrimPrefix(clean, "/")

	for _, e := range idx.Entries {
		eClean := filepath.ToSlash(strings.TrimSpace(e.Path))
		eClean = path.Clean(eClean)
		eClean = strings.TrimPrefix(eClean, "./")
		eClean = strings.TrimPrefix(eClean, "/")
		if e.Path == p || e.Path == clean || eClean == clean {
			return e, true
		}
	}
	return PayloadEntry{}, false
}

// Sort sorts index entries by Path in ascending order.
func (idx *PayloadIndex) Sort() {
	sort.Slice(idx.Entries, func(i, j int) bool {
		return idx.Entries[i].Path < idx.Entries[j].Path
	})
}

// TotalBytes returns the sum of bytes across all payload entries.
func (idx PayloadIndex) TotalBytes() int64 {
	var total int64
	for _, e := range idx.Entries {
		total += e.Bytes
	}
	return total
}

// CanMigrate enforces a fail-closed gate before externalizing a payload file:
// 1. TierOf(p) must be TierPayload
// 2. Entry must exist in idx
// 3. Entry must be Complete()
// 4. idx.StoreURI must not be empty
func CanMigrate(idx PayloadIndex, p string) error {
	if strings.TrimSpace(idx.StoreURI) == "" {
		return errors.New("cannot migrate: store URI is empty")
	}
	tier, reason := TierOf(p)
	if tier != TierPayload {
		return fmt.Errorf("cannot migrate %q: tier is %s (%s), must be payload", p, tier, reason)
	}
	entry, ok := idx.Lookup(p)
	if !ok {
		return fmt.Errorf("cannot migrate %q: entry not found in payload index", p)
	}
	if err := entry.Complete(); err != nil {
		return fmt.Errorf("cannot migrate %q: entry is incomplete: %w", p, err)
	}
	return nil
}

// MintPayloadIndex walks dir, hashes payload files, tallies the census, and returns a sorted PayloadIndex.
// Skips .git, .DS_Store, .gitignore, .gitkeep, and symlinks.
func MintPayloadIndex(dir string, storeURI string) (PayloadIndex, Census, error) {
	info, err := os.Stat(dir)
	if err != nil {
		return PayloadIndex{}, Census{}, fmt.Errorf("failed to stat directory %s: %w", dir, err)
	}
	if !info.IsDir() {
		return PayloadIndex{}, Census{}, fmt.Errorf("target is not a directory: %s", dir)
	}

	idx := PayloadIndex{
		Schema:   PayloadIndexSchema,
		StoreURI: storeURI,
		Entries:  make([]PayloadEntry, 0),
	}
	census := Census{
		UnknownExts: make(map[string]int),
	}

	err = filepath.WalkDir(dir, func(p string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if d.IsDir() {
			if d.Name() == ".git" {
				return filepath.SkipDir
			}
			return nil
		}

		// Skip symlinks
		if d.Type()&os.ModeSymlink != 0 {
			return nil
		}
		if !d.Type().IsRegular() {
			return nil
		}

		name := d.Name()
		if name == ".DS_Store" || name == ".gitignore" || name == ".gitkeep" || name == ".git" {
			return nil
		}

		rel, err := filepath.Rel(dir, p)
		if err != nil {
			return fmt.Errorf("failed to compute relative path for %s: %w", p, err)
		}
		relSlash := filepath.ToSlash(rel)
		relSlash = strings.TrimPrefix(relSlash, "./")

		fileInfo, err := d.Info()
		if err != nil {
			return fmt.Errorf("failed to get file info for %s: %w", p, err)
		}
		size := fileInfo.Size()

		tier, _ := TierOf(relSlash)
		switch tier {
		case TierClaim:
			census.ClaimFiles++
			census.ClaimBytes += size
		case TierPayload:
			if size <= 0 {
				// Zero-length files cannot detect truncated fetches; classify as unknown rather than minting an unverifiable zero-byte entry
				census.UnknownFiles++
				census.UnknownExts["<empty-payload>"]++
				return nil
			}
			f, err := os.Open(p)
			if err != nil {
				return fmt.Errorf("failed to open payload file %s: %w", p, err)
			}
			h := sha256.New()
			n, err := io.Copy(h, f)
			f.Close()
			if err != nil {
				return fmt.Errorf("failed to hash payload file %s: %w", p, err)
			}
			shaHex := hex.EncodeToString(h.Sum(nil))
			idx.Entries = append(idx.Entries, PayloadEntry{
				Path:   relSlash,
				Bytes:  n,
				SHA256: shaHex,
			})
			census.PayloadFiles++
			census.PayloadBytes += n
		default:
			census.UnknownFiles++
			census.UnknownBytes += size
			ext := strings.ToLower(filepath.Ext(name))
			if ext == "" {
				ext = "<none>"
			}
			census.UnknownExts[ext]++
		}

		return nil
	})
	if err != nil {
		return PayloadIndex{}, Census{}, err
	}

	idx.Sort()
	return idx, census, nil
}

// VerifyPayloadIndex checks each entry in idx.Entries against the filesystem at dir.
// It verifies existence, size equality, and SHA-256 integrity.
// Returns a slice of discrepancy messages if discrepancies are found.
func VerifyPayloadIndex(dir string, idx PayloadIndex) ([]string, error) {
	info, err := os.Stat(dir)
	if err != nil {
		return nil, fmt.Errorf("failed to stat directory %s: %w", dir, err)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("target is not a directory: %s", dir)
	}

	var discrepancies []string
	for _, entry := range idx.Entries {
		if err := entry.Complete(); err != nil {
			discrepancies = append(discrepancies, fmt.Sprintf("invalid payload entry %q: %v", entry.Path, err))
			continue
		}
		filePath := filepath.Join(dir, filepath.FromSlash(entry.Path))
		fileInfo, err := os.Stat(filePath)
		if err != nil {
			if os.IsNotExist(err) {
				discrepancies = append(discrepancies, fmt.Sprintf("missing payload file %q: does not exist on disk", entry.Path))
				continue
			}
			discrepancies = append(discrepancies, fmt.Sprintf("cannot stat payload file %q: %v", entry.Path, err))
			continue
		}

		if fileInfo.Size() != entry.Bytes {
			discrepancies = append(discrepancies, fmt.Sprintf("payload file %q size mismatch: recorded %d bytes, got %d bytes on disk", entry.Path, entry.Bytes, fileInfo.Size()))
		}

		f, err := os.Open(filePath)
		if err != nil {
			discrepancies = append(discrepancies, fmt.Sprintf("cannot open payload file %q: %v", entry.Path, err))
			continue
		}
		h := sha256.New()
		_, err = io.Copy(h, f)
		f.Close()
		if err != nil {
			discrepancies = append(discrepancies, fmt.Sprintf("cannot read payload file %q: %v", entry.Path, err))
			continue
		}

		actualSHA := hex.EncodeToString(h.Sum(nil))
		if !strings.EqualFold(actualSHA, entry.SHA256) {
			discrepancies = append(discrepancies, fmt.Sprintf("payload file %q sha256 mismatch: recorded %s, got %s on disk", entry.Path, entry.SHA256, actualSHA))
		}
	}

	return discrepancies, nil
}
