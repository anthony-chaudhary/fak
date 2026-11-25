package studymonitor

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// MaxRegistryShardBytes bounds one per-repository shard file.
const MaxRegistryShardBytes = 1 << 20

// MaxRegistryBytes bounds the assembled registry, header plus every shard.
const MaxRegistryBytes = 8 << 20

// ErrShardNameMismatch marks a shard whose repository field does not slug to
// its filename; the case-folded filename is what keeps rows unique.
var ErrShardNameMismatch = errors.New("SHARD_NAME_MISMATCH")

var registryRepositoryName = regexp.MustCompile(`^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$`)

// ShardDir is the per-repository shard directory beside a registry header.
func ShardDir(registryPath string) string {
	return strings.TrimSuffix(registryPath, ".json")
}

// ShardFileName is the shard filename for an owner/name repository.
func ShardFileName(repository string) (string, error) {
	name := strings.TrimSpace(repository)
	if !registryRepositoryName.MatchString(name) {
		return "", fmt.Errorf("repository %q is not a valid owner/name", repository)
	}
	owner, repo, _ := strings.Cut(name, "/")
	return strings.ToLower(owner) + "__" + strings.ToLower(repo) + ".json", nil
}

// overlayRegistryShards replaces header rows with same-named shard rows and
// appends shard-only rows in slug order. A missing shard directory leaves the
// registry untouched.
func overlayRegistryShards(registry *Registry, registryPath string, headerBytes int) error {
	if !strings.HasSuffix(registryPath, ".json") {
		return nil
	}
	shards, err := readRegistryShards(ShardDir(registryPath), MaxRegistryBytes-headerBytes)
	if err != nil || len(shards) == 0 {
		return err
	}
	byKey := make(map[string]int, len(shards))
	for i, shard := range shards {
		byKey[registryKey(shard.Repository)] = i
	}
	used := make([]bool, len(shards))
	for i, row := range registry.Repositories {
		if j, ok := byKey[registryKey(row.Repository)]; ok {
			registry.Repositories[i] = shards[j]
			used[j] = true
		}
	}
	for j, shard := range shards {
		if !used[j] {
			registry.Repositories = append(registry.Repositories, shard)
		}
	}
	return nil
}

func registryKey(repository string) string {
	return strings.ToLower(strings.TrimSpace(repository))
}

// readRegistryShards decodes direct *.json children in filename (slug) order.
// Subdirectories, dotfiles, writer locks (*.lock) and writer temp files
// (*.json.tmp-*) are not rows. budget bounds the summed shard bytes.
func readRegistryShards(dir string, budget int) ([]Repository, error) {
	entries, err := os.ReadDir(dir)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	// A shard directory opts into the assembled cap, so an oversize header fails even with no shards.
	if budget < 0 {
		return nil, fmt.Errorf("registry %s exceeds %d bytes assembled", dir, MaxRegistryBytes)
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].Name() < entries[j].Name() })
	var shards []Repository
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || strings.HasPrefix(name, ".") || !strings.HasSuffix(name, ".json") {
			continue
		}
		file := filepath.Join(dir, name)
		data, err := readCapped(file, MaxRegistryShardBytes)
		if err != nil {
			return nil, err
		}
		if budget -= len(data); budget < 0 {
			return nil, fmt.Errorf("registry %s exceeds %d bytes assembled", dir, MaxRegistryBytes)
		}
		var row Repository
		if err := json.Unmarshal(data, &row); err != nil {
			return nil, fmt.Errorf("decode registry shard %s: %w", file, err)
		}
		if want, err := ShardFileName(row.Repository); err != nil || want != name {
			return nil, fmt.Errorf("%w: %s holds repository %q", ErrShardNameMismatch, file, row.Repository)
		}
		shards = append(shards, row)
	}
	return shards, nil
}

func readCapped(file string, limit int) ([]byte, error) {
	f, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, int64(limit)+1))
	if err != nil {
		return nil, err
	}
	if len(data) > limit {
		return nil, fmt.Errorf("registry shard %s exceeds %d bytes", file, limit)
	}
	return data, nil
}
