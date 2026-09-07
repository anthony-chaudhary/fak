package armtracking

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

// DefaultPath is the canonical repository location for the shifting baselines and arms ledger.
const DefaultPath = "configs/shifting-baselines.json"

// LoadRegistry reads and decodes the shifting baselines registry from the specified file path.
// If the file does not exist, an empty registry initialized with SchemaVersion is returned.
func LoadRegistry(path string) (*Registry, error) {
	if path == "" {
		path = DefaultPath
	}

	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return NewRegistry(), nil
	}
	if err != nil {
		return nil, fmt.Errorf("armtracking: read %s: %w", path, err)
	}

	var reg Registry
	if err := json.Unmarshal(data, &reg); err != nil {
		return nil, fmt.Errorf("armtracking: parse %s: %w", path, err)
	}

	if reg.Schema != SchemaVersion {
		return nil, fmt.Errorf("armtracking: incompatible schema %q (want %q)", reg.Schema, SchemaVersion)
	}

	if reg.Workloads == nil {
		reg.Workloads = make(map[string]*WorkloadRegistry)
	}

	for _, w := range reg.Workloads {
		if w.Arms == nil {
			w.Arms = make(map[string]*ArmRecord)
		}
	}

	return &reg, nil
}

// SaveRegistry writes the registry to the specified file path atomically.
func SaveRegistry(r *Registry, path string) error {
	if r == nil {
		return errors.New("armtracking: cannot save nil registry")
	}
	if path == "" {
		path = DefaultPath
	}

	r.mu.RLock()
	defer r.mu.RUnlock()

	dir := filepath.Dir(path)
	if dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("armtracking: mkdir %s: %w", dir, err)
		}
	}

	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return fmt.Errorf("armtracking: marshal: %w", err)
	}
	data = append(data, '\n')

	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return fmt.Errorf("armtracking: write %s: %w", tmp, err)
	}

	if err := os.Rename(tmp, path); err != nil {
		// Fallback for systems where rename fails over existing file
		_ = os.Remove(path)
		if rerr := os.Rename(tmp, path); rerr != nil {
			_ = os.Remove(tmp)
			return fmt.Errorf("armtracking: rename %s to %s: %w", tmp, path, rerr)
		}
	}

	return nil
}
