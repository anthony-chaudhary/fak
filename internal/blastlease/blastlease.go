// Package blastlease projects live or fixture lease records into blast-radius inputs.
// It is runtime-shared because known-bad reporting and the development-only blast
// estimator consume the same lease-set contract.
package blastlease

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/blastradius"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

// Live reads the live lease ledger and projects records into blast-radius lease inputs.
func Live(root string, now time.Time) ([]blastradius.Lease, error) {
	live, _, err := leaseref.NewInDir(root).Live(context.Background(), now)
	if err != nil {
		return nil, err
	}
	out := make([]blastradius.Lease, 0, len(live))
	for _, record := range live {
		out = append(out, blastradius.Lease{Lane: record.ID, TreeGlobs: record.TreeGlobs})
	}
	return out, nil
}

// Read decodes a JSONL fixture of blast-radius lease inputs.
func Read(path string) ([]blastradius.Lease, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var out []blastradius.Lease
	for i, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var lease blastradius.Lease
		if err := json.Unmarshal([]byte(line), &lease); err != nil {
			return nil, fmt.Errorf("%s line %d: %w", path, i+1, err)
		}
		out = append(out, lease)
	}
	return out, nil
}
