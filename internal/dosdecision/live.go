package dosdecision

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
)

// LiveRow is one live lane row as reported by `dos lease-lane live`: the lane
// name and the tree globs its holder was granted.
type LiveRow struct {
	Lane string   `json:"lane"`
	Tree []string `json:"tree"`
}

// ReadLiveRows runs `dos lease-lane live` inside dir and returns its rows —
// the authoritative WAL live set of lane leases held by running sessions. The
// workspace is passed as the working directory, not as a path argument,
// because an explicit path makes `dos lease-lane live` re-resolve to a
// different `.dos/` (the same rule tools/dos_fleet_lease.py obeys). Any read
// or parse fault is returned as an error: an unreadable live plane must never
// be mistaken for an empty one. This leaf owns the subprocess boundary so
// Adjudicator packages can stay os/exec-free (TestEveryAdjudicatorIsExecFree).
func ReadLiveRows(ctx context.Context, dir string) ([]LiveRow, error) {
	cmd := exec.CommandContext(ctx, "dos", "lease-lane", "live")
	cmd.Dir = dir
	raw, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("dosdecision: dos lease-lane live: %w", err)
	}
	var rows []LiveRow
	if err := json.Unmarshal(raw, &rows); err != nil {
		return nil, fmt.Errorf("dosdecision: dos lease-lane live JSON: %w", err)
	}
	if rows == nil {
		return nil, errors.New("dosdecision: dos lease-lane live returned null")
	}
	return rows, nil
}
