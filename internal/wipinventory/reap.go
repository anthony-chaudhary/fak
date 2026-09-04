package wipinventory

import (
	"fmt"
	"strconv"
	"strings"
	"time"
)

type ReapResult struct {
	Reaped []string          `json:"reaped"`
	Kept   []string          `json:"kept"`
	DryRun bool              `json:"dry_run"`
	SHAs   map[string]string `json:"shas,omitempty"`
}

// Reap lists refs matching refs/fak/wip/* using for-each-ref, checks creation time,
// and deletes refs older than maxAge using git update-ref -d <ref>.
// If maxAge <= 0, it defaults to 7 days (7 * 24 * time.Hour).
func Reap(root string, maxAge time.Duration, dryRun bool, r Runner) (ReapResult, error) {
	if maxAge <= 0 {
		maxAge = 7 * 24 * time.Hour
	}
	res := ReapResult{
		Reaped: []string{},
		Kept:   []string{},
		DryRun: dryRun,
		SHAs:   make(map[string]string),
	}
	out, err := r.Run(root, "for-each-ref", "--format=%(refname)%00%(objectname)%00%(creatordate:unix)", "refs/fak/wip")
	if err != nil {
		return res, fmt.Errorf("list checkpoint refs: %w", err)
	}
	now := time.Now()
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		line = strings.TrimRight(line, "\r")
		if line == "" {
			continue
		}
		parts := strings.Split(line, "\x00")
		if len(parts) != 3 {
			continue
		}
		ref := parts[0]
		sha := parts[1]
		res.SHAs[ref] = sha
		unix, err := strconv.ParseInt(parts[2], 10, 64)
		if err != nil {
			res.Kept = append(res.Kept, ref)
			continue
		}
		createdAt := time.Unix(unix, 0)
		if now.Sub(createdAt) > maxAge {
			if dryRun {
				res.Reaped = append(res.Reaped, ref)
			} else {
				if _, derr := r.Run(root, "update-ref", "-d", ref); derr != nil {
					return res, fmt.Errorf("delete ref %s: %w", ref, derr)
				}
				res.Reaped = append(res.Reaped, ref)
			}
		} else {
			res.Kept = append(res.Kept, ref)
		}
	}
	return res, nil
}
