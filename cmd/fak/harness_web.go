package main

import (
	"context"
	"encoding/json"
	"io"
	"os/exec"
	"time"

	"github.com/anthony-chaudhary/fak/internal/harnessweb"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

func runHarnessWeb(stdout, stderr io.Writer, argv []string) int {
	return runHarnessWebWithCancel(context.Background(), stdout, stderr, argv)
}

func runHarnessWebWithCancel(ctx context.Context, stdout, stderr io.Writer, argv []string) int {
	return harnessweb.RunWithLocalWork(ctx, stdout, stderr, argv, harnessLocalWorkSource{})
}

type harnessLocalWorkSource struct{}

func (harnessLocalWorkSource) LiveIntentKeys(ctx context.Context, root string, now time.Time) ([]string, error) {
	live, _, err := leaseref.NewInDir(root).LiveIntents(ctx, now)
	if err != nil {
		return nil, err
	}
	keys := make([]string, 0, len(live))
	for _, intent := range live {
		keys = append(keys, intent.Key)
	}
	return keys, nil
}

var harnessDOSLive = func(ctx context.Context, root string) ([]byte, error) {
	timeoutScope, cancel := context.WithTimeout(ctx, 1500*time.Millisecond)
	defer cancel()
	return exec.CommandContext(timeoutScope, "dos", "--workspace", root, "lease-lane", "live").Output()
}

func (harnessLocalWorkSource) LiveDOSLeases(ctx context.Context, root string, _ time.Time) ([]harnessweb.LocalDOSLease, error) {
	data, err := harnessDOSLive(ctx, root)
	if err != nil {
		return nil, err
	}
	var rows []struct {
		Lane   string `json:"lane"`
		LoopID string `json:"loop_id"`
	}
	if err := json.Unmarshal(data, &rows); err != nil {
		return nil, err
	}
	leases := make([]harnessweb.LocalDOSLease, 0, len(rows))
	for _, row := range rows {
		leases = append(leases, harnessweb.LocalDOSLease{Lane: row.Lane, LoopID: row.LoopID})
	}
	return leases, nil
}
