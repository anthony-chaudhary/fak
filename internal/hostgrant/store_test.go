package hostgrant_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/hostgrant"
)

func TestStoreOrdinaryAdmissionPreservesControlReserve(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "grants.json")
	store := hostgrant.Store{
		Path:           path,
		Capacity:       hostgrant.Vector{Processes: 2},
		ControlReserve: hostgrant.Vector{Processes: 1},
	}
	first := hostgrant.Request{
		ID:    "ordinary-first",
		Owner: hostgrant.Owner{ID: "agent-first", PID: 101, StartedAt: time.Date(2026, time.September, 23, 8, 0, 0, 0, time.UTC)},
		Cost:  hostgrant.Vector{Processes: 1},
		TTL:   time.Minute,
	}
	grant, err := store.TryAcquire(context.Background(), first)
	if err != nil {
		t.Fatalf("first ordinary TryAcquire: %v", err)
	}
	if grant.UsesControlReserve {
		t.Fatal("public TryAcquire marked ordinary grant as using control reserve")
	}

	second := first
	second.ID = "ordinary-second"
	second.Owner.ID = "agent-second"
	second.Owner.PID = 202
	if _, err := store.TryAcquire(context.Background(), second); !errors.Is(err, hostgrant.ErrFull) {
		t.Fatalf("second ordinary TryAcquire error = %v, want ErrFull", err)
	}

	var persisted struct {
		ControlReserve hostgrant.Vector           `json:"control_reserve"`
		Grants         map[string]hostgrant.Grant `json:"grants"`
	}
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read persisted store: %v", err)
	}
	if err := json.Unmarshal(body, &persisted); err != nil {
		t.Fatalf("decode persisted store: %v", err)
	}
	if persisted.ControlReserve != store.ControlReserve {
		t.Fatalf("persisted control reserve = %+v, want %+v", persisted.ControlReserve, store.ControlReserve)
	}
	if len(persisted.Grants) != 1 || persisted.Grants[first.ID].ID != first.ID {
		t.Fatalf("persisted grants = %#v, want only %q", persisted.Grants, first.ID)
	}
}

func TestStoreInvalidControlReserveFailsClosed(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "grants.json")
	store := hostgrant.Store{
		Path:           path,
		Capacity:       hostgrant.Vector{Processes: 1},
		ControlReserve: hostgrant.Vector{Processes: 2},
	}
	_, err := store.TryAcquire(context.Background(), hostgrant.Request{
		ID:    "ordinary",
		Owner: hostgrant.Owner{ID: "agent", PID: 101, StartedAt: time.Date(2026, time.September, 23, 8, 0, 0, 0, time.UTC)},
		Cost:  hostgrant.Vector{Processes: 1},
		TTL:   time.Minute,
	})
	if err == nil {
		t.Fatal("TryAcquire with reserve above capacity succeeded")
	}
	if _, statErr := os.Stat(path); !errors.Is(statErr, os.ErrNotExist) {
		t.Fatalf("invalid configuration persisted state: stat error = %v, want not-exist", statErr)
	}
}

func TestStoreRestartWithDifferentControlReserveFailsClosed(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "grants.json")
	configured := hostgrant.Store{
		Path:           path,
		Capacity:       hostgrant.Vector{Processes: 2},
		ControlReserve: hostgrant.Vector{Processes: 1},
	}
	request := hostgrant.Request{
		ID:    "ordinary-first",
		Owner: hostgrant.Owner{ID: "agent-first", PID: 101, StartedAt: time.Date(2026, time.September, 23, 8, 0, 0, 0, time.UTC)},
		Cost:  hostgrant.Vector{Processes: 1},
		TTL:   time.Minute,
	}
	if _, err := configured.TryAcquire(context.Background(), request); err != nil {
		t.Fatalf("initialize store: %v", err)
	}

	restarted := hostgrant.Store{Path: path, Capacity: configured.Capacity}
	request.ID = "ordinary-after-restart"
	request.Owner.ID = "agent-after-restart"
	request.Owner.PID = 202
	if _, err := restarted.TryAcquire(context.Background(), request); !errors.Is(err, hostgrant.ErrCapacityMismatch) {
		t.Fatalf("TryAcquire after reserve mismatch error = %v, want ErrCapacityMismatch", err)
	}
}

func TestStoreRejectsPersistedOrdinaryOvercommitIntoReserve(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "grants.json")
	expires := time.Date(2026, time.September, 24, 8, 0, 0, 0, time.UTC)
	state := map[string]any{
		"schema":          "fak.hostgrant/v1",
		"capacity":        hostgrant.Vector{Processes: 2},
		"control_reserve": hostgrant.Vector{Processes: 1},
		"next_generation": uint64(3),
		"grants": map[string]hostgrant.Grant{
			"ordinary-a": {ID: "ordinary-a", Generation: 1, Owner: hostgrant.Owner{ID: "agent-a", PID: 101, StartedAt: expires.Add(-time.Hour)}, Cost: hostgrant.Vector{Processes: 1}, ExpiresAt: expires},
			"ordinary-b": {ID: "ordinary-b", Generation: 2, Owner: hostgrant.Owner{ID: "agent-b", PID: 202, StartedAt: expires.Add(-time.Hour)}, Cost: hostgrant.Vector{Processes: 1}, ExpiresAt: expires},
		},
	}
	body, err := json.Marshal(state)
	if err != nil {
		t.Fatalf("encode overcommitted state: %v", err)
	}
	if err := os.WriteFile(path, body, 0o600); err != nil {
		t.Fatalf("write overcommitted state: %v", err)
	}

	store := hostgrant.Store{
		Path:           path,
		Capacity:       hostgrant.Vector{Processes: 2},
		ControlReserve: hostgrant.Vector{Processes: 1},
	}
	_, err = store.TryAcquire(context.Background(), hostgrant.Request{
		ID:    "ordinary-c",
		Owner: hostgrant.Owner{ID: "agent-c", PID: 303, StartedAt: expires.Add(-time.Hour)},
		Cost:  hostgrant.Vector{Processes: 1},
		TTL:   time.Minute,
	})
	if err == nil {
		t.Fatal("TryAcquire accepted store with ordinary usage in control reserve")
	}
	if !bytes.Contains([]byte(err.Error()), []byte("persisted ordinary usage")) {
		t.Fatalf("TryAcquire error = %v, want persisted ordinary usage failure", err)
	}
}

func TestStoreCrossProcessCapacity(t *testing.T) {
	dir := t.TempDir()
	children := make([]struct {
		cmd    *exec.Cmd
		output bytes.Buffer
	}, 2)
	for i := range children {
		id := fmt.Sprintf("%d", i)
		children[i].cmd = exec.Command(os.Args[0], "-test.run=^TestHostgrantHelperProcess$")
		children[i].cmd.Env = append(os.Environ(),
			"FAK_HOSTGRANT_HELPER_DIR="+dir,
			"FAK_HOSTGRANT_HELPER_ID="+id,
		)
		children[i].cmd.Stdout = &children[i].output
		children[i].cmd.Stderr = &children[i].output
		if err := children[i].cmd.Start(); err != nil {
			t.Fatalf("start helper %s: %v", id, err)
		}
		defer children[i].cmd.Process.Kill()
	}

	for i := range children {
		waitForPath(t, filepath.Join(dir, fmt.Sprintf("ready-%d", i)))
	}
	if err := os.WriteFile(filepath.Join(dir, "start"), []byte("go"), 0o600); err != nil {
		t.Fatalf("open start gate: %v", err)
	}
	for i := range children {
		if err := children[i].cmd.Wait(); err != nil {
			t.Fatalf("helper %d: %v\n%s", i, err, children[i].output.String())
		}
	}

	counts := map[string]int{}
	for i := range children {
		body, err := os.ReadFile(filepath.Join(dir, fmt.Sprintf("result-%d", i)))
		if err != nil {
			t.Fatalf("read helper %d result: %v", i, err)
		}
		counts[string(body)]++
	}
	if counts["admitted"] != 1 || counts["full"] != 1 {
		t.Fatalf("cross-process results = %v, want one admitted and one full", counts)
	}
}

func TestHostgrantHelperProcess(t *testing.T) {
	dir := os.Getenv("FAK_HOSTGRANT_HELPER_DIR")
	if dir == "" {
		t.Skip("helper process only")
	}
	id := os.Getenv("FAK_HOSTGRANT_HELPER_ID")
	if err := os.WriteFile(filepath.Join(dir, "ready-"+id), []byte("ready"), 0o600); err != nil {
		t.Fatalf("signal ready: %v", err)
	}
	waitForPath(t, filepath.Join(dir, "start"))

	store := hostgrant.Store{
		Path:     filepath.Join(dir, "grants.json"),
		Capacity: hostgrant.Vector{Processes: 1},
	}
	_, err := store.TryAcquire(context.Background(), hostgrant.Request{
		ID: "request-" + id,
		Owner: hostgrant.Owner{
			ID:        "agent-" + id,
			PID:       os.Getpid(),
			StartedAt: time.Date(2026, time.September, 23, 8, 0, 0, 0, time.UTC),
		},
		Cost: hostgrant.Vector{Processes: 1},
		TTL:  time.Minute,
	})
	result := "admitted"
	if errors.Is(err, hostgrant.ErrFull) {
		result = "full"
	} else if err != nil {
		t.Fatalf("TryAcquire: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "result-"+id), []byte(result), 0o600); err != nil {
		t.Fatalf("write result: %v", err)
	}
}

func waitForPath(t *testing.T, path string) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for {
		if _, err := os.Stat(path); err == nil {
			return
		} else if !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("stat %s: %v", path, err)
		}
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", path)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestStoreIndependentInstancesEnforceSharedCapacity(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "grants.json")
	capacity := hostgrant.Vector{Processes: 1}
	stores := [2]hostgrant.Store{
		{Path: path, Capacity: capacity},
		{Path: path, Capacity: capacity},
	}
	startedAt := time.Date(2026, time.September, 23, 8, 0, 0, 0, time.FixedZone("PDT", -7*60*60))
	requests := [2]hostgrant.Request{
		{ID: "request-a", Owner: hostgrant.Owner{ID: "agent-a", PID: 101, StartedAt: startedAt}, Cost: capacity, TTL: time.Minute},
		{ID: "request-b", Owner: hostgrant.Owner{ID: "agent-b", PID: 202, StartedAt: startedAt}, Cost: capacity, TTL: time.Minute},
	}

	start := make(chan struct{})
	type result struct {
		request hostgrant.Request
		grant   hostgrant.Grant
		err     error
	}
	results := make(chan result, len(stores))
	var ready sync.WaitGroup
	ready.Add(len(stores))
	for i := range stores {
		go func(store hostgrant.Store, request hostgrant.Request) {
			ready.Done()
			<-start
			grant, err := store.TryAcquire(ctx, request)
			results <- result{request: request, grant: grant, err: err}
		}(stores[i], requests[i])
	}
	ready.Wait()
	close(start)

	var admitted *result
	full := 0
	for range stores {
		got := <-results
		switch {
		case got.err == nil:
			copy := got
			admitted = &copy
		case errors.Is(got.err, hostgrant.ErrFull):
			full++
		default:
			t.Fatalf("TryAcquire(%q) returned unexpected error: %v", got.request.ID, got.err)
		}
	}
	if admitted == nil || full != 1 {
		t.Fatalf("admission results: admitted=%v full=%d; want exactly one of each", admitted != nil, full)
	}

	retryRequest := admitted.request
	retryRequest.Owner.StartedAt = retryRequest.Owner.StartedAt.UTC()
	retry, err := stores[0].TryAcquire(ctx, retryRequest)
	if err != nil {
		t.Fatalf("idempotent TryAcquire(%q): %v", admitted.request.ID, err)
	}
	if retry.ID != admitted.grant.ID || retry.Generation != admitted.grant.Generation {
		t.Fatalf("idempotent retry returned a different grant: got (%q,%d), want (%q,%d)", retry.ID, retry.Generation, admitted.grant.ID, admitted.grant.Generation)
	}

	transferred, err := stores[1].Transfer(ctx, retry, hostgrant.Owner{ID: "agent-c", PID: 303, StartedAt: startedAt})
	if err != nil {
		t.Fatalf("Transfer(%q): %v", retry.ID, err)
	}
	if transferred.Generation == retry.Generation {
		t.Fatalf("transfer retained stale generation %d", retry.Generation)
	}
	if err := stores[0].Release(ctx, retry); !errors.Is(err, hostgrant.ErrFenced) {
		t.Fatalf("Release(stale generation) error = %v, want ErrFenced", err)
	}
	retransferred, err := stores[0].Transfer(ctx, transferred, hostgrant.Owner{ID: "agent-d", PID: 404, StartedAt: startedAt})
	if err != nil {
		t.Fatalf("second Transfer(%q): %v", transferred.ID, err)
	}
	if err := stores[1].Release(ctx, transferred); !errors.Is(err, hostgrant.ErrFenced) {
		t.Fatalf("Release(first transfer generation) error = %v, want ErrFenced", err)
	}
	if err := stores[0].Release(ctx, retransferred); err != nil {
		t.Fatalf("Release(current generation): %v", err)
	}
}
