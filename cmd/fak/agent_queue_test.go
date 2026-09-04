package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agentqueue"
)

func init() {
	if len(os.Args) > 1 && os.Args[1] == "dispatch" {
		os.Exit(0)
	}
}

func TestAgentQueueRunOnceJSON(t *testing.T) {
	dir := t.TempDir()
	statePath := filepath.Join(dir, "queue.json")
	store := agentqueue.FileStore(statePath)

	snapshot := agentqueue.Snapshot{
		Schema:     agentqueue.Schema,
		Generation: "g0",
		Pool:       agentqueue.PoolSpec{ID: "pool-alpha", Min: 1, Desired: 2, Max: 2},
		Intents: []agentqueue.Intent{
			{
				ID:     "intent-1",
				State:  agentqueue.IntentQueued,
				Launch: agentqueue.LaunchSpec{Issue: 8942, Lane: "agentqueue"},
			},
			{
				ID:     "intent-2",
				State:  agentqueue.IntentQueued,
				Launch: agentqueue.LaunchSpec{Issue: 8943, Lane: "agentqueue"},
			},
		},
	}
	if err := store.Save(snapshot); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runAgentQueue(&stdout, &stderr, []string{
		"run",
		"--state", statePath,
		"--fak", os.Args[0],
		"--interval", "50ms",
		"--once",
		"--json",
	})
	if code != 0 {
		t.Fatalf("expected code 0, got %d, stderr: %s", code, stderr.String())
	}
	if stderr.Len() > 0 {
		t.Fatalf("unexpected stderr: %s", stderr.String())
	}

	var receipt agentqueue.TickReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("decode receipt json: %v, raw: %s", err, stdout.String())
	}

	if receipt.Plan.PoolID != "pool-alpha" {
		t.Fatalf("expected pool ID 'pool-alpha', got %q", receipt.Plan.PoolID)
	}
	if receipt.Plan.Desired != 2 {
		t.Fatalf("expected desired 2, got %d", receipt.Plan.Desired)
	}
	if receipt.Plan.Observed != 0 {
		t.Fatalf("expected observed 0, got %d", receipt.Plan.Observed)
	}
	if len(receipt.Plan.Start) != 2 {
		t.Fatalf("expected 2 start actions, got %d", len(receipt.Plan.Start))
	}
	if receipt.Plan.Start[0].IntentID != "intent-1" {
		t.Fatalf("expected start[0] intent-1, got %q", receipt.Plan.Start[0].IntentID)
	}
	if receipt.Plan.Start[1].IntentID != "intent-2" {
		t.Fatalf("expected start[1] intent-2, got %q", receipt.Plan.Start[1].IntentID)
	}

	if len(receipt.Launches) != 2 {
		t.Fatalf("expected 2 launches, got %d", len(receipt.Launches))
	}
	if receipt.Launches[0].IntentID != "intent-1" {
		t.Fatalf("expected launch[0] intent-1, got %q", receipt.Launches[0].IntentID)
	}
	if receipt.Launches[0].IdempotencyKey != receipt.Plan.Start[0].IdempotencyKey {
		t.Fatalf("launch idempotency key mismatch: launch=%q plan=%q", receipt.Launches[0].IdempotencyKey, receipt.Plan.Start[0].IdempotencyKey)
	}
	if len(receipt.Launches[0].Command) < 3 || receipt.Launches[0].Command[1] != "dispatch" || receipt.Launches[0].Command[2] != "tick" {
		t.Fatalf("unexpected command structure: %#v", receipt.Launches[0].Command)
	}
	if receipt.Launches[1].IntentID != "intent-2" {
		t.Fatalf("expected launch[1] intent-2, got %q", receipt.Launches[1].IntentID)
	}

	// Verify state file persisted reservations
	updated, err := store.Load()
	if err != nil {
		t.Fatal(err)
	}
	if updated.Generation == "g0" {
		t.Fatalf("expected updated generation, got %q", updated.Generation)
	}
	if len(updated.Attempts) != 2 {
		t.Fatalf("expected 2 attempts recorded in store, got %d", len(updated.Attempts))
	}
}

func TestAgentQueueRunOnceText(t *testing.T) {
	dir := t.TempDir()
	statePath := filepath.Join(dir, "queue.json")
	store := agentqueue.FileStore(statePath)

	snapshot := agentqueue.Snapshot{
		Schema:     agentqueue.Schema,
		Generation: "g0",
		Pool:       agentqueue.PoolSpec{ID: "pool-beta", Min: 0, Desired: 1, Max: 1},
		Intents: []agentqueue.Intent{
			{
				ID:     "intent-a",
				State:  agentqueue.IntentQueued,
				Launch: agentqueue.LaunchSpec{Issue: 100, Lane: "docs"},
			},
		},
	}
	if err := store.Save(snapshot); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runAgentQueueContext(context.Background(), &stdout, &stderr, []string{
		"run",
		"--state", statePath,
		"--fak", os.Args[0],
		"--once",
	})
	if code != 0 {
		t.Fatalf("expected code 0, got %d, stderr: %s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "pool=pool-beta") || !strings.Contains(out, "launches=1") {
		t.Fatalf("unexpected output: %s", out)
	}
}

func TestAgentQueueRunCleanCancellation(t *testing.T) {
	dir := t.TempDir()
	statePath := filepath.Join(dir, "queue.json")
	store := agentqueue.FileStore(statePath)

	snapshot := agentqueue.Snapshot{
		Schema:     agentqueue.Schema,
		Generation: "g0",
		Pool:       agentqueue.PoolSpec{ID: "pool-idle", Min: 0, Desired: 0, Max: 1},
	}
	if err := store.Save(snapshot); err != nil {
		t.Fatal(err)
	}

	// 1. Cancel during ticker loop
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(25 * time.Millisecond)
		cancel()
	}()

	var stdout, stderr bytes.Buffer
	code := runAgentQueueContext(ctx, &stdout, &stderr, []string{
		"run",
		"--state", statePath,
		"--interval", "10ms",
	})
	if code != 0 {
		t.Fatalf("expected exit 0 on cancel, got %d, stderr: %s", code, stderr.String())
	}

	// 2. Pre-cancelled context stops cleanly immediately
	ctxPre, cancelPre := context.WithCancel(context.Background())
	cancelPre()
	stdout.Reset()
	stderr.Reset()
	codePre := runAgentQueueContext(ctxPre, &stdout, &stderr, []string{
		"run",
		"--state", statePath,
		"--interval", "10ms",
	})
	if codePre != 0 {
		t.Fatalf("expected exit 0 on pre-canceled context, got %d, stderr: %s", codePre, stderr.String())
	}
}

func TestAgentQueueUsageAndErrors(t *testing.T) {
	dir := t.TempDir()

	// Missing --state
	{
		var stdout, stderr bytes.Buffer
		code := runAgentQueueContext(context.Background(), &stdout, &stderr, []string{"run"})
		if code != 2 {
			t.Fatalf("expected code 2, got %d", code)
		}
		if !strings.Contains(stderr.String(), "--state is required") {
			t.Fatalf("expected '--state is required', got: %s", stderr.String())
		}
	}

	// Empty --state
	{
		var stdout, stderr bytes.Buffer
		code := runAgentQueueContext(context.Background(), &stdout, &stderr, []string{"run", "--state", ""})
		if code != 2 {
			t.Fatalf("expected code 2, got %d", code)
		}
		if !strings.Contains(stderr.String(), "--state is required") {
			t.Fatalf("expected '--state is required', got: %s", stderr.String())
		}
	}

	// Missing subcommand
	{
		var stdout, stderr bytes.Buffer
		code := runAgentQueueContext(context.Background(), &stdout, &stderr, []string{})
		if code != 2 {
			t.Fatalf("expected code 2, got %d", code)
		}
		if !strings.Contains(stderr.String(), "usage: fak agent-queue") {
			t.Fatalf("expected usage message, got: %s", stderr.String())
		}
	}

	// Top-level --help
	{
		var stdout, stderr bytes.Buffer
		code := runAgentQueueContext(context.Background(), &stdout, &stderr, []string{"--help"})
		if code != 0 {
			t.Fatalf("expected code 0, got %d", code)
		}
		if !strings.Contains(stdout.String(), "usage: fak agent-queue") {
			t.Fatalf("expected usage message, got: %s", stdout.String())
		}
	}

	// run --help
	{
		var stdout, stderr bytes.Buffer
		code := runAgentQueueContext(context.Background(), &stdout, &stderr, []string{"run", "--help"})
		if code != 0 {
			t.Fatalf("expected code 0, got %d", code)
		}
	}

	// Unknown subcommand
	{
		var stdout, stderr bytes.Buffer
		code := runAgentQueueContext(context.Background(), &stdout, &stderr, []string{"unknown"})
		if code != 2 {
			t.Fatalf("expected code 2, got %d", code)
		}
		if !strings.Contains(stderr.String(), "unknown subcommand") {
			t.Fatalf("expected 'unknown subcommand', got: %s", stderr.String())
		}
	}

	// Missing state file error
	{
		var stdout, stderr bytes.Buffer
		code := runAgentQueueContext(context.Background(), &stdout, &stderr, []string{
			"run",
			"--state", filepath.Join(dir, "nonexistent.json"),
			"--once",
		})
		if code != 1 {
			t.Fatalf("expected code 1, got %d", code)
		}
		if !strings.Contains(stderr.String(), "agent-queue run:") {
			t.Fatalf("expected 'agent-queue run:' error, got: %s", stderr.String())
		}
	}
}
