package main

import (
	"reflect"
	"testing"
	"time"
)

func TestTUIGuardGrantUndoRemovesOnlyReceiptMutation(t *testing.T) {
	path := t.TempDir() + "/allow.json"
	now := time.Date(2026, 9, 1, 12, 0, 0, 0, time.UTC)
	savedNow := guardAllowNow
	guardAllowNow = func() time.Time { return now }
	t.Cleanup(func() { guardAllowNow = savedNow })

	baseline := guardAllowOverlay{
		Version:     guardAllowOverlayVersion,
		Allow:       []string{"preexisting_a", "preexisting_b"},
		AllowPrefix: []string{"mcp__safe_"},
		Expiry:      map[string]string{"preexisting_b": now.Add(time.Hour).Format(time.RFC3339)},
	}
	if err := saveGuardAllowOverlay(path, baseline); err != nil {
		t.Fatalf("save baseline overlay: %v", err)
	}

	receipt, err := applyTUIGuardGrant(path, "owned_grant", now)
	if err != nil {
		t.Fatalf("apply TUI guard grant: %v", err)
	}
	if !receipt.Added || receipt.OverlayPath != path || receipt.Tool != "owned_grant" || receipt.ExpiresAt != now.Add(tuiGuardGrantTTL).Format(time.RFC3339) {
		t.Fatalf("receipt = %+v, want exact owned mutation", receipt)
	}

	current, err := loadGuardAllowOverlay(path)
	if err != nil {
		t.Fatalf("load overlay after apply: %v", err)
	}
	current.Allow = append(current.Allow, "later_grant")
	if err := saveGuardAllowOverlay(path, current); err != nil {
		t.Fatalf("save later unrelated grant: %v", err)
	}

	if err := undoTUIGuardGrant(receipt); err != nil {
		t.Fatalf("undo TUI guard grant: %v", err)
	}
	got, err := loadGuardAllowOverlay(path)
	if err != nil {
		t.Fatalf("load overlay after undo: %v", err)
	}
	want := baseline
	want.Allow = append(want.Allow, "later_grant")
	want.Allow = guardAllowNormalize(want.Allow)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("overlay after undo = %#v, want %#v", got, want)
	}
}

func TestTUIGuardGrantUndoRefusesLaterReplacement(t *testing.T) {
	path := t.TempDir() + "/allow.json"
	now := time.Date(2026, 9, 1, 12, 0, 0, 0, time.UTC)
	savedNow := guardAllowNow
	guardAllowNow = func() time.Time { return now }
	t.Cleanup(func() { guardAllowNow = savedNow })

	receipt, err := applyTUIGuardGrant(path, "owned_grant", now)
	if err != nil {
		t.Fatalf("apply TUI guard grant: %v", err)
	}
	current, err := loadGuardAllowOverlay(path)
	if err != nil {
		t.Fatalf("load overlay: %v", err)
	}
	current.Expiry[receipt.Tool] = now.Add(10 * time.Minute).Format(time.RFC3339)
	if err := saveGuardAllowOverlay(path, current); err != nil {
		t.Fatalf("save replacement: %v", err)
	}

	if err := undoTUIGuardGrant(receipt); err == nil {
		t.Fatal("stale undo removed or accepted a later replacement")
	}
	got, err := loadGuardAllowOverlay(path)
	if err != nil {
		t.Fatalf("load overlay after stale undo: %v", err)
	}
	if len(got.Allow) != 1 || got.Allow[0] != receipt.Tool || got.Expiry[receipt.Tool] != now.Add(10*time.Minute).Format(time.RFC3339) {
		t.Fatalf("later replacement changed after stale undo: %#v", got)
	}
}
