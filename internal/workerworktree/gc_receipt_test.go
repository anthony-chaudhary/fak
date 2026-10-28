package workerworktree

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// fak-test:runtime fast est=100ms
func TestGCReceiptNarrowSelectionAndKeepReasons(t *testing.T) {
	root := gcReceiptTempDir(t)
	now := time.Now().UTC()
	owner := OwnerStamp{PID: 101, LeaseID: "released", CreatedAt: now.Add(-2 * time.Hour)}
	chosen := makeGCWorktree(t, root, "chosen", "1", owner)
	other := makeGCWorktree(t, root, "other", "2", owner)
	dirty := makeGCWorktree(t, root, "dirty", "3", owner)
	evidence := filepath.Join(dirty, "source-evidence.txt")
	if err := os.WriteFile(evidence, []byte("preserve"), 0600); err != nil {
		t.Fatal(err)
	}
	g := &gcGit{list: gcPorcelain(chosen, other, dirty), status: map[string]string{dirty: "?? source-evidence.txt\n"}, heads: map[string]string{chosen: "a", other: "a", dirty: "a"}, ancestors: map[string]int{"a": 0}}
	opts := baseGCOptions(now, root)
	opts.Targets = []string{chosen}
	opts.ReceiptDir = filepath.Join(gcReceiptTempDir(t), "preview")
	preview := GarbageCollect(root, g.run, opts)
	if len(preview.Failures) != 0 || preview.WouldReap != 1 || preview.Reaped != 0 || len(preview.Kept) != 2 {
		t.Fatalf("preview: %+v", preview)
	}
	reasons := map[string]GCReason{}
	for _, row := range preview.Kept {
		reasons[row.Path] = row.ReasonCode
	}
	if reasons[dirty] != "working_tree_dirty" || reasons[other] != "not_selected" {
		t.Fatalf("kept reasons: %v", reasons)
	}
	opts.Apply = true
	opts.ReceiptDir = filepath.Join(gcReceiptTempDir(t), "apply")
	// The destructive seam must see a complete pre-action receipt, not a result.
	g.pruneCheck = func() bool {
		raw, err := os.ReadFile(filepath.Join(opts.ReceiptDir, "before.json"))
		var receipt gcReceipt
		return err == nil && json.Unmarshal(raw, &receipt) == nil && receipt.Phase == "before" && receipt.Report.Reaped == 0 && receipt.Report.WouldReap == 1
	}
	result := GarbageCollect(root, g.run, opts)
	if len(result.Failures) != 0 || result.Reaped != 1 {
		t.Fatalf("apply: %+v", result)
	}
	for _, path := range []string{other, dirty, evidence} {
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("preserved %s: %v", path, err)
		}
	}
	raw, err := os.ReadFile(filepath.Join(opts.ReceiptDir, "result.json"))
	var receipt gcReceipt
	if err != nil || json.Unmarshal(raw, &receipt) != nil || receipt.Phase != "result" || receipt.Report.Reaped != 1 {
		t.Fatalf("result receipt: %s, %v", raw, err)
	}
}

// fak-test:runtime fast est=100ms
func TestGCReceiptAndSelectionRefusalsNeverRemove(t *testing.T) {
	for _, mode := range []string{"existing", "inside", "missing-parent", "duplicate", "unknown", "traversal", "symlink", "inventory-error", "ambiguous", "mixed-unknown", "symlink-target", "whitespace-target"} {
		t.Run(mode, func(t *testing.T) {
			root := gcReceiptTempDir(t)
			now := time.Now().UTC()
			wt := makeGCWorktree(t, root, "safe", "1", OwnerStamp{PID: 101, LeaseID: "released", CreatedAt: now.Add(-2 * time.Hour)})
			g := &gcGit{list: gcPorcelain(wt), heads: map[string]string{wt: "a"}, ancestors: map[string]int{"a": 0}}
			opts := baseGCOptions(now, root)
			opts.Apply = true
			opts.Targets = []string{wt}
			parent := gcReceiptTempDir(t)
			opts.ReceiptDir = filepath.Join(parent, "receipt")
			switch mode {
			case "existing":
				if err := os.Mkdir(opts.ReceiptDir, 0700); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(opts.ReceiptDir, "before.json"), []byte("retained"), 0600); err != nil {
					t.Fatal(err)
				}
			case "inside":
				opts.ReceiptDir = filepath.Join(wt, "receipt")
			case "missing-parent":
				opts.ReceiptDir = filepath.Join(parent, "missing", "receipt")
			case "duplicate":
				opts.Targets = []string{wt, wt}
			case "unknown":
				opts.Targets = []string{parent}
			case "mixed-unknown":
				opts.Targets = []string{wt, parent}
			case "symlink-target":
				alias := filepath.Join(parent, "target")
				if err := os.Symlink(wt, alias); err != nil {
					t.Skipf("symlink unavailable: %v", err)
				}
				opts.Targets = []string{alias}
			case "whitespace-target":
				opts.Targets = []string{wt + " "}
			case "symlink":
				alias := filepath.Join(parent, "alias")
				if err := os.Symlink(root, alias); err != nil {
					t.Skipf("symlink unavailable: %v", err)
				}
				opts.ReceiptDir = filepath.Join(alias, "receipt")
			case "ambiguous":
				g.list = gcPorcelain(wt, wt)
			case "traversal":
				opts.ReceiptDir = parent + string(filepath.Separator) + ".." + string(filepath.Separator) + filepath.Base(parent) + string(filepath.Separator) + "receipt"
			}
			runner := GitRunner(g.run)
			if mode == "inventory-error" {
				runner = func(string, []string) (int, string) { return 1, "unavailable" }
			}
			report := GarbageCollect(root, runner, opts)
			if len(report.Failures) == 0 || report.Reaped != 0 || g.callsWithPrefix("worktree", "prune") != 0 {
				t.Fatalf("refusal: %+v", report)
			}
			if _, err := os.Stat(wt); err != nil {
				t.Fatal(err)
			}
			if mode == "existing" {
				raw, err := os.ReadFile(filepath.Join(opts.ReceiptDir, "before.json"))
				if err != nil || string(raw) != "retained" {
					t.Fatalf("existing evidence changed: %s %v", raw, err)
				}
			}
		})
	}
}

// fak-test:runtime fast est=100ms
func TestGCReceiptPreservesApplyRevalidation(t *testing.T) {
	for _, changed := range []string{"owner", "consumer"} {
		t.Run(changed, func(t *testing.T) {
			root := gcReceiptTempDir(t)
			now := time.Now().UTC()
			owner := OwnerStamp{PID: 101, LeaseID: "released", CreatedAt: now.Add(-2 * time.Hour)}
			wt := makeGCWorktree(t, root, "safe", "1", owner)
			g := &gcGit{list: gcPorcelain(wt), heads: map[string]string{wt: "a"}, ancestors: map[string]int{"a": 0}}
			opts := baseGCOptions(now, root)
			opts.Apply = true
			opts.Targets = []string{wt}
			opts.ReceiptDir = filepath.Join(gcReceiptTempDir(t), "receipt")
			probes := 0
			opts.PathActive = func(string) (bool, error) {
				probes++
				if probes == 1 && changed == "owner" {
					owner.PID = 303
					if err := writeOwnerStamp(wt, owner); err != nil {
						t.Fatal(err)
					}
				}
				return probes > 1 && changed == "consumer", nil
			}
			report := GarbageCollect(root, g.run, opts)
			if len(report.Failures) != 1 || report.Reaped != 0 {
				t.Fatalf("revalidation: %+v", report)
			}
			if _, err := os.Stat(wt); err != nil {
				t.Fatal(err)
			}
		})
	}
}

// Resolve platform temp aliases so the fixture exercises canonical CLI inputs.
func gcReceiptTempDir(t *testing.T) string {
	t.Helper()
	path, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	return path
}

// fak-test:runtime fast est=100ms
func TestGCReceiptSyncFailureAndPartialOutcome(t *testing.T) {
	for _, mode := range []string{"parent-sync", "before-sync", "result-sync", "prune"} {
		t.Run(mode, func(t *testing.T) {
			root := gcReceiptTempDir(t)
			now := time.Now().UTC()
			wt := makeGCWorktree(t, root, "safe", "1", OwnerStamp{PID: 101, LeaseID: "released", CreatedAt: now.Add(-2 * time.Hour)})
			g := &gcGit{list: gcPorcelain(wt), heads: map[string]string{wt: "a"}, ancestors: map[string]int{"a": 0}}
			opts := baseGCOptions(now, root)
			opts.Apply = true
			opts.Targets = []string{wt}
			opts.ReceiptDir = filepath.Join(gcReceiptTempDir(t), "receipt")
			syncs := 0
			opts.syncReceiptDirectory = func(string) error {
				syncs++
				// Both required pre-action barriers must occur while the target still exists.
				if syncs <= 2 {
					if _, err := os.Stat(wt); err != nil {
						t.Fatalf("mutation preceded receipt durability: %v", err)
					}
				}
				if (mode == "parent-sync" && syncs == 1) || (mode == "before-sync" && syncs == 2) || (mode == "result-sync" && syncs == 3) {
					return fmt.Errorf("injected sync failure")
				}
				return nil
			}
			if mode == "prune" {
				g.pruneCheck = func() bool { return false }
			}
			report := GarbageCollect(root, g.run, opts)
			if len(report.Failures) != 1 {
				t.Fatalf("expected one failure: %+v", report)
			}
			if mode == "parent-sync" || mode == "before-sync" {
				if report.Reaped != 0 || g.callsWithPrefix("worktree", "prune") != 0 {
					t.Fatalf("mutated after failed barrier: %+v", report)
				}
				if _, err := os.Stat(wt); err != nil {
					t.Fatal(err)
				}
			} else {
				if report.Reaped != 1 || len(report.Worktrees) != 1 || !report.Worktrees[0].Removed {
					t.Fatalf("lost removal outcome: %+v", report)
				}
				if _, err := os.Stat(wt); !os.IsNotExist(err) {
					t.Fatalf("expected verified absence: %v", err)
				}
				if mode == "prune" {
					if report.Worktrees[0].ReasonCode != "removed_with_cleanup_failure" {
						t.Fatalf("partial result: %+v", report)
					}
					raw, err := os.ReadFile(filepath.Join(opts.ReceiptDir, "result.json"))
					var receipt gcReceipt
					if err != nil || json.Unmarshal(raw, &receipt) != nil || receipt.Report.Reaped != 1 || len(receipt.Report.Failures) != 1 {
						t.Fatalf("untruthful receipt: %s %v", raw, err)
					}
				}
			}
		})
	}
}
