package ops

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/projectassets"
)

func setupTestAssetRepo(t *testing.T) string {
	t.Helper()
	r := t.TempDir()
	m := projectassets.Manifest{
		Schema: "fak-project-assets/1",
		Skills: projectassets.SkillPolicy{
			CanonicalRoot: ".claude/skills",
			CodexRoot:     ".agents/skills",
			Include:       []string{"SKILL.md"},
		},
		Memories: projectassets.Policy{
			CanonicalRoot:  ".claude/memory",
			Include:        []string{"*.md"},
			Exclude:        []projectassets.Exclusion{{Pattern: "MEMORY.md", Reason: "index"}},
			StartupCommand: "fak memory recall --intent <task> --json",
		},
		GoalPrompts: projectassets.Policy{
			CanonicalRoot: ".claude/goal-prompts",
			Include:       []string{"template.md"},
			Exclude:       []projectassets.Exclusion{{Pattern: "resolve-[0-9]*.md", Reason: "run fuel"}},
		},
		Harnesses: map[string]projectassets.Harness{
			"codex":      {},
			"fak-native": {},
			"opencode":   {},
		},
	}
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}

	writeFile := func(relPath, body string) {
		p := filepath.Join(r, filepath.FromSlash(relPath))
		if err := os.MkdirAll(filepath.Dir(p), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0644); err != nil {
			t.Fatal(err)
		}
	}

	writeFile(projectassets.ManifestPath, string(b))
	writeFile(".claude/skills/demo/SKILL.md", "---\nname: demo\ndescription: Demo skill.\n---\nbody\n")
	writeFile(".claude/memory/MEMORY.md", "index\n")
	writeFile(".claude/memory/durable.md", "memory\n")
	writeFile(".claude/goal-prompts/template.md", "template\n")
	return r
}

func TestEngineTickAssetSync(t *testing.T) {
	repo := setupTestAssetRepo(t)
	adapterPath := filepath.Join(repo, ".agents", "skills", "demo", "SKILL.md")
	if _, err := os.Stat(adapterPath); !os.IsNotExist(err) {
		t.Fatalf("expected adapter to not exist before sync, got err=%v", err)
	}

	engine, err := NewEngine(repo, DefaultConfig())
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}

	// 1. First Tick with forceAll = true should detect missing adapters, synchronize them, and record ActionAssetSync.
	ctx := context.Background()
	if err := engine.Tick(ctx, true); err != nil {
		t.Fatalf("engine.Tick failed: %v", err)
	}

	if _, err := os.Stat(adapterPath); err != nil {
		t.Fatalf("expected adapter %s to be created by background asset sync: %v", adapterPath, err)
	}

	events, err := engine.Ledger.QueryEvents(1 * time.Hour)
	if err != nil {
		t.Fatalf("QueryEvents: %v", err)
	}

	var assetSyncEvents []Event
	for _, ev := range events {
		if ev.ActionType == ActionAssetSync {
			assetSyncEvents = append(assetSyncEvents, ev)
		}
	}

	if len(assetSyncEvents) != 1 {
		t.Fatalf("expected 1 ActionAssetSync event, got %d", len(assetSyncEvents))
	}

	ev := assetSyncEvents[0]
	if ev.Schema != EventSchemaV1 {
		t.Errorf("expected schema %s, got %s", EventSchemaV1, ev.Schema)
	}
	expectedDetails := "synchronized portable skill adapters for opencode and codex workers (3 skills)"
	if !strings.Contains(ev.Details, expectedDetails) {
		t.Errorf("expected details %q, got %q", expectedDetails, ev.Details)
	}

	// 2. Second Tick when adapters are in parity should NOT record another ActionAssetSync event.
	if err := engine.Tick(ctx, true); err != nil {
		t.Fatalf("second engine.Tick failed: %v", err)
	}

	eventsAfter, err := engine.Ledger.QueryEvents(1 * time.Hour)
	if err != nil {
		t.Fatalf("QueryEvents: %v", err)
	}

	var assetSyncEventsAfter []Event
	for _, ev := range eventsAfter {
		if ev.ActionType == ActionAssetSync {
			assetSyncEventsAfter = append(assetSyncEventsAfter, ev)
		}
	}

	if len(assetSyncEventsAfter) != 1 {
		t.Fatalf("expected still 1 ActionAssetSync event after second tick, got %d", len(assetSyncEventsAfter))
	}

	// 3. Fail-soft verification: an uninitialized repo without manifest does not fail Tick.
	emptyRepo := t.TempDir()
	emptyEngine, err := NewEngine(emptyRepo, DefaultConfig())
	if err != nil {
		t.Fatalf("NewEngine for emptyRepo: %v", err)
	}
	if err := emptyEngine.Tick(ctx, true); err != nil {
		t.Errorf("expected Tick on unconfigured repo to fail-soft, got error: %v", err)
	}
}
