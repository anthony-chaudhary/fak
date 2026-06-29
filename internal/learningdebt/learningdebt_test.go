package learningdebt

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func fixturePayload(t *testing.T) map[string]any {
	t.Helper()
	const raw = `{
	  "schema": "fleet-learning-scorecard/1",
	  "verdict": "ACTION",
	  "corpus": {
	    "priorities": [
	      {"path": "docs/fak/tutorial.md", "priority": 1.5},
	      {"path": "docs/fak/policy-guide.md", "priority": 0.5}
	    ]
	  },
	  "docs": [
	    {
	      "path": "docs/fak/tutorial.md",
	      "score": 55.2,
	      "grade": "D",
	      "defects": [
	        "orientation: no orientation signpost (audience / prereq / TL;DR / 'you'll be able to')",
	        "runnable: teaches no runnable command or code block (prose-only)"
	      ]
	    },
	    {
	      "path": "docs/fak/policy-guide.md",
	      "score": 61,
	      "grade": "C",
	      "defects": [
	        "worked: no worked example / lab / checkpoint / expected-output (tells but never shows)"
	      ]
	    }
	  ],
	  "coverage": {
	    "defects": [
	      "orphan lesson (unreachable from any front door): docs/fak/orphan.md",
	      "uncovered learning topic: migration"
	    ]
	  },
	  "stamp_freshness": {
	    "stale_stamp": true,
	    "flag": "stale-stamp",
	    "doc": "docs/LEARNING-SCORECARD.md",
	    "reason": "stale-stamp: docs/LEARNING-SCORECARD.md stamp 2026-06-01 is 28d old"
	  }
	}`
	var m map[string]any
	if err := json.Unmarshal([]byte(raw), &m); err != nil {
		t.Fatalf("fixture parse: %v", err)
	}
	return m
}

func TestExtractDefectsIncludesDocCoverageAndStamp(t *testing.T) {
	defects := ExtractDefects(fixturePayload(t))
	if len(defects) != 6 {
		t.Fatalf("got %d defects, want 6", len(defects))
	}
	first := defects[0]
	if first.Doc != "docs/fak/tutorial.md" || first.Class != "orientation" {
		t.Fatalf("first defect = doc %q class %q, want prioritized tutorial orientation", first.Doc, first.Class)
	}
	var sawCoverage, sawTopic, sawStamp bool
	for _, d := range defects {
		switch {
		case d.Doc == "docs/fak/orphan.md" && d.Source == "coverage":
			sawCoverage = true
		case d.Doc == "topic:migration" && d.Class == "uncovered learning topic":
			sawTopic = true
		case d.Doc == "docs/LEARNING-SCORECARD.md" && d.Class == "stale-stamp":
			sawStamp = true
		}
	}
	if !sawCoverage || !sawTopic || !sawStamp {
		t.Fatalf("coverage/topic/stamp defects not all found: coverage=%v topic=%v stamp=%v", sawCoverage, sawTopic, sawStamp)
	}
}

func TestPlanAppliesCapAfterDedup(t *testing.T) {
	defects := ExtractDefects(fixturePayload(t))
	plan, stats := BuildPlan(defects, SeenCache{}, nil, 2, "scorecard.json")
	if len(plan) != 2 {
		t.Fatalf("planned %d, want cap 2", len(plan))
	}
	if stats.SkippedCap != len(defects)-2 {
		t.Fatalf("cap skipped %d, want %d", stats.SkippedCap, len(defects)-2)
	}
	if plan[0].Doc != "docs/fak/tutorial.md" {
		t.Fatalf("first planned doc = %q, want highest-priority doc", plan[0].Doc)
	}
}

func TestSeenCacheDedupsSecondRun(t *testing.T) {
	payload := fixturePayload(t)
	defects := ExtractDefects(payload)[:2]
	cache := SeenCache{Schema: SeenSchema, Seen: map[string]SeenRecord{}}
	plan, stats := BuildPlan(defects, cache, nil, 10, "scorecard.json")
	if len(plan) != 2 || stats.Planned != 2 {
		t.Fatalf("first plan len=%d stats=%+v, want 2", len(plan), stats)
	}
	synced := []SyncRow{
		{Key: plan[0].Key, OK: true, Stdout: "https://example.test/issues/1"},
		{Key: plan[1].Key, OK: true, Stdout: "https://example.test/issues/2"},
	}
	MarkSuccessful(&cache, plan, synced, time.Date(2026, 6, 29, 1, 2, 3, 0, time.UTC))

	dir := t.TempDir()
	path := filepath.Join(dir, "seen.json")
	if err := SaveSeen(path, cache); err != nil {
		t.Fatalf("save seen: %v", err)
	}
	loaded, err := LoadSeen(path)
	if err != nil {
		t.Fatalf("load seen: %v", err)
	}
	again, stats := BuildPlan(defects, loaded, nil, 10, "scorecard.json")
	if len(again) != 0 {
		t.Fatalf("second plan len=%d, want 0", len(again))
	}
	if stats.SkippedSeen != 2 {
		t.Fatalf("second skipped_seen=%d, want 2", stats.SkippedSeen)
	}
}

func TestIssueBodyCitesExactDocClassAndDefect(t *testing.T) {
	defect := ExtractDefects(fixturePayload(t))[0]
	body := IssueBody(defect, "scorecard.json")
	for _, want := range []string{
		"Doc/topic: `docs/fak/tutorial.md`",
		"Defect class: `orientation`",
		"orientation: no orientation signpost",
		"<!-- fak-learning-debt-key: " + defect.Key + " -->",
	} {
		if !strings.Contains(body, want) {
			t.Fatalf("body missing %q\n%s", want, body)
		}
	}
}

func TestExistingIssueMarkerDedups(t *testing.T) {
	defects := ExtractDefects(fixturePayload(t))
	existing := []Issue{{Number: 12, Body: "<!-- fak-learning-debt-key: " + defects[0].Key + " -->"}}
	plan, stats := BuildPlan(defects[:1], SeenCache{}, existing, 10, "scorecard.json")
	if len(plan) != 0 {
		t.Fatalf("plan len=%d, want 0", len(plan))
	}
	if stats.SkippedIssueBody != 1 {
		t.Fatalf("issue-body skipped=%d, want 1", stats.SkippedIssueBody)
	}
}

func TestSyncUsesInjectedRunner(t *testing.T) {
	defect := ExtractDefects(fixturePayload(t))[0]
	plan, _ := BuildPlan([]Defect{defect}, SeenCache{}, nil, 1, "scorecard.json")
	var calls [][]string
	rows := Sync(plan, "owner/repo", []string{"learning-debt"}, func(args []string) (string, string, bool) {
		calls = append(calls, args)
		return "https://example.test/issues/1", "", true
	})
	if len(rows) != 1 || !rows[0].OK {
		t.Fatalf("sync rows = %+v, want one ok", rows)
	}
	joined := strings.Join(calls[0], " ")
	for _, want := range []string{"issue create", "--repo owner/repo", "--label learning-debt", plan[0].Title} {
		if !strings.Contains(joined, want) {
			t.Fatalf("gh args missing %q: %v", want, calls[0])
		}
	}
}

func TestDryRunDoesNotWriteSeenCacheViaPlanOnly(t *testing.T) {
	path := filepath.Join(t.TempDir(), "missing", "seen.json")
	if _, err := LoadSeen(path); err != nil {
		t.Fatalf("missing cache should load empty: %v", err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("dry-run planning should not create cache, stat err=%v", err)
	}
}
