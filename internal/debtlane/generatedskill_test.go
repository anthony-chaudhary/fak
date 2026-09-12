package debtlane

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// adapterContent is the exact shape emitted by `fak project-assets sync` for a
// generated discovery adapter under .agents/skills/<name>/SKILL.md.
const adapterContent = `---
name: realskill
metadata:
  generated-by: fak project-assets sync
  canonical: ../../../.claude/skills/realskill/SKILL.md
---

# realskill

This generated discovery adapter contains no maintained workflow body.
`

// canonicalContent is an authored skill body with a witness command, so it is a
// healthy canonical lane that must survive generated-adapter filtering.
const canonicalContent = `---
name: realskill
description: canonical authored skill
---

# realskill

## Verification

go test ./...
`

// TestGeneratedSkillAdapterYieldsNoLane is the second phantom-debt regression:
// a generated .agents/skills adapter yields neither a lane nor findings, while
// the canonical .claude/skills source lane is preserved.
func TestGeneratedSkillAdapterYieldsNoLane(t *testing.T) {
	tmp := t.TempDir()

	canonicalDir := filepath.Join(tmp, ".claude", "skills", "realskill")
	if err := os.MkdirAll(canonicalDir, 0o755); err != nil {
		t.Fatalf("mkdir canonical skill dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(canonicalDir, "SKILL.md"), []byte(canonicalContent), 0o644); err != nil {
		t.Fatalf("write canonical SKILL.md: %v", err)
	}

	adapterDir := filepath.Join(tmp, ".agents", "skills", "realskill")
	if err := os.MkdirAll(adapterDir, 0o755); err != nil {
		t.Fatalf("mkdir adapter skill dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(adapterDir, "SKILL.md"), []byte(adapterContent), 0o644); err != nil {
		t.Fatalf("write adapter SKILL.md: %v", err)
	}

	report, err := Scan(Options{WorkspaceRoot: tmp, ExpandedBreadth: true})
	if err != nil {
		t.Fatalf("Scan returned error: %v", err)
	}

	for _, lane := range report.Lanes {
		unit := filepath.ToSlash(lane.UnitOfWork)
		if unit == ".agents/skills/realskill" || strings.HasPrefix(unit, ".agents/skills/realskill/") {
			t.Errorf("generated adapter produced a debt lane: %q (lane %q)", unit, lane.Lane)
		}
		if strings.Contains(lane.Lane, "realskill") && strings.Contains(lane.Lane, "agents") {
			t.Errorf("generated adapter produced a debt lane by name: %q (unit %q)", lane.Lane, unit)
		}
	}

	foundCanonical := false
	for _, lane := range report.Lanes {
		unit := filepath.ToSlash(lane.UnitOfWork)
		if unit == ".claude/skills/realskill" || strings.HasPrefix(unit, ".claude/skills/realskill/") {
			foundCanonical = true
			break
		}
	}
	if !foundCanonical {
		t.Errorf("expected canonical .claude/skills/realskill lane to be preserved; got %d lanes", len(report.Lanes))
	}

	if isGeneratedArtifact("x/SKILL.md", []byte(adapterContent)) != true {
		t.Errorf("isGeneratedArtifact did not detect generated project-assets adapter content")
	}
}
