package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/skilldiscovery"
)

// fak-test:runtime fast est=1s lane=default
// Declared estimate only; the command dependency is injected, with no network/client launch.
func TestSkillPullCommand(t *testing.T) {
	pin := strings.Repeat("a", 64)
	root := t.TempDir()
	var out, errw bytes.Buffer
	calls := 0
	puller := func(ctx context.Context, o skilldiscovery.Options) (skilldiscovery.Receipt, error) {
		calls++
		if o.Source != "https://skills.example.test/catalog/" || o.IndexSHA256 != pin || o.CacheRoot != root || o.ValidateSkill == nil {
			t.Fatalf("lost explicit operator binding: %+v", o)
		}
		entry := []byte("---\nname: safe\ndescription: Harmless fixture\n---\nOnly fixture data.\n")
		if err := o.ValidateSkill("safe", entry); err != nil {
			t.Fatal(err)
		}
		skillDir := filepath.Join(root, "safe")
		if err := os.Mkdir(skillDir, 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(skillDir, "SKILL.md"), entry, 0600); err != nil {
			t.Fatal(err)
		}
		return skilldiscovery.Receipt{Schema: "fak-skill-pull/1", IndexSHA256: pin, SkillsRoot: root, Activated: false}, nil
	}
	args := []string{"--source", "https://skills.example.test/catalog/", "--index-sha256", pin, "--cache-root", root, "--json"}
	if code := runSkillPullWith(&out, &errw, args, puller); code != 0 || calls != 1 {
		t.Fatalf("code=%d calls=%d error=%q", code, calls, errw.String())
	}
	var receipt skilldiscovery.Receipt
	if err := json.Unmarshal(out.Bytes(), &receipt); err != nil {
		t.Fatal(err)
	}
	if receipt.Activated || receipt.SkillsRoot != root {
		t.Fatalf("receipt=%+v", receipt)
	}
	// This is the same extra-dir registry seam used by agent/chat --skills-dir.
	// Discover/Execute reads a harmless definition; it does not grant tool permissions.
	reg, err := agent.DiscoverSkills(t.TempDir(), receipt.SkillsRoot)
	if err != nil {
		t.Fatal(err)
	}
	got, err := reg.Execute("safe")
	if err != nil || !strings.Contains(got, "Only fixture data.") {
		t.Fatalf("consumer=%q err=%v", got, err)
	}
	t.Run("argument refusal precedes acquisition", func(t *testing.T) {
		for _, argv := range [][]string{{}, {"--source", "https://skills.example.test/"}, {"--source", "x", "--index-sha256", pin, "--cache-root", root, "unexpected"}, {"--json=https://secret-token.invalid/"}} {
			called := false
			var out, errw bytes.Buffer
			code := runSkillPullWith(&out, &errw, argv, func(context.Context, skilldiscovery.Options) (skilldiscovery.Receipt, error) {
				called = true
				return skilldiscovery.Receipt{}, nil
			})
			if code != 2 || called || strings.Contains(errw.String(), "secret-token") {
				t.Fatalf("code=%d called=%v", code, called)
			}
		}
	})
	t.Run("parser refuses canonical import and redacts diagnostics", func(t *testing.T) {
		for _, entry := range []string{
			"---\nname: wrong\n---\nbody",
			"---\nname: safe\nmetadata:\n  canonical: ../../private-secret\n---\nbody",
			"secret-not-frontmatter",
		} {
			err := validatePulledSkill("safe", []byte(entry))
			if err == nil || strings.Contains(err.Error(), "secret") {
				t.Fatalf("bad or leaked error: %v", err)
			}
		}
		var out, errw bytes.Buffer
		code := runSkillPullWith(&out, &errw, args, func(context.Context, skilldiscovery.Options) (skilldiscovery.Receipt, error) {
			return skilldiscovery.Receipt{}, errors.New("URL https://secret-token.invalid/")
		})
		if code != 1 || strings.Contains(errw.String(), "secret-token") || out.Len() != 0 {
			t.Fatalf("code=%d stderr=%q", code, errw.String())
		}
	})
}
