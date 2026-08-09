package contextq

import (
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
)

var claudeCatalog = ToolCatalog{
	"Read": {}, "Bash": {}, "PowerShell": {}, "Write": {}, "Edit": {},
	"Grep": {}, "Glob": {}, "WebFetch": {}, "WebSearch": {}, "Agent": {},
	"AskUserQuestion": {}, "mcp__fak__fak_feature_query": {},
	"mcp__fak__fak_capabilities": {}, "mcp__fak__fak_index_docs": {},
	"mcp__fak__fak_index_leaves": {}, "mcp__fak__fak_index_verbs": {},
	"mcp__fak__fak_index_claims": {},
}

func TestResolveAllowedToolsExactAndUnknownFails(t *testing.T) {
	src := []byte("---\nname: x\nallowed-tools: Write, Read, Write\n---\nbody\n")
	got, err := ResolveAllowedTools(src, claudeCatalog)
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"Read", "Write"}; !reflect.DeepEqual(got.Tools, want) {
		t.Fatalf("tools=%v want=%v", got.Tools, want)
	}
	_, err = ResolveAllowedTools([]byte("---\nallowed-tools: Read, Mystery\n---\n"), claudeCatalog)
	if err == nil || !strings.Contains(err.Error(), `unknown allowed-tool "Mystery"`) {
		t.Fatalf("unknown tool error=%v", err)
	}
}

func TestCheckedInSkillsFaultSetEqualsFrontmatterFence(t *testing.T) {
	root := filepath.Join("..", "..", ".claude", "skills")
	paths, err := filepath.Glob(filepath.Join(root, "*", "SKILL.md"))
	if err != nil || len(paths) == 0 {
		t.Fatalf("skill corpus: paths=%d err=%v", len(paths), err)
	}
	for _, path := range paths {
		path := path
		t.Run(filepath.Base(filepath.Dir(path)), func(t *testing.T) {
			src, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(string(src), "allowed-tools:") {
				t.Skip("skill declares no allowed-tools fence")
			}
			got, err := ResolveAllowedTools(src, claudeCatalog)
			if err != nil {
				t.Fatal(err)
			}
			var want []string
			for _, line := range strings.Split(string(src), "\n") {
				if strings.HasPrefix(strings.TrimSpace(line), "allowed-tools:") {
					for _, v := range strings.Split(strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(line), "allowed-tools:")), ",") {
						want = append(want, strings.TrimSpace(v))
					}
					break
				}
			}
			want = uniqueSorted(want)
			if !reflect.DeepEqual(got.Tools, want) {
				t.Fatalf("fault set=%v fence=%v", got.Tools, want)
			}
		})
	}
}

func uniqueSorted(in []string) []string {
	m := map[string]bool{}
	for _, v := range in {
		if v != "" {
			m[v] = true
		}
	}
	out := make([]string, 0, len(m))
	for v := range m {
		out = append(out, v)
	}
	sort.Strings(out)
	return out
}
