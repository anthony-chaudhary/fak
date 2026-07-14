package evebridge

import (
	"bytes"
	"os"
	"testing"
	"testing/fstest"
)

func TestInspectFixtureTrees(t *testing.T) {
	for _, root := range []string{"testdata/source-agent", "testdata/compiled"} {
		manifest, err := InspectFS(os.DirFS(root))
		if err != nil {
			t.Fatalf("%s: %v", root, err)
		}
		if !manifest.OK {
			t.Fatalf("%s: %+v", root, manifest.Diagnostics)
		}
		first := manifest.JSON()
		again, err := InspectFS(os.DirFS(root))
		if err != nil || !bytes.Equal(first, again.JSON()) {
			t.Fatalf("%s fixture is not deterministic: %v", root, err)
		}
	}
}
func TestInspectSourceDeterministicAndComplete(t *testing.T) {
	root := fstest.MapFS{
		"agent/agent.ts":                          {},
		"agent/tools/greet.ts":                    {},
		"agent/connections/github.ts":             {},
		"agent/subagents/researcher/agent.ts":     {},
		"agent/subagents/researcher/tools/sum.ts": {},
		"agent/schedules/daily.ts":                {},
		"agent/channels/web.ts":                   {},
		"agent/evals/quality.ts":                  {},
		"agent/sandbox/workspace/README.md":       {},
	}
	first, err := InspectFS(root)
	if err != nil {
		t.Fatal(err)
	}
	second, err := InspectFS(root)
	if err != nil {
		t.Fatal(err)
	}
	if !first.OK {
		t.Fatalf("inspection failed: %+v", first.Diagnostics)
	}
	if !bytes.Equal(first.JSON(), second.JSON()) {
		t.Fatalf("manifest is not deterministic\nfirst: %s\nsecond: %s", first.JSON(), second.JSON())
	}
	assertSurface(t, first.Tools, "greet", "agent/tools/greet.ts")
	assertSurface(t, first.Tools, "researcher__sum", "agent/subagents/researcher/tools/sum.ts")
	assertSurface(t, first.Connections, "github", "agent/connections/github.ts")
	assertSurface(t, first.Subagents, "researcher", "agent/subagents/researcher/agent.ts")
	assertSurface(t, first.Schedules, "daily", "agent/schedules/daily.ts")
	assertSurface(t, first.Channels, "web", "agent/channels/web.ts")
	assertSurface(t, first.EvalIDs, "quality", "agent/evals/quality.ts")
	if len(first.SandboxMounts) != 1 || first.SandboxMounts[0].RuntimePath != "/workspace" {
		t.Fatalf("sandbox mounts = %+v", first.SandboxMounts)
	}
	if !first.Policy.DefaultDeny || !first.Policy.Network || !first.Policy.File {
		t.Fatalf("policy implications = %+v", first.Policy)
	}
}

func TestInspectCompiledCurrentShapeAndBuiltinActions(t *testing.T) {
	root := fstest.MapFS{
		".eve/compile/compiled-agent-manifest.json": {Data: []byte(`{
			"kind":"eve-agent-compiled-manifest","version":36,
			"tools":[
				{"name":"read_file","logicalPath":"tools/read_file.ts"},
				{"name":"greet","logicalPath":"tools/greet.ts"}
			],
			"connections":[{"connectionName":"github","logicalPath":"connections/github.ts"}],
			"subagents":[{"name":"researcher","logicalPath":"subagents/researcher/agent.ts","agent":{"tools":[{"name":"summarize","logicalPath":"subagents/researcher/tools/summarize.ts"}]}}],
			"schedules":[{"name":"daily","logicalPath":"schedules/daily.ts"}],
			"channels":[{"name":"web","logicalPath":"channels/web.ts"}],
			"eval_ids":[{"name":"quality","logicalPath":"evals/quality.ts"}],
			"sandboxWorkspaces":[{"logicalPath":"sandbox/workspace","sourcePath":"/src/agent/sandbox/workspace"}],
			"disabledFrameworkTools":["read_file","write_file"]
		}`)},
	}
	manifest, err := InspectFS(root)
	if err != nil {
		t.Fatal(err)
	}
	if !manifest.OK {
		t.Fatalf("inspection failed: %+v", manifest.Diagnostics)
	}
	assertSurface(t, manifest.Tools, "researcher__summarize", "subagents/researcher/tools/summarize.ts")
	assertBuiltin(t, manifest.BuiltinTools, "read_file", "override")
	assertBuiltin(t, manifest.BuiltinTools, "write_file", "disable")
	if len(manifest.SandboxMounts) != 1 || manifest.SandboxMounts[0].SourcePath != "/src/agent/sandbox/workspace" {
		t.Fatalf("sandbox mounts = %+v", manifest.SandboxMounts)
	}
	draft, err := manifest.PolicyDraft()
	if err != nil {
		t.Fatal(err)
	}
	if got := string(draft.JSON()); got == "" {
		t.Fatal("empty policy draft")
	}
}

func TestInspectFailuresAreTypedAndFailClosed(t *testing.T) {
	tests := []struct {
		name string
		root fstest.MapFS
		code string
	}{
		{"unsupported layout", fstest.MapFS{"README.md": {}}, CodeLayoutUnsupported},
		{"name collision", fstest.MapFS{"agent/agent.ts": {}, "agent/tools/foo-bar.ts": {}, "agent/tools/foo_bar.ts": {}}, CodeNameCollision},
		{"root-only misplaced", fstest.MapFS{"agent/agent.ts": {}, "agent/subagents/a/connections/x.ts": {}}, CodeRootOnlyMisplaced},
		{"unknown compiled version", fstest.MapFS{".eve/compile/compiled-agent-manifest.json": {Data: []byte(`{"kind":"eve-agent-compiled-manifest","version":999}`)}}, CodeManifestVersionUnsupported},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			manifest, err := InspectFS(tc.root)
			if err != nil {
				t.Fatal(err)
			}
			if manifest.OK {
				t.Fatal("inspection unexpectedly succeeded")
			}
			for _, diagnostic := range manifest.Diagnostics {
				if diagnostic.Code == tc.code && diagnostic.Severity == "fail" {
					return
				}
			}
			t.Fatalf("missing diagnostic %s: %+v", tc.code, manifest.Diagnostics)
		})
	}
}

func assertSurface(t *testing.T, surfaces []Surface, name, evidence string) {
	t.Helper()
	for _, surface := range surfaces {
		if surface.Name == name && surface.Path == evidence {
			return
		}
	}
	t.Fatalf("missing surface %q at %q in %+v", name, evidence, surfaces)
}

func assertBuiltin(t *testing.T, tools []BuiltinTool, name, action string) {
	t.Helper()
	for _, tool := range tools {
		if tool.Name == name && tool.Action == action {
			return
		}
	}
	t.Fatalf("missing builtin action %s=%s in %+v", name, action, tools)
}
