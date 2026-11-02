package agent

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/codetools"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

type repoMapWitnessPlanner struct {
	seen             [][]Message
	readAfterPreload bool
}

func (p *repoMapWitnessPlanner) Model() string { return "repomap-fixture" }
func (p *repoMapWitnessPlanner) Complete(_ context.Context, messages []Message, _ []ToolDef, _ ...SampleOpt) (*Completion, error) {
	p.seen = append(p.seen, append([]Message(nil), messages...))
	if p.readAfterPreload && len(p.seen) == 1 {
		var call ToolCall
		call.ID = "ordinary-read"
		call.Type = "function"
		call.Function.Name = codetools.ToolRead
		call.Function.Arguments = `{"file_path":"a.go"}`
		return &Completion{Message: Message{Role: RoleAssistant, ToolCalls: []ToolCall{call}}}, nil
	}
	return &Completion{Message: Message{Role: RoleAssistant, Content: "done"}}, nil
}

// fak-test:runtime fast est=1s lane=default
func TestRepoMapPreloadNativeBoundary(t *testing.T) {
	DisarmCodeTools()
	vdso.Default.BumpWorld()
	t.Cleanup(func() { DisarmCodeTools(); vdso.Default.BumpWorld(); Configure() })
	root := t.TempDir()
	// No server, HTTP planner, project discovery or external process is involved.
	const a = "package p\n// BODY_PRIVATE_TO_READ_RECEIPT\nfunc Entry(){ Target() }\n"
	const b = "package p\nfunc Target(){}\n"
	for name, body := range map[string]string{"a.go": a, "b.go": b} {
		if err := os.WriteFile(filepath.Join(root, name), []byte(body), 0600); err != nil {
			t.Fatal(err)
		}
	}
	catalog, err := ArmCodeToolsWithOptions(CodeToolsOptions{Root: root, EnableSkills: false})
	if err != nil {
		t.Fatal(err)
	}
	opt, err := WithRepoMapPreload(RepoMapPreload{Files: []string{"a.go", "b.go"}, Tokens: 256})
	if err != nil {
		t.Fatal(err)
	}
	p := &repoMapWitnessPlanner{readAfterPreload: true}
	var log []traceEvent
	m, err := RunArm(context.Background(), p, "inspect Target", true, 3, &log, WithToolCatalog(catalog), opt)
	if err != nil {
		t.Fatal(err)
	}
	if len(p.seen) != 2 || m.ToolCalls != 3 || m.EngineCalls < 2 {
		t.Fatalf("normal native read boundary not used: calls=%d engines=%d model=%d", m.ToolCalls, m.EngineCalls, len(p.seen))
	}
	preloadReads, maps := 0, 0
	for _, msg := range p.seen[0] {
		if msg.Role == RoleSystem && strings.Contains(msg.Content, "untrusted-repository-map") {
			t.Fatal("repository map was promoted to system authority")
		}
		if msg.Role == RoleTool && strings.HasPrefix(msg.ToolCallID, "fak-repomap-preload-") {
			var rec struct {
				Schema, Status, Version, SHA256 string
				Bytes                           int
			}
			if err := json.Unmarshal([]byte(msg.Content), &rec); err != nil {
				t.Fatal(err)
			}
			if rec.Schema != "fak.repomap.read/1" || rec.Status != "ok" ||
				!isFullFileVersion(rec.Version) || len(rec.SHA256) != 64 || rec.Bytes <= 0 {
				t.Fatalf("bad compact native read receipt: %s", msg.Content)
			}
			preloadReads++
		}
		if strings.HasPrefix(msg.Content, "[untrusted-repository-map]") {
			maps++
			if msg.Role != RoleUser || len(msg.Content) > 256*4 || !strings.Contains(msg.Content, `"Target"`) ||
				strings.Contains(msg.Content, "BODY_PRIVATE_TO_READ_RECEIPT") {
				t.Fatal("bad derived map")
			}
		}
	}
	if preloadReads != 2 || maps != 1 {
		t.Fatalf("preload=%d map=%d", preloadReads, maps)
	}
	foundOrdinary := false
	for _, msg := range p.seen[1] {
		if msg.ToolCallID == "ordinary-read" && msg.Role == RoleTool {
			var payload struct{ Content string }
			if json.Unmarshal([]byte(msg.Content), &payload) != nil || payload.Content != a {
				t.Fatal("ordinary tool result changed")
			}
			foundOrdinary = true
		}
	}
	if !foundOrdinary {
		t.Fatal("ordinary read missing")
	}
	if len(codeToolRows(log)) != 3 {
		t.Fatal("native audit trace missing")
	}

	// An ordinary session stop precedes even the first synthetic Read.
	stopped := &repoMapWitnessPlanner{}
	sm, err := RunArm(context.Background(), stopped, "inspect", true, 1, nil, WithToolCatalog(catalog), opt,
		WithSessionGate(SessionGate{Decide: func(string) (int, bool, int, string) { return 0, false, 0, "STOPPED" }}, "repomap-stop"))
	if err != nil || len(stopped.seen) != 0 || sm.ToolCalls != 0 {
		t.Fatal("preload bypassed session gate")
	}

	// Existing native denial remains decisive; no raw/file fallback or model call.
	denied := &repoMapWitnessPlanner{}
	dm, err := RunArm(context.Background(), denied, "inspect", true, 1, nil, WithToolCatalog(catalog), opt,
		WithPolicySnapshot(adjudicator.Policy{Posture: adjudicator.PostureFailClosed}))
	if err == nil || len(denied.seen) != 0 || dm.EngineCalls != 0 || dm.Denies != 1 {
		t.Fatalf("native refusal not preserved: err=%v engines=%d denies=%d model=%d", err, dm.EngineCalls, dm.Denies, len(denied.seen))
	}

	// An oversized first selected file stops serial preload before the next
	// call, while the native read itself still retains its ordinary receipt.
	if err := os.WriteFile(filepath.Join(root, "large.go"), []byte("package p\n"+strings.Repeat(" ", ctxmmu.RepoMapMaxFileBytes)), 0600); err != nil {
		t.Fatal(err)
	}
	largeOpt, err := WithRepoMapPreload(RepoMapPreload{Files: []string{"large.go", "b.go"}})
	if err != nil {
		t.Fatal(err)
	}
	large := &repoMapWitnessPlanner{}
	lm, err := RunArm(context.Background(), large, "inspect", true, 1, nil, WithToolCatalog(catalog), largeOpt)
	if err == nil || len(large.seen) != 0 || lm.ToolCalls != 1 {
		t.Fatal("oversize did not stop before the next call/model")
	}

	// A partial/ranged or differently-routed receipt is never a whole-file
	// preload input, even if its JSON contains plausible Go text.
	for _, tc := range []struct {
		name, args, engine, admit string
		truncated                 bool
	}{
		{"truncated", `{"file_path":"a.go"}`, codetools.EngineRead, "", true},
		{"range", `{"file_path":"a.go","offset":1}`, codetools.EngineRead, "", false},
		{"fallback engine", `{"file_path":"a.go"}`, FakReadEngineID, "", false},
		{"result transformed", `{"file_path":"a.go"}`, codetools.EngineRead, "transformed", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := &armRunner{repoMapPreload: &repoMapPreloadState{tools: armedCodeTools.Load(), selected: map[string]bool{"a.go": true}}}
			body, _ := json.Marshal(map[string]any{"file_path": "a.go", "content": a, "bytes": len(a), "version": "fv1:" + strings.Repeat("0", 64), "truncated": tc.truncated})
			result := &abi.Result{Status: abi.StatusOK, Meta: map[string]string{"engine": tc.engine, "admit": tc.admit}}
			if _, err := r.compactRepoMapRead(codetools.ToolRead, tc.args, string(body), result, false, false); err == nil || len(r.repoMapPreload.files) != 0 {
				t.Fatal("incomplete/unqualified input accepted")
			}
		})
	}

	// The zero option adds neither Read calls nor new transcript entries.
	plain := &repoMapWitnessPlanner{}
	pm, err := RunArm(context.Background(), plain, "plain", true, 1, nil, WithToolCatalog(catalog))
	if err != nil || pm.ToolCalls != 0 || len(plain.seen) != 1 {
		t.Fatal("unset path changed")
	}
	want := resolveRunConfig([]RunOption{WithToolCatalog(catalog)}).seedMessages("plain")
	if !reflect.DeepEqual(plain.seen[0], want) {
		t.Fatal("unset transcript not byte-identical")
	}

	// Validator cannot grant an escaping path or a too-large context budget.
	if _, err := WithRepoMapPreload(RepoMapPreload{Files: []string{"../escape.go"}}); err == nil {
		t.Fatal("path accepted")
	}
	if _, err := WithRepoMapPreload(RepoMapPreload{Files: []string{"a.go"}, Tokens: ctxmmu.RepoMapMaxTokens + 1}); err == nil {
		t.Fatal("budget accepted")
	}
}
