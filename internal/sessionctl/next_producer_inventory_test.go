package sessionctl_test

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
)

type syntheticProducer struct {
	File    string
	Role    string
	Payload string
}

type producerWitness struct {
	Producer    syntheticProducer
	Authorities []callAuthority
}

type callAuthority struct {
	File string
	Call string
}

func TestModelFacingSyntheticProducersHaveNextWitnesses(t *testing.T) {
	root := filepath.Clean(filepath.Join("..", ".."))
	var got []syntheticProducer
	for _, dir := range []string{"internal/agent", "internal/gateway"} {
		err := filepath.WalkDir(filepath.Join(root, filepath.FromSlash(dir)), func(path string, entry os.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if entry.IsDir() || filepath.Ext(path) != ".go" || strings.HasSuffix(path, "_test.go") {
				return nil
			}
			name, err := filepath.Rel(root, path)
			if err != nil {
				return err
			}
			name = filepath.ToSlash(name)
			b, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			producers, err := collectSyntheticProducers(name, b)
			if err != nil {
				return err
			}
			got = append(got, producers...)
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	sortProducers(got)

	// This is the full denominator of direct USER/SYSTEM Message appends in the two
	// model-facing packages. Request adaptation is deliberately classified here too:
	// otherwise moving a new synthetic producer into another file could evade a
	// hand-maintained file list. Initial transcript literals and assistant/tool rows
	// are not mid-session synthetic appends and therefore remain outside this ratchet.
	requestAdapters := []syntheticProducer{
		{File: "internal/agent/anthropic_server.go", Role: "RoleSystem", Payload: "out.System"},
		{File: "internal/agent/anthropic_server.go", Role: "RoleUser", Payload: "text.String()"},
		{File: "internal/agent/gemini_server.go", Role: "RoleSystem", Payload: "out.System"},
		{File: "internal/agent/loop_wire.go", Role: "RoleSystem", Payload: "c.memoryDigest"},
		{File: "internal/agent/loop_wire.go", Role: "RoleSystem", Payload: "c.seedSystemPrompt()"},
		{File: "internal/agent/loop_wire.go", Role: "RoleUser", Payload: "task"},
		{File: "internal/gateway/responses.go", Role: "agent.RoleSystem", Payload: "instructions"},
		{File: "internal/gateway/responses.go", Role: "agent.RoleUser", Payload: "s"},
	}
	witnessed := []producerWitness{
		{
			Producer:    syntheticProducer{File: "internal/agent/loop_turn.go", Role: "RoleUser", Payload: "toolTerminalPayload"},
			Authorities: []callAuthority{{File: "internal/agent/loop_turn.go", Call: "RecordToolTerminalWakeNext"}},
		},
		{
			Producer:    syntheticProducer{File: "internal/agent/loop_directives.go", Role: "RoleUser", Payload: "steer"},
			Authorities: []callAuthority{{File: "internal/agent/loop_session.go", Call: "RecordSteerNext"}},
		},
		{
			Producer: syntheticProducer{File: "internal/agent/loop_directives.go", Role: "RoleSystem", Payload: "objective"},
			Authorities: []callAuthority{
				{File: "internal/agent/loop_redirect.go", Call: "ApplyPendingRedirect"},
				{File: "internal/sessionctl/redirect.go", Call: "WitnessMove"},
			},
		},
		{
			Producer: syntheticProducer{File: "internal/agent/loop_directives.go", Role: "RoleSystem", Payload: "floor"},
			Authorities: []callAuthority{
				{File: "internal/agent/loop_constraint.go", Call: "ApplyPendingConstraints"},
				{File: "internal/sessionctl/constraint.go", Call: "WitnessMove"},
			},
		},
		{
			Producer:    syntheticProducer{File: "internal/agent/loop_directives.go", Role: "RoleUser", Payload: "nudge"},
			Authorities: []callAuthority{{File: "internal/agent/loop_directives.go", Call: "RecordContextAdvisoryNext"}},
		},
		{
			Producer:    syntheticProducer{File: "internal/agent/loop_turn.go", Role: "RoleUser", Payload: "continuation"},
			Authorities: []callAuthority{{File: "internal/agent/loop_turn.go", Call: "RecordStopWitnessNext"}},
		},
		{
			Producer:    syntheticProducer{File: "internal/gateway/messages.go", Role: "agent.RoleUser", Payload: "prompt"},
			Authorities: []callAuthority{{File: "internal/gateway/messages.go", Call: "RecordGuardRecoveryNext"}},
		},
	}
	want := append([]syntheticProducer(nil), requestAdapters...)
	for _, entry := range witnessed {
		want = append(want, entry.Producer)
	}
	sortProducers(want)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("model-facing USER/SYSTEM append inventory changed\n got: %+v\nwant: %+v\nclassify every new append as request adaptation or lower its exact payload to the shared Next witness before updating this denominator", got, want)
	}

	for _, entry := range witnessed {
		for _, authority := range entry.Authorities {
			b, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(authority.File)))
			if err != nil {
				t.Fatal(err)
			}
			if !hasCall(b, authority.Call) {
				t.Errorf("producer %+v lost shared Next authority %s::%s", entry.Producer, authority.File, authority.Call)
			}
		}
	}
}

func sortProducers(producers []syntheticProducer) {
	sort.Slice(producers, func(i, j int) bool {
		return fmt.Sprint(producers[i]) < fmt.Sprint(producers[j])
	})
}

func TestSyntheticProducerInventoryRejectsUnclassifiedAppend(t *testing.T) {
	source := []byte(`package p
func f(messages []Message, surprise string) {
	messages = append(messages, Message{Role: RoleUser, Content: surprise})
}`)
	got, err := collectSyntheticProducers("surprise.go", source)
	if err != nil {
		t.Fatal(err)
	}
	want := []syntheticProducer{{File: "surprise.go", Role: "RoleUser", Payload: "surprise"}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("unclassified producer escaped inventory: got %+v want %+v", got, want)
	}
}

func collectSyntheticProducers(name string, source []byte) ([]syntheticProducer, error) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, name, source, 0)
	if err != nil {
		return nil, err
	}
	var out []syntheticProducer
	ast.Inspect(file, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok || len(call.Args) < 2 {
			return true
		}
		fun, ok := call.Fun.(*ast.Ident)
		if !ok || fun.Name != "append" {
			return true
		}
		lit, ok := call.Args[1].(*ast.CompositeLit)
		if !ok || !isMessageType(lit.Type) {
			return true
		}
		var role, payload string
		for _, elt := range lit.Elts {
			kv, ok := elt.(*ast.KeyValueExpr)
			if !ok {
				continue
			}
			key, ok := kv.Key.(*ast.Ident)
			if !ok {
				continue
			}
			switch key.Name {
			case "Role":
				role = renderExpr(fset, kv.Value)
			case "Content":
				payload = renderExpr(fset, kv.Value)
			}
		}
		if strings.HasSuffix(role, "RoleUser") || strings.HasSuffix(role, "RoleSystem") {
			out = append(out, syntheticProducer{File: name, Role: role, Payload: payload})
		}
		return true
	})
	return out, nil
}

func isMessageType(expr ast.Expr) bool {
	switch x := expr.(type) {
	case *ast.Ident:
		return x.Name == "Message"
	case *ast.SelectorExpr:
		return x.Sel.Name == "Message"
	default:
		return false
	}
}

func renderExpr(fset *token.FileSet, expr ast.Expr) string {
	var b bytes.Buffer
	_ = format.Node(&b, fset, expr)
	return b.String()
}

func hasCall(source []byte, want string) bool {
	file, err := parser.ParseFile(token.NewFileSet(), "authority.go", source, 0)
	if err != nil {
		return false
	}
	found := false
	ast.Inspect(file, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		switch fun := call.Fun.(type) {
		case *ast.Ident:
			found = found || fun.Name == want
		case *ast.SelectorExpr:
			found = found || fun.Sel.Name == want
		}
		return !found
	})
	return found
}
