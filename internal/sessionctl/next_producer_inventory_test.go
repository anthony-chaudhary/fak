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
		// Both late-system folding branches re-encode existing transcript directives;
		// they do not introduce a new synthetic payload. Keep both occurrences counted.
		{File: "internal/agent/adapters.go", Role: "RoleUser", Payload: "text"},
		{File: "internal/agent/adapters.go", Role: "RoleUser", Payload: "text"},
		{File: "internal/agent/anthropic_server.go", Role: "RoleSystem", Payload: "out.System"},
		{File: "internal/agent/anthropic_server.go", Role: "RoleUser", Payload: "text.String()"},
		{File: "internal/agent/gemini_server.go", Role: "RoleSystem", Payload: "out.System"},
		// Warm-prefix descriptors re-encode the request's own instructions and resident
		// blocks to find the stable token boundary; nothing is spliced into a session.
		{File: "internal/agent/inkernel_warm.go", Role: "RoleSystem", Payload: "string(block)"},
		{File: "internal/agent/inkernel_warm.go", Role: "RoleSystem", Payload: "string(w.Instructions)"},
		{File: "internal/agent/warm_prefix.go", Role: "RoleSystem", Payload: "string(block)"},
		{File: "internal/agent/warm_prefix.go", Role: "RoleSystem", Payload: "string(in.Instructions)"},
		{File: "internal/agent/loop_wire.go", Role: "RoleSystem", Payload: "c.memoryDigest"},
		{File: "internal/agent/loop_wire.go", Role: "RoleSystem", Payload: "c.seedSystemPrompt()"},
		{File: "internal/agent/loop_wire.go", Role: "RoleUser", Payload: "task"},
		// Qwen ChatML re-renders a client tool row as a user turn carrying the same bytes.
		{File: "internal/gateway/anthropic_messages.go", Role: "agent.RoleUser", Payload: "respText"},
		// Prompt ordering moves client-supplied instructions and volatile text; it adds none.
		{File: "internal/gateway/prompt_order.go", Role: "agent.RoleSystem", Payload: "cleanedInstructions"},
		{File: "internal/gateway/prompt_order.go", Role: "agent.RoleUser", Payload: "hoistedText"},
		{File: "internal/gateway/prompt_order.go", Role: "agent.RoleUser", Payload: "volatileText"},
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
			Producer:    syntheticProducer{File: "internal/agent/loop_turn.go", Role: "RoleUser", Payload: "infraContinuation"},
			Authorities: []callAuthority{{File: "internal/agent/loop_turn.go", Call: "RecordInfraRepromptNext"}},
		},
		{
			Producer:    syntheticProducer{File: "internal/agent/loop_turn.go", Role: "RoleSystem", Payload: "guidanceMsg"},
			Authorities: []callAuthority{{File: "internal/agent/loop_turn.go", Call: "RecordCircuitBreakerNext"}},
		},
		{
			Producer:    syntheticProducer{File: "internal/agent/loop_turn.go", Role: "RoleSystem", Payload: "tripMsg"},
			Authorities: []callAuthority{{File: "internal/agent/loop_turn.go", Call: "RecordCircuitBreakerNext"}},
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
		for _, arg := range call.Args[1:] {
			lit, ok := arg.(*ast.CompositeLit)
			if !ok || !isMessageType(lit.Type) {
				continue
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

// fak-test:runtime fast est=1ms lane=default
func TestSyntheticProducerInventoryPreservesDuplicateMultiplicity(t *testing.T) {
	t.Parallel()
	producer := syntheticProducer{File: "internal/agent/adapters.go", Role: "RoleUser", Payload: "text"}
	classified := []syntheticProducer{producer, producer}
	for _, count := range []int{1, 2, 3} {
		source := []byte("package p\nfunc f(out []Message, text string) {\n" +
			strings.Repeat("out = append(out, Message{Role: RoleUser, Content: text})\n", count) + "}\n")
		got, err := collectSyntheticProducers(producer.File, source)
		if err != nil {
			t.Fatal(err)
		}
		want := make([]syntheticProducer, count)
		for i := range want {
			want[i] = producer
		}
		sortProducers(got)
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("%d identical appends: got %+v want %+v", count, got, want)
		}
		if matches := reflect.DeepEqual(got, classified); matches != (count == len(classified)) {
			t.Fatalf("%d identical appends: classified multiplicity is %d, inventory matches = %v", count, len(classified), matches)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestSyntheticProducerInventoryCollectsEveryAppendArgument(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name   string
		source string
		want   []syntheticProducer
	}{
		{
			name: "identical values in one append",
			source: `package p
func f(messages []Message, text string) {
	messages = append(messages, Message{Role: RoleUser, Content: text}, Message{Role: RoleUser, Content: text})
}`,
			want: []syntheticProducer{
				{File: "surprise.go", Role: "RoleUser", Payload: "text"},
				{File: "surprise.go", Role: "RoleUser", Payload: "text"},
			},
		},
		{
			name: "unclassified value after variable",
			source: `package p
func f(messages []Message, existing Message, surprise string) {
	messages = append(messages, existing, Message{Role: RoleSystem, Content: surprise})
}`,
			want: []syntheticProducer{{File: "surprise.go", Role: "RoleSystem", Payload: "surprise"}},
		},
		{
			name: "qualified value after ignored role",
			source: `package p
func f(messages []agent.Message, surprise string) {
	messages = append(messages, agent.Message{Role: agent.RoleTool}, agent.Message{Role: agent.RoleUser, Content: surprise})
}`,
			want: []syntheticProducer{{File: "surprise.go", Role: "agent.RoleUser", Payload: "surprise"}},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := collectSyntheticProducers("surprise.go", []byte(tc.source))
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("append argument escaped inventory: got %+v want %+v", got, tc.want)
			}
		})
	}
}
