package conversationprofile_test

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"testing"

	cp "github.com/anthony-chaudhary/fak/pkg/conversationprofile"
	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

type mapAdapter struct {
	name, prefix string
	supported    map[string]bool
}

func (a mapAdapter) Name() string { return a.name }
func (a mapAdapter) Resolve(k, v string) (cp.Binding, bool) {
	if !a.supported[k+"="+v] {
		return cp.Binding{}, false
	}
	return cp.Binding{Key: k, Value: v, Effect: a.prefix + ":" + k + "=" + v, Provenance: a.name + "@v1"}, true
}

type service struct{ calls []harnesskit.Invocation }

func (s *service) Invoke(_ context.Context, in harnesskit.Invocation) (harnesskit.Result, error) {
	s.calls = append(s.calls, in)
	var got struct {
		Bindings []cp.Binding `json:"bindings"`
	}
	if err := json.Unmarshal(in.Arguments, &got); err != nil {
		return harnesskit.Result{}, err
	}
	normalized := make(map[string]string, len(got.Bindings))
	for _, b := range got.Bindings {
		normalized[b.Key] = b.Value
	}
	out, _ := json.Marshal(struct {
		Applied map[string]string `json:"applied"`
		Reply   string            `json:"reply"`
	}{normalized, "booked"})
	return harnesskit.Result{Content: out}, nil
}

var portable = func() string {
	data, err := os.ReadFile("testdata/portable-profile.json")
	if err != nil {
		panic(err)
	}
	return string(data)
}()

func TestPortableProfileRunsThroughTwoDifferentBindings(t *testing.T) {
	p, err := cp.Parse([]byte(portable))
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{"Caveman", "Ponytail", "plugin", "skill"} {
		if contains(portable, forbidden) {
			t.Fatalf("intent leaked mechanism %q", forbidden)
		}
	}
	supported := map[string]bool{"response.detail=brief": true, "interaction.questions=when_blocked": true, "tone=warm": true}
	adapters := []cp.Adapter{
		mapAdapter{name: "instruction-template", prefix: "prompt", supported: supported},
		mapAdapter{name: "runtime-controls", prefix: "field", supported: supported},
	}
	var outcomes []json.RawMessage
	for _, a := range adapters {
		svc := &service{}
		r, err := cp.Run(context.Background(), p, a, svc)
		if err != nil {
			t.Fatalf("%s: %v", a.Name(), err)
		}
		if r.Adapter != a.Name() || len(r.Bindings) != 3 || len(svc.calls) != 1 || svc.calls[0].Tool != "conversation.apply" {
			t.Fatalf("bad receipt/call: %+v %+v", r, svc.calls)
		}
		outcomes = append(outcomes, r.Outcome)
		blob, _ := json.Marshal(r)
		if !json.Valid(blob) {
			t.Fatal("receipt is not machine readable")
		}
	}
	var a, b any
	json.Unmarshal(outcomes[0], &a)
	json.Unmarshal(outcomes[1], &b)
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("normalized outcomes differ: %s != %s", outcomes[0], outcomes[1])
	}
}

func TestRequiredSemanticRefusesBeforeExecution(t *testing.T) {
	p, _ := cp.Parse([]byte(portable))
	svc := &service{}
	_, err := cp.Run(context.Background(), p, mapAdapter{name: "lossy", supported: map[string]bool{"tone=warm": true}}, svc)
	var typed *cp.Error
	if !errors.As(err, &typed) || typed.Code != cp.UnsupportedRequired || len(svc.calls) != 0 {
		t.Fatalf("want prelaunch typed refusal, got %v calls=%d", err, len(svc.calls))
	}
}

func TestProfileValidationAndOptionalGap(t *testing.T) {
	bad := []string{
		`{"schema":"wrong","id":"x","settings":{"tone":{"value":"warm","fidelity":"required"}}}`,
		`{"schema":"fak.conversation-profile/v1","id":"x","settings":{"tone":{"value":"host-specific","fidelity":"required"}}}`,
		`{"schema":"fak.conversation-profile/v1","id":"x","settings":{"unknown":{"value":"warm","fidelity":"required"}}}`,
	}
	for _, raw := range bad {
		if _, err := cp.Parse([]byte(raw)); err == nil {
			t.Fatalf("accepted %s", raw)
		}
	}
	p, _ := cp.Parse([]byte(portable))
	svc := &service{}
	a := mapAdapter{name: "partial", prefix: "p", supported: map[string]bool{"response.detail=brief": true, "interaction.questions=when_blocked": true}}
	r, err := cp.Run(context.Background(), p, a, svc)
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Gaps) != 1 || r.Gaps[0].Key != "tone" || r.Gaps[0].Fidelity != cp.Optional {
		t.Fatalf("bad optional gap: %+v", r.Gaps)
	}
}

func TestAmbiguousAndWideningBindingsRefuse(t *testing.T) {
	p, _ := cp.Parse([]byte(portable))
	cases := []cp.Adapter{
		badAdapter{mode: "ambiguous"}, badAdapter{mode: "widen"},
	}
	want := []cp.ErrorCode{cp.AmbiguousBinding, cp.AuthorityWidening}
	for i, a := range cases {
		svc := &service{}
		_, err := cp.Run(context.Background(), p, a, svc)
		var e *cp.Error
		if !errors.As(err, &e) || e.Code != want[i] || len(svc.calls) != 0 {
			t.Fatalf("case %d: %v calls=%d", i, err, len(svc.calls))
		}
	}
}

type badAdapter struct{ mode string }

func (a badAdapter) Name() string { return a.mode }
func (a badAdapter) Resolve(k, v string) (cp.Binding, bool) {
	if a.mode == "widen" {
		return cp.Binding{Key: k, Value: "proactive", Effect: k, Provenance: "bad@1"}, true
	}
	return cp.Binding{Key: k, Value: v, Effect: "same", Provenance: "bad@1"}, true
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

func TestDirectConfigIsExplicitlyNonPortable(t *testing.T) {
	direct := cp.DirectConfig{Adapter: "runtime-controls", Raw: json.RawMessage(`{"detail":1}`)}
	blob, err := json.Marshal(direct)
	if err != nil || !contains(string(blob), `"adapter":"runtime-controls"`) {
		t.Fatalf("direct config must identify its non-portable adapter: %s %v", blob, err)
	}
	if _, err := cp.Parse(blob); err == nil {
		t.Fatal("direct adapter config must not parse as portable intent")
	}
}
