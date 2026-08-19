package llamacppinterop

import (
	"context"
	"errors"
	"github.com/anthony-chaudhary/fak/internal/quantmeta"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
)

type fakeRunner struct {
	out string
	err error
	got []string
}

func (f *fakeRunner) Output(_ context.Context, n string, a ...string) ([]byte, error) {
	f.got = append([]string{n}, a...)
	return []byte(f.out), f.err
}
func TestDiscoverAndPlan(t *testing.T) {
	f := &fakeRunner{out: "llama-server version: 0.0.6123"}
	r := Discover(context.Background(), f, "llama-server")
	if r.Outcome != OutcomeDelegate || r.Capability.Version != "0.0.6123" || !reflect.DeepEqual(f.got, []string{"llama-server", "--version"}) {
		t.Fatalf("%+v got=%v", r, f.got)
	}
	p := Plan(r.Capability, "tiny.gguf", quantmeta.Descriptor{Artifact: &quantmeta.ArtifactSpec{ContainerID: "gguf"}})
	if p.Outcome != OutcomeDelegate || len(p.Argv) < 3 {
		t.Fatalf("%+v", p)
	}
}
func TestDiscoverFailsClosed(t *testing.T) {
	if r := Discover(context.Background(), &fakeRunner{err: errors.New("missing")}, "llama-cli"); r.Outcome != OutcomeRefuse {
		t.Fatalf("%+v", r)
	}
	if r := Discover(context.Background(), &fakeRunner{out: "unknown"}, "llama-cli"); r.Outcome != OutcomeAbstain {
		t.Fatalf("%+v", r)
	}
}
func TestPlanRejectsNonGGUF(t *testing.T) {
	r := Plan(Capability{Binary: "llama-cli", Version: "1"}, "m.bin", quantmeta.Descriptor{})
	if r.Outcome != OutcomeAbstain {
		t.Fatalf("%+v", r)
	}
}
func TestHealth(t *testing.T) {
	for _, tc := range []struct {
		name string
		code int
		body string
		want Outcome
	}{{"ready", 200, `{"status":"ready"}`, OutcomeDelegate}, {"down", 503, `{}`, OutcomeRefuse}, {"unknown", 200, `{"status":"loading"}`, OutcomeAbstain}} {
		t.Run(tc.name, func(t *testing.T) {
			s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(tc.code); _, _ = w.Write([]byte(tc.body)) }))
			defer s.Close()
			if r := CheckHealth(context.Background(), s.Client(), s.URL); r.Outcome != tc.want {
				t.Fatalf("%+v", r)
			}
		})
	}
}
