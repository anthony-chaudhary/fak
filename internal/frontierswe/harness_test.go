package frontierswe

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// TestFrontierHarnessRoutingBaseURLOnly is the #1725 proof: the shim routes at
// least three distinct harness classes (claude-code + 2 others) through the fak
// gateway, and for each the ONLY delta versus the raw job.yaml is the base URL.
func TestFrontierHarnessRoutingBaseURLOnly(t *testing.T) {
	const (
		gateway  = "http://127.0.0.1:8080/v1"
		upstream = "http://127.0.0.1:11434/v1"
	)
	harnesses := FrontierHarnesses()
	if len(harnesses) < 3 {
		t.Fatalf("want >=3 wired harnesses, got %d: %+v", len(harnesses), harnesses)
	}
	seenClass := map[string]bool{}
	for _, h := range harnesses {
		if seenClass[h.ImportPath] {
			t.Errorf("duplicate wrapped class %q", h.ImportPath)
		}
		seenClass[h.ImportPath] = true

		routing := BuildHarnessRouting(h.Name, gateway, upstream, false)
		if routing.WrappedAgent != h.ImportPath {
			t.Errorf("harness %q wrapped = %q, want %q", h.Name, routing.WrappedAgent, h.ImportPath)
		}
		diffs, baseURLOnly := routing.BaseURLOnlyDelta()
		if !baseURLOnly {
			t.Errorf("harness %q: routing changes more than the base URL: %+v", h.Name, diffs)
		}
		if !strings.Contains(routing.FakJobYAML, "wrapped: "+h.ImportPath) {
			t.Errorf("harness %q: fak job.yaml missing wrapped class:\n%s", h.Name, routing.FakJobYAML)
		}
		if !strings.Contains(routing.FakJobYAML, "fak_base_url: "+gateway) {
			t.Errorf("harness %q: fak job.yaml not routed to gateway:\n%s", h.Name, routing.FakJobYAML)
		}
		if !strings.Contains(routing.RawJobYAML, "fak_base_url: "+upstream) {
			t.Errorf("harness %q: raw job.yaml not pointed at upstream:\n%s", h.Name, routing.RawJobYAML)
		}
		if !strings.Contains(routing.FakJobYAML, "name: fak-routed-"+h.Name) {
			t.Errorf("harness %q: fak job.yaml agent name not per-harness:\n%s", h.Name, routing.FakJobYAML)
		}
	}
}

// TestFrontierHarnessMockEndpoint verifies each wired harness against a mock
// endpoint: pointing the routing at a local httptest server sends the harness's
// model traffic to that gateway URL and nowhere else — the routing override is
// real, not just a string in the recipe.
func TestFrontierHarnessMockEndpoint(t *testing.T) {
	const upstream = "http://127.0.0.1:11434/v1"
	for _, h := range FrontierHarnesses() {
		h := h
		t.Run(h.Name, func(t *testing.T) {
			var gotPath, gotBody string
			hits := 0
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				hits++
				gotPath = r.URL.Path
				b, _ := io.ReadAll(r.Body)
				gotBody = string(b)
				w.Header().Set("content-type", "application/json")
				io.WriteString(w, `{"choices":[{"message":{"content":"ok"}}]}`)
			}))
			defer srv.Close()

			routing := BuildHarnessRouting(h.Name, srv.URL+"/v1", upstream, false)
			endpoint := strings.TrimRight(routing.FakBaseURL, "/") + "/chat/completions"
			resp, err := http.Post(endpoint, "application/json",
				strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"`+h.Name+` smoke"}]}`))
			if err != nil {
				t.Fatalf("post to routed gateway: %v", err)
			}
			resp.Body.Close()
			if hits != 1 {
				t.Fatalf("mock gateway hit %d times, want 1 — routing override did not reach the mock", hits)
			}
			if gotPath != "/v1/chat/completions" {
				t.Errorf("mock got path %q, want /v1/chat/completions", gotPath)
			}
			if !strings.Contains(gotBody, h.Name+" smoke") {
				t.Errorf("mock did not receive the harness payload: %q", gotBody)
			}
		})
	}
}

// TestHarnessCoverageHonest guards the coverage claim: at least three harnesses
// wired, a non-empty remaining set, and no overlap between the two.
func TestHarnessCoverageHonest(t *testing.T) {
	wired, remaining := HarnessCoverage()
	if len(wired) < 3 {
		t.Fatalf("want >=3 wired harnesses, got %v", wired)
	}
	if len(remaining) == 0 {
		t.Fatalf("remaining set must be explicit, not a blanket all-harnesses claim")
	}
	wiredSet := map[string]bool{}
	for _, w := range wired {
		wiredSet[w] = true
	}
	for _, r := range remaining {
		if wiredSet[r] {
			t.Errorf("%q is listed both wired and remaining", r)
		}
	}
	if _, ok := WrappedAgentForHarness("claude-code"); !ok {
		t.Error("claude-code must resolve to a wrapped class")
	}
	if _, ok := WrappedAgentForHarness("gemini_cli"); !ok {
		t.Error("harness lookup should tolerate underscore spelling")
	}
	if _, ok := WrappedAgentForHarness("no-such-harness"); ok {
		t.Error("unknown harness must not resolve")
	}
	if md := RenderHarnessCoverageMarkdown(); !strings.Contains(md, "Wired") || !strings.Contains(md, "codex") {
		t.Errorf("coverage markdown missing wired section:\n%s", md)
	}
}

// TestEnvAdapterJobYAMLUnchangedForClaudeCode is a regression guard on the
// job.yaml refactor: the default (claude-code) path still renders the exact
// agent name and shim contract the env-adapter plan test asserts.
func TestEnvAdapterJobYAMLUnchangedForClaudeCode(t *testing.T) {
	got := envAdapterJobYAML(DefaultWrappedAgent, "http://127.0.0.1:8080/v1", false)
	for _, want := range []string{
		"name: fak-routed-claude-code",
		"import_path: harbor_ext.fak_routed:FakRoutedAgent",
		"wrapped: " + DefaultWrappedAgent,
		"fak_base_url: http://127.0.0.1:8080/v1",
		"allow_internet: false",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("default job yaml missing %q:\n%s", want, got)
		}
	}
}
