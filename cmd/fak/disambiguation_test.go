package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/disambiguation"
)

func TestDisambiguationSchemaPublicSeam(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runDisambiguation(&stdout, &stderr, []string{"schema", "--json"}); code != 0 {
		t.Fatalf("exit %d, stderr=%s", code, stderr.String())
	}
	var got disambiguation.SchemaDescriptor
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("decode descriptor: %v", err)
	}
	if got.Schema != disambiguation.EntrySchemaVersion || got.Compatibility != "exact-version" {
		t.Fatalf("unexpected descriptor: %+v", got)
	}
}

func TestDisambiguationSchemaSelfTestPublicSeam(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runDisambiguation(&stdout, &stderr, []string{"schema", "--self-test", "--json"}); code != 0 {
		t.Fatalf("exit %d, stderr=%s", code, stderr.String())
	}
	var got disambiguation.SelfTestReport
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("decode report: %v", err)
	}
	if !got.CompleteAccepted || len(got.OmissionsRejected) != len(disambiguation.Descriptor().Required) {
		t.Fatalf("self-test did not exercise complete required-field contract: %+v", got)
	}
	for _, required := range []string{"identity.canonical_term", "owner.leaf", "contrasts[].explanation", "sources[].revision", "freshness.probe"} {
		if !containsDisambiguationString(got.OmissionsRejected, required) {
			t.Fatalf("self-test omitted required rejection %q: %v", required, got.OmissionsRejected)
		}
	}
}

func TestDisambiguationUsageIsDiscoverable(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runDisambiguation(&stdout, &stderr, nil); code != 2 {
		t.Fatalf("exit %d, want 2", code)
	}
	if !strings.Contains(stderr.String(), "fak disambiguation schema") {
		t.Fatalf("usage missing public seam: %q", stderr.String())
	}
}

func containsDisambiguationString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

func TestRunDisambiguationQueryJSON(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runDisambiguation(&stdout, &stderr, []string{"query", "agent kernel", "--json"})
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, stderr.String())
	}
	var got disambiguation.QueryResponse
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("decode query JSON: %v\n%s", err, stdout.String())
	}
	if got.Schema != disambiguation.QuerySchemaVersion || got.Entry.Identity.CanonicalTerm != "agent kernel" {
		t.Fatalf("query response = %#v", got)
	}
	if got.Entry.Definition == "" || len(got.Entry.Contrasts) == 0 || got.Entry.Owner.Leaf == "" || len(got.Entry.Sources) == 0 || got.Entry.Freshness.Verdict == "" {
		t.Fatalf("query JSON omitted contract fields: %#v", got.Entry)
	}
}

func TestRunDisambiguationQuerySelfTestJSON(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runDisambiguation(&stdout, &stderr, []string{"query", "--self-test", "--json"})
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, stderr.String())
	}
	var got disambiguation.QuerySelfTestReport
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("decode self-test JSON: %v", err)
	}
	if !got.Complete || got.CanonicalTerm != "agent kernel" || got.MatchedAlias != "fused agent kernel" || got.Schema != disambiguation.QuerySchemaVersion {
		t.Fatalf("self-test report = %#v", got)
	}
}

func TestRunDisambiguationQueryAliasReturnsCanonicalIdentityAndMatch(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runDisambiguation(&stdout, &stderr, []string{"query", "fused agent kernel", "--json"}); code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, stderr.String())
	}
	var got disambiguation.QueryResponse
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("decode alias JSON: %v", err)
	}
	if got.Entry.Identity.CanonicalTerm != "agent kernel" || got.MatchedAlias != "fused agent kernel" {
		t.Fatalf("alias response = %#v", got)
	}
	if got.Entry.Owner.Leaf == "" || got.Entry.Owner.Lane == "" {
		t.Fatalf("canonical ownership hidden: %#v", got.Entry.Owner)
	}
}

func TestRunDisambiguationQueryRejectsUnknown(t *testing.T) {
	for _, term := range []string{"Fused Agent Kernel", "unknown"} {
		var stdout, stderr bytes.Buffer
		if code := runDisambiguation(&stdout, &stderr, []string{"query", term, "--json"}); code != 3 {
			t.Errorf("query %q exit = %d, want 3; stderr = %q", term, code, stderr.String())
		}
	}
}

func TestRunDisambiguationQueryContrastCLI(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runDisambiguation(&stdout, &stderr, []string{"query", "fused agent kernel", "--json"})
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, stderr.String())
	}
	var got struct {
		MatchedAlias string `json:"matched_alias"`
		Entry        struct {
			Contrasts []struct {
				CanonicalTerm       string `json:"canonical_term"`
				Explanation         string `json:"explanation"`
				RequiredPair        bool   `json:"required_pair"`
				ForbiddenConflation bool   `json:"forbidden_conflation"`
			} `json:"contrasts"`
		} `json:"entry"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("decode stdout: %v\n%s", err, stdout.String())
	}
	if got.MatchedAlias != "fused agent kernel" || len(got.Entry.Contrasts) != 1 {
		t.Fatalf("unexpected alias contrast response: %+v", got)
	}
	contrast := got.Entry.Contrasts[0]
	if contrast.CanonicalTerm != "compute kernel" || contrast.Explanation == "" || !contrast.RequiredPair || !contrast.ForbiddenConflation {
		t.Fatalf("contrast = %+v, want explicit required forbidden pair", contrast)
	}
}

func TestRunDisambiguationQuerySelfTestJSONIncludesScopeWitness(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runDisambiguation(&stdout, &stderr, []string{"query", "--self-test", "--json"}); code != 0 {
		t.Fatalf("code = %d, stderr = %s", code, stderr.String())
	}
	var report disambiguation.QuerySelfTestReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.OverloadedTerm != "kernel" || !report.UnscopedAmbiguous {
		t.Fatalf("scope witness = %#v", report)
	}
	want := disambiguation.Scope{Kind: "package", Value: "internal/disambiguation"}
	if report.Scope != want {
		t.Fatalf("scope = %#v, want %#v", report.Scope, want)
	}
}

func TestRunDisambiguationQueryScopeFlagsRoundTrip(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runDisambiguation(&stdout, &stderr, []string{"query", "agent kernel", "--scope-kind", "product", "--scope-value", "fak", "--json"})
	if code != 0 {
		t.Fatalf("code = %d, stderr = %s", code, stderr.String())
	}
	var response disambiguation.QueryResponse
	if err := json.Unmarshal(stdout.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	want := disambiguation.Scope{Kind: "product", Value: "fak"}
	if response.Entry.Scope != want {
		t.Fatalf("scope = %#v, want %#v", response.Entry.Scope, want)
	}
}

func TestRunDisambiguationQueryRejectsPartialScope(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runDisambiguation(&stdout, &stderr, []string{"query", "agent kernel", "--scope-kind", "product"}); code != 2 {
		t.Fatalf("code = %d, stderr = %s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "required together") {
		t.Fatalf("stderr = %q", stderr.String())
	}
}

func TestRunDisambiguationQueryOverloadRequiresScope(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runDisambiguation(&stdout, &stderr, []string{"query", "kernel", "--json"}); code != 3 {
		t.Fatalf("code = %d, stderr = %s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "scope required") {
		t.Fatalf("stderr = %q", stderr.String())
	}
	stdout.Reset()
	stderr.Reset()
	code := runDisambiguation(&stdout, &stderr, []string{"query", "kernel", "--scope-kind", "package", "--scope-value", "internal/disambiguation", "--json"})
	if code != 0 {
		t.Fatalf("scoped code = %d, stderr = %s", code, stderr.String())
	}
	var response disambiguation.QueryResponse
	if err := json.Unmarshal(stdout.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	want := disambiguation.Scope{Kind: "package", Value: "internal/disambiguation"}
	if response.Entry.Scope != want {
		t.Fatalf("scope = %#v, want %#v", response.Entry.Scope, want)
	}
}
