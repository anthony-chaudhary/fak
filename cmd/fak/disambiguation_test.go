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
	if !got.Complete || got.CanonicalTerm != "agent kernel" || got.Schema != disambiguation.QuerySchemaVersion {
		t.Fatalf("self-test report = %#v", got)
	}
}

func TestRunDisambiguationQueryRejectsAliasAndUnknown(t *testing.T) {
	for _, term := range []string{"fused agent kernel", "unknown"} {
		var stdout, stderr bytes.Buffer
		if code := runDisambiguation(&stdout, &stderr, []string{"query", term, "--json"}); code != 3 {
			t.Errorf("query %q exit = %d, want 3; stderr = %q", term, code, stderr.String())
		}
	}
}
