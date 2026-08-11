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
