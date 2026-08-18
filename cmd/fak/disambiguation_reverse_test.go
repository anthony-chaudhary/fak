package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/disambiguation"
)

func TestDisambiguationReverseCLIResolvesSourceAndSymbol(t *testing.T) {
	for _, tc := range []struct{ kind, input, want string }{
		{"source-path", "internal/disambiguation/query.go", "disambiguation package"},
		{"symbol", "Query", "disambiguation package"},
		{"cli-token", "disambiguation", "fak CLI kernel"},
		{"reason-code", "SOURCE_CURRENT", "agent kernel"},
	} {
		t.Run(tc.kind, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			code := runDisambiguation(&stdout, &stderr, []string{"reverse", "--kind", tc.kind, tc.input, "--json"})
			if code != 0 {
				t.Fatalf("code=%d stderr=%s", code, stderr.String())
			}
			var response disambiguation.ReverseLookupResponse
			if err := json.Unmarshal(stdout.Bytes(), &response); err != nil {
				t.Fatal(err)
			}
			found := false
			for _, match := range response.Matches {
				if match.Entry.Identity.CanonicalTerm == tc.want {
					found = true
				}
			}
			if !found {
				t.Fatalf("%q absent from %#v", tc.want, response.Matches)
			}
		})
	}
}

func TestDisambiguationReverseCLIUnknownIsNotFound(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runDisambiguation(&stdout, &stderr, []string{"reverse", "--kind", "symbol", "DefinitelyAbsent", "--json"})
	if code != 1 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var response disambiguation.ReverseLookupResponse
	if err := json.Unmarshal(stdout.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if len(response.Matches) != 0 {
		t.Fatalf("fabricated matches = %#v", response.Matches)
	}
	if !strings.Contains(stderr.String(), "reverse locator not found") {
		t.Fatalf("stderr=%s", stderr.String())
	}
}

func TestDisambiguationReverseCLISelfTest(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runDisambiguation(&stdout, &stderr, []string{"reverse", "--self-test", "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var report disambiguation.ReverseSelfTestReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if len(report.Cases) != 4 || !report.UnknownRejected {
		t.Fatalf("report=%#v", report)
	}
}
