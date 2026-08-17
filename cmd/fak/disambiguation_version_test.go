package main

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/disambiguation"
)

func TestDisambiguationVersionJSON(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runDisambiguation(&stdout, &stderr, []string{"version", "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var got disambiguation.IndexVersion
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != disambiguation.IndexVersionSchema || got.IndexSchema != disambiguation.GeneratedIndexSchemaVersion || got.EntryCount == 0 || len(got.ContentSHA256) != 64 {
		t.Fatalf("version=%+v", got)
	}
}
