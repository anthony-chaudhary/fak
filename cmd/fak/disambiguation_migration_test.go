package main

import (
	"bytes"
	"encoding/json"
	"github.com/anthony-chaudhary/fak/internal/disambiguation"
	"testing"
)

func TestDisambiguationMigrationSelfTestCLI(t *testing.T) {
	var out, errout bytes.Buffer
	if code := runDisambiguation(&out, &errout, []string{"migration-self-test", "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errout.String())
	}
	var report disambiguation.MigrationSelfTestReport
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if !report.SilentRemovalRejected || !report.VersionedAliasAccepted || report.ReplacementTarget == "" {
		t.Fatalf("report=%#v", report)
	}
}
