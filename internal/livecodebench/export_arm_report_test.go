package livecodebench

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestLoadArmReportFixture(t *testing.T) {
	path := filepath.Join(t.TempDir(), "raw.json")
	report := `{"arm":"raw","release":"release_v6","problems":[{"question_id":"q-2","completions":["def two(): return 2"]},{"question_id":"q-1","completions":["def one(): return 1","def one():\n return 1"]}]}`
	if err := os.WriteFile(path, []byte(report), 0o600); err != nil {
		t.Fatal(err)
	}
	got, err := LoadArmReportFixture(path)
	if err != nil {
		t.Fatal(err)
	}
	if got.Schema != FixtureSchema || got.ReleaseVersion != "release_v6" {
		t.Fatalf("header = %#v", got)
	}
	if ids := []string{got.Items[0].QuestionID, got.Items[1].QuestionID}; !reflect.DeepEqual(ids, []string{"q-2", "q-1"}) {
		t.Fatalf("ids = %v", ids)
	}
	if !reflect.DeepEqual(got.Items[1].CodeList, []string{"def one(): return 1", "def one():\n return 1"}) {
		t.Fatalf("code = %#v", got.Items[1].CodeList)
	}
}

func TestLoadArmReportFixtureRejectsUngradeableReports(t *testing.T) {
	tests := map[string]string{
		"unknown arm":      `{"arm":"other","release":"release_v6","problems":[{"question_id":"q","completions":["x"]}]}`,
		"missing release":  `{"arm":"fak","problems":[{"question_id":"q","completions":["x"]}]}`,
		"empty completion": `{"arm":"raw","release":"release_v6","problems":[{"question_id":"q","completions":[""]}]}`,
		"duplicate":        `{"arm":"raw","release":"release_v6","problems":[{"question_id":"q","completions":["x"]},{"question_id":"q","completions":["y"]}]}`,
	}
	for name, body := range tests {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "report.json")
			if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
				t.Fatal(err)
			}
			if _, err := LoadArmReportFixture(path); err == nil || strings.TrimSpace(err.Error()) == "" {
				t.Fatalf("err = %v", err)
			}
		})
	}
}
