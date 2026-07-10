package fleetverify

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/loopfleet"
)

// writeFile drops content into dir under name and returns the full path.
func writeFile(t *testing.T, dir, name, content string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
	return path
}

func TestLoadBriefJSON(t *testing.T) {
	type payload struct {
		Name  string `json:"name"`
		Count int    `json:"count"`
	}

	dir := t.TempDir()
	goodPath := writeFile(t, dir, "good.json", `{"name":"alpha","count":3}`)
	badPath := writeFile(t, dir, "bad.json", `{"name":`)

	tests := []struct {
		name    string
		path    string
		stdin   string
		want    payload
		wantErr bool
	}{
		{
			name: "reads from file when path is a filename",
			path: goodPath,
			// stdin content must be ignored for a real path
			stdin: `{"name":"stdin-should-not-be-read","count":99}`,
			want:  payload{Name: "alpha", Count: 3},
		},
		{
			name:  "reads from stdin when path is dash",
			path:  "-",
			stdin: `{"name":"beta","count":7}`,
			want:  payload{Name: "beta", Count: 7},
		},
		{
			name:    "missing file is an error",
			path:    filepath.Join(dir, "no-such-file.json"),
			wantErr: true,
		},
		{
			name:    "malformed JSON is an error",
			path:    badPath,
			wantErr: true,
		},
		{
			name:    "malformed JSON on stdin is an error",
			path:    "-",
			stdin:   `not json at all`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got payload
			err := loadBriefJSON(tt.path, strings.NewReader(tt.stdin), &got)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("loadBriefJSON(%q) = nil error, want error", tt.path)
				}
				return
			}
			if err != nil {
				t.Fatalf("loadBriefJSON(%q) unexpected error: %v", tt.path, err)
			}
			if got != tt.want {
				t.Errorf("loadBriefJSON(%q) = %+v, want %+v", tt.path, got, tt.want)
			}
		})
	}
}

func TestLoadFleetBriefReport(t *testing.T) {
	dir := t.TempDir()

	mkReport := func(schema string, dark int) string {
		r := loopfleet.Report{
			Schema:     schema,
			TSUnixNano: 42,
			Rollup:     loopfleet.Rollup{Loops: 1, Dark: dark, Ledgers: 1},
		}
		b, err := json.Marshal(r)
		if err != nil {
			t.Fatalf("marshal fixture: %v", err)
		}
		return string(b)
	}

	tests := []struct {
		name       string
		content    string
		wantErr    string // substring the error must carry; "" means no error
		wantSchema string
		wantDark   int
	}{
		{
			name:       "matching schema is accepted",
			content:    mkReport(loopfleet.Schema, 2),
			wantSchema: loopfleet.Schema,
			wantDark:   2,
		},
		{
			name:       "empty schema is tolerated",
			content:    mkReport("", 1),
			wantSchema: "",
			wantDark:   1,
		},
		{
			name:    "mismatched schema is refused",
			content: mkReport("some.other.schema.v9", 0),
			wantErr: `schema "some.other.schema.v9", want "` + loopfleet.Schema + `"`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := writeFile(t, dir, "report.json", tt.content)
			got, err := loadFleetBriefReport(path, strings.NewReader(""))
			if tt.wantErr != "" {
				if err == nil {
					t.Fatalf("loadFleetBriefReport = nil error, want error containing %q", tt.wantErr)
				}
				if !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("loadFleetBriefReport error = %q, want it to contain %q", err.Error(), tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("loadFleetBriefReport unexpected error: %v", err)
			}
			if got.Schema != tt.wantSchema {
				t.Errorf("Schema = %q, want %q", got.Schema, tt.wantSchema)
			}
			if got.Rollup.Dark != tt.wantDark {
				t.Errorf("Rollup.Dark = %d, want %d", got.Rollup.Dark, tt.wantDark)
			}
		})
	}
}

func TestLoadFleetBriefReportFromStdin(t *testing.T) {
	in := loopfleet.Report{
		Schema:     loopfleet.Schema,
		TSUnixNano: 7,
		Loops: []loopfleet.LoopHealth{
			{Kind: "nightrun", Ledger: "nightrun", Runs: 4, Keep: 3, Witness: 2, KeepRate: 0.75},
		},
		Rollup: loopfleet.Rollup{Loops: 1, Live: 1, Ledgers: 1},
	}
	b, err := json.Marshal(in)
	if err != nil {
		t.Fatalf("marshal fixture: %v", err)
	}

	got, err := loadFleetBriefReport("-", strings.NewReader(string(b)))
	if err != nil {
		t.Fatalf("loadFleetBriefReport(\"-\") unexpected error: %v", err)
	}
	if !reflect.DeepEqual(got, in) {
		t.Errorf("round-trip via stdin = %+v, want %+v", got, in)
	}
}

func TestCollectFleetBriefReportEmptyRoot(t *testing.T) {
	root := t.TempDir()

	got, err := collectFleetBriefReport(root)
	if err != nil {
		t.Fatalf("collectFleetBriefReport unexpected error: %v", err)
	}
	if got.Schema != loopfleet.Schema {
		t.Errorf("Schema = %q, want %q", got.Schema, loopfleet.Schema)
	}
	if got.TSUnixNano == 0 {
		t.Error("TSUnixNano = 0, want a real fold timestamp")
	}
	if len(got.Loops) != 0 {
		t.Errorf("Loops = %d rows on an empty root, want 0", len(got.Loops))
	}
	if got.Rollup.Loops != 0 {
		t.Errorf("Rollup.Loops = %d on an empty root, want 0", got.Rollup.Loops)
	}
	// An empty root has no ledgers; every registered ledger must be SURFACED as
	// skipped (absence is a known gap, never a silent healthy zero).
	if len(got.Skipped) == 0 {
		t.Fatal("Skipped is empty on an empty root, want every ledger surfaced")
	}
	if got.Rollup.Skipped != len(got.Skipped) {
		t.Errorf("Rollup.Skipped = %d, want %d (len(Skipped))", got.Rollup.Skipped, len(got.Skipped))
	}
	seen := map[string]bool{}
	for _, s := range got.Skipped {
		if s.Ledger == "" || s.Reason == "" {
			t.Errorf("skipped entry %+v missing ledger or reason", s)
		}
		if seen[s.Ledger] {
			t.Errorf("ledger %q surfaced twice in Skipped", s.Ledger)
		}
		seen[s.Ledger] = true
	}
	for _, ledger := range []string{"loopmgr", "nightrun", "dojo", "cadence", "dispatch"} {
		if !seen[ledger] {
			t.Errorf("ledger %q not surfaced in Skipped on an empty root; got %v", ledger, got.Skipped)
		}
	}
}

func TestExercise(t *testing.T) {
	root := t.TempDir()
	dir := t.TempDir()

	want := loopfleet.Report{
		Schema:     loopfleet.Schema,
		TSUnixNano: 99,
		Rollup:     loopfleet.Rollup{Loops: 2, Live: 1, Stale: 1, Ledgers: 2},
	}
	b, err := json.Marshal(want)
	if err != nil {
		t.Fatalf("marshal fixture: %v", err)
	}
	path := writeFile(t, dir, "brief.json", string(b))

	got, err := Exercise(root, path, strings.NewReader(""))
	if err != nil {
		t.Fatalf("Exercise unexpected error: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Exercise = %+v, want %+v", got, want)
	}

	// A brief carrying a foreign schema must propagate the load refusal.
	badPath := writeFile(t, dir, "bad-brief.json", `{"schema":"wrong.v0"}`)
	if _, err := Exercise(root, badPath, strings.NewReader("")); err == nil {
		t.Error("Exercise with a mismatched-schema brief = nil error, want error")
	}
}
