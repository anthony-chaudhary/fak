package perfledger

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func rmarkWrite(t *testing.T, p, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func rmarkNames(arts []ResultArtifact) []string {
	out := []string{}
	for _, a := range arts {
		out = append(out, a.Name)
	}
	return out
}

func TestReadyMarkerRunKey(t *testing.T) {
	t.Parallel()
	cases := []struct {
		key  string
		want bool
	}{
		{"1714069323", true},
		{"999999999", true},
		{"1714069323123456", true},
		{"999999999123456", true},
		{"1714069323\n", false},
		{"17140693231", false},
		{"17140693231234", false},
		{"12345678", false},
		{"17140693231234567", false},
		{"", false},
		{"171406932a", false},
	}
	for _, c := range cases {
		if got := ValidResultRunKey(c.key); got != c.want {
			t.Errorf("ValidResultRunKey(%q) = %v, want %v", c.key, got, c.want)
		}
	}
}

func TestReadyMarkerLifecycle(t *testing.T) {
	t.Parallel()
	dir := filepath.Join(t.TempDir(), "run")
	if ResultsReady(dir) || ResultsProcessing(dir) {
		t.Fatal("missing dir must read as neither ready nor processing")
	}
	if err := WriteProcessingMarker(dir); err != nil {
		t.Fatal(err)
	}
	if ResultsReady(dir) || !ResultsProcessing(dir) {
		t.Fatal("after processing marker: want processing, not ready")
	}
	if err := WriteReadyMarker(dir, ReadyMarker{}); err != nil {
		t.Fatal(err)
	}
	if !ResultsReady(dir) || ResultsProcessing(dir) {
		t.Fatal("after ready marker: want ready, processing cleared")
	}
	got, err := os.ReadFile(ReadyMarkerPath(dir))
	if err != nil {
		t.Fatal(err)
	}
	// Upstream golden: orjson.dumps of the default payload.
	if want := `{"ready":true,"was_cancelled":false,"partial":false}`; string(got) != want {
		t.Fatalf("ready payload = %s, want %s", got, want)
	}
	// Restarting an export retracts readiness before anything else.
	if err := WriteProcessingMarker(dir); err != nil {
		t.Fatal(err)
	}
	if ResultsReady(dir) || !ResultsProcessing(dir) {
		t.Fatal("re-export must hide results again")
	}
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".tmp") {
			t.Fatalf("temporary marker left behind: %s", e.Name())
		}
	}
	if err := ClearProcessingMarker(dir); err != nil {
		t.Fatal(err)
	}
	if err := ClearProcessingMarker(dir); err != nil {
		t.Fatalf("clearing an absent marker: %v", err)
	}
}

func TestReadyMarkerFailedCommitRollsBack(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	// A non-empty directory squatting on the marker name makes the rename fail.
	rmarkWrite(t, filepath.Join(ReadyMarkerPath(dir), "blocker"), "x")
	if err := WriteReadyMarker(dir, ReadyMarker{}); err == nil {
		t.Fatal("want commit failure")
	}
	if ResultsReady(dir) {
		t.Fatal("failed commit must not read as ready")
	}
	if !ResultsProcessing(dir) {
		t.Fatal("rollback must restore the processing marker")
	}
	if _, err := ResolveResultFile(dir, "profile.json"); err == nil {
		t.Fatal("top-level artifact served after failed commit")
	}
}

func TestReadyMarkerInfo(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name        string
		body        string // empty: no marker
		wantPartial bool
		wantFailed  []string
	}{
		{"missing", "", false, nil},
		{"malformed", "{not json", false, nil},
		{"non-object", `[1,2]`, false, nil},
		{"clean", `{"ready":true}`, false, nil},
		{"partial", `{"ready":true,"partial":true,"failed_exporters":["csv","parquet"]}`, true, []string{"csv", "parquet"}},
		{"failed not list", `{"partial":true,"failed_exporters":"csv"}`, true, nil},
		{"non-string items", `{"failed_exporters":["a",1]}`, false, []string{"a", "1"}},
		{"null list", `{"partial":true,"failed_exporters":null}`, true, nil},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			t.Parallel()
			dir := t.TempDir()
			if c.body != "" {
				rmarkWrite(t, ReadyMarkerPath(dir), c.body)
			}
			p, f := ReadReadyMarkerInfo(dir)
			if p != c.wantPartial || !reflect.DeepEqual(f, c.wantFailed) {
				t.Fatalf("got (%v, %v), want (%v, %v)", p, f, c.wantPartial, c.wantFailed)
			}
		})
	}

	dir := t.TempDir()
	phase := "profiling"
	m := ReadyMarker{Partial: true, WasCancelled: true, TerminalPhase: &phase, FailedExporters: []string{"parquet"}}
	if err := WriteReadyMarker(dir, m); err != nil {
		t.Fatal(err)
	}
	if p, f := ReadReadyMarkerInfo(dir); !p || !reflect.DeepEqual(f, []string{"parquet"}) {
		t.Fatalf("round trip got (%v, %v)", p, f)
	}
	raw, _ := os.ReadFile(ReadyMarkerPath(dir))
	want := `{"ready":true,"was_cancelled":true,"partial":true,"terminal_phase":"profiling","failed_exporters":["parquet"]}`
	if string(raw) != want {
		t.Fatalf("payload = %s, want %s", raw, want)
	}
}

func TestReadyMarkerResolveResultFile(t *testing.T) {
	t.Parallel()
	notReady := t.TempDir()
	ready := t.TempDir()
	if err := WriteReadyMarker(ready, ReadyMarker{}); err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name       string
		dir        string
		file       string
		wantStatus int // 0: success
		wantRel    string
	}{
		{"ready top-level", ready, "profile.json", 0, "profile.json"},
		{"not ready top-level", notReady, "profile.json", 404, ""},
		{"not ready checkpoint", notReady, "checkpoints/records.parquet", 0, "checkpoints/records.parquet"},
		{"backslash checkpoint", notReady, `checkpoints\records.parquet`, 0, "checkpoints/records.parquet"},
		{"checkpoint lookalike", notReady, "checkpoints2/x", 404, ""},
		{"traversal", ready, "../etc/passwd", 400, ""},
		{"inner traversal", ready, "a/../../x", 400, ""},
		{"posix absolute", ready, "/etc/passwd", 400, ""},
		{"windows absolute", ready, `C:\Windows\x`, 400, ""},
		{"unc", ready, `\\host\share\x`, 400, ""},
		{"nul", ready, "a\x00b", 400, ""},
		{"empty", ready, "", 400, ""},
		{"ready marker", ready, ReadyMarkerName, 400, ""},
		{"processing marker nested", ready, "sub/" + ProcessingMarkerName, 400, ""},
		{"marker temp", ready, "." + ReadyMarkerName + ".abc.tmp", 400, ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			t.Parallel()
			got, err := ResolveResultFile(c.dir, c.file)
			if c.wantStatus == 0 {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if want := filepath.Join(c.dir, filepath.FromSlash(c.wantRel)); got != want {
					t.Fatalf("path = %s, want %s", got, want)
				}
				return
			}
			var ue *ResultFileUnavailableError
			if !errors.As(err, &ue) || ue.Status != c.wantStatus {
				t.Fatalf("err = %v, want status %d", err, c.wantStatus)
			}
		})
	}
}

func TestReadyMarkerNotReadyDetail(t *testing.T) {
	t.Parallel()
	dir := filepath.Join(t.TempDir(), "1714069323")
	want := "Results not ready for 1714069323; marker file .aiperf_results_ready.json not present — retry after completion"
	if got := ResultsNotReadyDetail(dir); got != want {
		t.Fatalf("detail = %q", got)
	}
	if err := WriteProcessingMarker(dir); err != nil {
		t.Fatal(err)
	}
	want = "Results not ready for 1714069323; export still processing; marker file .aiperf_results_ready.json not present — retry after completion"
	if got := ResultsNotReadyDetail(dir); got != want {
		t.Fatalf("processing detail = %q", got)
	}
}

func TestReadyMarkerCollectArtifacts(t *testing.T) {
	t.Parallel()
	build := func(t *testing.T, ready bool) string {
		dir := t.TempDir()
		rmarkWrite(t, filepath.Join(dir, "profile.json"), "{}")
		rmarkWrite(t, filepath.Join(dir, "export.csv"), "a,b\n")
		rmarkWrite(t, filepath.Join(dir, "sweeps", "s1", "r.json"), "1")
		rmarkWrite(t, filepath.Join(dir, ResultCheckpointsName, "records.parquet"), "pq")
		rmarkWrite(t, filepath.Join(dir, ResultCheckpointsName, "deep", "c.jsonl"), "x")
		if err := WriteProcessingMarker(dir); err != nil {
			t.Fatal(err)
		}
		if ready {
			if err := WriteReadyMarker(dir, ReadyMarker{}); err != nil {
				t.Fatal(err)
			}
		}
		return dir
	}
	cases := []struct {
		name      string
		ready     bool
		recursive bool
		want      []string
	}{
		{"not ready lists only checkpoints", false, true, []string{"checkpoints/deep/c.jsonl", "checkpoints/records.parquet"}},
		{"ready top-level", true, false, []string{"checkpoints/deep/c.jsonl", "checkpoints/records.parquet", "export.csv", "profile.json"}},
		{"ready recursive", true, true, []string{"checkpoints/deep/c.jsonl", "checkpoints/records.parquet", "export.csv", "profile.json", "sweeps/s1/r.json"}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			t.Parallel()
			dir := build(t, c.ready)
			arts, err := CollectReadyArtifacts(dir, c.recursive)
			if err != nil {
				t.Fatal(err)
			}
			if got := rmarkNames(arts); !reflect.DeepEqual(got, c.want) {
				t.Fatalf("names = %v, want %v", got, c.want)
			}
			for _, a := range arts {
				if a.Name == "export.csv" && a.Size != 4 {
					t.Fatalf("export.csv size = %d", a.Size)
				}
			}
		})
	}

	arts, err := CollectReadyArtifacts(filepath.Join(t.TempDir(), "absent"), true)
	if err != nil || len(arts) != 0 {
		t.Fatalf("absent dir: %v %v", arts, err)
	}
}

func TestReadyMarkerLineSeparatorsRaw(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	errText := "a\u2028b\u2029c" + `\u2028`
	if err := WriteReadyMarker(dir, ReadyMarker{TerminalError: &errText}); err != nil {
		t.Fatal(err)
	}
	raw, _ := os.ReadFile(ReadyMarkerPath(dir))
	// Upstream golden: orjson keeps U+2028/U+2029 raw and escapes only the
	// literal backslash.
	want := `{"ready":true,"was_cancelled":false,"partial":false,"terminal_error":"a` +
		"\u2028b\u2029c" + `\\u2028"}`
	if string(raw) != want {
		t.Fatalf("payload = %q, want %q", raw, want)
	}
}

func TestReadyMarkerCheckpointSymlinkGated(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	rmarkWrite(t, filepath.Join(dir, "profile.json"), "{}")
	rmarkWrite(t, filepath.Join(dir, ResultCheckpointsName, "keep"), "x")
	link := filepath.Join(dir, ResultCheckpointsName, "leak.json")
	if err := os.Symlink(filepath.Join(dir, "profile.json"), link); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	var ue *ResultFileUnavailableError
	if _, err := ResolveResultFile(dir, "checkpoints/leak.json"); !errors.As(err, &ue) || ue.Status != 404 {
		t.Fatalf("checkpoint symlink to top-level artifact served before ready: %v", err)
	}
}
