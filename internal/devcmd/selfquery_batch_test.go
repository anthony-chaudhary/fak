package devcmd

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/selfquery"
)

// writeSelfQueryFixtureRoot builds a small repo root the dev catalog can load: one
// lane, one indexed doc, one claim. lane distinguishes two roots in one batch.
func writeSelfQueryFixtureRoot(t *testing.T, lane string) string {
	t.Helper()
	root := t.TempDir()
	files := map[string]string{
		"dos.toml":             "[lanes.trees]\n" + lane + " = [\"internal/" + lane + "/**\"] # the " + lane + " landing surface\n",
		"INDEX.md":             "# INDEX\n- [" + lane + " guide](docs/" + lane + ".md) - how the " + lane + " landing path works.\n",
		"docs/" + lane + ".md": "# " + lane + " guide\nLanding notes.\n",
		"CLAIMS.md":            "# CLAIMS\n## " + lane + "\n- [SHIPPED] internal/" + lane + " lands changes.\n",
	}
	for rel, body := range files {
		full := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

type selfQuerySingleRun struct {
	code           int
	stdout, stderr string
}

func runSelfQuerySingle(argv []string) selfQuerySingleRun {
	var out, errOut bytes.Buffer
	var code int
	switch argv[0] {
	case "feature":
		code = RunFeature(&out, &errOut, argv[1:])
	case "capabilities":
		code = RunCapabilities(&out, &errOut, argv[1:])
	}
	return selfQuerySingleRun{code: code, stdout: out.String(), stderr: errOut.String()}
}

func decodeFeatureBatch(t *testing.T, raw string) featureBatchOutput {
	t.Helper()
	var out featureBatchOutput
	if err := json.Unmarshal([]byte(raw), &out); err != nil {
		t.Fatalf("batch output is not JSON: %v\n%s", err, raw)
	}
	if out.Schema != featureBatchSchema {
		t.Fatalf("schema = %q, want %q", out.Schema, featureBatchSchema)
	}
	return out
}

// Every batch result carries exactly what the standalone command prints for the
// same argv: stdout bytes, stderr bytes, and exit code, including a failing segment.
func TestFeatureBatchMatchesSingleCommands(t *testing.T) {
	alpha, beta := writeSelfQueryFixtureRoot(t, "alpha"), writeSelfQueryFixtureRoot(t, "beta")
	singles := [][]string{
		{"feature", "query", "landing surface", "--root", alpha, "--json"},
		{"feature", "query", "landing surface", "--root", beta, "--json"},
		{"capabilities", "--root", beta, "landing surface"},
		{"feature", "query", "beta guide", "--root", beta},
		{"feature", "query", "landing", "--root", alpha, "--plane", "bogus"},
		{"capabilities", "--root", beta, "--json", "memory"},
	}

	batchArgv := []string{"batch", "--json"}
	for i, single := range singles {
		if i > 0 {
			batchArgv = append(batchArgv, featureBatchSeparator)
		}
		if single[0] == "feature" {
			batchArgv = append(batchArgv, single[1:]...)
		} else {
			batchArgv = append(batchArgv, single...)
		}
	}
	var out, errOut bytes.Buffer
	code := RunFeature(&out, &errOut, batchArgv)
	batch := decodeFeatureBatch(t, out.String())
	if len(batch.Results) != len(singles) {
		t.Fatalf("batch returned %d results for %d segments (stderr=%s)", len(batch.Results), len(singles), errOut.String())
	}

	wantCode := 0
	for i, single := range singles {
		want := runSelfQuerySingle(single)
		got := batch.Results[i]
		if !reflect.DeepEqual(got.Argv, single) {
			t.Errorf("result %d argv = %v, want %v", i, got.Argv, single)
		}
		if got.ExitCode != want.code {
			t.Errorf("result %d (%v) exit = %d, single command exit = %d", i, single, got.ExitCode, want.code)
		}
		if got.Stdout != want.stdout {
			t.Errorf("result %d (%v) stdout differs from the single command\nbatch:  %q\nsingle: %q", i, single, got.Stdout, want.stdout)
		}
		if got.Stderr != want.stderr {
			t.Errorf("result %d (%v) stderr differs from the single command\nbatch:  %q\nsingle: %q", i, single, got.Stderr, want.stderr)
		}
		if wantCode == 0 {
			wantCode = want.code
		}
	}
	if wantCode == 0 {
		t.Fatal("the bogus-plane segment should fail on its own; the fixture no longer covers a failing segment")
	}
	if code != wantCode {
		t.Errorf("batch exit = %d, want the first failing segment's %d", code, wantCode)
	}
	if first := batch.Results[0]; first.ExitCode != 0 || !strings.Contains(first.Stdout, "alpha") {
		t.Errorf("first segment should answer from the alpha root: exit=%d stdout=%s", first.ExitCode, first.Stdout)
	}
	if second := batch.Results[1]; !strings.Contains(second.Stdout, "beta") || strings.Contains(second.Stdout, "alpha guide") {
		t.Errorf("second segment should answer from the beta root only: %s", second.Stdout)
	}
}

func TestFeatureBatchTextModeLabelsEachSegment(t *testing.T) {
	alpha := writeSelfQueryFixtureRoot(t, "alpha")
	single := runSelfQuerySingle([]string{"feature", "query", "alpha guide", "--root", alpha})

	var out, errOut bytes.Buffer
	code := RunFeature(&out, &errOut, []string{"batch", "query", "alpha guide", "--root", alpha, featureBatchSeparator, "capabilities", "--root", alpha})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	text := out.String()
	header := "=== fak-dev " + quoteBatchArgv([]string{"feature", "query", "alpha guide", "--root", alpha}) + " (exit 0) ===\n"
	if !strings.Contains(header, `"alpha guide"`) {
		t.Fatalf("an argument with a space should be quoted in the header: %s", header)
	}
	if !strings.HasPrefix(text, header+single.stdout) {
		t.Errorf("text batch should open with the labelled single-command output\ngot:\n%s\nwant prefix:\n%s", text, header+single.stdout)
	}
	if !strings.Contains(text, "=== fak-dev "+quoteBatchArgv([]string{"capabilities", "--root", alpha})+" (exit 0) ===\n") {
		t.Errorf("text batch is missing the capabilities section header:\n%s", text)
	}
}

func TestFeatureBatchRejectsMalformedSegments(t *testing.T) {
	cases := [][]string{
		{"batch"},
		{"batch", "--json"},
		{"batch", "index", "verbs"},
		{"batch", "query", "x", featureBatchSeparator},
		{"batch", "query", "x", featureBatchSeparator, featureBatchSeparator, "capabilities"},
	}
	for _, argv := range cases {
		var out, errOut bytes.Buffer
		if code := RunFeature(&out, &errOut, argv); code != 2 {
			t.Errorf("RunFeature(%v) = %d, want 2 (stderr=%s)", argv, code, errOut.String())
		}
		if out.Len() != 0 {
			t.Errorf("RunFeature(%v) wrote to stdout on a usage error: %s", argv, out.String())
		}
	}
}

func TestMemoizedSelfQueryCatalogLoaderLoadsEachRootOnce(t *testing.T) {
	loads := map[string]int{}
	failure := errors.New("unreadable root")
	load := memoizedSelfQueryCatalogLoader(func(root string) (*selfquery.Catalog, error) {
		loads[root]++
		if root == "bad" {
			return nil, failure
		}
		return selfquery.Load(root, selfquery.Options{})
	})

	first, err := load("good")
	if err != nil {
		t.Fatal(err)
	}
	second, err := load("good")
	if err != nil {
		t.Fatal(err)
	}
	if first != second {
		t.Error("the same root should share one catalog")
	}
	respelled, err := load("good/.")
	if err != nil {
		t.Fatal(err)
	}
	if respelled == first {
		t.Error("another spelling of the root needs its own catalog, since responses echo the root string")
	}
	for i := 0; i < 2; i++ {
		if _, err := load("bad"); !errors.Is(err, failure) {
			t.Errorf("load(bad) attempt %d = %v, want the original failure replayed", i, err)
		}
	}
	if want := map[string]int{"good": 1, "good/.": 1, "bad": 1}; !reflect.DeepEqual(loads, want) {
		t.Errorf("loads = %v, want %v", loads, want)
	}
}
