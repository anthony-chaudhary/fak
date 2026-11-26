package scorecard

import (
	"bytes"
	"encoding/json"
	"flag"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
)

var update = flag.Bool("update", false, "rewrite testdata/golden from the Go scorecard")

// copyFixture copies testdata/fixture out of the testdata tree: the scorecard
// skips any path with a testdata component, so it must be scored elsewhere.
func copyFixture(t *testing.T) string {
	t.Helper()
	dst := filepath.Join(t.TempDir(), "workspace")
	src := filepath.Join("testdata", "fixture")
	walkErr := filepath.WalkDir(src, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, relErr := filepath.Rel(src, p)
		if relErr != nil {
			return relErr
		}
		target := filepath.Join(dst, rel)
		if d.IsDir() {
			return os.MkdirAll(target, 0o755)
		}
		b, readErr := os.ReadFile(p)
		if readErr != nil {
			return readErr
		}
		// A CRLF checkout must not change the bytes the oracle reads.
		return os.WriteFile(target, bytes.ReplaceAll(b, []byte("\r\n"), []byte("\n")), 0o644)
	})
	if walkErr != nil {
		t.Fatal(walkErr)
	}
	return ResolvePath(dst)
}

type mode struct {
	name string
	args []string
}

func fixtureModes(baseline string) []mode {
	return []mode{
		{"json", []string{"--json"}},
		{"default", nil},
		{"chart", []string{"--chart"}},
		{"critical", []string{"--critical"}},
		{"rollup", []string{"--rollup"}},
		{"gaps", []string{"--gaps"}},
		{"pairs", []string{"--pairs"}},
		{"index", []string{"--index"}},
		{"index-limit", []string{"--index", "--limit", "3"}},
		{"index-negative-limit", []string{"--ind", "--limit=-2"}},
		{"lookup-exact", []string{"--lookup", "kv cache"}},
		{"lookup-near", []string{"--lookup", "Sess"}},
		{"lookup-miss", []string{"--lookup", "zzqqxx"}},
		{"compare", []string{"--compare", baseline}},
	}
}

func runGo(t *testing.T, args ...string) (string, int) {
	t.Helper()
	var out, errw bytes.Buffer
	code := Main(args, &out, &errw, ".")
	if errw.Len() > 0 {
		t.Logf("stderr: %s", errw.String())
	}
	return out.String(), code
}

// scrubWorkspace replaces the scratch workspace path, which differs per run.
func scrubWorkspace(s, ws string) string {
	enc, _ := json.Marshal(ws)
	s = strings.ReplaceAll(s, string(enc[1:len(enc)-1]), "<WORKSPACE>")
	return strings.ReplaceAll(s, ws, "<WORKSPACE>")
}

func checkGolden(t *testing.T, name, got string) {
	t.Helper()
	path := filepath.Join("testdata", "golden", name)
	if *update {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(got), 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read golden %s: %v (run with -update)", name, err)
	}
	if w := strings.ReplaceAll(string(want), "\r\n", "\n"); w != got {
		t.Fatalf("%s differs from golden:\n%s", name, firstDiff(w, got))
	}
}

func firstDiff(want, got string) string {
	wl, gl := strings.Split(want, "\n"), strings.Split(got, "\n")
	for i := 0; i < len(wl) || i < len(gl); i++ {
		var w, g string
		if i < len(wl) {
			w = wl[i]
		}
		if i < len(gl) {
			g = gl[i]
		}
		if w != g {
			return "line " + strconv.Itoa(i+1) + ":\nwant: " + w + "\n got: " + g
		}
	}
	return "(lengths differ)"
}

func writeBaseline(t *testing.T, ws string) string {
	t.Helper()
	out, _ := runGo(t, "--workspace", ws, "--json")
	p := filepath.Join(t.TempDir(), "baseline.json")
	if err := os.WriteFile(p, []byte(out), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// fak-test:runtime fast est=2s
func TestFixtureMatchesGolden(t *testing.T) {
	ws := copyFixture(t)
	baseline := writeBaseline(t, ws)
	for _, m := range fixtureModes(baseline) {
		out, code := runGo(t, append([]string{"--workspace", ws}, m.args...)...)
		if code != 1 {
			t.Fatalf("%s: exit %d, want 1 (the fixture carries debt)", m.name, code)
		}
		ext := ".txt"
		if m.name == "json" {
			ext = ".json"
		}
		checkGolden(t, m.name+ext, scrubWorkspace(out, ws))
	}
	md := filepath.Join(t.TempDir(), "md")
	if out, _ := runGo(t, "--workspace", ws, "--markdown-dir", md, "--stamp", "fixture"); !strings.HasPrefix(out, "wrote concept-disambiguation doc folder -> ") {
		t.Fatalf("markdown-dir announcement = %q", out)
	}
	for _, name := range []string{ReadmeName, IndexName} {
		b, err := os.ReadFile(filepath.Join(md, name))
		if err != nil {
			t.Fatal(err)
		}
		checkGolden(t, name, string(b))
	}
}

func findPython(t *testing.T) string {
	t.Helper()
	names := []string{"python3", "python"}
	if runtime.GOOS == "windows" {
		names = []string{"python", "python3"}
	}
	for _, n := range names {
		if p, err := exec.LookPath(n); err == nil {
			return p
		}
	}
	t.Skip("python is not on PATH; the parity oracle cannot run")
	return ""
}

func oracleScript(t *testing.T) string {
	t.Helper()
	p, err := filepath.Abs(filepath.Join("..", "..", "..", "tools", "concept_disambiguation_scorecard.py"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(p); err != nil {
		t.Skipf("parity oracle %s is absent", p)
	}
	return p
}

// fak-test:runtime fast est=15s
func TestFixtureParityWithPythonOracle(t *testing.T) {
	python := findPython(t)
	script := oracleScript(t)
	ws := copyFixture(t)
	baseline := writeBaseline(t, ws)
	runPy := func(args ...string) (string, int) {
		cmd := exec.Command(python, append([]string{"-I", "-B", script, "--workspace", ws}, args...)...)
		var stderr bytes.Buffer
		cmd.Stderr = &stderr
		out, err := cmd.Output()
		code := 0
		if err != nil {
			exitErr, ok := err.(*exec.ExitError)
			if !ok {
				t.Fatalf("python oracle: %v", err)
			}
			code = exitErr.ExitCode()
		}
		if stderr.Len() > 0 {
			t.Logf("python stderr: %s", stderr.String())
		}
		// Python's text-mode stdout writes CRLF on Windows; that is transport.
		return strings.ReplaceAll(string(out), "\r\n", "\n"), code
	}
	for _, m := range fixtureModes(baseline) {
		want, wantCode := runPy(m.args...)
		got, gotCode := runGo(t, append([]string{"--workspace", ws}, m.args...)...)
		if wantCode != gotCode {
			t.Errorf("%s: exit go=%d python=%d", m.name, gotCode, wantCode)
		}
		if want != got {
			t.Errorf("%s: output differs from the Python oracle:\n%s", m.name, firstDiff(want, got))
		}
	}
	pyMD, goMD := filepath.Join(t.TempDir(), "py"), filepath.Join(t.TempDir(), "go")
	runPy("--markdown-dir", pyMD, "--stamp", "fixture")
	runGo(t, "--workspace", ws, "--markdown-dir", goMD, "--stamp", "fixture")
	for _, name := range []string{ReadmeName, IndexName} {
		want, err := os.ReadFile(filepath.Join(pyMD, name))
		if err != nil {
			t.Fatal(err)
		}
		got, err := os.ReadFile(filepath.Join(goMD, name))
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(want, got) {
			t.Errorf("%s differs from the Python oracle:\n%s", name, firstDiff(string(want), string(got)))
		}
	}
}

func TestAuditErrorPayloadMatchesPythonShape(t *testing.T) {
	ws := t.TempDir()
	out, code := runGo(t, "--workspace", ws, "--json", "--data", filepath.Join(ws, "absent"))
	if code != 1 {
		t.Fatalf("exit %d, want 1", code)
	}
	var doc map[string]any
	if err := json.Unmarshal([]byte(out), &doc); err != nil {
		t.Fatal(err)
	}
	if doc["verdict"] != "AUDIT_ERROR" || doc["finding"] != "tooling_error" {
		t.Fatalf("payload = %v", doc)
	}
	if !strings.HasPrefix(doc["reason"].(string), "missing data directory: ") {
		t.Fatalf("reason = %v", doc["reason"])
	}
	if c, ok := doc["corpus"].(map[string]any); !ok || len(c) != 0 {
		t.Fatalf("corpus = %v, want {}", doc["corpus"])
	}
	if _, ok := doc["_data"]; ok {
		t.Fatal("an AUDIT_ERROR payload carries no _data")
	}
}

func TestPythonFormattingHelpers(t *testing.T) {
	floats := map[float64]string{1e16: "1e+16", 1234567890123456.0: "1234567890123456.0", 0.0001: "0.0001",
		0.00001: "1e-05", 100: "100.0", 87.25: "87.25", -0.5: "-0.5", 0: "0.0"}
	for f, want := range floats {
		if got := pyFloatRepr(f); got != want {
			t.Errorf("pyFloatRepr(%v) = %q, want %q", f, got, want)
		}
	}
	if pyRound(2.675, 2) != 2.67 || pyRound(0.125, 2) != 0.12 || pyRoundInt(2.5) != 2 || pyRoundInt(3.5) != 4 {
		t.Error("round must be correctly rounded with ties to even")
	}
	reprs := map[string]string{"it's": `"it's"`, `say "hi"`: `'say "hi"'`, `both ' "`: `'both \' "'`,
		"tab\there": `'tab\there'`, "\u200b\u00a0\u00e9": "'\\u200b\\xa0\u00e9'"}
	for s, want := range reprs {
		if got := pyStrString(s); got != want {
			t.Errorf("repr(%q) = %s, want %s", s, got, want)
		}
	}
	glosses := map[string]string{"Decision (kernel)": "Decision", "f (a) (b)": "f (a)", "f (a (b)": "f",
		"plain": "plain", "x (y) z": "x (y) z", "  q (r)  \n": "q"}
	for s, want := range glosses {
		if got := bareName(s); got != want {
			t.Errorf("bareName(%q) = %q, want %q", s, got, want)
		}
	}
	words := map[string]string{"KVCacheShape": "kv cache shape", "witnessPath": "witness path",
		"HTTPServer2x": "http server2x", "q4_kernel.v2": "q4 kernel v2", "ABC": "abc", "Ab": "ab"}
	for s, want := range words {
		if got := strings.Join(splitWords(s), " "); got != want {
			t.Errorf("splitWords(%q) = %q, want %q", s, got, want)
		}
	}
	if got := pySum([]float64{0.1, 0.2, 0.3}); got != 0.6 {
		t.Errorf("compensated sum = %v, want 0.6", got)
	}
	if got := string(asciiJSON([]byte("\"\u00e9\U0001F642\x7f<\""))); got != `"\u00e9\ud83d\ude42\u007f<"` {
		t.Errorf("asciiJSON = %s", got)
	}
}
