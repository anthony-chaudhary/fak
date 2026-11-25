package aeogen

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// goldenOutputs maps each generated file to its committed golden (rendered by the Python
// originals over the prepared fixture).
var goldenOutputs = map[string]string{
	"llms-full.txt":                   "llms-full.txt.golden",
	"llms.txt":                        "llms.txt.golden",
	"docs/_includes/head-custom.html": "head-custom.html.golden",
	"docs/FAQ.md":                     "FAQ.md.golden",
	"docs/index.md":                   "index.md.golden",
}

func copyTree(t *testing.T, src, dst string) {
	t.Helper()
	walkErr := filepath.WalkDir(src, func(p string, d os.DirEntry, werr error) error {
		if werr != nil {
			return werr
		}
		rel, rerr := filepath.Rel(src, p)
		if rerr != nil {
			return rerr
		}
		target := filepath.Join(dst, rel)
		if d.IsDir() {
			return os.MkdirAll(target, 0o755)
		}
		b, rerr := os.ReadFile(p)
		if rerr != nil {
			return rerr
		}
		return os.WriteFile(target, b, 0o644)
	})
	if walkErr != nil {
		t.Fatal(walkErr)
	}
}

// prepareFixture copies the fixture and adds the inputs a checkout could not keep
// byte-exact: CRLF/lone-CR line endings and invalid UTF-8.
func prepareFixture(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	copyTree(t, filepath.Join("testdata", "fixture"), dir)
	writeFile(t, dir, "docs/crlf.md", "# CRLF\r\n\r\nline one\r\nline two\rlone cr [next](guide/next.md)\r\n")
	writeFile(t, dir, "docs/bad.md", "# Bad\n\nbytes \xe2\x82 then \xff\xfe end \xf0\x9f\x98 [next](guide/next.md)\n")
	return dir
}

func writeFile(t *testing.T, dir, rel, body string) {
	t.Helper()
	p := filepath.Join(dir, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func readFile(t *testing.T, dir, rel string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(dir, filepath.FromSlash(rel)))
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func goGenerate(t *testing.T, dir string) {
	t.Helper()
	var out bytes.Buffer
	if rc := RunStructuredData(dir, false, &out, &out); rc != 0 {
		t.Fatalf("structured data write rc=%d: %s", rc, out.String())
	}
	if rc := RunLLMSFull(dir, false, &out, &out); rc != 0 {
		t.Fatalf("llms-full write rc=%d: %s", rc, out.String())
	}
}

func goCheck(dir string) (sdRC int, sdOut string, lfRC int, lfOut string) {
	var a, b bytes.Buffer
	sdRC = RunStructuredData(dir, true, &a, &a)
	lfRC = RunLLMSFull(dir, true, &b, &b)
	return sdRC, a.String(), lfRC, b.String()
}

// makeStale perturbs an input of each generator and the corpus itself.
func makeStale(t *testing.T, dir string) {
	t.Helper()
	writeFile(t, dir, "VERSION", "9.9.9-stale\n")
	writeFile(t, dir, "llms-full.txt", readFile(t, dir, "llms-full.txt")+"\n# tampered\n")
}

func addArtifactAnswer(t *testing.T, dir string) {
	t.Helper()
	writeFile(t, dir, "docs/FAQ.md", readFile(t, dir, "docs/FAQ.md")+
		"\n## Is 2 ** 3 eight?\n\nYes, 2 ** 3 is eight in power notation, plus a `` stray pair.\n")
}

func TestGeneratorsMatchGolden(t *testing.T) {
	dir := prepareFixture(t)
	goGenerate(t, dir)
	for rel, golden := range goldenOutputs {
		want, err := os.ReadFile(filepath.Join("testdata", "golden", golden))
		if err != nil {
			t.Fatal(err)
		}
		if got := readFile(t, dir, rel); got != string(want) {
			t.Errorf("%s differs from golden %s\n--- got ---\n%s", rel, golden, got)
		}
	}

	sdRC, sdOut, lfRC, lfOut := goCheck(dir)
	if sdRC != 0 || lfRC != 0 {
		t.Fatalf("clean tree check: structured rc=%d %q, llms-full rc=%d %q", sdRC, sdOut, lfRC, lfOut)
	}
	var out bytes.Buffer
	if rc := RunStructuredData(dir, false, &out, &out); rc != 0 || !strings.Contains(out.String(), "(no changes — already up to date)") {
		t.Fatalf("idempotent rewrite rc=%d: %s", rc, out.String())
	}

	makeStale(t, dir)
	sdRC, sdOut, lfRC, lfOut = goCheck(dir)
	if sdRC != 1 || !strings.Contains(sdOut, "  - docs/_includes/head-custom.html\n") {
		t.Errorf("stale structured check rc=%d out=%q", sdRC, sdOut)
	}
	if lfRC != 1 || lfOut != "warn: linked file not found, skipping: docs/missing.md\nllms-full.txt OUT OF DATE; run: python tools/gen_llms_full.py\n" {
		t.Errorf("stale llms-full check rc=%d out=%q", lfRC, lfOut)
	}
	if err := os.Remove(filepath.Join(dir, "llms-full.txt")); err != nil {
		t.Fatal(err)
	}
	if _, _, rc, msg := goCheck(dir); rc != 1 || !strings.Contains(msg, "(missing)") {
		t.Errorf("missing llms-full check rc=%d out=%q", rc, msg)
	}
}

func TestStructuredDataArtifactGuard(t *testing.T) {
	dir := prepareFixture(t)
	goGenerate(t, dir)
	addArtifactAnswer(t, dir)
	goGenerate(t, dir)
	rc, out, _, _ := goCheck(dir)
	if rc != 1 || !strings.Contains(out, "FAQPage answers carry raw markdown artifacts (fix _plain()):\n  - 'Is 2 ** 3 eight?': ") {
		t.Fatalf("artifact guard rc=%d out=%q", rc, out)
	}
	if strings.Contains(out, "OUT OF DATE") {
		t.Fatalf("freshly written tree reported drift: %q", out)
	}
}

func TestPythonTextHelpers(t *testing.T) {
	if got := decodeReplace([]byte("a\xe2\x82b\xff\xfec\xf0\x9f\x98")); got != "a�b��c�" {
		t.Errorf("decodeReplace = %q", got)
	}
	if got := urlQuote("docs/a b+ü~@:.md", "/:@"); got != "docs/a%20b%2B%C3%BC~@:.md" {
		t.Errorf("urlQuote = %q", got)
	}
	for in, want := range map[string]string{"": ".", "a/./b/../c": "a/c", "../x": "../x", "a/../../x": "../x", "//a//b": "//a/b", "///a": "/a", "./": "."} {
		if got := posixNormpath(in); got != want {
			t.Errorf("posixNormpath(%q) = %q, want %q", in, got, want)
		}
	}
	for n, want := range map[int]string{0: "0", 999: "999", 1000: "1,000", 2177714: "2,177,714"} {
		if got := thousands(n); got != want {
			t.Errorf("thousands(%d) = %q", n, got)
		}
	}
	if got := pyJSONDumps(jsonObj{{"s", "<a>& \"\\\x01é"}, {"e", []any{}}, {"o", jsonObj{}}}); got != "{\n  \"s\": \"<a>& \\\"\\\\\\u0001é\",\n  \"e\": [],\n  \"o\": {}\n}" {
		t.Errorf("pyJSONDumps = %q", got)
	}
}

func findPython() string {
	for _, name := range []string{"python", "python3"} {
		if p, err := exec.LookPath(name); err == nil {
			return p
		}
	}
	return ""
}

type pyRun struct {
	rc  int
	out string
}

func runPython(t *testing.T, py, dir string, args ...string) pyRun {
	t.Helper()
	cmd := exec.Command(py, args...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "PYTHONIOENCODING=utf-8")
	out, err := cmd.CombinedOutput()
	rc := 0
	if err != nil {
		ee, ok := err.(*exec.ExitError)
		if !ok {
			t.Fatalf("python %v: %v", args, err)
		}
		rc = ee.ExitCode()
	}
	return pyRun{rc: rc, out: strings.ReplaceAll(string(out), "\r\n", "\n")}
}

// TestPythonParity runs the Python originals and the port over the same fixture and
// compares every generated file byte-for-byte plus the --check verdicts. With
// AEOGEN_UPDATE_GOLDEN=1 it rewrites the goldens from the Python output.
func TestPythonParity(t *testing.T) {
	py := findPython()
	if py == "" {
		t.Skip("python not on PATH")
	}
	repoTools, err := filepath.Abs(filepath.Join("..", "..", "..", "tools"))
	if err != nil {
		t.Fatal(err)
	}
	pyDir, goDir := prepareFixture(t), prepareFixture(t)
	for _, script := range []string{"gen_structured_data.py", "gen_llms_full.py"} {
		b, rerr := os.ReadFile(filepath.Join(repoTools, script))
		if rerr != nil {
			t.Skipf("python original not present: %v", rerr)
		}
		writeFile(t, pyDir, "tools/"+script, string(b))
		writeFile(t, goDir, "tools/"+script, string(b))
	}
	sdScript := filepath.Join(pyDir, "tools", "gen_structured_data.py")
	lfScript := filepath.Join(pyDir, "tools", "gen_llms_full.py")
	pyCheck := func() (pyRun, pyRun) {
		return runPython(t, py, pyDir, sdScript, "--check"), runPython(t, py, pyDir, lfScript, "--check", "--root", pyDir)
	}
	compareChecks := func(stage string) {
		t.Helper()
		sd, lf := pyCheck()
		sdRC, sdOut, lfRC, lfOut := goCheck(goDir)
		if sd.rc != sdRC || sd.out != sdOut {
			t.Errorf("%s structured check: python rc=%d %q, go rc=%d %q", stage, sd.rc, sd.out, sdRC, sdOut)
		}
		if lf.rc != lfRC || lf.out != lfOut {
			t.Errorf("%s llms-full check: python rc=%d %q, go rc=%d %q", stage, lf.rc, lf.out, lfRC, lfOut)
		}
	}
	compareFiles := func(stage string) {
		t.Helper()
		for rel := range goldenOutputs {
			if p, g := readFile(t, pyDir, rel), readFile(t, goDir, rel); p != g {
				t.Errorf("%s: %s differs between python and go\n--- python ---\n%s\n--- go ---\n%s", stage, rel, p, g)
			}
		}
	}
	generate := func() {
		t.Helper()
		if r := runPython(t, py, pyDir, sdScript); r.rc != 0 {
			t.Fatalf("python structured write: %+v", r)
		}
		if r := runPython(t, py, pyDir, lfScript, "--root", pyDir); r.rc != 0 {
			t.Fatalf("python llms-full write: %+v", r)
		}
		goGenerate(t, goDir)
	}

	compareChecks("pristine")
	generate()
	compareFiles("first write")
	compareChecks("clean")
	if os.Getenv("AEOGEN_UPDATE_GOLDEN") == "1" {
		for rel, golden := range goldenOutputs {
			writeFile(t, filepath.Join("testdata", "golden"), golden, readFile(t, pyDir, rel))
		}
	}

	makeStale(t, pyDir)
	makeStale(t, goDir)
	compareChecks("stale")
	generate()
	compareFiles("stale rewrite")

	addArtifactAnswer(t, pyDir)
	addArtifactAnswer(t, goDir)
	compareChecks("artifact")
	generate()
	compareFiles("artifact rewrite")
	compareChecks("artifact rewritten")
}
