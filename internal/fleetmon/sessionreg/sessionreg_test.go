package sessionreg

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"
	"unicode/utf8"
)

var update = flag.Bool("update", false, "rewrite testdata/golden from the Go port")

const (
	fixtureDir = "testdata/fixture"
	goldenDir  = "testdata/golden"
	// fixtureCwd is the cwd every fixture transcript records. It is relative so the bytes
	// (and so the task signatures and transcript sizes) are identical on every checkout;
	// both the Go test and the Python child run from this package directory.
	fixtureCwd = "testdata/fixture/work"
)

var frozenNow = time.Date(2026, 11, 24, 12, 0, 0, 250000*1000, time.UTC)

// modes runs the read-only modes first and registry last, so one fixture copy serves every
// mode: registry is the only one that writes, and runGoModes checks that.
var modes = []string{ModeJSON, ModePlan, ModeSummary, ModeResume, ModeRegistry}

var readOnlyModes = modes[:len(modes)-1]

var registryFiles = []string{"sessions.json", "resume_plan.json", "decisions.log", "transitions.log", "_prev_disp.json"}

// setupFixture copies the synthetic home + registry into dst, stamps every transcript's
// mtime relative to now, and points the fleet env knobs at the copy.
func setupFixture(t *testing.T, dst string, now time.Time) {
	t.Helper()
	for _, sub := range []string{"home", "reg"} {
		src := filepath.Join(fixtureDir, sub)
		err := filepath.WalkDir(src, func(p string, d fs.DirEntry, walkErr error) error {
			if walkErr != nil {
				return walkErr
			}
			rel, _ := filepath.Rel(fixtureDir, p)
			out := filepath.Join(dst, rel)
			if d.IsDir() {
				return os.MkdirAll(out, 0o755)
			}
			data, rerr := os.ReadFile(p)
			if rerr != nil {
				return rerr
			}
			data = bytes.ReplaceAll(data, []byte("{{WORK}}"), []byte(fixtureCwd))
			return os.WriteFile(out, data, 0o644)
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	for _, d := range []string{filepath.Join("home", ".claude-nopro"), "cfg"} {
		if err := os.MkdirAll(filepath.Join(dst, d), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	raw, err := os.ReadFile(filepath.Join(fixtureDir, "mtimes.json"))
	if err != nil {
		t.Fatal(err)
	}
	var ages map[string]float64
	if err := json.Unmarshal(raw, &ages); err != nil {
		t.Fatal(err)
	}
	for rel, minutes := range ages {
		mt := now.Add(-time.Duration(minutes * float64(time.Minute)))
		if err := os.Chtimes(filepath.Join(dst, filepath.FromSlash(rel)), mt, mt); err != nil {
			t.Fatal(err)
		}
	}
	policy, _ := filepath.Abs(filepath.Join(fixtureDir, "policy.json"))
	for k, v := range map[string]string{
		"FLEET_USER_HOME":            filepath.Join(dst, "home"),
		"FLEET_CONFIG_HOME":          filepath.Join(dst, "cfg"),
		"FLEET_REG_DIR":              filepath.Join(dst, "reg"),
		"FLEET_POLICY_PATH":          policy,
		"FAK_ACCOUNTS_REGISTRY":      filepath.Join(dst, "no-accounts-registry.json"),
		"FLEET_ACCOUNT_CAP_RUNS_DIR": filepath.Join(dst, "no-runs"),
		"FAK_APP_VERSION":            "fixture",
		"FLEET_WINDOW_H":             "3",
		"FAK_REHOME_CAP":             "4",
		"FAK_RESUME_ESCALATE_AFTER":  "2",
		"FAK_LOAD_REHOME":            "1",
		"FAK_MAX_ATTEMPTS":           "8",
	} {
		t.Setenv(k, v)
	}
}

func runGo(t *testing.T, mode string, now time.Time) string {
	t.Helper()
	var out bytes.Buffer
	if _, err := Run(context.Background(), FromEnv(filepath.Join("..", "..", ".."), now), mode, &out); err != nil {
		t.Fatalf("Run(%s): %v", mode, err)
	}
	return out.String()
}

// normalize makes an output comparable across fixture roots: the root (raw and
// JSON-escaped) becomes <ROOT>. Line endings are left alone so parity stays byte-exact.
func normalize(s, root string) string {
	for _, r := range []string{strings.ReplaceAll(root, `\`, `\\`), root} {
		s = strings.ReplaceAll(s, r, "<ROOT>")
	}
	return s
}

func registryOutputs(t *testing.T, root string) map[string]string {
	t.Helper()
	out := map[string]string{}
	for _, f := range registryFiles {
		data, err := os.ReadFile(filepath.Join(root, "reg", f))
		if err != nil {
			t.Fatalf("registry file %s: %v", f, err)
		}
		out[f] = string(data)
	}
	return out
}

// goldenForm is normalize plus platform folding (CRLF, and every path separator written as
// "/"), so one committed golden serves Windows and Unix runs alike.
func goldenForm(s, root string) string {
	s = strings.ReplaceAll(normalize(s, root), "\r\n", "\n")
	return strings.ReplaceAll(strings.ReplaceAll(s, `\\`, `/`), `\`, `/`)
}

// fixtureOutputs is every mode's stdout over one fixture copy plus the registry files the
// final registry mode wrote.
type fixtureOutputs struct {
	stdout map[string]string
	files  map[string]string
}

// treeSnapshot reads every file under root, keyed by its slash path.
func treeSnapshot(t *testing.T, root string) map[string]string {
	t.Helper()
	out := map[string]string{}
	err := filepath.WalkDir(root, func(p string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil || d.IsDir() {
			return walkErr
		}
		data, rerr := os.ReadFile(p)
		if rerr != nil {
			return rerr
		}
		rel, _ := filepath.Rel(root, p)
		out[filepath.ToSlash(rel)] = string(data)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func requireTreeUnchanged(t *testing.T, root string, before map[string]string, after string) {
	t.Helper()
	if now := treeSnapshot(t, root); !reflect.DeepEqual(before, now) {
		t.Fatalf("the fixture changed after %s, so the modes cannot share one copy", after)
	}
}

// runGoModes runs every mode in order over the one fixture copy at root.
func runGoModes(t *testing.T, root string, now time.Time) fixtureOutputs {
	t.Helper()
	before := treeSnapshot(t, root)
	out := fixtureOutputs{stdout: map[string]string{}}
	for _, mode := range modes {
		if mode == ModeRegistry {
			requireTreeUnchanged(t, root, before, "the read-only go modes")
		}
		out.stdout[mode] = runGo(t, mode, now)
	}
	out.files = registryOutputs(t, root)
	return out
}

func TestFixtureGolden(t *testing.T) {
	root := t.TempDir()
	setupFixture(t, root, frozenNow)
	res := runGoModes(t, root, frozenNow)
	for _, mode := range modes {
		t.Run(mode, func(t *testing.T) {
			got := map[string]string{mode + ".out": res.stdout[mode]}
			if mode == ModeRegistry {
				for f, body := range res.files {
					got[f] = body
				}
			}
			for name, body := range got {
				body = goldenForm(body, root)
				path := filepath.Join(goldenDir, name)
				if *update {
					if err := os.MkdirAll(goldenDir, 0o755); err != nil {
						t.Fatal(err)
					}
					if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
						t.Fatal(err)
					}
					continue
				}
				want, err := os.ReadFile(path)
				if err != nil {
					t.Fatalf("golden %s: %v (run with -update)", name, err)
				}
				if body != string(want) {
					t.Errorf("%s differs from golden:\n%s", name, firstDiff(string(want), body))
				}
			}
		})
	}
}

// TestFixtureDecisions pins the load-bearing verdicts the golden files encode, so a golden
// regeneration that silently changes one fails loudly here.
func TestFixtureDecisions(t *testing.T) {
	root := t.TempDir()
	setupFixture(t, root, frozenNow)
	p, err := Evaluate(context.Background(), FromEnv(filepath.Join("..", "..", ".."), frozenNow))
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]string{}
	for _, r := range p.Rows {
		got[r.Account+"/"+r.Session[:8]] = r.Disp + " " + r.Action
	}
	want := map[string]string{
		".claude-alpha/11111111":   "LIVE SKIP_LIVE",
		".claude-alpha/22222222":   "DEAD_MIDTOOL AUTO_RESUME",
		".claude-alpha/33333333":   "DONE SKIP_DONE",
		".claude-alpha/44444444":   "USER_CLOSED SKIP_USER_CLOSED",
		".claude-alpha/55555555":   "STOPPED_APIERR AUTO_RESUME",
		".claude-alpha/66666666":   "NEVER_STARTED AUTO_RESUME",
		".claude-alpha/77777777":   "PARKED_WAIT SKIP_PARKED",
		".claude-alpha/88888888":   "DEAD_KILLED DEFER_DUPLICATE_TASK",
		".claude-alpha/99999999":   "DEAD_KILLED AUTO_RESUME",
		".claude-alpha/aaaaaaaa":   "DEAD_MIDTOOL SURFACE",
		".claude-alpha/cccccccc":   "STOPPED_QUIET SURFACE",
		".claude-beta/dddddddd":    "STOPPED_LIMIT AUTO_RESUME",
		".claude-beta/22222222":    "DEAD_MIDTOOL SKIP_STALE_COPY",
		".claude-beta/eeeeeeee":    "INFRA_AUTH DEFER_THROTTLED",
		".claude-delta/ffffffff":   "DONE SKIP_DONE",
		".claude-alpha/a0a0a0a0":   "INFRA_ORG_DISABLED AUTO_RESUME",
		".claude-delta/d0d0d0d0":   "STOPPED_CONTEXT_EXHAUSTED ESCALATE_FRESH_CONTINUATION",
		".claude-epsilon/e1e1e1e1": "INFRA_AUTH BLOCKED_AUTH",
		// "Fehler é529" and "ü503ö": no word edge beside the code, so no API error.
		".claude-alpha/f1f1f1f1": "STOPPED_QUIET SURFACE",
		// The final end_turn record carries NaN/Infinity constants and still counts.
		".claude-alpha/f2f2f2f2": "DONE SKIP_DONE",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("decisions:\n got %v\nwant %v", got, want)
	}
	for _, r := range p.Rows {
		if r.Session[:8] == "f1f1f1f1" && r.HTTPStatus != "" {
			t.Errorf("f1f1f1f1 http_status = %q, want none: each code abuts a word rune", r.HTTPStatus)
		}
		if r.Session[:8] == "f2f2f2f2" && r.LastKind != "assistant_end" {
			t.Errorf("f2f2f2f2 last_kind = %q, want assistant_end from the NaN-bearing record", r.LastKind)
		}
		switch r.Session[:8] {
		case "dddddddd", "22222222", "a0a0a0a0":
			if r.Account == ".claude-beta" && r.Session[:8] == "22222222" {
				continue
			}
			if !r.Rehomed || r.ResumeAccount != ".claude-delta" {
				t.Errorf("%s should re-home onto delta, got rehomed=%v -> %s", r.Session[:8], r.Rehomed, r.ResumeAccount)
			}
		}
	}
	if _, ok := p.Throttle.Get(".claude-beta"); !ok {
		t.Errorf("beta's live session-limit banner should throttle it: %v", p.Throttle.Keys())
	}
	if _, ok := p.Auth.Get(".claude-delta"); ok {
		t.Errorf("delta's newer DONE turn should clear its carried auth block")
	}
}

func pythonExe() (string, []string) {
	for _, c := range []struct {
		bin  string
		args []string
	}{{"python3", nil}, {"python", nil}, {"py", []string{"-3"}}} {
		if p, err := exec.LookPath(c.bin); err == nil {
			if out, verr := exec.Command(p, append(c.args, "-c", "print(1)")...).Output(); verr == nil && strings.TrimSpace(string(out)) == "1" {
				return p, c.args
			}
		}
	}
	return "", nil
}

// pythonRun is one oracle process: each requested mode's stdout, decoded with the codec the
// process reported for sys.stdout, and that codec's name.
type pythonRun struct {
	stdout   map[string]string
	encoding string
}

// runPythonModes runs the oracle once over the fixture the env points at, one mode after
// another in a single interpreter, with its clock pinned to now (testdata/frozen_now.py), the
// instant the Go pass is handed, so neither side's start-up latency shifts an age. utf8Stdout
// forces PYTHONIOENCODING=utf-8; otherwise stdout keeps the host's default encoding.
func runPythonModes(t *testing.T, now time.Time, utf8Stdout bool, runModes []string) pythonRun {
	t.Helper()
	py, args := pythonExe()
	if py == "" {
		t.Skip("python not on PATH; the Python oracle cannot run")
	}
	script := filepath.Join("..", "..", "..", "tools", "fleet_sessions.py")
	if _, err := os.Stat(script); err != nil {
		t.Skipf("oracle %s absent: %v", script, err)
	}
	driver := filepath.Join("testdata", "frozen_now.py")
	argv := append(append([]string{}, args...), driver, now.UTC().Format(time.RFC3339Nano), script)
	cmd := exec.Command(py, append(argv, runModes...)...)
	cmd.Env = withoutEnv(os.Environ(), "PYTHONIOENCODING")
	if utf8Stdout {
		cmd.Env = append(cmd.Env, "PYTHONIOENCODING=utf-8")
	}
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	raw, err := cmd.Output()
	if err != nil {
		t.Fatalf("python %v: %v\n%s", runModes, err, stderr.String())
	}
	parts := bytes.Split(raw, []byte{0})
	if len(parts) != len(runModes)+1 {
		t.Fatalf("python driver wrote %d NUL-separated parts, want %d", len(parts), len(runModes)+1)
	}
	run := pythonRun{stdout: map[string]string{}, encoding: string(parts[len(runModes)])}
	for i, mode := range runModes {
		run.stdout[mode] = decodePython(t, py, args, run.encoding, parts[i])
	}
	return run
}

// decodePython decodes b from Python's codec enc with Python's own codec, so the comparison
// does not rest on a Go reimplementation of the host code page.
func decodePython(t *testing.T, py string, args []string, enc string, b []byte) string {
	t.Helper()
	if enc == "utf-8" {
		if !utf8.Valid(b) {
			t.Fatalf("python reported utf-8 but wrote invalid UTF-8")
		}
		return string(b)
	}
	const decode = "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read().decode(sys.argv[1]).encode('utf-8'))"
	cmd := exec.Command(py, append(append([]string{}, args...), "-c", decode, enc)...)
	cmd.Stdin = bytes.NewReader(b)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("decode python output as %s: %v\n%s", enc, err, stderr.String())
	}
	return string(out)
}

func withoutEnv(env []string, key string) []string {
	out := make([]string, 0, len(env))
	prefix := strings.ToUpper(key) + "="
	for _, kv := range env {
		if !strings.HasPrefix(strings.ToUpper(kv), prefix) {
			out = append(out, kv)
		}
	}
	return out
}

// TestFixtureParityWithPython runs the Python oracle and the Go port over identical copies
// of the fixture at one pinned instant and requires identical registry JSON and
// byte-identical text. The instant is the real clock (with a sub-second part, which the
// isoformat writers must render alike) so any clock read neither side injects still agrees.
//
// The read-only modes also run once with Python's stdout at the host default encoding;
// decoded with the codec Python reports, that text must equal the Go text too.
func TestFixtureParityWithPython(t *testing.T) {
	if py, _ := pythonExe(); py == "" {
		t.Skip("python not on PATH; the Python oracle cannot run")
	}
	base := t.TempDir()
	pyRoot, goRoot := filepath.Join(base, "py"), filepath.Join(base, "go")
	now := time.Now().UTC().Truncate(time.Microsecond)
	setupFixture(t, pyRoot, now)
	before := treeSnapshot(t, pyRoot)
	host := runPythonModes(t, now, false, readOnlyModes)
	requireTreeUnchanged(t, pyRoot, before, "the read-only python modes")
	pyRes := runPythonModes(t, now, true, modes)
	pyFiles := registryOutputs(t, pyRoot)
	setupFixture(t, goRoot, now)
	goRes := runGoModes(t, goRoot, now)
	t.Logf("python host-default stdout encoding: %s", host.encoding)
	for _, mode := range readOnlyModes {
		t.Run("host-encoding/"+mode, func(t *testing.T) {
			if a, b := normalize(host.stdout[mode], pyRoot), normalize(goRes.stdout[mode], goRoot); a != b {
				t.Errorf("%s stdout written as %s differs from go once decoded:\n%s", mode, host.encoding, firstDiff(a, b))
			}
		})
	}
	for _, mode := range modes {
		t.Run(mode, func(t *testing.T) {
			if a, b := normalize(pyRes.stdout[mode], pyRoot), normalize(goRes.stdout[mode], goRoot); a != b {
				t.Errorf("%s stdout differs (python vs go):\n%s", mode, firstDiff(a, b))
			}
			if mode != ModeRegistry {
				return
			}
			for _, f := range registryFiles {
				a, b := normalize(pyFiles[f], pyRoot), normalize(goRes.files[f], goRoot)
				if strings.HasSuffix(f, ".json") {
					if d := jsonDiff(a, b); d != "" {
						t.Errorf("%s differs semantically: %s", f, d)
					}
				}
				if a != b {
					t.Errorf("%s bytes differ:\n%s", f, firstDiff(a, b))
				}
			}
		})
	}
}

// jsonDiff reports the first path at which two JSON documents differ, or "".
func jsonDiff(a, b string) string {
	var va, vb any
	if err := json.Unmarshal([]byte(a), &va); err != nil {
		return "python: " + err.Error()
	}
	if err := json.Unmarshal([]byte(b), &vb); err != nil {
		return "go: " + err.Error()
	}
	return valueDiff("$", va, vb)
}

func valueDiff(path string, a, b any) string {
	switch ta := a.(type) {
	case map[string]any:
		tb, ok := b.(map[string]any)
		if !ok {
			return fmt.Sprintf("%s: %T vs %T", path, a, b)
		}
		keys := map[string]bool{}
		for k := range ta {
			keys[k] = true
		}
		for k := range tb {
			keys[k] = true
		}
		sorted := make([]string, 0, len(keys))
		for k := range keys {
			sorted = append(sorted, k)
		}
		sort.Strings(sorted)
		for _, k := range sorted {
			if d := valueDiff(path+"."+k, ta[k], tb[k]); d != "" {
				return d
			}
		}
		return ""
	case []any:
		tb, ok := b.([]any)
		if !ok || len(ta) != len(tb) {
			return fmt.Sprintf("%s: %v vs %v", path, a, b)
		}
		for i := range ta {
			if d := valueDiff(fmt.Sprintf("%s[%d]", path, i), ta[i], tb[i]); d != "" {
				return d
			}
		}
		return ""
	}
	if !reflect.DeepEqual(a, b) {
		return fmt.Sprintf("%s: %#v vs %#v", path, a, b)
	}
	return ""
}

func firstDiff(a, b string) string {
	la, lb := strings.Split(a, "\n"), strings.Split(b, "\n")
	for i := 0; i < len(la) || i < len(lb); i++ {
		var x, y string
		if i < len(la) {
			x = la[i]
		}
		if i < len(lb) {
			y = lb[i]
		}
		if x != y {
			return fmt.Sprintf("line %d:\n  want/python: %q\n  got/go:      %q", i+1, x, y)
		}
	}
	return "(no line difference)"
}

func TestPythonTextPrimitives(t *testing.T) {
	cases := map[float64]string{3: "3.0", 0.1: "0.1", 12.35: "12.35", 1e16: "1e+16", 1e-5: "1e-05", 0.0001: "0.0001", -0.5: "-0.5"}
	for f, want := range cases {
		if got := pyFloatRepr(f); got != want {
			t.Errorf("pyFloatRepr(%v) = %q, want %q", f, got, want)
		}
	}
	if got := pyRound(0.25, 1); got != 0.2 {
		t.Errorf("pyRound(0.25,1) = %v, want ties-to-even 0.2", got)
	}
	if got := pySplitlines("a\r\nb\u2028c\rd\n"); !reflect.DeepEqual(got, []string{"a", "b", "c", "d"}) {
		t.Errorf("pySplitlines = %q", got)
	}
	o := NewObj()
	o.Set("k", "é\u2028\U0001F600\x7f")
	o.Set("n", []any{})
	if got, want := dumpPy(o, 1), "{\n \"k\": \"\\u00e9\\u2028\\ud83d\\ude00\\u007f\",\n \"n\": []\n}"; got != want {
		t.Errorf("dumpPy = %q, want %q", got, want)
	}
	if !descLess("b", "a") || !descLess("a", "ab") || descLess("a", "a") {
		t.Errorf("descLess must order like tuple(-ord(c))")
	}
	if got := mtimeUTC(time.Unix(5, 1999)); !got.Equal(time.Unix(5, 2000)) {
		t.Errorf("mtimeUTC rounds to whole microseconds: %v", got)
	}
}
