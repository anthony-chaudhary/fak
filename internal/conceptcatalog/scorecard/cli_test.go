package scorecard

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCLIHelpDescribesEveryOption(t *testing.T) {
	for _, flag := range []string{"-h", "--help"} {
		var out, errw bytes.Buffer
		if code := Main([]string{flag}, &out, &errw, t.TempDir()); code != 0 || errw.Len() != 0 {
			t.Fatalf("%s: code=%d stderr=%q", flag, code, errw.String())
		}
		got := out.String()
		if !strings.HasPrefix(got, cliUsage+"\n\n"+cliDescription+"\n\noptions:\n") {
			t.Fatalf("%s: missing usage/description header:\n%s", flag, got)
		}
		// argparse layout: a short invocation shares its line with the help.
		for _, line := range []string{
			"  -h, --help            show this help message and exit\n",
			"  --json                emit machine-readable JSON\n",
			"  --limit LIMIT         cap the --index listing (0 = all)\n",
			"  --compare COMPARE     baseline JSON to prove disambiguation-debt dropped\n",
			// A long invocation puts its help on the next line at column 24.
			"  --workspace WORKSPACE\n                        workspace root (default: repo root)\n",
		} {
			if !strings.Contains(got, line) {
				t.Fatalf("%s: help lacks %q:\n%s", flag, line, got)
			}
		}
		flat := strings.Join(strings.Fields(got), " ")
		for _, o := range cliOptions {
			if o.help == "" || !strings.Contains(flat, o.help) {
				t.Fatalf("%s: option %s has no rendered description:\n%s", flag, o.name, got)
			}
		}
		for _, line := range strings.Split(strings.TrimRight(got, "\n"), "\n") {
			if strings.HasPrefix(line, "  ") && len(line) > cliHelpWidth {
				t.Fatalf("%s: option line exceeds %d columns: %q", flag, cliHelpWidth, line)
			}
		}
	}
}

func TestCLICompareRefusesNonObjectBaseline(t *testing.T) {
	dir := t.TempDir()
	for name, body := range map[string]string{
		"array":  "[1, 2]",
		"string": `"corpus"`,
		"number": "3",
		"null":   "null",
	} {
		t.Run(name, func(t *testing.T) {
			baseline := filepath.Join(dir, name+".json")
			if err := os.WriteFile(baseline, []byte(body), 0o600); err != nil {
				t.Fatal(err)
			}
			var out, errw bytes.Buffer
			code := Main([]string{"--workspace", dir, "--compare", baseline}, &out, &errw, dir)
			if code != 2 {
				t.Fatalf("code=%d want 2; stdout=%q stderr=%q", code, out.String(), errw.String())
			}
			if out.Len() != 0 {
				t.Fatalf("compared against a non-object baseline: %q", out.String())
			}
			if !strings.HasPrefix(errw.String(), "cannot read baseline "+baseline+": ") {
				t.Fatalf("stderr=%q", errw.String())
			}
		})
	}
}

func TestCLICompareAcceptsObjectBaseline(t *testing.T) {
	dir := t.TempDir()
	baseline := filepath.Join(dir, "base.json")
	if err := os.WriteFile(baseline, []byte(`{"corpus": {"disambiguation_debt": 0}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	var out, errw bytes.Buffer
	code := Main([]string{"--workspace", dir, "--compare", baseline}, &out, &errw, dir)
	if code == 2 || !strings.HasPrefix(out.String(), "disambiguation-debt: ") {
		t.Fatalf("code=%d stdout=%q stderr=%q", code, out.String(), errw.String())
	}
}
