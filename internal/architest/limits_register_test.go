package architest

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// Named meters must resolve to real checks, or the register rots into a list
// of comfortable names.
const limitsRegisterPath = "docs/LIMITS-REGISTER.json"

const limitsUnmetered = "UNMETERED"

type limitsRegister struct {
	Rows []limitsRow `json:"rows"`
}

type limitsRow struct {
	Limit    string   `json:"limit"`
	Value    string   `json:"value"`
	Owner    string   `json:"owner"`
	Breaks   string   `json:"breaks"`
	Meters   []string `json:"meters"`
	Budget   string   `json:"budget"`
	Evidence string   `json:"evidence"`
}

var limitsTestFuncRe = regexp.MustCompile(`(?m)^func (Test\w+)\(`)

func TestLimitsRegisterMetersExist(t *testing.T) {
	root := filepath.Join("..", "..")
	raw, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(limitsRegisterPath)))
	if err != nil {
		t.Fatalf("read %s: %v", limitsRegisterPath, err)
	}
	for _, problem := range limitsRegisterProblems(t, root, raw) {
		t.Error(problem)
	}
}

func TestLimitsRegisterProblems(t *testing.T) {
	const okMeters = `["test:TestPresent"]`
	rowWith := func(meters string) string {
		return fmt.Sprintf(`{"limit":"L","value":"V","owner":"O","breaks":"B","meters":%s,"budget":"B","evidence":"E"}`, meters)
	}
	cases := []struct {
		name string
		row  string
		want string
	}{
		{"valid test meter", rowWith(okMeters), ""},
		{"valid ci meter", rowWith(`["ci:.github/workflows/ci.yml#module zip builds (what go install compiles)"]`), ""},
		{"ci meter matches a line with trailing whitespace", rowWith(`["ci:.github/workflows/ci.yml#plain step"]`), ""},
		{"unmetered alone is accepted", rowWith(`["UNMETERED"]`), ""},
		{"missing test function", rowWith(`["test:TestDoesNotExistXyz"]`), "no Test function named TestDoesNotExistXyz"},
		{"ci name is only a prefix of a step", rowWith(`["ci:.github/workflows/ci.yml#module zip"]`), "has no step or job named module zip"},
		{"ci step is commented out", rowWith(`["ci:.github/workflows/ci.yml#commented step"]`), "has no step or job named commented step"},
		{"ci step absent", rowWith(`["ci:.github/workflows/ci.yml#absent step"]`), "has no step or job named absent step"},
		{"unmetered combined with a meter", rowWith(`["UNMETERED","test:TestPresent"]`), "UNMETERED cannot be combined"},
		{"empty meters list", rowWith(`[]`), "meters must name at least one"},
		{"empty column", `{"limit":"L","value":"","owner":"O","breaks":"B","meters":["test:TestPresent"],"budget":"B","evidence":"E"}`, "every column except meters must be non-empty"},
		{"unknown meter kind", rowWith(`["bogus:x"]`), "unknown meter kind"},
		{"meter without a kind separator", rowWith(`["TestPresent"]`), "meter must be"},
		{"ci meter without a step", rowWith(`["ci:.github/workflows/ci.yml"]`), "ci meter must be"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			root := writeLimitsFixture(t)
			problems := limitsRegisterProblems(t, root, []byte(fmt.Sprintf(`{"rows":[%s]}`, tc.row)))
			if tc.want == "" {
				if len(problems) != 0 {
					t.Fatalf("want no problems, got %q", problems)
				}
				return
			}
			if !strings.Contains(strings.Join(problems, "\n"), tc.want) {
				t.Fatalf("want a problem containing %q, got %q", tc.want, problems)
			}
		})
	}
}

// writeLimitsFixture builds a tiny tree with one Test function and one workflow
// whose step lines include a commented-out name and a trailing-space name.
func writeLimitsFixture(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	files := map[string]string{
		"pkg/present_test.go": "package pkg\n\nimport \"testing\"\n\nfunc TestPresent(t *testing.T) {}\n",
		".github/workflows/ci.yml": "jobs:\n  build:\n    steps:\n" +
			"      - name: module zip builds (what go install compiles)\n" +
			"      # - name: commented step\n" +
			"      - name: plain step  \n",
	}
	for rel, body := range files {
		path := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

// limitsRegisterProblems returns every violation in the register so the
// rejection paths can be exercised against temp trees.
func limitsRegisterProblems(t *testing.T, root string, raw []byte) []string {
	t.Helper()
	var reg limitsRegister
	if err := json.Unmarshal(raw, &reg); err != nil {
		return []string{fmt.Sprintf("parse %s: %v", limitsRegisterPath, err)}
	}
	if len(reg.Rows) == 0 {
		return []string{fmt.Sprintf("%s has no rows", limitsRegisterPath)}
	}
	testNames := limitsTestNames(t, root)

	var problems []string
	for i, row := range reg.Rows {
		if strings.TrimSpace(row.Limit) == "" || strings.TrimSpace(row.Value) == "" ||
			strings.TrimSpace(row.Owner) == "" || strings.TrimSpace(row.Breaks) == "" ||
			strings.TrimSpace(row.Budget) == "" || strings.TrimSpace(row.Evidence) == "" {
			problems = append(problems, fmt.Sprintf("row %d (%q): every column except meters must be non-empty", i, row.Limit))
		}
		if len(row.Meters) == 0 {
			problems = append(problems, fmt.Sprintf("row %d (%q): meters must name at least one meter or UNMETERED", i, row.Limit))
			continue
		}
		for _, m := range row.Meters {
			if m == limitsUnmetered {
				if len(row.Meters) != 1 {
					problems = append(problems, fmt.Sprintf("row %d (%q): UNMETERED cannot be combined with other meters", i, row.Limit))
				}
				continue
			}
			if err := checkLimitsMeter(root, testNames, m); err != nil {
				problems = append(problems, fmt.Sprintf("row %d (%q): meter %q: %v", i, row.Limit, m, err))
			}
		}
	}
	return problems
}

func checkLimitsMeter(root string, testNames map[string]bool, meter string) error {
	kind, target, ok := strings.Cut(meter, ":")
	if !ok {
		return fmt.Errorf("meter must be test:<TestName>, ci:<path>#<name>, or UNMETERED")
	}
	switch kind {
	case "test":
		if !testNames[target] {
			return fmt.Errorf("no Test function named %s in any *_test.go", target)
		}
	case "ci":
		file, step, ok := strings.Cut(target, "#")
		if !ok || step == "" {
			return fmt.Errorf("ci meter must be ci:<workflow path>#<step or job name>")
		}
		body, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(file)))
		if err != nil {
			return err
		}
		// Whole-line match: a prefix of a step name or a commented-out line must not count.
		nameLine := regexp.MustCompile(`(?m)^[ \t]*(- )?name: ` + regexp.QuoteMeta(step) + `[ \t\r]*$`)
		if !nameLine.Match(body) {
			return fmt.Errorf("workflow %s has no step or job named %s", file, step)
		}
	default:
		return fmt.Errorf("unknown meter kind %q", kind)
	}
	return nil
}

// limitsTestNames walks the tree for Test functions. It skips dot-directories
// (including linked worktrees) so one lane's state cannot satisfy another's
// meter.
func limitsTestNames(t *testing.T, root string) map[string]bool {
	t.Helper()
	names := map[string]bool{}
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if path != root && (strings.HasPrefix(d.Name(), ".") || d.Name() == "_scratch") {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(d.Name(), "_test.go") {
			return nil
		}
		body, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		for _, m := range limitsTestFuncRe.FindAllStringSubmatch(string(body), -1) {
			names[m[1]] = true
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk %s: %v", root, err)
	}
	return names
}
