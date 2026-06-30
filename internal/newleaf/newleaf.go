package newleaf

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

const (
	ModulePrefix = "github.com/anthony-chaudhary/fak/internal"
	TierMarker   = "// new-leaf:tier"
	LaneMarker   = "# new-leaf:lane"
	TreeMarker   = "# new-leaf:tree"
)

var (
	NameRE = regexp.MustCompile(`^[a-z][a-z0-9]*$`)
	Tiers  = map[string]int{"root": 0, "foundation": 1, "mechanism": 2, "composer": 3, "integrator": 4}
)

type Options struct {
	Root     string
	Name     string
	Tier     string
	Register bool
	Summary  string
	DryRun   bool
}

type Report struct {
	Name      string   `json:"name"`
	Tier      string   `json:"tier"`
	Register  bool     `json:"register"`
	DryRun    bool     `json:"dry_run"`
	Edits     []string `json:"edits"`
	NextSteps []string `json:"next_steps"`
}

func DocGo(name, tier string, n int, summary string) string {
	return fmt.Sprintf(
		"// Package %s is %s.\n//\n// Tier: %s (%d) - see internal/architest. This package may import only\n// packages whose tier is <= %d; an upward import fails the architest gate.\n// See AGENTS.md and internal/architest for the layering contract.\npackage %s\n",
		name, summary, tier, n, n, name,
	)
}

func ImplGo(name string, register bool) string {
	head := "package " + name + "\n"
	if register {
		head += "\nimport \"" + ModulePrefix + "/abi\"\n"
	}
	body := "\n// Ready reports that the leaf is wired. Replace this placeholder with the\n" +
		"// real surface this package exists to provide.\n" +
		"func Ready() bool { return true }\n"
	if register {
		body += "\n// init registers this leaf's driver against the frozen ABI before the\n" +
			"// kernel boots. Replace this placeholder with the real abi.Register* call.\n" +
			"func init() {\n\t_ = abi.ABIMinor\n}\n"
	}
	return head + body
}

func TestGo(name string) string {
	return "package " + name + "\n\n" +
		"import \"testing\"\n\n" +
		"func TestReady(t *testing.T) {\n" +
		"\tif !Ready() {\n" +
		"\t\tt.Fatal(\"Ready() should report true for the generated skeleton\")\n" +
		"\t}\n" +
		"}\n"
}

func InsertBeforeMarker(text, marker, line string) (string, error) {
	var out strings.Builder
	done := false
	for _, ln := range splitKeepLines(text) {
		if !done && strings.Contains(ln, marker) {
			out.WriteString(line)
			done = true
		}
		out.WriteString(ln)
	}
	if !done {
		return "", fmt.Errorf("marker %q not found", marker)
	}
	return out.String(), nil
}

func InsertBeforeAllMarkers(text, marker, line string) (string, error) {
	var out strings.Builder
	hits := 0
	for _, ln := range splitKeepLines(text) {
		if strings.Contains(ln, marker) {
			out.WriteString(line)
			hits++
		}
		out.WriteString(ln)
	}
	if hits == 0 {
		return "", fmt.Errorf("marker %q not found", marker)
	}
	return out.String(), nil
}

func AddRegistration(text, name string) (string, error) {
	imp := "\t_ \"" + ModulePrefix + "/" + name + "\"\n"
	if strings.Contains(text, imp) {
		return text, nil
	}
	idx := strings.LastIndex(text, "\n)")
	if idx < 0 {
		return "", fmt.Errorf("could not find import block close")
	}
	return text[:idx+1] + imp + text[idx+1:], nil
}

func AddLeafLane(text, name string) (string, error) {
	if strings.Contains(text, "[\"internal/"+name+"/**\"]") || strings.Contains(text, "fak/internal/"+name+"/**") {
		return text, nil
	}
	var err error
	text, err = InsertBeforeAllMarkers(text, LaneMarker, "  \""+name+"\",\n")
	if err != nil {
		return "", err
	}
	text, err = InsertBeforeMarker(text, TreeMarker, name+" = [\"internal/"+name+"/**\"]\n")
	if err != nil {
		return "", err
	}
	return text, nil
}

func Apply(opts Options) (Report, error) {
	name := opts.Name
	tier := opts.Tier
	n, ok := Tiers[tier]
	if !ok {
		return Report{}, fmt.Errorf("unknown tier %q", tier)
	}
	if !NameRE.MatchString(name) {
		return Report{}, fmt.Errorf("%q is not a valid lowercase Go package name", name)
	}
	if tier == "root" {
		return Report{}, fmt.Errorf("'root' is reserved for internal/abi; pick foundation or higher")
	}
	root := opts.Root
	if root == "" {
		var err error
		root, err = os.Getwd()
		if err != nil {
			return Report{}, err
		}
	}
	summary := opts.Summary
	if summary == "" {
		summary = "a tier-" + tier + " leaf (describe its single responsibility)"
	}

	leafDir := filepath.Join(root, "internal", name)
	if _, err := os.Stat(leafDir); err == nil {
		return Report{}, fmt.Errorf("%s already exists - refusing to overwrite", leafDir)
	}
	archPath := filepath.Join(root, "internal", "architest", "architest_test.go")
	regPath := filepath.Join(root, "internal", "registrations", "registrations.go")
	dosPath := filepath.Join(root, "dos.toml")

	archText, err := os.ReadFile(archPath)
	if err != nil {
		return Report{}, err
	}
	if strings.Contains(string(archText), "\""+name+"\":") {
		return Report{}, fmt.Errorf("tier table already declares %q", name)
	}
	newArch, err := InsertBeforeMarker(string(archText), TierMarker, fmt.Sprintf("\t%q: %d,\n", name, n))
	if err != nil {
		return Report{}, err
	}

	var newReg string
	if opts.Register {
		regText, err := os.ReadFile(regPath)
		if err != nil {
			return Report{}, err
		}
		newReg, err = AddRegistration(string(regText), name)
		if err != nil {
			return Report{}, err
		}
	}

	var newDos string
	if raw, err := os.ReadFile(dosPath); err == nil {
		newDos, err = AddLeafLane(string(raw), name)
		if err != nil {
			return Report{}, err
		}
	}

	files := map[string]string{
		filepath.Join(leafDir, "doc.go"):        DocGo(name, tier, n, summary),
		filepath.Join(leafDir, name+".go"):      ImplGo(name, opts.Register),
		filepath.Join(leafDir, name+"_test.go"): TestGo(name),
	}
	if !opts.DryRun {
		if err := os.MkdirAll(leafDir, 0o755); err != nil {
			return Report{}, err
		}
		for path, content := range files {
			if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
				return Report{}, err
			}
		}
		if err := os.WriteFile(archPath, []byte(newArch), 0o644); err != nil {
			return Report{}, err
		}
		if opts.Register {
			if err := os.WriteFile(regPath, []byte(newReg), 0o644); err != nil {
				return Report{}, err
			}
		}
		if newDos != "" {
			if err := os.WriteFile(dosPath, []byte(newDos), 0o644); err != nil {
				return Report{}, err
			}
		}
	}

	report := Report{Name: name, Tier: tier, Register: opts.Register, DryRun: opts.DryRun}
	for _, p := range []string{
		filepath.Join("internal", name, "doc.go"),
		filepath.Join("internal", name, name+".go"),
		filepath.Join("internal", name, name+"_test.go"),
	} {
		report.Edits = append(report.Edits, filepath.ToSlash(p))
	}
	report.Edits = append(report.Edits, "internal/architest/architest_test.go (tier table)")
	if opts.Register {
		report.Edits = append(report.Edits, "internal/registrations/registrations.go (defconfig)")
	}
	if newDos != "" {
		report.Edits = append(report.Edits, "dos.toml (concurrency lane)")
	}
	report.NextSteps = []string{
		"implement the leaf in internal/" + name + "/" + name + ".go",
		"go test ./internal/" + name + " ./internal/architest",
		"the architest gate now enforces this leaf's tier on every CI run",
	}
	return report, nil
}

func (r Report) JSON() ([]byte, error) {
	return json.MarshalIndent(r, "", "  ")
}

func splitKeepLines(s string) []string {
	if s == "" {
		return nil
	}
	parts := strings.SplitAfter(s, "\n")
	if parts[len(parts)-1] == "" {
		parts = parts[:len(parts)-1]
	}
	return parts
}
