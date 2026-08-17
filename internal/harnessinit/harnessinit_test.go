package harnessinit

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestInitCreatesIdempotentPublicProductAndPreservesUserFiles(t *testing.T) {
	root := t.TempDir()
	result, err := Init(Options{Dir: root, Module: "example.test/acme"})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Created) != 9 {
		t.Fatalf("created=%v", result.Created)
	}
	mainBody, err := os.ReadFile(filepath.Join(root, "generated", "runtime.go"))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(mainBody), "github.com/anthony-chaudhary/fak/internal/") {
		t.Fatal("generated product imports internal package")
	}
	userPath := filepath.Join(root, "product", "config.go")
	if err := os.WriteFile(userPath, []byte("package product\n// user edit\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	second, err := Init(Options{Dir: root, Module: "example.test/acme"})
	if err != nil {
		t.Fatal(err)
	}
	got, _ := os.ReadFile(userPath)
	if !strings.Contains(string(got), "user edit") {
		t.Fatalf("user file overwritten: %s", got)
	}
	if len(second.Created) != 0 || len(second.Updated) != 0 {
		t.Fatalf("rerun changed generated files: %+v", second)
	}
}

func TestGeneratedLaunchIsProgressOnlyAndDashboardIsAgentScoped(t *testing.T) {
	root := t.TempDir()
	if _, err := Init(Options{Dir: root, Module: "example.test/product"}); err != nil {
		t.Fatal(err)
	}
	launch := exec.Command("go", "run", "./cmd/product", "--launch", "--agent-id", "agent 42")
	launch.Dir = root
	got, err := launch.CombinedOutput()
	if err != nil {
		t.Fatalf("launch: %v\n%s", err, got)
	}
	text := string(got)
	for _, forbidden := range []string{"turn.started", "model.response", "tool.requested", "tool.completed", "turn.completed", "harness.locked"} {
		if strings.Contains(text, forbidden) {
			t.Fatalf("clean launch leaked %q:\n%s", forbidden, text)
		}
	}
	if !strings.Contains(text, "[--------------------] Launching agent") || !strings.Contains(text, "[####################] Launching agent") {
		t.Fatalf("captured progress render=%q", text)
	}
	link := exec.Command("go", "run", "./cmd/product", "--dashboard-link", "--agent-id", "agent 42")
	link.Dir = root
	got, err = link.CombinedOutput()
	if err != nil {
		t.Fatalf("dashboard link: %v\n%s", err, got)
	}
	want := "http://localhost:3000/d/fak-fleet-session/fak-fleet-session-drill-down?var-session=agent+42"
	if strings.TrimSpace(string(got)) != want {
		t.Fatalf("dashboard link=%q want=%q", strings.TrimSpace(string(got)), want)
	}
}

func TestInitPreservesUserLaunchCustomization(t *testing.T) {
	root := t.TempDir()
	opts := Options{Dir: root, Module: "example.test/product"}
	if _, err := Init(opts); err != nil {
		t.Fatal(err)
	}
	launchPath := filepath.Join(root, "product", "launch.go")
	custom := []byte("package product\n\n// my launch UI\n")
	if err := os.WriteFile(launchPath, custom, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Init(opts); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(launchPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(custom) {
		t.Fatalf("user launch customization overwritten:\n%s", got)
	}
}

func TestInitRefusesUnownedGeneratedPath(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "generated", "runtime.go")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("package generated\n// mine\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := Init(Options{Dir: root, Module: "example.test/acme"})
	if err == nil || !strings.Contains(err.Error(), "refusing to overwrite user-owned") {
		t.Fatalf("err=%v", err)
	}
	body, _ := os.ReadFile(path)
	if !strings.Contains(string(body), "mine") {
		t.Fatal("foreign file changed")
	}
}

func TestInitPublishesProvenanceAndOwnershipMetadata(t *testing.T) {
	root := t.TempDir()
	if _, err := Init(Options{Dir: root, Module: "example.test/acme"}); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{"go.mod", "go.sum", "cmd/product/main.go", "generated/runtime.go", "harness.lock.json"} {
		body, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(path)))
		if err != nil {
			t.Fatal(err)
		}
		if path != "go.sum" && !strings.Contains(string(body), generatorID) {
			t.Fatalf("%s lacks generator provenance", path)
		}
	}
	lock, err := os.ReadFile(filepath.Join(root, "harness.lock.json"))
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{ContractVersion, DefaultFAKVersion, `"README.md": "user"`, `"go.sum": "generated"`, `"build":`, `"run":`, `"upgrade":`} {
		if !strings.Contains(string(lock), want) {
			t.Fatalf("manifest lacks %q: %s", want, lock)
		}
	}
}

func TestInitRefusesUnrecognizedGoSum(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "go.sum")
	if err := os.WriteFile(path, []byte("example.test/user v1.0.0 h1:user-owned\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := Init(Options{Dir: root, Module: "example.test/acme"})
	if err == nil || !strings.Contains(err.Error(), "refusing to overwrite") {
		t.Fatalf("err=%v", err)
	}
	body, _ := os.ReadFile(path)
	if string(body) != "example.test/user v1.0.0 h1:user-owned\n" {
		t.Fatal("foreign go.sum changed")
	}
}

func TestInitUpgradesOwnedGeneratedGoSum(t *testing.T) {
	root := t.TempDir()
	if _, err := Init(Options{Dir: root, Module: "example.test/acme"}); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(root, "go.sum")
	if err := os.WriteFile(path, []byte("old generated sums\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	result, err := Init(Options{Dir: root, Module: "example.test/acme"})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Updated) != 1 || result.Updated[0] != "go.sum" {
		t.Fatalf("updated=%v", result.Updated)
	}
	body, _ := os.ReadFile(path)
	if !strings.Contains(string(body), DefaultFAKVersion) {
		t.Fatalf("go.sum not restored: %s", body)
	}
}

func TestInitUserConfigSupportsTaskCardCustomization(t *testing.T) {
	root := t.TempDir()
	if _, err := Init(Options{Dir: root, Module: "example.test/acme"}); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(root, "product", "config.go")
	custom := `package product

type Config struct {
 ID string
 Version string
 Profile string
 SystemPrompt string
 Task string
}

func DefaultConfig() Config {
 return Config{
  ID: "local-support-harness",
  Version: "0.1.0",
  Profile: "support-summary",
  SystemPrompt: "Answer from admitted context.",
  Task: "Summarize the support request.",
 }
}

func OfflineReply(prompt string) string { return "admitted support summary: " + prompt }
`
	if err := os.WriteFile(path, []byte(custom), 0o644); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Init(Options{Dir: root, Module: "example.test/acme"}); err != nil {
		t.Fatal(err)
	}
	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(before) != string(after) {
		t.Fatal("generator rerun changed task-card customization")
	}
	for _, generated := range []string{"generated/runtime.go", "README.md"} {
		body, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(generated)))
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(body), "DefaultConfig") {
			t.Fatalf("%s does not use or document stable config seam", generated)
		}
	}
}
