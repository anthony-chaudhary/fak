package main

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/devcmd"
	"github.com/anthony-chaudhary/fak/internal/devindex"
)

func TestStudyOperationsDispatcherMatchesHandlers(t *testing.T) {
	type handler func(io.Writer, io.Writer, []string) int
	commands := map[string]handler{
		"study-monitor":       devcmd.RunStudyMonitor,
		"study-inventory":     devcmd.RunStudyInventory,
		"study-forge":         devcmd.RunStudyForge,
		"study-classify":      devcmd.RunStudyClassify,
		"study-link":          devcmd.RunStudyLink,
		"study-priority":      devcmd.RunStudyPriority,
		"study-tickets":       devcmd.RunStudyTickets,
		"study-adjacency":     devcmd.RunStudyAdjacency,
		"idea-scout":          devcmd.RunIdeaScout,
		"borrow-provenance":   devcmd.RunBorrowProvenance,
		"customization-index": devcmd.RunCustomizationIndex,
	}
	for name, direct := range commands {
		t.Run(name, func(t *testing.T) {
			args := []string{"--definitely-invalid"}
			var wantOut, wantErr bytes.Buffer
			wantCode := direct(&wantOut, &wantErr, args)
			var gotOut, gotErr bytes.Buffer
			gotCode := run(&gotOut, &gotErr, append([]string{name}, args...))
			if gotCode != wantCode || gotOut.String() != wantOut.String() || gotErr.String() != wantErr.String() {
				t.Fatalf("dispatcher parity mismatch\ncode: got %d want %d\nstdout: got %q want %q\nstderr: got %q want %q", gotCode, wantCode, gotOut.String(), wantOut.String(), gotErr.String(), wantErr.String())
			}
		})
	}
}

func TestRuntimeSourceExcludesStudyOperations(t *testing.T) {
	root := devindex.FindRoot(".")
	mainBody, err := os.ReadFile(filepath.Join(root, "cmd", "fak", "main.go"))
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{
		"study-monitor", "study-inventory", "study-forge", "study-classify",
		"study-link", "study-priority", "study-tickets", "study-adjacency",
		"idea-scout", "borrow-provenance", "customization-index",
	} {
		if strings.Contains(string(mainBody), `case "`+name+`"`) {
			t.Errorf("runtime dispatch still contains %s", name)
		}
	}
	for _, base := range []string{
		"study_monitor.go", "study_inventory.go", "study_forge.go", "study_classify.go",
		"study_link.go", "study_priority.go", "study_tickets.go", "study_adjacency.go",
		"study_import.go", "ideascout.go", "borrow_provenance.go", "customization_index.go",
	} {
		if _, err := os.Stat(filepath.Join(root, "cmd", "fak", base)); !os.IsNotExist(err) {
			t.Errorf("runtime source %s still exists (err=%v)", base, err)
		}
	}
	if !strings.Contains(string(mainBody), `case "study":`) {
		t.Fatal("product study command was removed with the dev-only study operations")
	}
}

func TestHelpIdentifiesIndependentDevelopmentArtifact(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := run(&out, &errOut, []string{"help"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	for _, want := range []string{"fak-dev — repository-development tooling", "index [docs] <query>", "index ownership", "wiki <structure|verify|fresh|score>", "separately buildable 'fak' artifact"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("help missing %q:\n%s", want, out.String())
		}
	}
}

func TestVersionEmitsMachineReadableBuildStamp(t *testing.T) {
	var out bytes.Buffer
	if code := run(&out, &bytes.Buffer{}, []string{"version"}); code != 0 {
		t.Fatalf("code=%d output=%s", code, out.String())
	}
	if !strings.Contains(out.String(), "\nbuild: ") {
		t.Fatalf("version output lacks self-update provenance line:\n%s", out.String())
	}
}

func TestOwnershipCommandUsesInventory(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := run(&out, &errOut, []string{"index", "ownership", "--json", "--root", devindex.FindRoot(".")}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var got devindex.OwnershipReport
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	found := false
	for _, command := range got.Commands {
		if command.Name == "index" && command.Owner == devindex.OwnerDev && command.DispatchTarget == "fak-dev" {
			found = true
		}
	}
	if !found {
		t.Fatal("ownership inventory does not authorize index on fak-dev")
	}
}

func TestIndexLaneExecutesThroughDevelopmentArtifact(t *testing.T) {
	var out, errOut bytes.Buffer
	root := devindex.FindRoot(".")
	if code := run(&out, &errOut, []string{"index", "lane", "cmd/fak/main.go", "--root", root, "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	if !strings.Contains(out.String(), `"lane"`) {
		t.Fatalf("index lane did not execute through fak-dev:\n%s", out.String())
	}
}

func TestWikiStructureExecutesThroughDevelopmentArtifact(t *testing.T) {
	root := t.TempDir()
	files := map[string]string{
		"dos.toml": `[lanes.trees]
gateway = ["internal/gateway/**"]
`,
		"README.md":                   "# fixture\n",
		"internal/gateway/gateway.go": "package gateway\n",
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

	var out, errOut bytes.Buffer
	if code := run(&out, &errOut, []string{"wiki", "structure", "--root", root, "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var got struct {
		Repo     string `json:"repo"`
		Sections []any  `json:"sections"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("decode output: %v\n%s", err, out.String())
	}
	if got.Repo == "" || len(got.Sections) == 0 {
		t.Fatalf("wiki structure did not execute: %+v", got)
	}
}

func TestCatchupExecutesThroughDevelopmentArtifact(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := run(&out, &errOut, []string{"catchup", "--workspace", t.TempDir(), "--no-index", "--intake-behind", "3", "--intake-total", "10", "--json"}); code != 1 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	if !strings.Contains(out.String(), `"catchup_backlog": 3`) {
		t.Fatalf("catchup did not execute through fak-dev:\n%s", out.String())
	}
}

func TestBackendScaffoldExecutesThroughDevelopmentArtifact(t *testing.T) {
	dir := t.TempDir()
	var out, errOut bytes.Buffer
	if code := run(&out, &errOut, []string{"backend", "scaffold", "artifacttest", "--lane", "custom", "--dir", dir}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	if _, err := os.Stat(filepath.Join(dir, "artifacttest_backend.go")); err != nil {
		t.Fatalf("fak-dev backend did not write scaffold: %v", err)
	}
}

func TestOrientExecutesThroughDevelopmentArtifact(t *testing.T) {
	var out, errOut bytes.Buffer
	root := devindex.FindRoot(".")
	if code := run(&out, &errOut, []string{"orient", "--root", root, "--leases=false", "--json", "--paths", "cmd/fak/main.go"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	if !strings.Contains(out.String(), `"lane"`) {
		t.Fatalf("orient did not execute through fak-dev:\n%s", out.String())
	}
}

func TestRunDispatchesWhatsChangedUsage(t *testing.T) {
	var out, errOut bytes.Buffer
	code := run(&out, &errOut, []string{"whats-changed", "--since", "HEAD"})
	if code != 2 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	if !strings.Contains(errOut.String(), "--paths is required") {
		t.Fatalf("stderr=%s", errOut.String())
	}
}
func TestRunDispatchesBoundaryUsage(t *testing.T) {
	var out, errOut bytes.Buffer
	code := run(&out, &errOut, []string{"boundary", "extra"})
	if code != 2 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
}

func TestRunDispatchesDevelopmentDiagnosticsUsage(t *testing.T) {
	for _, verb := range []string{"amd-gpu-facts", "commit-subject-coverage"} {
		t.Run(verb, func(t *testing.T) {
			var out, errOut bytes.Buffer
			code := run(&out, &errOut, []string{verb, "--help"})
			if code != 2 {
				t.Fatalf("code=%d stderr=%s", code, errOut.String())
			}
			if !strings.Contains(errOut.String(), "Usage of "+verb) {
				t.Fatalf("stderr=%s", errOut.String())
			}
		})
	}
}

func TestRunDispatchesBlastUsage(t *testing.T) {
	var out, errOut bytes.Buffer
	code := run(&out, &errOut, []string{"blast"})
	if code != 2 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	if !strings.Contains(errOut.String(), "expected a subcommand (estimate)") {
		t.Fatalf("stderr=%s", errOut.String())
	}
}

func TestRunDispatchesBuildcheckUsage(t *testing.T) {
	var out, errOut bytes.Buffer
	code := run(&out, &errOut, []string{"buildcheck", "--help"})
	if code != 2 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	if !strings.Contains(errOut.String(), "Usage of fak buildcheck:") {
		t.Fatalf("stderr=%s", errOut.String())
	}
}

func TestRunDispatchesBuildWithArgvAndExitPassthrough(t *testing.T) {
	var out, errOut bytes.Buffer
	code := run(&out, &errOut, []string{"build", "--profile", "not-a-profile"})
	if code != 2 {
		t.Fatalf("code=%d, want handler exit 2; stderr=%s", code, errOut.String())
	}
	if !strings.Contains(errOut.String(), `--profile must be release, dev, or race (got "not-a-profile")`) {
		t.Fatalf("stderr does not prove argv reached RunBuild: %s", errOut.String())
	}
}

func TestHelpAdvertisesTimedProductBuild(t *testing.T) {
	var out bytes.Buffer
	writeHelp(&out)
	for _, want := range []string{"build [--profile P]", "record phase timings"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("help missing %q:\n%s", want, out.String())
		}
	}
}

func TestRunDispatchesCIPreflightUsage(t *testing.T) {
	var out, errOut bytes.Buffer
	code := run(&out, &errOut, []string{"ci-preflight", "--repo", t.TempDir(), "--ref", "missing", "--json"})
	if code != 2 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
}

func TestRuntimeSourceDoesNotDispatchDevelopmentDiagnostics(t *testing.T) {
	src, err := os.ReadFile(filepath.Join("..", "fak", "main.go"))
	if err != nil {
		t.Fatal(err)
	}
	for _, verb := range []string{"amd-gpu-facts", "commit-subject-coverage"} {
		if bytes.Contains(src, []byte(`case "`+verb+`":`)) {
			t.Fatalf("runtime fak still dispatches dev-owned %s", verb)
		}
	}
}

func TestRuntimeSourceDoesNotDispatchBlast(t *testing.T) {
	src, err := os.ReadFile(filepath.Join("..", "fak", "main.go"))
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(src, []byte(`case "blast":`)) {
		t.Fatal("runtime fak still dispatches dev-owned blast")
	}
}

func TestRuntimeSourceDoesNotDispatchBuildcheck(t *testing.T) {
	src, err := os.ReadFile(filepath.Join("..", "fak", "main.go"))
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(src, []byte(`case "buildcheck":`)) {
		t.Fatal("runtime fak still dispatches dev-owned buildcheck")
	}
}

func TestRuntimeSourceDoesNotDispatchCIPreflight(t *testing.T) {
	src, err := os.ReadFile(filepath.Join("..", "fak", "main.go"))
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(src, []byte(`case "ci-preflight":`)) {
		t.Fatal("runtime fak still dispatches dev-owned ci-preflight")
	}
}
func TestRuntimeSourceDoesNotDispatchBoundary(t *testing.T) {
	src, err := os.ReadFile(filepath.Join("..", "fak", "main.go"))
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(src, []byte(`case "boundary":`)) {
		t.Fatal("runtime fak still dispatches dev-owned boundary")
	}
}
func TestRuntimeSourceDoesNotDispatchWhatsChanged(t *testing.T) {
	src, err := os.ReadFile(filepath.Join("..", "fak", "main.go"))
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(src, []byte(`case "whats-changed":`)) {
		t.Fatal("runtime fak still dispatches dev-owned whats-changed")
	}
}
func TestRuntimeSourceDoesNotDispatchBackend(t *testing.T) {
	mainPath := filepath.Join(devindex.FindRoot("."), "cmd", "fak", "main.go")
	body, err := os.ReadFile(mainPath)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(body), `case "backend":`) || strings.Contains(string(body), "cmdBackend(") {
		t.Fatal("runtime fak still dispatches the dev-only backend command")
	}
}

func TestRunDispatchesWorkflowAuditUsage(t *testing.T) {
	var out, errOut bytes.Buffer
	code := run(&out, &errOut, []string{"workflow-audit", "--help"})
	if code != 2 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	if !strings.Contains(errOut.String(), "Usage of fak workflow-audit") {
		t.Fatalf("stderr=%s", errOut.String())
	}
}

func TestRuntimeSourceDoesNotDispatchWorkflowAudit(t *testing.T) {
	mainPath := filepath.Join(devindex.FindRoot("."), "cmd", "fak", "main.go")
	body, err := os.ReadFile(mainPath)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(body), `case "workflow-audit":`) || strings.Contains(string(body), "cmdWorkflowAudit(") {
		t.Fatal("runtime fak still dispatches the dev-only workflow-audit command")
	}
}

func TestRuntimeSourceDoesNotDispatchToolCoverageAudit(t *testing.T) {
	mainPath := filepath.Join(devindex.FindRoot("."), "cmd", "fak", "main.go")
	body, err := os.ReadFile(mainPath)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(body), `case "tool-coverage-audit":`) || strings.Contains(string(body), "cmdToolCoverageAudit(") {
		t.Fatal("runtime fak still dispatches the dev-only tool-coverage-audit command")
	}
}

func TestRunDispatchesRefactorVerify(t *testing.T) {
	var out, errOut bytes.Buffer
	code := run(&out, &errOut, []string{"refactor-verify", "--ref", "definitely-not-a-commit"})
	if code != 2 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	if !strings.Contains(errOut.String(), "refactor-verify: not a commit") {
		t.Fatalf("stderr=%s", errOut.String())
	}
}

func TestRuntimeSourceDoesNotDispatchRefactorVerify(t *testing.T) {
	mainPath := filepath.Join(devindex.FindRoot("."), "cmd", "fak", "main.go")
	body, err := os.ReadFile(mainPath)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(body), `case "refactor-verify":`) || strings.Contains(string(body), "cmdRefactorVerify(") {
		t.Fatal("runtime fak still dispatches the dev-only refactor-verify command")
	}
}

func TestRunDispatchesCodexHookProfileUsage(t *testing.T) {
	var out, errOut bytes.Buffer
	code := run(&out, &errOut, []string{"codex-hook-profile", "--help"})
	if code != 2 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	if !strings.Contains(errOut.String(), "usage: fak-dev codex-hook-profile") {
		t.Fatalf("stderr=%s", errOut.String())
	}
}

func TestHelpListsCodexHookProfile(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := run(&out, &errOut, []string{"help"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	if !strings.Contains(out.String(), "codex-hook-profile [flags]") {
		t.Fatalf("help=%s", out.String())
	}
}

func TestRunDispatchesSessionDiagUsage(t *testing.T) {
	var out, errOut bytes.Buffer
	code := run(&out, &errOut, []string{"sessiondiag", "--help"})
	if code != 2 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	if !strings.Contains(errOut.String(), "Usage of sessiondiag") {
		t.Fatalf("stderr=%s", errOut.String())
	}
}

func TestRuntimeSourceDoesNotDispatchSessionDiag(t *testing.T) {
	mainPath := filepath.Join(devindex.FindRoot("."), "cmd", "fak", "main.go")
	body, err := os.ReadFile(mainPath)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(body), `case "sessiondiag":`) || strings.Contains(string(body), "cmdSessionDiag(") {
		t.Fatal("runtime fak still dispatches the dev-only sessiondiag command")
	}
}

func TestRunDispatchesCodexMemoryUsage(t *testing.T) {
	var out, errOut bytes.Buffer
	code := run(&out, &errOut, []string{"codex-memory", "--help"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	if !strings.Contains(errOut.String(), "usage: fak codex-memory doctor") {
		t.Fatalf("stderr=%s", errOut.String())
	}
}

func TestRuntimeSourceDoesNotDispatchCodexMemory(t *testing.T) {
	mainPath := filepath.Join(devindex.FindRoot("."), "cmd", "fak", "main.go")
	body, err := os.ReadFile(mainPath)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(body), `case "codex-memory":`) || strings.Contains(string(body), "cmdCodexMemory(") {
		t.Fatal("runtime fak still dispatches the dev-only codex-memory command")
	}
}

func TestRunDispatchesPlanAuditUsage(t *testing.T) {
	var out, errOut bytes.Buffer
	code := run(&out, &errOut, []string{"plan-audit", "--help"})
	if code != 2 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	if !strings.Contains(errOut.String(), "Usage of plan-audit") {
		t.Fatalf("stderr=%s", errOut.String())
	}
}

func TestRuntimeSourceDoesNotDispatchPlanAudit(t *testing.T) {
	mainPath := filepath.Join(devindex.FindRoot("."), "cmd", "fak", "main.go")
	body, err := os.ReadFile(mainPath)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(body), `case "plan-audit":`) || strings.Contains(string(body), "cmdPlanAudit(") {
		t.Fatal("runtime fak still dispatches the dev-only plan-audit command")
	}
}

func TestRunDispatchesReadmeVisualAuditUsage(t *testing.T) {
	var out, errOut bytes.Buffer
	code := run(&out, &errOut, []string{"readme-visual-audit", "--help"})
	if code != 2 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	if !strings.Contains(errOut.String(), "Usage of readme-visual-audit") {
		t.Fatalf("stderr=%s", errOut.String())
	}
}

func TestRuntimeSourceDoesNotDispatchReadmeVisualAudit(t *testing.T) {
	mainPath := filepath.Join(devindex.FindRoot("."), "cmd", "fak", "main.go")
	body, err := os.ReadFile(mainPath)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(body), `case "readme-visual-audit":`) || strings.Contains(string(body), "cmdReadmeVisualAudit(") {
		t.Fatal("runtime fak still dispatches the dev-only readme-visual-audit command")
	}
}

// TestRunDispatchesFleetcapPlanner proves the capacity planner answers from the
// development artifact now that the runtime arm is gone (#6022 DoD row 4).
func TestRunDispatchesFleetcapPlanner(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := run(&out, &errOut, []string{"fleetcap", "--rate", "400", "--session", "10", "--available", "40"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	if !strings.Contains(out.String(), "UNDER_CAPACITY") {
		t.Fatalf("stdout=%s", out.String())
	}
}

func TestRuntimeSourceDoesNotDispatchFleetcap(t *testing.T) {
	mainPath := filepath.Join(devindex.FindRoot("."), "cmd", "fak", "main.go")
	body, err := os.ReadFile(mainPath)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(body), `case "fleetcap":`) || strings.Contains(string(body), "cmdFleetcap(") {
		t.Fatal("runtime fak still dispatches the dev-only fleetcap command")
	}
}

func TestRuntimeSourceDoesNotDispatchOrient(t *testing.T) {
	mainPath := filepath.Join(devindex.FindRoot("."), "cmd", "fak", "main.go")
	body, err := os.ReadFile(mainPath)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(body), `case "orient":`) || strings.Contains(string(body), "cmdOrient(") {
		t.Fatal("runtime fak still dispatches the dev-only orient command")
	}
}

func TestRuntimeSourceDoesNotDispatchIndex(t *testing.T) {
	mainPath := filepath.Join(devindex.FindRoot("."), "cmd", "fak", "main.go")
	body, err := os.ReadFile(mainPath)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(body), `case "index"`) || strings.Contains(string(body), `case "index", "devindex"`) || strings.Contains(string(body), "cmdIndex(") {
		t.Fatal("runtime fak still dispatches the dev-only index command")
	}
}

func TestRuntimeSourceDoesNotDispatchWiki(t *testing.T) {
	mainPath := filepath.Join(devindex.FindRoot("."), "cmd", "fak", "main.go")
	body, err := os.ReadFile(mainPath)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(body), `case "wiki":`) || strings.Contains(string(body), "cmdWiki(") {
		t.Fatal("runtime fak still dispatches the dev-only wiki command")
	}
}
