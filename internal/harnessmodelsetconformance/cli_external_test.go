package harnessmodelsetconformance_test

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelinventory"
	"github.com/anthony-chaudhary/fak/internal/modelsetreceipt"
	"github.com/anthony-chaudhary/fak/internal/modelsetresolve"
)

func TestPublicCLICleanDirectoryConformance(t *testing.T) {
	repoRoot := findRepoRoot(t)
	binary := filepath.Join(t.TempDir(), "fak-model-set-conformance")
	if runtime.GOOS == "windows" {
		binary += ".exe"
	}
	build := exec.Command("go", "build", "-o", binary, "./cmd/fak")
	build.Dir = repoRoot
	if output, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build public fak CLI: %v\n%s", err, output)
	}

	cleanDir := t.TempDir()
	copyFixture(t, filepath.Join("testdata", "harness.model-set.json"), filepath.Join(cleanDir, "harness.model-set.json"))
	copyFixture(t, filepath.Join("testdata", "model-inventory.json"), filepath.Join(cleanDir, "model-inventory.json"))
	resolveArgs := []string{
		"harness", "model-set", "resolve",
		"--intent", "harness.model-set.json", "--inventory", "model-inventory.json",
		"--out", "harness.model-set.lock.json", "--as-of", conformanceTime.Format("2006-01-02T15:04:05Z07:00"),
		"--os", "linux", "--arch", "amd64", "--accelerator", "cpu", "--runtime", "mixed-runtime", "--json",
	}
	firstResolve := runPublicCLI(t, binary, cleanDir, resolveArgs...)
	if firstResolve.code != 0 || firstResolve.stderr != "" {
		t.Fatalf("resolve exit=%d\nstdout=%s\nstderr=%s", firstResolve.code, firstResolve.stdout, firstResolve.stderr)
	}
	assertGolden(t, "cli-resolve.stdout.json", []byte(firstResolve.stdout))
	firstLock := readFile(t, filepath.Join(cleanDir, "harness.model-set.lock.json"))
	if want := readFile(t, filepath.Join("testdata", "harness.model-set.lock.json")); !bytes.Equal(firstLock, want) {
		t.Fatal("public CLI lock differs from package-level captured lock")
	}
	if got := readFile(t, filepath.Join(cleanDir, "harness.model-set.expectation.json")); !bytes.Equal(got, readFile(t, filepath.Join("testdata", "harness.model-set.expectation.json"))) {
		t.Fatal("public CLI expectation differs from package-level captured expectation")
	}

	secondResolve := runPublicCLI(t, binary, cleanDir, resolveArgs...)
	if secondResolve.code != 0 || secondResolve.stdout != firstResolve.stdout || secondResolve.stderr != "" {
		t.Fatalf("deterministic resolve changed: first=%+v second=%+v", firstResolve, secondResolve)
	}
	if got := readFile(t, filepath.Join(cleanDir, "harness.model-set.lock.json")); !bytes.Equal(got, firstLock) {
		t.Fatal("second public CLI resolve changed canonical lock bytes")
	}

	selfcheck := runPublicCLI(t, binary, cleanDir,
		"harness", "model-set", "selfcheck",
		"--lock", "harness.model-set.lock.json", "--inventory", "model-inventory.json",
		"--receipt", "harness.model-set.receipt.json", "--as-of", conformanceTime.Format("2006-01-02T15:04:05Z07:00"), "--json",
	)
	if selfcheck.code != 0 || selfcheck.stderr != "" {
		t.Fatalf("selfcheck exit=%d\nstdout=%s\nstderr=%s", selfcheck.code, selfcheck.stdout, selfcheck.stderr)
	}
	assertGolden(t, "cli-selfcheck.stdout.json", []byte(selfcheck.stdout))
	receiptRaw := readFile(t, filepath.Join(cleanDir, "harness.model-set.receipt.json"))
	if !bytes.Equal(receiptRaw, readFile(t, filepath.Join("testdata", "harness.model-set.receipt.json"))) {
		t.Fatal("public CLI receipt differs from package-level captured receipt")
	}
	if receipt, err := modelsetreceipt.ParseJSON(receiptRaw); err != nil || receipt.Status != modelsetreceipt.StatusCompatible {
		t.Fatalf("independent CLI receipt read-back = (%+v, %v)", receipt, err)
	}

	t.Run("exact model mismatch", func(t *testing.T) {
		inventory, _ := normalize(t, successObservations())
		candidateByID(t, &inventory, "executor-exact").Identity.Digest = digest("executor-exact-repacked")
		writeInventory(t, filepath.Join(cleanDir, "exact-mismatch.json"), inventory)
		result := runPublicCLI(t, binary, cleanDir,
			"harness", "model-set", "selfcheck",
			"--lock", "harness.model-set.lock.json", "--inventory", "exact-mismatch.json",
			"--receipt", "exact-mismatch.receipt.json", "--as-of", conformanceTime.Format("2006-01-02T15:04:05Z07:00"), "--json",
		)
		if result.code != 3 {
			t.Fatalf("exact mismatch exit=%d\nstdout=%s\nstderr=%s", result.code, result.stdout, result.stderr)
		}
		receipt, err := modelsetreceipt.ParseJSON(readFile(t, filepath.Join(cleanDir, "exact-mismatch.receipt.json")))
		if err != nil {
			t.Fatalf("exact mismatch receipt read-back: %v", err)
		}
		assertReceiptFailure(t, receipt.Failures, modelsetreceipt.CodeIdentityMismatch, "", "executor", false)
		if got := readFile(t, filepath.Join(cleanDir, "harness.model-set.lock.json")); !bytes.Equal(got, firstLock) {
			t.Fatal("exact-model mismatch mutated prior lock")
		}
	})

	t.Run("incompatible alternatives", func(t *testing.T) {
		inventory, _ := normalize(t, incompatibleObservations())
		writeInventory(t, filepath.Join(cleanDir, "incompatible.json"), inventory)
		failedResolveArgs := append([]string(nil), resolveArgs...)
		for index := range failedResolveArgs {
			if failedResolveArgs[index] == "model-inventory.json" {
				failedResolveArgs[index] = "incompatible.json"
			}
		}
		failedResolve := runPublicCLI(t, binary, cleanDir, failedResolveArgs...)
		if failedResolve.code != 3 {
			t.Fatalf("incompatible resolve exit=%d\nstdout=%s\nstderr=%s", failedResolve.code, failedResolve.stdout, failedResolve.stderr)
		}
		var resolveResult struct {
			Status     string                     `json:"status"`
			Resolution modelsetresolve.Resolution `json:"resolution"`
		}
		if err := json.Unmarshal([]byte(failedResolve.stdout), &resolveResult); err != nil || resolveResult.Status != "incompatible" {
			t.Fatalf("decode incompatible resolve: status=%q err=%v\n%s", resolveResult.Status, err, failedResolve.stdout)
		}
		for _, code := range []modelsetresolve.RejectionCode{
			modelsetresolve.CodeServingProtocol,
			modelsetresolve.CodeMemory,
			modelsetresolve.CodeEvidenceStale,
		} {
			assertRejection(t, resolveResult.Resolution.Rejections(), code, code == modelsetresolve.CodeEvidenceStale)
		}
		if got := readFile(t, filepath.Join(cleanDir, "harness.model-set.lock.json")); !bytes.Equal(got, firstLock) {
			t.Fatal("failed public CLI resolve replaced prior lock")
		}

		failedSelfcheck := runPublicCLI(t, binary, cleanDir,
			"harness", "model-set", "selfcheck",
			"--lock", "harness.model-set.lock.json", "--inventory", "incompatible.json",
			"--receipt", "incompatible.receipt.json", "--as-of", conformanceTime.Format("2006-01-02T15:04:05Z07:00"), "--json",
		)
		if failedSelfcheck.code != 3 {
			t.Fatalf("incompatible selfcheck exit=%d\nstdout=%s\nstderr=%s", failedSelfcheck.code, failedSelfcheck.stdout, failedSelfcheck.stderr)
		}
		receipt, err := modelsetreceipt.ParseJSON(readFile(t, filepath.Join(cleanDir, "incompatible.receipt.json")))
		if err != nil {
			t.Fatalf("incompatible receipt read-back: %v", err)
		}
		assertReceiptFailure(t, receipt.Failures, modelsetreceipt.CodeRuntimeMismatch, string(modelsetresolve.CodeServingProtocol), "executor", false)
		assertReceiptFailure(t, receipt.Failures, modelsetreceipt.CodeRequiredFactMismatch, string(modelsetresolve.CodeMemory), "executor", false)
		assertReceiptFailure(t, receipt.Failures, modelsetreceipt.CodeEvidenceStale, string(modelsetresolve.CodeEvidenceStale), "executor", true)
		for _, want := range []string{"role=executor", "field=", "evidence=", "remediation="} {
			if !strings.Contains(failedSelfcheck.stderr, want) {
				t.Fatalf("CLI diagnostics omit %q:\n%s", want, failedSelfcheck.stderr)
			}
		}
		if got := readFile(t, filepath.Join(cleanDir, "harness.model-set.lock.json")); !bytes.Equal(got, firstLock) {
			t.Fatal("failed public CLI selfcheck mutated prior lock")
		}
	})
}

type publicCLIResult struct {
	code           int
	stdout, stderr string
}

func runPublicCLI(t *testing.T, binary, dir string, args ...string) publicCLIResult {
	t.Helper()
	command := exec.Command(binary, args...)
	command.Dir = dir
	command.Env = append(os.Environ(), "HTTP_PROXY=http://127.0.0.1:1", "HTTPS_PROXY=http://127.0.0.1:1", "NO_PROXY=")
	var stdout, stderr bytes.Buffer
	command.Stdout = &stdout
	command.Stderr = &stderr
	err := command.Run()
	code := 0
	if err != nil {
		var exitErr *exec.ExitError
		if !errors.As(err, &exitErr) {
			t.Fatalf("run public CLI: %v", err)
		}
		code = exitErr.ExitCode()
	}
	return publicCLIResult{code: code, stdout: stdout.String(), stderr: stderr.String()}
}

func findRepoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("repository root not found")
		}
		dir = parent
	}
}

func copyFixture(t *testing.T, source, destination string) {
	t.Helper()
	if err := os.WriteFile(destination, readFile(t, source), 0o600); err != nil {
		t.Fatal(err)
	}
}

func writeInventory(t *testing.T, path string, inventory modelinventory.Inventory) {
	t.Helper()
	raw, diagnostics := inventory.CanonicalJSON()
	if len(diagnostics) != 0 {
		t.Fatal(diagnostics)
	}
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
}
