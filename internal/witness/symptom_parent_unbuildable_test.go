package witness

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

const unbuildableParentSign = "package m\n\nfunc Sign(n int) int { return 0 }\n"

const unbuildableFixSign = `package m

func NewHelper() bool { return true }

func Sign(n int) int {
	if n < 0 { return -1 }
	return 0
}
`

const unbuildableHelperTest = `package m

import "testing"

func TestNewHelper(t *testing.T) {
	if !NewHelper() { t.Fatal("helper") }
}
`

// commitUnbuildableSibling builds a fix whose changed tests split into one file
// naming a fix-only API and one regression that compiles at the parent.
func commitUnbuildableSibling(t *testing.T, regression string) string {
	t.Helper()
	requireGoAndGit(t)
	dir := newGoModuleRepo(t)
	writeRepoFile(t, dir, "sign.go", unbuildableParentSign)
	gitIn(t, dir, "add", "sign.go")
	gitIn(t, dir, "commit", "-q", "-m", "parent: Sign bug")

	writeRepoFile(t, dir, "sign.go", unbuildableFixSign)
	writeRepoFile(t, dir, "helper_test.go", unbuildableHelperTest)
	writeRepoFile(t, dir, "sign_regression_test.go", regression)
	gitIn(t, dir, "add", "sign.go", "helper_test.go", "sign_regression_test.go")
	gitIn(t, dir, "commit", "-q", "-m", "fix(m): NewHelper and negative Sign")
	return dir
}

// fak-test:runtime slow est=20s lane=default
func TestSymptomExcludesParentUnbuildableSiblingFile(t *testing.T) {
	dir := commitUnbuildableSibling(t, `package m

import "testing"

func TestSignNegativeRegression(t *testing.T) {
	if Sign(-3) != -1 { t.Fatalf("Sign(-3)=%d", Sign(-3)) }
}
`)
	got, detail := NewWithRunner(gitRunner, dir).ResolveSymptomWithDetail(context.Background(), "HEAD", true)
	if got != abi.WitnessConfirmed {
		t.Fatalf("sibling regression with unbuildable helper test = %v (%s), want confirmed", got, detail)
	}
	if !strings.Contains(detail, "1 parent-unbuildable changed test file(s) excluded") {
		t.Fatalf("detail %q does not report the excluded file", detail)
	}
	assertRepoClean(t, dir)
}

// fak-test:runtime slow est=20s lane=default
func TestSymptomExcludedSiblingNeverCountsAsRed(t *testing.T) {
	dir := commitUnbuildableSibling(t, `package m

import "testing"

func TestSignPositive(t *testing.T) {
	if Sign(3) != 0 { t.Fatalf("Sign(3)=%d", Sign(3)) }
}
`)
	if got, detail := NewWithRunner(gitRunner, dir).ResolveSymptomWithDetail(context.Background(), "HEAD", true); got != abi.WitnessRefuted {
		t.Fatalf("tautological regression beside unbuildable helper = %v (%s), want refuted", got, detail)
	}
	assertRepoClean(t, dir)
}

// fak-test:runtime slow est=20s lane=default
func TestSymptomBlamesCompilerNamedSiblingFile(t *testing.T) {
	requireGoAndGit(t)
	dir := newGoModuleRepo(t)
	writeRepoFile(t, dir, "sign.go", "package m\n\ntype Opts struct{}\n\nfunc Sign(n int) int { return 0 }\n")
	gitIn(t, dir, "add", "sign.go")
	gitIn(t, dir, "commit", "-q", "-m", "parent: Sign bug")

	writeRepoFile(t, dir, "sign.go", "package m\n\ntype Opts struct{ Strict bool }\n\nfunc Sign(n int) int {\n\tif n < 0 { return -1 }\n\treturn 0\n}\n")
	writeRepoFile(t, dir, "opts_test.go", "package m\n\nimport \"testing\"\n\nfunc TestOptsStrict(t *testing.T) {\n\tif !(Opts{Strict: true}).Strict { t.Fatal(\"strict\") }\n}\n")
	writeRepoFile(t, dir, "sign_regression_test.go", "package m\n\nimport \"testing\"\n\nfunc TestSignNegativeRegression(t *testing.T) {\n\tif Sign(-3) != -1 { t.Fatalf(\"Sign(-3)=%d\", Sign(-3)) }\n}\n")
	gitIn(t, dir, "add", "sign.go", "opts_test.go", "sign_regression_test.go")
	gitIn(t, dir, "commit", "-q", "-m", "fix(m): Strict field and negative Sign")

	got, detail := NewWithRunner(gitRunner, dir).ResolveSymptomWithDetail(context.Background(), "HEAD", true)
	if got != abi.WitnessConfirmed || !strings.Contains(detail, "1 parent-unbuildable changed test file(s) excluded") {
		t.Fatalf("field-only sibling = %v (%s), want confirmed with one excluded file", got, detail)
	}
	assertRepoClean(t, dir)
}

// fak-test:runtime fast est=3s lane=default
func TestSymptomAbstainsBeforeCheckoutWhenEveryTestNamesFixOnlyAPI(t *testing.T) {
	requireGoAndGit(t)
	dir := newGoModuleRepo(t)
	if err := os.MkdirAll(filepath.Join(dir, "a"), 0o755); err != nil {
		t.Fatal(err)
	}
	writeRepoFile(t, dir, "a/sign.go", strings.Replace(unbuildableParentSign, "package m", "package a", 1))
	gitIn(t, dir, "add", "a/sign.go")
	gitIn(t, dir, "commit", "-q", "-m", "parent: Sign bug")

	writeRepoFile(t, dir, "a/sign.go", strings.Replace(unbuildableFixSign, "package m", "package a", 1))
	writeRepoFile(t, dir, "a/sign_ext_test.go", "package a_test\n\nimport (\n\t\"testing\"\n\n\t\"m/a\"\n)\n\nfunc TestHelperExternal(t *testing.T) {\n\tif !a.NewHelper() { t.Fatal(\"helper\") }\n}\n")
	writeRepoFile(t, dir, "a/helper_test.go", strings.Replace(unbuildableHelperTest, "package m", "package a", 1))
	gitIn(t, dir, "add", "a/sign.go", "a/sign_ext_test.go", "a/helper_test.go")
	gitIn(t, dir, "commit", "-q", "-m", "fix(a): NewHelper")

	runs := 0
	exec := func(ctx context.Context, d string, argv ...string) (string, int, error) {
		runs++
		return commandRunner(ctx, d, argv...)
	}
	got, detail := NewWithRunners(gitRunner, exec, dir).ResolveSymptomWithDetail(context.Background(), "HEAD", true)
	if got != abi.WitnessAbstain || !strings.Contains(detail, "did not build") || !strings.Contains(detail, "NewHelper") {
		t.Fatalf("fix-only API witness = %v (%s), want static parent-build abstain naming NewHelper", got, detail)
	}
	if runs != 0 {
		t.Fatalf("static abstain ran %d test command(s), want 0", runs)
	}
	assertRepoClean(t, dir)
}
