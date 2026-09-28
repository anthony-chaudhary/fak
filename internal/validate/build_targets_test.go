package validate

import (
	"reflect"
	"testing"
)

// TestSelectValidatePackagesVetsButNeverBuildsTestOnlyImporters pins the gate against
// the timeoutregression COMMITTED_RED: a changed package whose importer closure reaches a
// test-only package (its tests import validate, so it is an importer, but it owns no
// non-test Go file) must still vet that package, yet never hand it to `go build`, which
// refuses it with "no non-test Go files" regardless of the delta under validation.
func TestSelectValidatePackagesVetsButNeverBuildsTestOnlyImporters(t *testing.T) {
	const mod = "example.test/m"
	fileToPkg := map[string]string{
		"internal/leaf/leaf.go":                         mod + "/internal/leaf",
		"internal/leaf/leaf_test.go":                    mod + "/internal/leaf",
		"internal/validate/validate.go":                 mod + "/internal/validate",
		"internal/validate/timeoutregression/a_test.go": mod + "/internal/validate/timeoutregression",
		"cmd/tool/main.go":                              mod + "/cmd/tool",
	}
	edges := map[string][]string{
		mod + "/internal/validate":                   {mod + "/internal/leaf"},
		mod + "/internal/validate/timeoutregression": {mod + "/internal/validate"},
		mod + "/cmd/tool":                            {mod + "/internal/validate"},
	}
	res, recorder := newStrixPhaseTestRecorder()

	vet, build := selectValidatePackages(res, recorder, t.TempDir(), []string{"internal/leaf/leaf.go"}, fileToPkg, edges, nil, nil)

	wantVet := []string{"./cmd/tool", "./internal/leaf", "./internal/validate", "./internal/validate/timeoutregression"}
	wantBuild := []string{"./cmd/tool", "./internal/leaf", "./internal/validate"}
	if !reflect.DeepEqual(vet, wantVet) {
		t.Errorf("vet targets = %v, want %v (a test-only importer must still be vetted)", vet, wantVet)
	}
	if !reflect.DeepEqual(build, wantBuild) {
		t.Errorf("build targets = %v, want %v (go build refuses a test-only package)", build, wantBuild)
	}
	if want := []string{mod + "/internal/leaf"}; !reflect.DeepEqual(res.Tested, want) {
		t.Errorf("tested = %v, want %v", res.Tested, want)
	}
}

// TestSelectValidatePackagesTestOnlyChangeHasNoBuildTarget covers a delta that edits only a
// test-only package: there is nothing `go build` may name, but vet still covers it.
func TestSelectValidatePackagesTestOnlyChangeHasNoBuildTarget(t *testing.T) {
	const mod = "example.test/m"
	fileToPkg := map[string]string{
		"cmd/tool/startuptest/startup_test.go": mod + "/cmd/tool/startuptest",
	}
	res, recorder := newStrixPhaseTestRecorder()

	vet, build := selectValidatePackages(res, recorder, t.TempDir(), []string{"cmd/tool/startuptest/startup_test.go"}, fileToPkg, map[string][]string{}, nil, nil)

	if want := []string{"./cmd/tool/startuptest"}; !reflect.DeepEqual(vet, want) {
		t.Errorf("vet targets = %v, want %v", vet, want)
	}
	for _, target := range build {
		t.Errorf("build target %q selected, want none for a test-only delta", target)
	}
}
