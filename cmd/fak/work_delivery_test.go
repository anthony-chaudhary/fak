package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestWorkDeliveryStatusKeepsAxesIndependent(t *testing.T) {
	file := writeWorkDeliveryFixture(t, `{"schema":"fak.work-delivery/v1","id":"issue-7104","axes":{"authoring":"recorded","compile_admission":"excluded","verification":"unverified","integration":"unintegrated","release":"not_ready"}}`)
	var out, errOut bytes.Buffer
	if code := runWorkDelivery(&out, &errOut, []string{"status", "--file", file}); code != 0 {
		t.Fatalf("code=%d err=%s", code, errOut.String())
	}
	for _, want := range []string{"unit issue-7104", "authoring: recorded (declared)", "compile admission: excluded (declared)", "verification: unverified (observed)", "release: not_ready (observed)"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("missing %q in\n%s", want, out.String())
		}
	}
}

func TestWorkDeliveryTransitionChangesOnlyNamedAxis(t *testing.T) {
	file := writeWorkDeliveryFixture(t, `{"schema":"fak.work-delivery/v1","id":"unit","axes":{"authoring":"draft","compile_admission":"undeclared","verification":"unverified","integration":"unintegrated","release":"not_ready"}}`)
	var out, errOut bytes.Buffer
	if code := runWorkDelivery(&out, &errOut, []string{"transition", "--file", file, "--axis", "authoring", "--to", "recorded", "--gate", "git.commit", "--json"}); code != 0 {
		t.Fatalf("code=%d err=%s", code, errOut.String())
	}
	for _, want := range []string{`"authoring": "recorded"`, `"compile_admission": "undeclared"`, `"release": "not_ready"`, `"gate": "git.commit"`} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("missing %q in\n%s", want, out.String())
		}
	}
}

func TestWorkDeliveryDiagnoseRendersExactLeaf(t *testing.T) {
	file := writeWorkDeliveryFixture(t, `{"scope":{"id":"ci","units":[{"id":"gateway"},{"id":"engine"}]},"failed_unit_id":"engine","class":"compile","gate":"go-build","check_command":"check {unit}"}`)
	var out, errOut bytes.Buffer
	if code := runWorkDelivery(&out, &errOut, []string{"diagnose", "--file", file}); code != 0 {
		t.Fatalf("code=%d err=%s", code, errOut.String())
	}
	if got := out.String(); !strings.Contains(got, "unit engine blocked at go-build (compile)") || !strings.Contains(got, "next: check engine") {
		t.Fatalf("render:\n%s", got)
	}
}

func TestWorkDeliveryDiagnoseRendersRecursiveSplitJSON(t *testing.T) {
	file := writeWorkDeliveryFixture(t, `{"scope":{"id":"ci","units":[{"id":"agent","tree":"internal/agent","independently_checkable":true},{"id":"gateway","tree":"internal/gateway","independently_checkable":true}]},"class":"unknown","gate":"make-ci"}`)
	var out, errOut bytes.Buffer
	if code := runWorkDelivery(&out, &errOut, []string{"diagnose", "--file", file, "--json"}); code != 0 {
		t.Fatalf("code=%d err=%s", code, errOut.String())
	}
	for _, want := range []string{`"kind": "split"`, `"id": "ci/1"`, `"id": "ci/2"`, `"next_action"`} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("missing %q in\n%s", want, out.String())
		}
	}
}

func TestWorkDeliveryStagesQueriesCanonicalRegistry(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := runWorkDelivery(&out, &errOut, []string{"stages", "--id", "recording"}); code != 0 {
		t.Fatalf("code=%d err=%s", code, errOut.String())
	}
	if got := out.String(); !strings.Contains(got, "stage recording: Checkpoint and commit recording") || strings.Contains(got, "compile-stream admission") {
		t.Fatalf("render:\n%s", got)
	}
	out.Reset()
	errOut.Reset()
	if code := runWorkDelivery(&out, &errOut, []string{"stages", "--local", "CI red"}); code != 0 {
		t.Fatalf("code=%d err=%s", code, errOut.String())
	}
	if got := out.String(); !strings.Contains(got, "stage full-tests, bottleneck unknown-irreducible") {
		t.Fatalf("render:\n%s", got)
	}
}

func writeWorkDeliveryFixture(t *testing.T, data string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "fixture.json")
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestWorkDeliveryTransitionRejectsInvalidState(t *testing.T) {
	path := writeWorkDeliveryFixture(t, `{"schema":"fak.work-delivery/v1","id":"unit-invalid","axes":{"authoring":"draft","compile_admission":"undeclared","verification":"unverified","integration":"unintegrated","release":"not_ready"}}`)
	var stdout, stderr bytes.Buffer
	code := runWorkDelivery(&stdout, &stderr, []string{"transition", "--file", path, "--axis", "verification", "--to", "verified"})
	if code == 0 || !strings.Contains(stderr.String(), "illegal verification transition") {
		t.Fatalf("code=%d stderr=%q", code, stderr.String())
	}
}

func TestWorkDeliveryStagesRejectsUnknownQueries(t *testing.T) {
	for _, args := range [][]string{{"stages", "--id", "ci-red"}, {"stages", "--local", "mystery-red"}} {
		var stdout, stderr bytes.Buffer
		if code := runWorkDelivery(&stdout, &stderr, args); code == 0 || !strings.Contains(stderr.String(), "unknown") {
			t.Fatalf("args=%v code=%d stderr=%q", args, code, stderr.String())
		}
	}
}
