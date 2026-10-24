package gateway

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// fak-test:runtime fast est=1s
func TestCodexPatchFootprintCompleteOperationsAndAliases(t *testing.T) {
	patch := strings.Join([]string{
		"*** Begin Patch",
		"*** Add File: add.txt", "+added",
		"*** Update File: old.txt", "*** Move to: moved.txt", "@@", "-old", "+new",
		"*** Delete File: delete.txt",
		"*** End Patch",
	}, "\n")
	want := []string{"add.txt", "old.txt", "moved.txt", "delete.txt"}
	for _, field := range []string{"input", "patch"} {
		raw, err := json.Marshal(map[string]string{field: patch})
		if err != nil {
			t.Fatal(err)
		}
		_, got, mutating, known := adjudicateRawWriteTargets("apply_patch", string(raw), false)
		if !mutating || !known || !reflect.DeepEqual(got, want) {
			t.Fatalf("%s footprint=%v mutating=%t known=%t, want %v/true/true", field, got, mutating, known, want)
		}
	}
}

// fak-test:runtime fast est=1s
func TestCodexPatchFootprintRejectsAmbiguityAndMalformedInput(t *testing.T) {
	valid := "*** Begin Patch\n*** Add File: safe.txt\n+safe\n*** End Patch"
	tests := []struct {
		name string
		raw  string
	}{
		{"conflicting aliases", `{"input":` + quotedJSON(valid) + `,"patch":` + quotedJSON(strings.Replace(valid, "safe.txt", "hidden.txt", 1)) + `}`},
		{"duplicate alias", `{"input":` + quotedJSON(valid) + `,"input":` + quotedJSON(valid) + `}`},
		{"safe path masks extra field", `{"input":` + quotedJSON(valid) + `,"path":"safe.txt"}`},
		{"missing end envelope", `{"input":"*** Begin Patch\n*** Add File: safe.txt\n+safe"}`},
		{"empty path", `{"input":"*** Begin Patch\n*** Add File:   \n+safe\n*** End Patch"}`},
		{"move without update", `{"input":"*** Begin Patch\n*** Add File: safe.txt\n*** Move to: hidden.txt\n+safe\n*** End Patch"}`},
		{"unknown structural header", `{"input":"*** Begin Patch\n*** Add File: safe.txt\n+safe\n*** Copy to: hidden.txt\n*** End Patch"}`},
		{"update without hunk", `{"input":"*** Begin Patch\n*** Update File: safe.txt\n*** End Patch"}`},
		{"move after body", `{"input":"*** Begin Patch\n*** Update File: safe.txt\n@@\n-old\n+new\n*** Move to: hidden.txt\n*** End Patch"}`},
		{"duplicate eof", `{"input":"*** Begin Patch\n*** Update File: safe.txt\n@@\n-old\n+new\n*** End of File\n*** End of File\n*** End Patch"}`},
		{"content after eof", `{"input":"*** Begin Patch\n*** Update File: safe.txt\n@@\n-old\n+new\n*** End of File\n+hidden\n*** End Patch"}`},
		{"leading target whitespace", `{"input":"*** Begin Patch\n*** Add File:  safe.txt\n+safe\n*** End Patch"}`},
		{"trailing target whitespace", `{"input":"*** Begin Patch\n*** Add File: safe.txt \n+safe\n*** End Patch"}`},
		{"missing canonical separator", `{"input":"*** Begin Patch\n*** Add File:safe.txt\n+safe\n*** End Patch"}`},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, targets, mutating, known := adjudicateRawWriteTargets("functions.apply_patch", tc.raw, false)
			if !mutating || known || len(targets) != 0 {
				t.Fatalf("footprint=%v mutating=%t known=%t, want unknown mutation", targets, mutating, known)
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestCodexPatchFootprintPreservesInternalPathSpaces(t *testing.T) {
	patch := "*** Begin Patch\n*** Add File: dir/file with space.txt\n+safe\n*** End Patch"
	raw, _ := json.Marshal(map[string]string{"input": patch})
	_, targets, mutating, known := adjudicateRawWriteTargets("apply_patch", string(raw), false)
	if !mutating || !known || !reflect.DeepEqual(targets, []string{"dir/file with space.txt"}) {
		t.Fatalf("footprint=%v mutating=%t known=%t", targets, mutating, known)
	}
}

// fak-test:runtime fast est=1s
func TestCodexPatchFootprintHeaderlessUpdateAliasParity(t *testing.T) {
	patch := "*** Begin Patch\n*** Update File: source/headerless.txt\n-before\n+after\n*** End Patch"
	for _, field := range []string{"input", "patch"} {
		raw, _ := json.Marshal(map[string]string{field: patch})
		_, targets, mutating, known := adjudicateRawWriteTargets("apply_patch", string(raw), false)
		if !mutating || !known || !reflect.DeepEqual(targets, []string{"source/headerless.txt"}) {
			t.Fatalf("%s footprint=%v mutating=%t known=%t", field, targets, mutating, known)
		}
	}
}

// fak-test:runtime fast est=1s
func TestCodexPatchFootprintContentCannotInventStructuralTarget(t *testing.T) {
	patch := "*** Begin Patch\n*** Add File: visible.txt\n+*** Delete File: hidden.txt\n*** End Patch"
	raw, _ := json.Marshal(map[string]string{"input": patch})
	_, targets, mutating, known := adjudicateRawWriteTargets("apply_patch", string(raw), false)
	if !mutating || !known || !reflect.DeepEqual(targets, []string{"visible.txt"}) {
		t.Fatalf("footprint=%v mutating=%t known=%t", targets, mutating, known)
	}
}

// fak-test:runtime fast est=1s
func TestCodexPatchFootprintGatewayRejectsOutsideRootAndHeldMoveDestination(t *testing.T) {
	root := t.TempDir()
	for _, dir := range []string{"source", "protected"} {
		if err := os.MkdirAll(filepath.Join(root, dir), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	outside := t.TempDir()
	SetLeasePlaneProviders(nil, func(context.Context) (LeasePresenceView, error) {
		return LeasePresenceView{ClassifiedLeases: json.RawMessage(`[]`)}, nil
	})
	SetWorkspaceLeaseAdmissionProvider(func(context.Context) (WorkspaceLeaseAdmissionView, error) {
		return WorkspaceLeaseAdmissionView{
			WorkspaceRoot: root,
			Leases:        []WorkspaceAdmissionLease{{TreeGlobs: []string{"protected/**"}, SessionID: "peer"}},
		}, nil
	})
	t.Cleanup(func() {
		SetLeasePlaneProviders(nil, nil)
		SetWorkspaceLeaseAdmissionProvider(nil)
	})
	srv := &Server{}

	outsidePatch := "*** Begin Patch\n*** Add File: " + filepath.ToSlash(filepath.Join(outside, "escape.txt")) + "\n+escape\n*** End Patch"
	outsideArgs, _ := json.Marshal(map[string]string{"input": outsidePatch})
	_, outsideTargets, mutating, known := adjudicateRawWriteTargets("apply_patch", string(outsideArgs), false)
	if !mutating || !known || !reflect.DeepEqual(outsideTargets, []string{filepath.ToSlash(filepath.Join(outside, "escape.txt"))}) {
		t.Fatalf("outside-root extraction=%v mutating=%t known=%t", outsideTargets, mutating, known)
	}
	outsideVerdict, refused := srv.adjudicateLeaseAdmission(context.Background(), "apply_patch", string(outsideArgs), false)
	if !refused || outsideVerdict.Kind != "DENY" || outsideVerdict.Reason != abi.ReasonName(abi.ReasonDefaultDeny) ||
		outsideVerdict.By != "lease-admission" {
		t.Fatalf("outside-root verdict=%+v refused=%t", outsideVerdict, refused)
	}

	movePatch := "*** Begin Patch\n*** Update File: source/file.txt\n*** Move to: protected/moved.txt\n@@\n-old\n+new\n*** End Patch"
	moveArgs, _ := json.Marshal(map[string]string{"input": movePatch})
	moveVerdict, refused := srv.adjudicateLeaseAdmission(context.Background(), "apply_patch", string(moveArgs), false)
	if !refused || moveVerdict.Kind != "DENY" || moveVerdict.Reason != abi.ReasonName(abi.ReasonLeaseHeld) || moveVerdict.By != "lease-admission" {
		t.Fatalf("held move verdict=%+v refused=%t", moveVerdict, refused)
	}

	safePatch := "*** Begin Patch\n*** Add File: source/safe.txt\n+safe\n*** End Patch"
	for _, field := range []string{"input", "patch"} {
		safeArgs, _ := json.Marshal(map[string]string{field: safePatch})
		verdict, refused := srv.adjudicateLeaseAdmission(context.Background(), "apply_patch", string(safeArgs), false)
		if refused || verdict.Kind != "" {
			t.Fatalf("safe %s gateway verdict=%+v refused=%t, want admitted", field, verdict, refused)
		}
	}

	headerlessPatch := "*** Begin Patch\n*** Update File: source/headerless.txt\n-before\n+after\n*** End Patch"
	for _, field := range []string{"input", "patch"} {
		headerlessArgs, _ := json.Marshal(map[string]string{field: headerlessPatch})
		verdict, refused := srv.adjudicateLeaseAdmission(context.Background(), "apply_patch", string(headerlessArgs), false)
		if refused || verdict.Kind != "" {
			t.Fatalf("headerless %s gateway verdict=%+v refused=%t, want admitted", field, verdict, refused)
		}
	}
}

func quotedJSON(value string) string {
	b, _ := json.Marshal(value)
	return string(b)
}
