package repocompose

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestComposePublicOnly(t *testing.T) {
	public := repository(t, "public", "example.test/fak")
	missingPrivate := filepath.Join(t.TempDir(), "fak-private")

	plan, err := Compose(
		Candidate{Name: "public", Root: public, Identity: moduleIdentity("example.test/fak"), Source: "workspace"},
		Candidate{Name: "private", Root: missingPrivate, Identity: moduleIdentity("example.test/fak-private"), Source: "sibling-default"},
	)
	if err != nil {
		t.Fatalf("Compose() error = %v", err)
	}
	if len(plan.Roots) != 1 || plan.Roots[0] != canonical(t, public) {
		t.Fatalf("Roots = %#v, want public root only", plan.Roots)
	}
	if plan.Receipt.Schema != ReceiptSchema {
		t.Fatalf("receipt schema = %q, want %q", plan.Receipt.Schema, ReceiptSchema)
	}
	if got := plan.Receipt.Entries[1]; got.State != StateAbsent || got.Reason != "root_not_found" || got.Source != "sibling-default" {
		t.Fatalf("private entry = %#v, want absent sibling-default", got)
	}
}

func TestComposeValidOverlayAndStablePrecedence(t *testing.T) {
	public := repository(t, "public", "example.test/fak")
	private := repository(t, "private", "example.test/fak-private")

	primary := Candidate{Name: "public", Root: public, Mode: ModeRequired, Identity: moduleIdentity("example.test/fak")}
	overlays := []Candidate{
		{Name: "private", Root: private, Mode: ModeAuto, Identity: moduleIdentity("example.test/fak-private")},
		{Name: "public-alias", Root: filepath.Join(public, "."), Mode: ModeAuto, Identity: moduleIdentity("example.test/fak")},
	}
	want, err := Compose(primary, overlays...)
	if err != nil {
		t.Fatalf("Compose() error = %v", err)
	}
	if got := want.Roots; !reflect.DeepEqual(got, []string{canonical(t, public), canonical(t, private)}) {
		t.Fatalf("Roots = %#v, want caller precedence with duplicate collapsed", got)
	}
	if got := want.Receipt.Entries[2]; got.State != StateSelected || got.Reason != "duplicate_root" {
		t.Fatalf("duplicate entry = %#v", got)
	}

	again, err := Compose(primary, overlays...)
	if err != nil {
		t.Fatalf("second Compose() error = %v", err)
	}
	firstJSON, _ := json.Marshal(want)
	secondJSON, _ := json.Marshal(again)
	if string(firstJSON) != string(secondJSON) {
		t.Fatalf("composition is not deterministic:\n%s\n%s", firstJSON, secondJSON)
	}
}

func TestComposePresentRequiredOverlayIsSelected(t *testing.T) {
	public := repository(t, "public", "example.test/fak")
	private := repository(t, "private", "example.test/fak-private")

	plan, err := Compose(
		Candidate{Name: "public", Root: public, Identity: moduleIdentity("example.test/fak")},
		Candidate{Name: "private", Root: private, Mode: ModeRequired, Identity: moduleIdentity("example.test/fak-private")},
	)
	if err != nil {
		t.Fatalf("Compose() error = %v", err)
	}
	want := []string{public, private}
	if !reflect.DeepEqual(plan.Roots, want) {
		t.Fatalf("Roots = %#v, want %#v", plan.Roots, want)
	}
	if got := plan.Receipt.Entries[1]; got.Mode != ModeRequired || got.State != StateSelected || got.Reason != "identity_verified" {
		t.Fatalf("required overlay entry = %#v, want selected required overlay", got)
	}
}

func TestComposeRequiredOrExplicitAbsenceFailsTyped(t *testing.T) {
	public := repository(t, "public", "example.test/fak")
	missing := filepath.Join(t.TempDir(), "missing")
	tests := []struct {
		name      string
		candidate Candidate
	}{
		{name: "required", candidate: Candidate{Name: "required", Root: missing, Mode: ModeRequired, Identity: moduleIdentity("example.test/overlay")}},
		{name: "explicit auto", candidate: Candidate{Name: "explicit", Root: missing, Mode: ModeAuto, Explicit: true, Identity: moduleIdentity("example.test/overlay")}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			plan, err := Compose(
				Candidate{Name: "public", Root: public, Identity: moduleIdentity("example.test/fak")},
				tt.candidate,
			)
			var missingErr *MissingError
			if !errors.As(err, &missingErr) {
				t.Fatalf("error = %T %v, want *MissingError", err, err)
			}
			if got := plan.Receipt.Entries[1]; got.State != StateAbsent || got.Reason != "root_not_found" {
				t.Fatalf("entry = %#v, want recorded absence", got)
			}
		})
	}
}

func TestComposePresentWrongIdentityFailsTyped(t *testing.T) {
	public := repository(t, "public", "example.test/fak")
	wrong := repository(t, "wrong", "example.test/not-private")

	plan, err := Compose(
		Candidate{Name: "public", Root: public, Identity: moduleIdentity("example.test/fak")},
		Candidate{Name: "private", Root: wrong, Identity: moduleIdentity("example.test/fak-private")},
	)
	var invalidErr *InvalidError
	if !errors.As(err, &invalidErr) {
		t.Fatalf("error = %T %v, want *InvalidError", err, err)
	}
	if invalidErr.Reason != "identity_mismatch" {
		t.Fatalf("invalid reason = %q", invalidErr.Reason)
	}
	if got := plan.Receipt.Entries[1]; got.State != StateInvalid || got.Reason != "identity_mismatch" {
		t.Fatalf("entry = %#v, want invalid identity_mismatch", got)
	}
}

func TestComposeOffDoesNotProbe(t *testing.T) {
	public := repository(t, "public", "example.test/fak")
	plan, err := Compose(
		Candidate{Name: "public", Root: public, Identity: moduleIdentity("example.test/fak")},
		Candidate{Name: "disabled", Root: "", Mode: ModeOff, Explicit: true},
	)
	if err != nil {
		t.Fatalf("Compose() error = %v", err)
	}
	if got := plan.Receipt.Entries[1]; got.State != StateDisabled || got.Reason != "mode_off" {
		t.Fatalf("disabled entry = %#v", got)
	}
}

func TestComposePrimaryMustBeRequired(t *testing.T) {
	public := repository(t, "public", "example.test/fak")
	for _, mode := range []Mode{ModeAuto, ModeOff} {
		t.Run(string(mode), func(t *testing.T) {
			plan, err := Compose(Candidate{
				Name:     "public",
				Root:     public,
				Mode:     mode,
				Identity: moduleIdentity("example.test/fak"),
			})
			var invalidErr *InvalidError
			if !errors.As(err, &invalidErr) {
				t.Fatalf("error = %T %v, want *InvalidError", err, err)
			}
			if invalidErr.Reason != "primary_mode_not_required" {
				t.Fatalf("invalid reason = %q", invalidErr.Reason)
			}
			if got := plan.Receipt.Entries[0]; got.State != StateInvalid || got.Reason != "primary_mode_not_required" {
				t.Fatalf("primary entry = %#v", got)
			}
		})
	}

	missing := filepath.Join(t.TempDir(), "missing-public")
	plan, err := Compose(Candidate{Name: "public", Root: missing, Identity: moduleIdentity("example.test/fak")})
	var missingErr *MissingError
	if !errors.As(err, &missingErr) {
		t.Fatalf("missing primary error = %T %v, want *MissingError", err, err)
	}
	if got := plan.Receipt.Entries[0]; got.Mode != ModeRequired || got.State != StateAbsent {
		t.Fatalf("missing primary entry = %#v", got)
	}
}

func TestComposeDanglingRootIsInvalid(t *testing.T) {
	public := repository(t, "public", "example.test/fak")
	parent := t.TempDir()
	target := filepath.Join(parent, "missing-target")
	link := filepath.Join(parent, "declared-overlay")
	if err := os.Symlink(target, link); err != nil {
		t.Skipf("symlinks unsupported: %v", err)
	}

	plan, err := Compose(
		Candidate{Name: "public", Root: public, Identity: moduleIdentity("example.test/fak")},
		Candidate{Name: "broken-default", Root: link, Identity: moduleIdentity("example.test/overlay"), Source: "default"},
	)
	var invalidErr *InvalidError
	if !errors.As(err, &invalidErr) {
		t.Fatalf("error = %T %v, want *InvalidError", err, err)
	}
	if got := plan.Receipt.Entries[1]; got.State != StateInvalid || got.Reason != "invalid_root" {
		t.Fatalf("dangling root entry = %#v, want invalid declaration", got)
	}
}

func TestComposeIdentityContainsSearchesPrefix(t *testing.T) {
	const needle = "repo-identity"

	writeMarker := func(t *testing.T, offset int) string {
		t.Helper()
		root := filepath.Join(t.TempDir(), "repo")
		if err := os.Mkdir(root, 0o755); err != nil {
			t.Fatal(err)
		}
		data := bytes.Repeat([]byte{'x'}, maxMarkerBytes+128)
		copy(data[offset:], needle)
		if err := os.WriteFile(filepath.Join(root, "identity"), data, 0o644); err != nil {
			t.Fatal(err)
		}
		return root
	}

	t.Run("match ending at prefix boundary", func(t *testing.T) {
		root := writeMarker(t, maxMarkerBytes-len(needle))
		plan, err := Compose(Candidate{Root: root, Identity: Identity{Path: "identity", Contains: needle}})
		if err != nil {
			t.Fatalf("Compose() error = %v", err)
		}
		if len(plan.Roots) != 1 {
			t.Fatalf("Roots = %#v", plan.Roots)
		}
	})

	t.Run("match beyond prefix boundary", func(t *testing.T) {
		root := writeMarker(t, maxMarkerBytes)
		plan, err := Compose(Candidate{Root: root, Identity: Identity{Path: "identity", Contains: needle}})
		var invalidErr *InvalidError
		if !errors.As(err, &invalidErr) || invalidErr.Reason != "identity_mismatch" {
			t.Fatalf("error = %T %v, want identity_mismatch", err, err)
		}
		if got := plan.Receipt.Entries[0]; got.State != StateInvalid {
			t.Fatalf("entry = %#v", got)
		}
	})
}

func TestComposeDedupeUsesFilesystemIdentity(t *testing.T) {
	root := repository(t, "real", "example.test/fak")
	alias := filepath.Join(t.TempDir(), "alias")
	if err := os.Symlink(root, alias); err != nil {
		t.Skipf("symlinks unsupported: %v", err)
	}

	rootInfo, err := os.Stat(root)
	if err != nil {
		t.Fatal(err)
	}
	aliasInfo, err := os.Stat(alias)
	if err != nil {
		t.Fatal(err)
	}
	if !sameSelectedRoot(alias, aliasInfo, []selectedRoot{{path: root, info: rootInfo}}) {
		t.Fatal("filesystem aliases were not recognized as the same root")
	}

	plan, err := Compose(
		Candidate{Name: "real", Root: root, Identity: moduleIdentity("example.test/fak")},
		Candidate{Name: "alias", Root: alias, Identity: moduleIdentity("example.test/fak")},
	)
	if err != nil {
		t.Fatalf("Compose() error = %v", err)
	}
	if len(plan.Roots) != 1 || plan.Receipt.Entries[1].Reason != "duplicate_root" {
		t.Fatalf("plan = %#v, want one selected filesystem identity", plan)
	}
}

func repository(t *testing.T, name, module string) string {
	t.Helper()
	root := filepath.Join(t.TempDir(), name)
	if err := os.Mkdir(root, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module "+module+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	return root
}

func moduleIdentity(module string) Identity {
	return Identity{Path: "go.mod", Contains: "module " + module + "\n"}
}

func canonical(t *testing.T, path string) string {
	t.Helper()
	got, err := canonicalPath(path)
	if err != nil {
		t.Fatal(err)
	}
	return got
}
