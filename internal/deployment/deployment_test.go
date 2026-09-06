package deployment

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestDeterministicRealizationActivationAndRollback(t *testing.T) {
	desired := representativeDesired()
	firstID, err := DerivationID(desired)
	if err != nil {
		t.Fatal(err)
	}
	desired.Objects[0], desired.Objects[1] = desired.Objects[1], desired.Objects[0]
	desired.Target.Capabilities[0], desired.Target.Capabilities[1] = desired.Target.Capabilities[1], desired.Target.Capabilities[0]
	secondID, err := DerivationID(desired)
	if err != nil {
		t.Fatal(err)
	}
	if firstID != secondID {
		t.Fatalf("normalized derivation changed with input order: %s != %s", firstID, secondID)
	}

	linux, err := NewRealization(firstID, desired.Target, map[string][]byte{"bin/agent": []byte("v1"), "context.json": []byte(`{"active":"one"}`)})
	if err != nil {
		t.Fatal(err)
	}
	if linux.ID == firstID {
		t.Fatal("realization identity must not conflate with derivation identity")
	}
	otherBytes, err := NewRealization(firstID, desired.Target, map[string][]byte{"bin/agent": []byte("v2"), "context.json": []byte(`{"active":"one"}`)})
	if err != nil {
		t.Fatal(err)
	}
	if otherBytes.ID == linux.ID {
		t.Fatal("different output bytes produced equal realization IDs")
	}

	root := t.TempDir()
	storePath, err := (Store{Root: filepath.Join(root, "store")}).Materialize(linux)
	if err != nil {
		t.Fatal(err)
	}
	if got, err := os.ReadFile(filepath.Join(storePath, "bin", "agent")); err != nil || string(got) != "v1" {
		t.Fatalf("materialized output = %q, %v", got, err)
	}

	activator := Activator{Root: filepath.Join(root, "machine")}
	one, err := activator.Activate("default", []string{linux.ID}, []byte("policy=v1"))
	if err != nil {
		t.Fatal(err)
	}
	two, err := activator.Activate("default", []string{otherBytes.ID}, []byte("policy=v2"))
	if err != nil {
		t.Fatal(err)
	}
	if two.Sequence != one.Sequence+1 {
		t.Fatalf("generation sequence = %d after %d", two.Sequence, one.Sequence)
	}
	rolledBack, err := activator.Rollback("default")
	if err != nil {
		t.Fatal(err)
	}
	if rolledBack.Sequence != one.Sequence || rolledBack.Realizations[0] != linux.ID {
		t.Fatalf("rollback selected %#v, want %#v", rolledBack, one)
	}
	active, err := activator.Active("default")
	if err != nil {
		t.Fatal(err)
	}
	if active.Sequence != one.Sequence {
		t.Fatalf("active generation = %d, want %d", active.Sequence, one.Sequence)
	}
}

func TestSubstitutionCompatibilityAndBoundedDeterminism(t *testing.T) {
	desired := representativeDesired()
	id, err := DerivationID(desired)
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewRealization(id, desired.Target, map[string][]byte{"engine.bin": []byte("gpu-specific")})
	if err != nil {
		t.Fatal(err)
	}
	if !r.Compatible(id, Target{OS: "linux", Architecture: "amd64", Capabilities: []string{"cuda-12.8", "sm-89"}}) {
		t.Fatal("matching trusted realization was not substitutable")
	}
	if r.Compatible(id, Target{OS: "linux", Architecture: "amd64", Capabilities: []string{"cuda-12.8", "sm-90"}}) {
		t.Fatal("incompatible hardware realization was substitutable")
	}
	if r.Compatible("different-derivation", desired.Target) {
		t.Fatal("different derivation was substitutable")
	}
}

func TestMaterializeRejectsTraversalAndTamperedIdentity(t *testing.T) {
	if _, err := NewRealization("d", Target{}, map[string][]byte{"../escape": []byte("bad")}); err == nil {
		t.Fatal("path traversal accepted")
	}
	r, err := NewRealization("d", Target{}, map[string][]byte{"safe": []byte("ok")})
	if err != nil {
		t.Fatal(err)
	}
	r.ID = "tampered"
	if _, err := (Store{Root: filepath.Join(t.TempDir(), "store")}).Materialize(r); err == nil {
		t.Fatal("tampered realization identity accepted")
	}
}

func TestFailClosedInvariants(t *testing.T) {
	if _, err := NewRealization("", Target{}, map[string][]byte{"ok": []byte("1")}); err == nil {
		t.Fatal("expected error on empty derivation ID")
	}
	if _, err := NewRealization("d", Target{}, map[string][]byte{}); err == nil {
		t.Fatal("expected error on empty files map")
	}
	if _, err := NewRealization("d", Target{}, map[string][]byte{"": []byte("empty")}); err == nil {
		t.Fatal("expected error on empty file path")
	}
	if _, err := NewRealization("d", Target{}, map[string][]byte{"realization.json": []byte("manifest")}); err == nil {
		t.Fatal("expected error on reserved realization.json file")
	}
	absPath, err := filepath.Abs("safe_rel")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := NewRealization("d", Target{}, map[string][]byte{absPath: []byte("abs")}); err == nil {
		t.Fatal("expected error on absolute path")
	}

	valid, err := NewRealization("d", Target{}, map[string][]byte{"payload": []byte("content")})
	if err != nil {
		t.Fatal(err)
	}

	if _, err := (Store{Root: ""}).Materialize(valid); err == nil {
		t.Fatal("expected error on empty store root")
	}

	store := Store{Root: filepath.Join(t.TempDir(), "store")}
	path1, err := store.Materialize(valid)
	if err != nil {
		t.Fatal(err)
	}
	path2, err := store.Materialize(valid)
	if err != nil {
		t.Fatal(err)
	}
	if path1 != path2 {
		t.Fatalf("idempotent materialize returned differing paths: %s != %s", path1, path2)
	}

	activator := Activator{Root: filepath.Join(t.TempDir(), "machine")}
	for _, badName := range []string{"", ".", "..", "invalid/name", "invalid\\name"} {
		if _, err := activator.Activate(badName, []string{valid.ID}, []byte("cfg")); err == nil {
			t.Fatalf("expected error on invalid name %q", badName)
		}
		if _, err := activator.Active(badName); err == nil {
			t.Fatalf("expected error on Active with invalid name %q", badName)
		}
		if _, err := activator.Rollback(badName); err == nil {
			t.Fatalf("expected error on Rollback with invalid name %q", badName)
		}
	}
	if _, err := activator.Activate("valid", nil, []byte("cfg")); err == nil {
		t.Fatal("expected error on empty realization IDs")
	}
	if _, err := activator.Rollback("valid"); err == nil {
		t.Fatal("expected error on rollback when no generation is active")
	}
	if _, err := activator.Activate("valid", []string{valid.ID}, []byte("cfg")); err != nil {
		t.Fatal(err)
	}
	if _, err := activator.Rollback("valid"); err == nil {
		t.Fatal("expected error on rollback when sequence <= 1")
	}
}

func TestMaterializeRejectsWindowsReservedDeviceNames(t *testing.T) {
	reservedPaths := []string{
		"aux",
		"nul.txt",
		"com1",
		"COM2",
		"prn",
		"con.json",
		"lpt1",
		"LPT9.txt",
		"dir/aux",
		"dir/nul.txt",
		"dir/com1/file.txt",
		"nested/PRN.log",
		"sub/LPT3.dat",
	}

	for _, badPath := range reservedPaths {
		t.Run(badPath, func(t *testing.T) {
			files := map[string][]byte{
				badPath: []byte("malicious payload"),
			}

			// 1. Direct validation check on NewRealization
			_, err := NewRealization("d1", Target{}, files)
			if err == nil {
				t.Fatalf("expected NewRealization error for reserved path %q, got nil", badPath)
			}
			var valErr *ValidationError
			if !errors.As(err, &valErr) {
				t.Fatalf("expected *ValidationError, got %T (%v)", err, err)
			}
			if !errors.Is(err, ErrReservedDeviceName) {
				t.Fatalf("expected errors.Is(err, ErrReservedDeviceName) for %q, got %v", badPath, err)
			}
			if valErr.Path != badPath {
				t.Fatalf("valErr.Path = %q, want %q", valErr.Path, badPath)
			}

			// 2. Materialize must fail before performing disk operations
			storeDir := filepath.Join(t.TempDir(), "store")
			store := Store{Root: storeDir}
			r := Realization{
				DerivationID: "d1",
				Target:       Target{OS: "linux", Architecture: "amd64"},
				Files:        files,
			}
			_, matErr := store.Materialize(r)
			if matErr == nil {
				t.Fatalf("expected Materialize error for reserved path %q, got nil", badPath)
			}
			if !errors.Is(matErr, ErrReservedDeviceName) {
				t.Fatalf("expected Materialize to return ErrReservedDeviceName for %q, got %v", badPath, matErr)
			}

			// 3. Verify zero disk operations occurred in store directory
			if _, statErr := os.Stat(storeDir); !errors.Is(statErr, os.ErrNotExist) {
				t.Fatalf("store directory was created on disk despite rejection of %q: %v", badPath, statErr)
			}
		})
	}

	// Verify that valid paths with overlapping substrings are not rejected
	validPaths := []string{
		"auxiliary.txt",
		"community.json",
		"console.log",
		"printer.go",
		"null.txt",
		"com0.bin",
		"com10.dat",
		"lpt0.log",
		"lpt10.log",
	}
	for _, goodPath := range validPaths {
		t.Run("valid_"+goodPath, func(t *testing.T) {
			goodFiles := map[string][]byte{goodPath: []byte("valid content")}
			r, err := NewRealization("d1", Target{OS: "linux", Architecture: "amd64"}, goodFiles)
			if err != nil {
				t.Fatalf("valid path %q unexpectedly rejected: %v", goodPath, err)
			}
			store := Store{Root: filepath.Join(t.TempDir(), "store")}
			outPath, err := store.Materialize(r)
			if err != nil {
				t.Fatalf("valid path %q failed to materialize: %v", goodPath, err)
			}
			if _, statErr := os.Stat(outPath); statErr != nil {
				t.Fatalf("materialized directory not found for valid path %q: %v", goodPath, statErr)
			}
		})
	}
}

func representativeDesired() DesiredContext {
	return DesiredContext{
		Objects:        []ManagedObject{{Type: "skill", ID: "review", Content: []byte("locked-skill")}, {Type: "policy", ID: "readonly", Content: []byte("deny-write")}},
		LockedInputs:   []LockedInput{{Name: "model", Digest: "sha256:model"}, {Name: "adapter", Digest: "sha256:adapter"}},
		AdapterVersion: "adapter/v1", SchemaVersion: "context/v1", Policy: "readonly",
		Target:         Target{OS: "linux", Architecture: "amd64", Capabilities: []string{"sm-89", "cuda-12.8"}},
		AllowedEffects: []string{"read-source"},
	}
}
