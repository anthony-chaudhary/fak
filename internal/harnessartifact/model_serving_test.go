package harnessartifact

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

func TestCanonicalLocalModelDeclaration(t *testing.T) {
	decl := harnesskit.LocalModelDeclaration{
		Schema: LocalModelDeclarationSchema, ModelID: "qwen", GGUFPath: filepath.Join(t.TempDir(), "model.gguf"),
		GGUFSHA256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		Runtime:    "llama.cpp", ContextTokens: 32768, RequiredDevices: []string{"gpu0", "cpu", "gpu0"},
	}
	first, err := CanonicalLocalModelDeclaration(decl)
	if err != nil {
		t.Fatal(err)
	}
	second, err := CanonicalLocalModelDeclaration(decl)
	if err != nil {
		t.Fatal(err)
	}
	if string(first) != string(second) {
		t.Fatalf("canonical bytes differ: %s != %s", first, second)
	}
	independent := sha256.Sum256(first)
	if LocalModelDeclarationDigest(first) != hex.EncodeToString(independent[:]) {
		t.Fatal("digest does not match independent SHA-256")
	}
}

func TestCanonicalLocalModelDeclarationRefusals(t *testing.T) {
	valid := harnesskit.LocalModelDeclaration{Schema: LocalModelDeclarationSchema, ModelID: "m", GGUFPath: filepath.Join(t.TempDir(), "m.gguf"), GGUFSHA256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", Runtime: "llama.cpp", ContextTokens: 1}
	cases := map[string]func(*harnesskit.LocalModelDeclaration){
		"schema":        func(d *harnesskit.LocalModelDeclaration) { d.Schema = "" },
		"model":         func(d *harnesskit.LocalModelDeclaration) { d.ModelID = "" },
		"relative path": func(d *harnesskit.LocalModelDeclaration) { d.GGUFPath = "m.gguf" },
		"format":        func(d *harnesskit.LocalModelDeclaration) { d.GGUFPath = filepath.Join(t.TempDir(), "m.bin") },
		"digest":        func(d *harnesskit.LocalModelDeclaration) { d.GGUFSHA256 = "bad" },
		"runtime":       func(d *harnesskit.LocalModelDeclaration) { d.Runtime = "" },
		"context":       func(d *harnesskit.LocalModelDeclaration) { d.ContextTokens = 0 },
		"device":        func(d *harnesskit.LocalModelDeclaration) { d.RequiredDevices = []string{""} },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			d := valid
			mutate(&d)
			if _, err := CanonicalLocalModelDeclaration(d); !errors.Is(err, ErrInvalidModelDeclaration) {
				t.Fatalf("error=%v", err)
			}
		})
	}
}
