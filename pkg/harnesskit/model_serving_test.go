package harnesskit

import "testing"

func TestLocalModelDeclarationIsDataOnly(t *testing.T) {
	decl := LocalModelDeclaration{Schema: "fak.harness.local-model-declaration.v1", ModelID: "m", GGUFPath: `C:\models\m.gguf`, GGUFSHA256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", Runtime: "llama.cpp", ContextTokens: 4096}
	if decl.GGUFPath == "" || decl.Runtime == "" {
		t.Fatal("declaration lost authored fields")
	}
}
