package qwen38quantrun

import (
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

func validTestPromptPacket() PromptTokenPacket {
	return PromptTokenPacket{
		Schema:            PromptTokenPacketSchema,
		PacketID:          "amd-rx7600-trial-001",
		ArtifactSHA256:    strings.Repeat("ab", 32),
		TokenizerIdentity: "synthetic-test-tokenizer",
		TokenizerDigest:   strings.Repeat("cd", 32),
		TemplateDigest:    strings.Repeat("ef", 32),
		PromptTokenIDs:    []int{151644, 872, 198, 2610, 525, 264, 10925, 13, 151645, 198},
		StopTokens:        []string{"<|im_end|>", "<|endoftext|>"},
		StopTokenIDs:      []int{151645, 151643},
		ContextBudget: ContextBudget{
			ContextTokens:      2048,
			ContextBudgetBytes: 4 << 30,
		},
		GenerationControls: GenerationControls{
			Temperature:     0.0,
			TopP:            1.0,
			TopK:            1,
			MaxOutputTokens: 128,
			StopTokens:      []string{"<|im_end|>", "<|endoftext|>"},
			StopTokenIDs:    []int{151645, 151643},
		},
	}
}

func TestPromptPacketCompatibilityBaselines(t *testing.T) {
	current, err := FreezePromptPacket(validTestPromptPacket())
	if err != nil {
		t.Fatal(err)
	}
	if got, want := current.PacketDigest, "1e21b686d2db6699aa1f3cfe6678c4e1564f048354f5fc0979a0b8b4fbb4b61c"; got != want {
		t.Fatalf("false/omitted v2 digest changed: got %s want %s", got, want)
	}
	currentJSON, err := ExportPromptPacket(current)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(currentJSON, []byte("ignore_eos")) {
		t.Fatalf("false ignore_eos changed historical v2 bytes: %s", currentJSON)
	}

	legacy := validTestPromptPacket()
	legacy.Schema = promptTokenPacketLegacySchema
	legacy.TemplateDigest = ""
	legacyTokens := slices.Clone(legacy.PromptTokenIDs)
	digest, err := ComputePromptPacketDigest(legacy)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := digest, "a56bce496206b2203e6bbdc201526c675f80cbb06720a1ae78aafe656ec53c20"; got != want {
		t.Fatalf("historical v1 digest changed: got %s want %s", got, want)
	}
	if !slices.Equal(legacy.PromptTokenIDs, legacyTokens) {
		t.Fatalf("historical v1 token IDs changed: got %v want %v", legacy.PromptTokenIDs, legacyTokens)
	}

	ignoreEOS := validTestPromptPacket()
	ignoreEOS.GenerationControls.IgnoreEOS = true
	ignoreEOS, err = FreezePromptPacket(ignoreEOS)
	if err != nil {
		t.Fatal(err)
	}
	if ignoreEOS.PacketDigest == current.PacketDigest {
		t.Fatal("ignore_eos=true did not change the packet digest")
	}
	ignoreJSON, err := ExportPromptPacket(ignoreEOS)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(ignoreJSON, []byte(`"ignore_eos": true`)) {
		t.Fatalf("ignore_eos=true absent from exported packet: %s", ignoreJSON)
	}
	imported, err := ImportPromptPacket(ignoreJSON)
	if err != nil {
		t.Fatal(err)
	}
	if !imported.GenerationControls.IgnoreEOS || imported.PacketDigest != ignoreEOS.PacketDigest {
		t.Fatalf("ignore_eos round trip lost its digest-bound value: %+v", imported.GenerationControls)
	}

	legacy.GenerationControls.IgnoreEOS = true
	legacy.PacketDigest, err = ComputePromptPacketDigest(legacy)
	if err != nil {
		t.Fatal(err)
	}
	if err := VerifyPromptPacket(legacy); err == nil || !strings.Contains(err.Error(), "legacy prompt packet") {
		t.Fatalf("legacy v1 packet claimed ignore_eos: %v", err)
	}
}

func TestDerivePromptPacketGGUFIdentityPinnedQwen38Header(t *testing.T) {
	compressed, err := os.ReadFile(filepath.Join("..", "ggufload", "testdata", "qwen38_ud_q2kxl_header.gguf.gz"))
	if err != nil {
		t.Fatal(err)
	}
	zr, err := gzip.NewReader(bytes.NewReader(compressed))
	if err != nil {
		t.Fatal(err)
	}
	raw, err := io.ReadAll(io.LimitReader(zr, 16<<20))
	closeErr := zr.Close()
	if err != nil {
		t.Fatal(err)
	}
	if closeErr != nil {
		t.Fatal(closeErr)
	}
	const headerSHA256 = "1fe82fda85430cca654a156e9ec2915baf460752197013563b426db2581dcc0f"
	if len(raw) != 10996640 || fmt.Sprintf("%x", sha256.Sum256(raw)) != headerSHA256 {
		t.Fatalf("pinned canonicalizer fixture identity mismatch: bytes=%d sha256=%x", len(raw), sha256.Sum256(raw))
	}
	headerPath := filepath.Join(t.TempDir(), "qwen38-canonicalizer-fixture.gguf")
	if err := os.WriteFile(headerPath, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	identity, err := DerivePromptPacketGGUFIdentity(headerPath)
	if err != nil {
		t.Fatal(err)
	}
	if identity.TokenizerIdentity != GGUFTokenizerIdentity || identity.TokenizerDigest != "839c662c4a47759df9150bd939b382767b1e36bc2372740ad6d02495a63fa5a0" || identity.TemplateDigest != "87049d017c4eee304541572ddfee78756784389fac4808d02039215f062d31d1" || identity.Architecture != "qwen35" || identity.TokenizerPre != "qwen35" || identity.VocabSize != 248320 {
		t.Fatalf("pinned canonicalizer identity = %+v", identity)
	}

	synthetic, err := FreezePromptPacket(validTestPromptPacket())
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidatePromptPacketGGUFIdentity(synthetic, identity); err == nil {
		t.Fatal("synthetic generic packet received exact GGUF comparison eligibility")
	}
	eligible := validTestPromptPacket()
	eligible.TokenizerIdentity = identity.TokenizerIdentity
	eligible.TokenizerDigest = identity.TokenizerDigest
	eligible.TemplateDigest = identity.TemplateDigest
	eligible, err = FreezePromptPacket(eligible)
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidatePromptPacketGGUFIdentity(eligible, identity); err != nil {
		t.Fatalf("exactly bound packet rejected: %v", err)
	}
	outOfVocab := validTestPromptPacket()
	outOfVocab.TokenizerIdentity = identity.TokenizerIdentity
	outOfVocab.TokenizerDigest = identity.TokenizerDigest
	outOfVocab.TemplateDigest = identity.TemplateDigest
	outOfVocab.PromptTokenIDs = []int{identity.VocabSize}
	outOfVocab, err = FreezePromptPacket(outOfVocab)
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidatePromptPacketGGUFIdentity(outOfVocab, identity); err == nil || !strings.Contains(err.Error(), "outside vocabulary") {
		t.Fatalf("out-of-vocabulary prompt token was not refused: %v", err)
	}
	outOfVocab = validTestPromptPacket()
	outOfVocab.TokenizerIdentity = identity.TokenizerIdentity
	outOfVocab.TokenizerDigest = identity.TokenizerDigest
	outOfVocab.TemplateDigest = identity.TemplateDigest
	outOfVocab.StopTokenIDs = []int{identity.VocabSize}
	outOfVocab.GenerationControls.StopTokenIDs = []int{identity.VocabSize}
	outOfVocab, err = FreezePromptPacket(outOfVocab)
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidatePromptPacketGGUFIdentity(outOfVocab, identity); err == nil || !strings.Contains(err.Error(), "outside vocabulary") {
		t.Fatalf("out-of-vocabulary mirrored stop token was not refused: %v", err)
	}

	gg, err := ggufload.Read(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}

	mutatedTokenizer := cloneGGUFHeader(gg)
	merges := mutatedTokenizer.Metadata["tokenizer.ggml.merges"]
	mergeItems := slices.Clone(merges.Value.([]ggufload.Value))
	mergeItems[0].Value = mergeItems[0].Value.(string) + "x"
	merges.Value = mergeItems
	mutatedTokenizer.Metadata["tokenizer.ggml.merges"] = merges
	mutated, err := derivePromptPacketGGUFIdentity(mutatedTokenizer)
	if err != nil {
		t.Fatal(err)
	}
	if mutated.TokenizerDigest == identity.TokenizerDigest || mutated.TemplateDigest != identity.TemplateDigest {
		t.Fatalf("tokenizer-only mutation was not isolated: original=%+v mutated=%+v", identity, mutated)
	}

	mutatedTemplate := cloneGGUFHeader(gg)
	chatTemplate := mutatedTemplate.Metadata["tokenizer.chat_template"]
	chatTemplate.Value = chatTemplate.Value.(string) + "\n"
	mutatedTemplate.Metadata["tokenizer.chat_template"] = chatTemplate
	mutated, err = derivePromptPacketGGUFIdentity(mutatedTemplate)
	if err != nil {
		t.Fatal(err)
	}
	if mutated.TemplateDigest == identity.TemplateDigest || mutated.TokenizerDigest != identity.TokenizerDigest {
		t.Fatalf("template-only mutation was not isolated: original=%+v mutated=%+v", identity, mutated)
	}

	for _, missing := range []string{"tokenizer.chat_template", "tokenizer.ggml.pre"} {
		incomplete := cloneGGUFHeader(gg)
		delete(incomplete.Metadata, missing)
		if _, err := derivePromptPacketGGUFIdentity(incomplete); err == nil {
			t.Fatalf("missing %s was accepted", missing)
		}
	}
}

func cloneGGUFHeader(src *ggufload.File) *ggufload.File {
	clone := *src
	clone.Metadata = make(map[string]ggufload.Value, len(src.Metadata))
	for key, value := range src.Metadata {
		clone.Metadata[key] = value
	}
	return &clone
}

func TestPromptPacketSerializationAndDeserialization(t *testing.T) {
	orig := validTestPromptPacket()
	frozen, err := FreezePromptPacket(orig)
	if err != nil {
		t.Fatalf("FreezePromptPacket failed: %v", err)
	}
	if frozen.PacketDigest == "" {
		t.Fatal("expected non-empty PacketDigest on frozen packet")
	}

	exported, err := ExportPromptPacket(frozen)
	if err != nil {
		t.Fatalf("ExportPromptPacket failed: %v", err)
	}

	imported, err := ImportPromptPacket(exported)
	if err != nil {
		t.Fatalf("ImportPromptPacket failed: %v", err)
	}

	if imported.PacketDigest != frozen.PacketDigest {
		t.Fatalf("imported packet digest = %q, want %q", imported.PacketDigest, frozen.PacketDigest)
	}
	if imported.ArtifactSHA256 != frozen.ArtifactSHA256 {
		t.Fatalf("imported artifact = %q, want %q", imported.ArtifactSHA256, frozen.ArtifactSHA256)
	}
	if imported.TokenizerDigest != frozen.TokenizerDigest {
		t.Fatalf("imported tokenizer digest = %q, want %q", imported.TokenizerDigest, frozen.TokenizerDigest)
	}
	if imported.TemplateDigest != frozen.TemplateDigest {
		t.Fatalf("imported template digest = %q, want %q", imported.TemplateDigest, frozen.TemplateDigest)
	}
	if !slices.Equal(imported.PromptTokenIDs, frozen.PromptTokenIDs) {
		t.Fatalf("imported prompt token IDs mismatch: got %v, want %v", imported.PromptTokenIDs, frozen.PromptTokenIDs)
	}
	if !slices.Equal(imported.StopTokens, frozen.StopTokens) {
		t.Fatalf("imported stop tokens mismatch: got %v, want %v", imported.StopTokens, frozen.StopTokens)
	}
	if imported.ContextBudget != frozen.ContextBudget {
		t.Fatalf("imported context budget = %+v, want %+v", imported.ContextBudget, frozen.ContextBudget)
	}
	if imported.GenerationControls.Temperature != frozen.GenerationControls.Temperature ||
		imported.GenerationControls.TopP != frozen.GenerationControls.TopP ||
		imported.GenerationControls.MaxOutputTokens != frozen.GenerationControls.MaxOutputTokens {
		t.Fatalf("imported generation controls = %+v, want %+v", imported.GenerationControls, frozen.GenerationControls)
	}

	// Test file export and import
	dir := t.TempDir()
	filePath := filepath.Join(dir, "prompt_packet.json")
	if err := WritePromptPacketFile(filePath, frozen); err != nil {
		t.Fatalf("WritePromptPacketFile failed: %v", err)
	}

	fromFile, err := ReadPromptPacketFile(filePath)
	if err != nil {
		t.Fatalf("ReadPromptPacketFile failed: %v", err)
	}
	if fromFile.PacketDigest != frozen.PacketDigest {
		t.Fatalf("file packet digest = %q, want %q", fromFile.PacketDigest, frozen.PacketDigest)
	}
}

func TestPromptPacketHashingAndTamperingDetection(t *testing.T) {
	orig := validTestPromptPacket()
	frozen, err := FreezePromptPacket(orig)
	if err != nil {
		t.Fatalf("FreezePromptPacket failed: %v", err)
	}

	// Verify digest is deterministic across multiple calls
	digest1, err := ComputePromptPacketDigest(frozen)
	if err != nil {
		t.Fatal(err)
	}
	digest2, err := ComputePromptPacketDigest(orig)
	if err != nil {
		t.Fatal(err)
	}
	if digest1 != digest2 || digest1 != frozen.PacketDigest {
		t.Fatalf("digest non-deterministic: d1=%s d2=%s frozen=%s", digest1, digest2, frozen.PacketDigest)
	}

	// Tampering test cases: mutating any field must cause VerifyPromptPacket to fail
	cases := []struct {
		name   string
		mutate func(p *PromptTokenPacket)
	}{
		{
			name: "tamper token IDs",
			mutate: func(p *PromptTokenPacket) {
				p.PromptTokenIDs[0]++
			},
		},
		{
			name: "tamper artifact SHA-256",
			mutate: func(p *PromptTokenPacket) {
				p.ArtifactSHA256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
			},
		},
		{
			name: "tamper tokenizer digest",
			mutate: func(p *PromptTokenPacket) {
				p.TokenizerDigest = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
			},
		},
		{
			name: "tamper template digest",
			mutate: func(p *PromptTokenPacket) {
				p.TemplateDigest = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
			},
		},
		{
			name: "tamper context tokens budget",
			mutate: func(p *PromptTokenPacket) {
				p.ContextBudget.ContextTokens += 100
			},
		},
		{
			name: "tamper context budget bytes",
			mutate: func(p *PromptTokenPacket) {
				p.ContextBudget.ContextBudgetBytes += 1024
			},
		},
		{
			name: "tamper temperature",
			mutate: func(p *PromptTokenPacket) {
				p.GenerationControls.Temperature = 0.7
			},
		},
		{
			name: "tamper top_p",
			mutate: func(p *PromptTokenPacket) {
				p.GenerationControls.TopP = 0.9
			},
		},
		{
			name: "tamper max output tokens",
			mutate: func(p *PromptTokenPacket) {
				p.GenerationControls.MaxOutputTokens = 256
			},
		},
		{
			name: "tamper ignore EOS",
			mutate: func(p *PromptTokenPacket) {
				p.GenerationControls.IgnoreEOS = !p.GenerationControls.IgnoreEOS
			},
		},
		{
			name: "tamper stop tokens",
			mutate: func(p *PromptTokenPacket) {
				p.StopTokens = append(p.StopTokens, "<|extra_stop|>")
			},
		},
		{
			name: "tamper stop token IDs",
			mutate: func(p *PromptTokenPacket) {
				p.StopTokenIDs = append(p.StopTokenIDs, 99999)
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tampered := frozen
			tampered.PromptTokenIDs = slices.Clone(frozen.PromptTokenIDs)
			tampered.StopTokens = slices.Clone(frozen.StopTokens)
			tampered.StopTokenIDs = slices.Clone(frozen.StopTokenIDs)
			tc.mutate(&tampered)

			err := VerifyPromptPacket(tampered)
			if err == nil {
				t.Fatalf("expected tampering error for %s, got nil", tc.name)
			}
			if !strings.Contains(err.Error(), "tamper") && !strings.Contains(err.Error(), "mismatch") {
				t.Fatalf("expected tamper/mismatch error message, got: %v", err)
			}
		})
	}
}

func TestPromptPacketFieldValidation(t *testing.T) {
	valid := validTestPromptPacket()

	t.Run("invalid schema", func(t *testing.T) {
		p := valid
		p.Schema = "wrong.schema"
		if _, err := FreezePromptPacket(p); err == nil {
			t.Fatal("expected error on invalid schema")
		}
	})

	t.Run("invalid artifact sha", func(t *testing.T) {
		p := valid
		p.ArtifactSHA256 = "not-a-valid-sha"
		if _, err := FreezePromptPacket(p); err == nil {
			t.Fatal("expected error on invalid artifact sha")
		}
	})

	t.Run("empty-input artifact sha", func(t *testing.T) {
		p := valid
		p.ArtifactSHA256 = emptySHA256
		if _, err := FreezePromptPacket(p); err == nil {
			t.Fatal("expected error on SHA-256 of empty artifact input")
		}
	})

	t.Run("uppercase artifact sha", func(t *testing.T) {
		p := valid
		p.ArtifactSHA256 = strings.ToUpper(p.ArtifactSHA256)
		if _, err := FreezePromptPacket(p); err == nil {
			t.Fatal("expected error on non-canonical uppercase artifact SHA-256")
		}
	})

	t.Run("empty tokenizer digest", func(t *testing.T) {
		p := valid
		p.TokenizerDigest = ""
		if _, err := FreezePromptPacket(p); err == nil {
			t.Fatal("expected error on empty tokenizer digest")
		}
	})

	t.Run("empty tokenizer identity", func(t *testing.T) {
		p := valid
		p.TokenizerIdentity = ""
		if _, err := FreezePromptPacket(p); err == nil {
			t.Fatal("expected error on empty tokenizer identity")
		}
	})

	t.Run("empty-input tokenizer digest", func(t *testing.T) {
		p := valid
		p.TokenizerDigest = emptySHA256
		if _, err := FreezePromptPacket(p); err == nil {
			t.Fatal("expected error on SHA-256 of empty tokenizer input")
		}
	})

	t.Run("uppercase tokenizer digest", func(t *testing.T) {
		p := valid
		p.TokenizerDigest = strings.ToUpper(p.TokenizerDigest)
		if _, err := FreezePromptPacket(p); err == nil {
			t.Fatal("expected error on non-canonical uppercase tokenizer digest")
		}
	})

	t.Run("malformed tokenizer digest", func(t *testing.T) {
		p := valid
		p.TokenizerDigest = "not-a-sha256"
		if _, err := FreezePromptPacket(p); err == nil {
			t.Fatal("expected error on malformed tokenizer digest")
		}
	})

	t.Run("empty template digest", func(t *testing.T) {
		p := valid
		p.TemplateDigest = ""
		if _, err := FreezePromptPacket(p); err == nil {
			t.Fatal("expected error on empty template digest")
		}
	})

	t.Run("empty-input template digest", func(t *testing.T) {
		p := valid
		p.TemplateDigest = emptySHA256
		if _, err := FreezePromptPacket(p); err == nil {
			t.Fatal("expected error on SHA-256 of empty template input")
		}
	})

	t.Run("empty prompt tokens", func(t *testing.T) {
		p := valid
		p.PromptTokenIDs = nil
		if _, err := FreezePromptPacket(p); err == nil {
			t.Fatal("expected error on empty prompt token IDs")
		}
	})

	t.Run("negative token ID", func(t *testing.T) {
		p := valid
		p.PromptTokenIDs = []int{10, -1, 20}
		if _, err := FreezePromptPacket(p); err == nil {
			t.Fatal("expected error on negative token ID")
		}
	})

	t.Run("tokens exceed context budget", func(t *testing.T) {
		p := valid
		p.ContextBudget.ContextTokens = 2
		p.PromptTokenIDs = []int{1, 2, 3}
		if _, err := FreezePromptPacket(p); err == nil {
			t.Fatal("expected error when prompt exceeds context tokens")
		}
	})

	t.Run("negative temperature", func(t *testing.T) {
		p := valid
		p.GenerationControls.Temperature = -0.5
		if _, err := FreezePromptPacket(p); err == nil {
			t.Fatal("expected error on negative temperature")
		}
	})

	t.Run("top_p out of bounds", func(t *testing.T) {
		p := valid
		p.GenerationControls.TopP = 1.5
		if _, err := FreezePromptPacket(p); err == nil {
			t.Fatal("expected error on top_p > 1.0")
		}
	})

	t.Run("unfrozen packet verification fails", func(t *testing.T) {
		p := valid
		p.PacketDigest = ""
		if err := VerifyPromptPacket(p); err == nil {
			t.Fatal("expected error verifying unfrozen packet without digest")
		}
	})
}

func TestPromptPacketArmAttestationRejection(t *testing.T) {
	orig := validTestPromptPacket()
	frozenCandidate, err := FreezePromptPacket(orig)
	if err != nil {
		t.Fatal(err)
	}
	frozenComparator, err := FreezePromptPacket(orig)
	if err != nil {
		t.Fatal(err)
	}

	// Matched packets must pass attestation
	if err := ValidatePromptPacketAttestation(frozenCandidate, frozenComparator); err != nil {
		t.Fatalf("identical packets must pass attestation: %v", err)
	}

	// 1. Token IDs differ between arms
	t.Run("token IDs differ", func(t *testing.T) {
		mismatched := orig
		mismatched.PromptTokenIDs = slices.Clone(orig.PromptTokenIDs)
		mismatched.PromptTokenIDs[0] = 999
		frozenMismatched, err := FreezePromptPacket(mismatched)
		if err != nil {
			t.Fatal(err)
		}
		err = ValidatePromptPacketAttestation(frozenCandidate, frozenMismatched)
		if err == nil || !strings.Contains(err.Error(), "token IDs mismatch") {
			t.Fatalf("expected token IDs mismatch, got: %v", err)
		}
	})

	// 2. Tokenizer digest differs between arms
	t.Run("tokenizer digest differs", func(t *testing.T) {
		mismatched := orig
		mismatched.TokenizerDigest = "1111111111111111111111111111111111111111111111111111111111111111"
		frozenMismatched, err := FreezePromptPacket(mismatched)
		if err != nil {
			t.Fatal(err)
		}
		err = ValidatePromptPacketAttestation(frozenCandidate, frozenMismatched)
		if err == nil || !strings.Contains(err.Error(), "tokenizer digest mismatch") {
			t.Fatalf("expected tokenizer digest mismatch, got: %v", err)
		}
	})

	// 3. Artifact SHA differs between arms
	t.Run("artifact SHA differs", func(t *testing.T) {
		mismatched := orig
		mismatched.ArtifactSHA256 = "2222222222222222222222222222222222222222222222222222222222222222"
		frozenMismatched, err := FreezePromptPacket(mismatched)
		if err != nil {
			t.Fatal(err)
		}
		err = ValidatePromptPacketAttestation(frozenCandidate, frozenMismatched)
		if err == nil || !strings.Contains(err.Error(), "artifact SHA-256 mismatch") {
			t.Fatalf("expected artifact SHA mismatch, got: %v", err)
		}
	})

	// A template change must never silently retain comparison eligibility even
	// when token IDs happen to remain the same.
	t.Run("template digest differs with identical tokens", func(t *testing.T) {
		mismatched := orig
		mismatched.TemplateDigest = "5555555555555555555555555555555555555555555555555555555555555555"
		frozenMismatched, err := FreezePromptPacket(mismatched)
		if err != nil {
			t.Fatal(err)
		}
		err = ValidatePromptPacketAttestation(frozenCandidate, frozenMismatched)
		if err == nil || !strings.Contains(err.Error(), "template digest mismatch") {
			t.Fatalf("expected template digest mismatch, got: %v", err)
		}
	})

	// 4. Stop tokens differ between arms
	t.Run("stop tokens differ", func(t *testing.T) {
		mismatched := orig
		mismatched.StopTokens = []string{"<|different_stop|>"}
		mismatched.GenerationControls.StopTokens = mismatched.StopTokens
		frozenMismatched, err := FreezePromptPacket(mismatched)
		if err != nil {
			t.Fatal(err)
		}
		err = ValidatePromptPacketAttestation(frozenCandidate, frozenMismatched)
		if err == nil || !strings.Contains(err.Error(), "stop tokens mismatch") {
			t.Fatalf("expected stop tokens mismatch, got: %v", err)
		}
	})

	t.Run("ignore EOS differs", func(t *testing.T) {
		mismatched := orig
		mismatched.GenerationControls.IgnoreEOS = true
		frozenMismatched, err := FreezePromptPacket(mismatched)
		if err != nil {
			t.Fatal(err)
		}
		err = ValidatePromptPacketAttestation(frozenCandidate, frozenMismatched)
		if err == nil || !strings.Contains(err.Error(), "generation controls mismatch") {
			t.Fatalf("expected ignore-EOS generation mismatch, got: %v", err)
		}
	})
}

func TestPromptPacketArmReceiptAttestation(t *testing.T) {
	orig := validTestPromptPacket()
	frozenPacket, err := FreezePromptPacket(orig)
	if err != nil {
		t.Fatal(err)
	}

	input := validAMDScoreboardInput()
	cand := input.Candidate
	cand.TokenizerDigest = frozenPacket.TokenizerDigest
	cand.TemplateDigest = frozenPacket.TemplateDigest
	cand.PromptPacketDigest = frozenPacket.PacketDigest
	cand.PromptTokenIDs = slices.Clone(frozenPacket.PromptTokenIDs)
	cand.ArtifactSHA256 = frozenPacket.ArtifactSHA256
	cand.PromptPacket = &frozenPacket
	cand.StopTokens = slices.Clone(frozenPacket.StopTokens)
	cand.StopTokenIDs = slices.Clone(frozenPacket.StopTokenIDs)
	cand.ContextTokens = frozenPacket.ContextBudget.ContextTokens
	cand.ContextBudgetBytes = frozenPacket.ContextBudget.ContextBudgetBytes
	cand.TopP = frozenPacket.GenerationControls.TopP
	cand.TopK = frozenPacket.GenerationControls.TopK
	cand.DecodeTokens = frozenPacket.GenerationControls.MaxOutputTokens

	ref := input.Reference
	ref.TokenizerDigest = frozenPacket.TokenizerDigest
	ref.TemplateDigest = frozenPacket.TemplateDigest
	ref.PromptPacketDigest = frozenPacket.PacketDigest
	ref.PromptTokenIDs = slices.Clone(frozenPacket.PromptTokenIDs)
	ref.ArtifactSHA256 = frozenPacket.ArtifactSHA256
	ref.PromptPacket = &frozenPacket
	ref.StopTokens = slices.Clone(frozenPacket.StopTokens)
	ref.StopTokenIDs = slices.Clone(frozenPacket.StopTokenIDs)
	ref.ContextTokens = frozenPacket.ContextBudget.ContextTokens
	ref.ContextBudgetBytes = frozenPacket.ContextBudget.ContextBudgetBytes
	ref.TopP = frozenPacket.GenerationControls.TopP
	ref.TopK = frozenPacket.GenerationControls.TopK
	ref.DecodeTokens = frozenPacket.GenerationControls.MaxOutputTokens

	// Matched receipts pass
	if err := ValidateArmPromptPacketAttestation(cand, ref); err != nil {
		t.Fatalf("matched arm receipts must pass attestation: %v", err)
	}

	// 1. Candidate not fak-native
	t.Run("candidate not fak-native", func(t *testing.T) {
		badCand := cand
		badCand.Engine = "llama.cpp"
		if err := ValidateArmPromptPacketAttestation(badCand, ref); err == nil {
			t.Fatal("expected error for non fak-native candidate")
		}
	})

	// 2. Tokenizer digest mismatch
	t.Run("receipt tokenizer digest mismatch", func(t *testing.T) {
		badRef := ref
		badRef.TokenizerDigest = "3333333333333333333333333333333333333333333333333333333333333333"
		if err := ValidateArmPromptPacketAttestation(cand, badRef); err == nil {
			t.Fatal("expected error on tokenizer digest mismatch")
		}
	})

	// 3. Prompt packet digest mismatch
	t.Run("receipt prompt packet digest mismatch", func(t *testing.T) {
		badRef := ref
		badRef.PromptPacketDigest = "4444444444444444444444444444444444444444444444444444444444444444"
		if err := ValidateArmPromptPacketAttestation(cand, badRef); err == nil {
			t.Fatal("expected error on prompt packet digest mismatch")
		}
	})

	t.Run("outer receipt identity cannot disagree with embedded packet", func(t *testing.T) {
		badRef := ref
		badRef.TemplateDigest = "6666666666666666666666666666666666666666666666666666666666666666"
		if err := ValidateArmPromptPacketAttestation(cand, badRef); err == nil || !strings.Contains(err.Error(), "does not bind embedded packet") {
			t.Fatalf("expected outer receipt binding error, got %v", err)
		}
	})

	t.Run("outer receipt ignore EOS cannot disagree with embedded packet", func(t *testing.T) {
		badRef := ref
		badRef.IgnoreEOS = !frozenPacket.GenerationControls.IgnoreEOS
		if err := ValidateArmPromptPacketAttestation(cand, badRef); err == nil || !strings.Contains(err.Error(), "ignore-EOS policy") {
			t.Fatalf("expected outer receipt ignore-EOS binding error, got %v", err)
		}
	})
}

func TestLegacyPromptPacketIsReadableButNotComparisonEligible(t *testing.T) {
	legacy := validTestPromptPacket()
	legacy.Schema = promptTokenPacketLegacySchema
	legacy.TemplateDigest = ""
	digest, err := ComputePromptPacketDigest(legacy)
	if err != nil {
		t.Fatal(err)
	}
	legacy.PacketDigest = digest
	raw, err := ExportPromptPacket(legacy)
	if err != nil {
		t.Fatalf("historical packet should remain exportable: %v", err)
	}
	imported, err := ImportPromptPacket(raw)
	if err != nil {
		t.Fatalf("historical packet should remain readable: %v", err)
	}
	if err := ValidatePromptPacketAttestation(imported, imported); err == nil || !strings.Contains(err.Error(), "not eligible for comparison credit") {
		t.Fatalf("historical packet must be non-credit, got %v", err)
	}
}
