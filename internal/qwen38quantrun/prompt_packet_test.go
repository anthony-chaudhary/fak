package qwen38quantrun

import (
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func validTestPromptPacket() PromptTokenPacket {
	return PromptTokenPacket{
		Schema:            PromptTokenPacketSchema,
		PacketID:          "amd-rx7600-trial-001",
		ArtifactSHA256:    "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169",
		TokenizerIdentity: "Qwen/Qwen2.5-Coder-7B-Instruct",
		TokenizerDigest:   "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
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

	t.Run("empty tokenizer digest", func(t *testing.T) {
		p := valid
		p.TokenizerDigest = ""
		if _, err := FreezePromptPacket(p); err == nil {
			t.Fatal("expected error on empty tokenizer digest")
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
	cand.PromptPacketDigest = frozenPacket.PacketDigest
	cand.PromptTokenIDs = slices.Clone(frozenPacket.PromptTokenIDs)
	cand.ArtifactSHA256 = frozenPacket.ArtifactSHA256
	cand.PromptPacket = &frozenPacket

	ref := input.Reference
	ref.TokenizerDigest = frozenPacket.TokenizerDigest
	ref.PromptPacketDigest = frozenPacket.PacketDigest
	ref.PromptTokenIDs = slices.Clone(frozenPacket.PromptTokenIDs)
	ref.ArtifactSHA256 = frozenPacket.ArtifactSHA256
	ref.PromptPacket = &frozenPacket

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
}
