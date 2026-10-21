package main

import (
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"runtime/debug"
	"slices"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// fak-test:runtime fast est=2s
func TestPrepareDFlash2AuthenticatesAndEncodesProductionFixture(t *testing.T) {
	declaration, request, declarationRaw, requestRaw, targetPath := prepareFixture(t)
	prepared, err := prepareDFlash2(t.Context(), declaration, request, declarationRaw, requestRaw, prepareSourceIdentity{Revision: strings.Repeat("a", 40)})
	if err != nil {
		t.Fatalf("prepareDFlash2: %v", err)
	}
	var receipt prepareReceipt
	if err := json.Unmarshal(prepared.Receipt, &receipt); err != nil {
		t.Fatalf("decode receipt: %v", err)
	}
	if receipt.Schema != prepareReceiptSchema || receipt.Status != "PREPARED" || receipt.PromptID != request.PromptID || receipt.ModelID != declaration.ModelID {
		t.Fatalf("receipt identity = %+v", receipt)
	}
	if receipt.TargetDeclarationSHA256 != digestHex(declarationRaw) || receipt.PromptRequestSHA256 != digestHex(requestRaw) ||
		receipt.TargetSpecSHA256 != digestHex(prepared.TargetSpec) || receipt.PromptIDsSHA256 != digestHex(prepared.PromptIDs) || receipt.TargetConfigSHA256 == "" {
		t.Fatalf("receipt digests do not bind emitted and input bytes: %+v", receipt)
	}
	if receipt.Encoding.TokenizerID == "" || receipt.Encoding.TokenizerID != receipt.TargetSpec.TokenizerID || receipt.Encoding.RendererID != declaration.ExpectedRendererID || receipt.Encoding.PromptTokens == 0 || receipt.Encoding.PromptTokens != len(receipt.Encoding.TokenIDs) {
		t.Fatalf("encoding identity = %+v", receipt.Encoding)
	}
	if receipt.TokenizerIdentitySource != "authenticated_embedded_gguf" || receipt.Scope != "input-preparation-only" || receipt.GPUUsed || receipt.SidecarAuthenticated || receipt.PhysicalEvidence {
		t.Fatalf("preparation overstated its scope: %+v", receipt)
	}
	var spec agent.DFlash2TargetSpec
	if err := json.Unmarshal(prepared.TargetSpec, &spec); err != nil {
		t.Fatalf("decode target spec: %v", err)
	}
	var ids []int
	if err := json.Unmarshal(prepared.PromptIDs, &ids); err != nil {
		t.Fatalf("decode prompt IDs: %v", err)
	}
	if spec.ExpectedShardSetSHA256 != declaration.ExpectedShardSetSHA256 || !slices.Equal(ids, receipt.Encoding.TokenIDs) {
		t.Fatalf("prepared outputs are not bound to receipt")
	}
	if err := os.Rename(targetPath, targetPath+".retired"); err != nil {
		t.Fatalf("authenticated target reader was not retired: %v", err)
	}
}

func TestPrepareDFlash2RefusesTrustAndIdentityMismatch(t *testing.T) {
	baseDeclaration, baseRequest, _, _, _ := prepareFixture(t)
	tests := []struct {
		name   string
		mutate func(*prepareTargetDeclaration, *preparePromptRequest, *prepareSourceIdentity)
	}{
		{"manifest digest", func(d *prepareTargetDeclaration, _ *preparePromptRequest, _ *prepareSourceIdentity) {
			d.ExpectedShardSetSHA256 = strings.Repeat("0", 64)
		}},
		{"observed bytes differ", func(d *prepareTargetDeclaration, _ *preparePromptRequest, _ *prepareSourceIdentity) {
			d.Shards[0].RawFileSHA256 = strings.Repeat("0", 64)
			raw, _ := json.Marshal(d.Shards)
			d.ExpectedShardSetSHA256 = digestHex(raw)
		}},
		{"tokenizer identity", func(d *prepareTargetDeclaration, _ *preparePromptRequest, _ *prepareSourceIdentity) {
			d.ExpectedTokenizerID = "sha256:" + strings.Repeat("0", 64)
		}},
		{"renderer identity", func(_ *prepareTargetDeclaration, r *preparePromptRequest, _ *prepareSourceIdentity) {
			r.RendererID = "qwen-chatml-v1:thinking"
		}},
		{"input schema", func(_ *prepareTargetDeclaration, r *preparePromptRequest, _ *prepareSourceIdentity) {
			r.Schema = "fak.dflash2.prompt-request.v2"
		}},
		{"modified source", func(_ *prepareTargetDeclaration, _ *preparePromptRequest, s *prepareSourceIdentity) {
			s.Modified = true
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			declaration := baseDeclaration
			declaration.Shards = slices.Clone(baseDeclaration.Shards)
			request := baseRequest
			source := prepareSourceIdentity{Revision: strings.Repeat("a", 40)}
			tt.mutate(&declaration, &request, &source)
			declarationRaw, _ := json.Marshal(declaration)
			requestRaw, _ := json.Marshal(request)
			if _, err := prepareDFlash2(context.Background(), declaration, request, declarationRaw, requestRaw, source); err == nil {
				t.Fatal("prepareDFlash2 accepted mismatched trust or identity")
			}
		})
	}
}

func TestPrepareDFlash2InputIsClosed(t *testing.T) {
	path := filepath.Join(t.TempDir(), "request.json")
	if err := os.WriteFile(path, []byte(`{"schema":"fak.dflash2.prompt-request.v1","unknown":true}`), 0o600); err != nil {
		t.Fatal(err)
	}
	var request preparePromptRequest
	if _, err := readJSON(path, &request); err == nil {
		t.Fatal("readJSON accepted an unknown prompt-request field")
	}
}

func TestPrepareSourceIdentityRequiresExplicitCleanVCSSettings(t *testing.T) {
	validRevision := strings.Repeat("a", 40)
	tests := []struct {
		name     string
		settings []debug.BuildSetting
		wantOK   bool
	}{
		{"clean", []debug.BuildSetting{{Key: "vcs.revision", Value: validRevision}, {Key: "vcs.modified", Value: "false"}}, true},
		{"missing modified", []debug.BuildSetting{{Key: "vcs.revision", Value: validRevision}}, false},
		{"missing revision", []debug.BuildSetting{{Key: "vcs.modified", Value: "false"}}, false},
		{"modified", []debug.BuildSetting{{Key: "vcs.revision", Value: validRevision}, {Key: "vcs.modified", Value: "true"}}, false},
		{"unknown modified", []debug.BuildSetting{{Key: "vcs.revision", Value: validRevision}, {Key: "vcs.modified", Value: "unknown"}}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := prepareSourceIdentityFromBuildInfo(&debug.BuildInfo{Settings: tt.settings})
			if tt.wantOK {
				if err != nil || got.Revision != validRevision || got.Modified {
					t.Fatalf("source identity = %+v, %v", got, err)
				}
				return
			}
			if err == nil {
				t.Fatalf("settings unexpectedly admitted: %+v", got)
			}
		})
	}
}

func TestPrepareHelpSucceeds(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runPrepareCLI(t.Context(), []string{"--help"}, &stdout, &stderr); code != 0 {
		t.Fatalf("prepare --help exit = %d, stderr=%q", code, stderr.String())
	}
}

func prepareFixture(t *testing.T) (prepareTargetDeclaration, preparePromptRequest, []byte, []byte, string) {
	t.Helper()
	compressed, err := os.ReadFile(filepath.Join("..", "..", "internal", "ggufload", "testdata", "qwen38_ud_q2kxl_header.gguf.gz"))
	if err != nil {
		t.Fatal(err)
	}
	zr, err := gzip.NewReader(bytes.NewReader(compressed))
	if err != nil {
		t.Fatal(err)
	}
	raw, err := io.ReadAll(zr)
	if closeErr := zr.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	targetPath := filepath.Join(dir, "Qwen3.8-27B-UD-Q2_K_XL.gguf")
	if err := os.WriteFile(targetPath, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	rawSum := sha256.Sum256(raw)
	shards := []ggufload.ShardIdentity{{Path: targetPath, Size: int64(len(raw)), RawFileSHA256: hex.EncodeToString(rawSum[:])}}
	manifestRaw, err := json.Marshal(shards)
	if err != nil {
		t.Fatal(err)
	}
	declaration := prepareTargetDeclaration{
		Schema: prepareTargetSchema, ModelID: "qwen3.8-27b-dflash2-target", Shards: shards,
		ExpectedShardSetSHA256: digestHex(manifestRaw), ExpectedTokenizerID: "",
		ExpectedRendererID: "ornith-qwen35-chatml-v1:no-thinking", SidecarPath: filepath.Join(dir, "model.safetensors"),
	}
	request := preparePromptRequest{
		Schema: preparePromptSchema, PromptID: "dflash2-diagnostic-smoke@v1", ModelID: declaration.ModelID,
		RendererID: declaration.ExpectedRendererID, Messages: []agent.Message{{Role: agent.RoleUser, Content: "Return one short greeting."}}, MaxOutputTokens: 64,
	}
	declarationRaw, err := json.Marshal(declaration)
	if err != nil {
		t.Fatal(err)
	}
	requestRaw, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}
	return declaration, request, declarationRaw, requestRaw, targetPath
}
