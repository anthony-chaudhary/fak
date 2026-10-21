package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime/debug"
	"slices"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

const (
	prepareTargetSchema  = "fak.dflash2.trusted-target.v1"
	preparePromptSchema  = "fak.dflash2.prompt-request.v1"
	prepareReceiptSchema = "fak.dflash2.preparation.v1"
)

var ErrPrepareDeviceBackendRegistered = errors.New("prepare: device backend already initialized")

type prepareTargetDeclaration struct {
	Schema                 string                   `json:"schema"`
	ModelID                string                   `json:"model_id"`
	Shards                 []ggufload.ShardIdentity `json:"shards"`
	ExpectedShardSetSHA256 string                   `json:"expected_shard_set_sha256"`
	ExpectedTokenizerID    string                   `json:"expected_tokenizer_id"`
	ExpectedRendererID     string                   `json:"expected_renderer_id"`
	SidecarPath            string                   `json:"sidecar_path"`
}

type preparePromptRequest struct {
	Schema          string          `json:"schema"`
	PromptID        string          `json:"prompt_id"`
	ModelID         string          `json:"model_id"`
	RendererID      string          `json:"renderer_id"`
	Messages        []agent.Message `json:"messages"`
	Tools           []agent.ToolDef `json:"tools,omitempty"`
	MaxOutputTokens int             `json:"max_output_tokens"`
}

type prepareSourceIdentity struct {
	Revision string `json:"revision"`
	Modified bool   `json:"modified"`
}

type prepareReceipt struct {
	Schema                  string                  `json:"schema"`
	Status                  string                  `json:"status"`
	Source                  prepareSourceIdentity   `json:"source"`
	PromptID                string                  `json:"prompt_id"`
	ModelID                 string                  `json:"model_id"`
	TargetDeclarationSHA256 string                  `json:"target_declaration_sha256"`
	PromptRequestSHA256     string                  `json:"prompt_request_sha256"`
	TargetSpecSHA256        string                  `json:"target_spec_sha256"`
	PromptIDsSHA256         string                  `json:"prompt_ids_sha256"`
	TargetConfigSHA256      string                  `json:"target_config_sha256"`
	TokenizerIdentitySource string                  `json:"tokenizer_identity_source"`
	Scope                   string                  `json:"scope"`
	GPUUsed                 bool                    `json:"gpu_used"`
	SidecarAuthenticated    bool                    `json:"sidecar_authenticated"`
	PhysicalEvidence        bool                    `json:"physical_evidence"`
	TargetSpec              agent.DFlash2TargetSpec `json:"target_spec"`
	Encoding                agent.PromptEncoding    `json:"encoding"`
}

type preparedDFlash2 struct {
	TargetSpec []byte
	PromptIDs  []byte
	Receipt    []byte
}

func runPrepareCLI(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("dflash2bench prepare", flag.ContinueOnError)
	fs.SetOutput(stderr)
	declarationPath := fs.String("target-declaration", "", "trusted versioned target declaration JSON")
	promptPath := fs.String("prompt-request", "", "versioned messages/tools prompt request JSON")
	targetOut := fs.String("target-spec-out", "", "new target-spec JSON path")
	promptOut := fs.String("prompt-ids-out", "", "new prompt-ID JSON path")
	receiptOut := fs.String("receipt-out", "", "new source-bound preparation receipt path")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 0 || *declarationPath == "" || *promptPath == "" || *targetOut == "" || *promptOut == "" || *receiptOut == "" {
		fmt.Fprintln(stderr, "required: --target-declaration, --prompt-request, --target-spec-out, --prompt-ids-out, --receipt-out")
		return 2
	}
	if err := distinctOutputPaths(*targetOut, *promptOut, *receiptOut); err != nil {
		fmt.Fprintln(stderr, err)
		return 2
	}
	var declaration prepareTargetDeclaration
	declarationRaw, err := readJSON(*declarationPath, &declaration)
	if err != nil {
		fmt.Fprintf(stderr, "target declaration: %v\n", err)
		return 1
	}
	var request preparePromptRequest
	requestRaw, err := readJSON(*promptPath, &request)
	if err != nil {
		fmt.Fprintf(stderr, "prompt request: %v\n", err)
		return 1
	}
	source, err := currentPrepareSourceIdentity()
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	prepared, err := prepareDFlash2(ctx, declaration, request, declarationRaw, requestRaw, source)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	outputs := []struct {
		path string
		data []byte
	}{{*targetOut, prepared.TargetSpec}, {*promptOut, prepared.PromptIDs}, {*receiptOut, prepared.Receipt}}
	created := make([]string, 0, len(outputs))
	for _, output := range outputs {
		if err := writeNewFile(output.path, output.data); err != nil {
			for _, path := range created {
				_ = os.Remove(path)
			}
			fmt.Fprintf(stderr, "write %s: %v\n", output.path, err)
			return 1
		}
		created = append(created, output.path)
	}
	_, _ = stdout.Write(prepared.Receipt)
	return 0
}

func prepareDFlash2(ctx context.Context, declaration prepareTargetDeclaration, request preparePromptRequest, declarationRaw, requestRaw []byte, source prepareSourceIdentity) (out preparedDFlash2, err error) {
	if ctx == nil {
		return out, errors.New("prepare: context required")
	}
	if compute.DeviceBackendRegistered() {
		return out, ErrPrepareDeviceBackendRegistered
	}
	if err := ctx.Err(); err != nil {
		return out, err
	}
	if err := validatePrepareInputs(declaration, request, source); err != nil {
		return out, err
	}
	spec := agent.DFlash2TargetSpec{
		Shards: slices.Clone(declaration.Shards), ExpectedShardSetSHA256: declaration.ExpectedShardSetSHA256,
		TokenizerID: declaration.ExpectedTokenizerID, RendererID: declaration.ExpectedRendererID, SidecarPath: declaration.SidecarPath,
	}
	sourceWeights, err := ggufload.OpenWeightsAuthenticated(ctx, spec.Shards, spec.ExpectedShardSetSHA256)
	if err != nil {
		return out, fmt.Errorf("prepare: authenticate target: %w", err)
	}
	defer func() {
		if closeErr := sourceWeights.Close(); closeErr != nil {
			err = errors.Join(err, fmt.Errorf("prepare: retire authenticated target: %w", closeErr))
		}
	}()
	cfg, err := sourceWeights.File.Config()
	if err != nil {
		return out, fmt.Errorf("prepare: target config: %w", err)
	}
	if !cfg.IsQwen35Hybrid() || cfg.IsDeepSeekV41() || cfg.NumExperts != 0 || cfg.NumLayers != 64 || cfg.HiddenSize != 5120 || cfg.VocabSize != 248320 || cfg.EnableResidualHook {
		return out, errors.New("prepare: exact dense Qwen3.8-27B target geometry required")
	}
	embedded, ok := sourceWeights.File.GGMLTokenizer()
	if !ok {
		return out, errors.New("prepare: authenticated embedded tokenizer required")
	}
	tok, err := tokenizer.FromGGML(embedded.Tokens, embedded.Merges, embedded.TokenTypes, embedded.Pre)
	if err != nil {
		return out, fmt.Errorf("prepare: embedded tokenizer: %w", err)
	}
	if tok.Identity() != spec.TokenizerID {
		if spec.TokenizerID != "" {
			return out, errors.New("prepare: tokenizer identity mismatch")
		}
	}
	spec.TokenizerID = tok.Identity()
	planner := agent.NewInKernelPlanner(&model.Model{Cfg: cfg}, tok, declaration.ModelID, false, nil, false)
	opts := []agent.SampleOpt{agent.WithMaxTokens(request.MaxOutputTokens)}
	if strings.HasSuffix(request.RendererID, ":no-thinking") {
		opts = append(opts, agent.WithReasoningEffort(agent.EffortTierNone))
	} else {
		opts = append(opts, agent.WithReasoningEffort(agent.EffortTierMedium))
	}
	encoding, err := planner.EncodePrompt(ctx, request.Messages, request.Tools, opts...)
	if err != nil {
		return out, fmt.Errorf("prepare: encode prompt: %w", err)
	}
	if encoding.TokenizerID != spec.TokenizerID || encoding.RendererID != spec.RendererID || encoding.ModelID != declaration.ModelID {
		return out, fmt.Errorf("prepare: authenticated identity mismatch: model=%q tokenizer=%q renderer=%q", encoding.ModelID, encoding.TokenizerID, encoding.RendererID)
	}
	if len(encoding.TokenIDs) == 0 || len(encoding.TokenIDs) > 8192 {
		return out, fmt.Errorf("prepare: prompt produced %d token IDs; require 1..8192", len(encoding.TokenIDs))
	}
	targetSpecRaw, err := marshalPreparedJSON(spec)
	if err != nil {
		return out, err
	}
	promptIDsRaw, err := marshalPreparedJSON(encoding.TokenIDs)
	if err != nil {
		return out, err
	}
	targetConfigSHA256, err := targetConfigIdentitySHA256(cfg)
	if err != nil {
		return out, err
	}
	receipt := prepareReceipt{
		Schema: prepareReceiptSchema, Status: "PREPARED", Source: source, PromptID: request.PromptID, ModelID: declaration.ModelID,
		TargetDeclarationSHA256: digestHex(declarationRaw), PromptRequestSHA256: digestHex(requestRaw),
		TargetSpecSHA256: digestHex(targetSpecRaw), PromptIDsSHA256: digestHex(promptIDsRaw), TargetConfigSHA256: targetConfigSHA256,
		TokenizerIdentitySource: "authenticated_embedded_gguf", Scope: "input-preparation-only",
		GPUUsed: false, SidecarAuthenticated: false, PhysicalEvidence: false,
		TargetSpec: spec, Encoding: encoding,
	}
	if compute.DeviceBackendRegistered() {
		return out, ErrPrepareDeviceBackendRegistered
	}
	receiptRaw, err := marshalPreparedJSON(receipt)
	if err != nil {
		return out, err
	}
	return preparedDFlash2{TargetSpec: targetSpecRaw, PromptIDs: promptIDsRaw, Receipt: receiptRaw}, nil
}

func targetConfigIdentitySHA256(cfg model.Config) (string, error) {
	raw, err := json.Marshal(struct {
		Config  model.Config `json:"config"`
		BOS     *int         `json:"bos"`
		EOS     int          `json:"eos"`
		EOSList []int        `json:"eos_list"`
	}{cfg, cfg.BOSTokenID, cfg.EOSTokenID, cfg.EOSTokenIDs})
	if err != nil {
		return "", fmt.Errorf("encode target config identity: %w", err)
	}
	return digestHex(raw), nil
}

func validatePrepareInputs(declaration prepareTargetDeclaration, request preparePromptRequest, source prepareSourceIdentity) error {
	if declaration.Schema != prepareTargetSchema || request.Schema != preparePromptSchema {
		return errors.New("prepare: unsupported target declaration or prompt request schema")
	}
	if !canonicalRevision(source.Revision) || source.Modified {
		return errors.New("prepare: a clean, source-revision-bound build is required")
	}
	if strings.TrimSpace(declaration.ModelID) == "" || len(declaration.ModelID) > 256 || declaration.ModelID != strings.TrimSpace(declaration.ModelID) || request.ModelID != declaration.ModelID {
		return errors.New("prepare: one bounded matching model_id is required")
	}
	if strings.TrimSpace(request.PromptID) == "" || len(request.PromptID) > 256 || request.PromptID != strings.TrimSpace(request.PromptID) {
		return errors.New("prepare: a bounded versioned prompt_id is required")
	}
	if len(declaration.Shards) == 0 || len(declaration.Shards) > 128 || !canonicalSHA256(declaration.ExpectedShardSetSHA256) {
		return errors.New("prepare: trusted ordered shards and canonical manifest SHA-256 are required")
	}
	for _, shard := range declaration.Shards {
		if !canonicalSHA256(shard.RawFileSHA256) {
			return errors.New("prepare: shard SHA-256 identities must be canonical lowercase")
		}
	}
	if declaration.ExpectedTokenizerID != "" && (!strings.HasPrefix(declaration.ExpectedTokenizerID, "sha256:") || !canonicalSHA256(strings.TrimPrefix(declaration.ExpectedTokenizerID, "sha256:"))) {
		return errors.New("prepare: canonical expected tokenizer identity required")
	}
	switch declaration.ExpectedRendererID {
	case "qwen-chatml-v1:thinking", "qwen-chatml-v1:no-thinking", "ornith-qwen35-chatml-v1:thinking", "ornith-qwen35-chatml-v1:no-thinking":
	default:
		return errors.New("prepare: unsupported expected renderer identity")
	}
	if request.RendererID != declaration.ExpectedRendererID {
		return errors.New("prepare: prompt renderer identity differs from trusted target declaration")
	}
	if !filepath.IsAbs(declaration.SidecarPath) || filepath.Clean(declaration.SidecarPath) != declaration.SidecarPath {
		return errors.New("prepare: absolute clean sidecar path required")
	}
	if len(request.Messages) == 0 || request.MaxOutputTokens < 1 || request.MaxOutputTokens > 4096 {
		return errors.New("prepare: messages and max_output_tokens 1..4096 are required")
	}
	return nil
}

func currentPrepareSourceIdentity() (prepareSourceIdentity, error) {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return prepareSourceIdentity{}, errors.New("prepare: Go build information unavailable")
	}
	return prepareSourceIdentityFromBuildInfo(info)
}

func prepareSourceIdentityFromBuildInfo(info *debug.BuildInfo) (prepareSourceIdentity, error) {
	if info == nil {
		return prepareSourceIdentity{}, errors.New("prepare: Go build information unavailable")
	}
	var source prepareSourceIdentity
	revisionSeen, modifiedSeen := false, false
	for _, setting := range info.Settings {
		switch setting.Key {
		case "vcs.revision":
			if revisionSeen {
				return prepareSourceIdentity{}, errors.New("prepare: duplicate source revision setting")
			}
			revisionSeen = true
			source.Revision = setting.Value
		case "vcs.modified":
			if modifiedSeen {
				return prepareSourceIdentity{}, errors.New("prepare: duplicate source modification setting")
			}
			modifiedSeen = true
			if setting.Value != "false" {
				return prepareSourceIdentity{}, errors.New("prepare: source build must carry vcs.modified=false")
			}
		}
	}
	if !revisionSeen || !modifiedSeen || !canonicalRevision(source.Revision) {
		return prepareSourceIdentity{}, errors.New("prepare: full clean source identity unavailable")
	}
	return source, nil
}

func marshalPreparedJSON(v any) ([]byte, error) {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("prepare: encode output: %w", err)
	}
	return append(b, '\n'), nil
}

func digestHex(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func canonicalSHA256(value string) bool {
	decoded, err := hex.DecodeString(value)
	return err == nil && len(decoded) == sha256.Size && value == strings.ToLower(value)
}

func canonicalRevision(value string) bool {
	decoded, err := hex.DecodeString(value)
	return err == nil && (len(decoded) == 20 || len(decoded) == sha256.Size) && value == strings.ToLower(value)
}

func distinctOutputPaths(paths ...string) error {
	seen := make(map[string]bool, len(paths))
	for _, path := range paths {
		if !filepath.IsAbs(path) || filepath.Clean(path) != path {
			return errors.New("prepare: output paths must be absolute and clean")
		}
		key := strings.ToLower(path)
		if seen[key] {
			return errors.New("prepare: output paths must be distinct")
		}
		seen[key] = true
	}
	return nil
}

func writeNewFile(path string, data []byte) error {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return err
	}
	_, writeErr := f.Write(data)
	closeErr := f.Close()
	if err := errors.Join(writeErr, closeErr); err != nil {
		_ = os.Remove(path)
		return err
	}
	return nil
}
