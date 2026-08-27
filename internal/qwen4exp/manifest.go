package qwen4exp

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strings"
)

const ManifestSchema = "fak-qwen4exp-manifest/1"

var immutableRevision = regexp.MustCompile(`^[0-9a-f]{40}$`)

type ManifestArtifact struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
	Size   int64  `json:"size"`
}

type TensorInventory struct {
	Count  int    `json:"count"`
	SHA256 string `json:"sha256"`
}

type Manifest struct {
	Schema           string             `json:"schema"`
	Repository       string             `json:"repository"`
	Revision         string             `json:"revision"`
	SourceRepository string             `json:"source_repository"`
	SourceRevision   string             `json:"source_revision"`
	Architecture     string             `json:"architecture"`
	DType            string             `json:"dtype"`
	Artifacts        []ManifestArtifact `json:"artifacts"`
	TensorInventory  TensorInventory    `json:"tensor_inventory"`
}

type ManifestBinding struct {
	ArtifactIdentity  string `json:"artifact_identity"`
	TokenizerIdentity string `json:"tokenizer_identity"`
	TemplateIdentity  string `json:"template_identity"`
	EngineIdentity    string `json:"engine_identity"`
	ManifestIdentity  string `json:"manifest_identity"`
}

func ParseManifest(raw []byte) (Manifest, error) {
	var m Manifest
	if err := json.Unmarshal(raw, &m); err != nil {
		return Manifest{}, fmt.Errorf("qwen4exp manifest: decode: %w", err)
	}
	if err := m.Validate(); err != nil {
		return Manifest{}, err
	}
	return m, nil
}

func (m Manifest) Validate() error {
	if m.Schema != ManifestSchema {
		return fmt.Errorf("qwen4exp manifest: unsupported schema %q", m.Schema)
	}
	if !immutableRevision.MatchString(m.Revision) || !immutableRevision.MatchString(m.SourceRevision) {
		return errors.New("qwen4exp manifest: repository and source revisions must be immutable 40-hex commits")
	}
	if m.Repository == "" || m.SourceRepository == "" || m.Architecture == "" || m.DType == "" {
		return errors.New("qwen4exp manifest: repository, source repository, architecture, and dtype are required")
	}
	if m.TensorInventory.Count <= 0 || !validDigest(m.TensorInventory.SHA256) {
		return errors.New("qwen4exp manifest: complete tensor inventory identity is required")
	}
	required := map[string]bool{"config.json": false, "generation_config.json": false, "tokenizer.json": false, "tokenizer_config.json": false, "chat_template.jinja": false, "LICENSE": false, "model.safetensors.index.json": false}
	seen := make(map[string]struct{}, len(m.Artifacts))
	shards := 0
	for _, a := range m.Artifacts {
		if a.Path == "" || a.Size < 0 || !validDigest(a.SHA256) {
			return fmt.Errorf("qwen4exp manifest: artifact %q lacks an exact digest or size", a.Path)
		}
		if _, ok := seen[a.Path]; ok {
			return fmt.Errorf("qwen4exp manifest: duplicate artifact %q", a.Path)
		}
		seen[a.Path] = struct{}{}
		if _, ok := required[a.Path]; ok {
			required[a.Path] = true
		}
		if strings.HasPrefix(a.Path, "model-") && strings.HasSuffix(a.Path, ".safetensors") {
			shards++
		}
	}
	for name, ok := range required {
		if !ok {
			return fmt.Errorf("qwen4exp manifest: required artifact %q is missing", name)
		}
	}
	if shards == 0 {
		return errors.New("qwen4exp manifest: no checkpoint shards")
	}
	return nil
}

func (m Manifest) Identity() (string, error) {
	if err := m.Validate(); err != nil {
		return "", err
	}
	copyM := m
	copyM.Artifacts = append([]ManifestArtifact(nil), m.Artifacts...)
	sort.Slice(copyM.Artifacts, func(i, j int) bool { return copyM.Artifacts[i].Path < copyM.Artifacts[j].Path })
	raw, _ := json.Marshal(copyM)
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:]), nil
}

func (m Manifest) Binding(engine string) (ManifestBinding, error) {
	if engine == "" {
		return ManifestBinding{}, errors.New("qwen4exp manifest: engine identity is required")
	}
	id, err := m.Identity()
	if err != nil {
		return ManifestBinding{}, err
	}
	artifact := sha256.Sum256([]byte(m.Repository + "@" + m.Revision))
	find := func(path string) string {
		for _, a := range m.Artifacts {
			if a.Path == path {
				return a.SHA256
			}
		}
		return ""
	}
	return ManifestBinding{ArtifactIdentity: hex.EncodeToString(artifact[:]), TokenizerIdentity: find("tokenizer.json"), TemplateIdentity: find("chat_template.jinja"), EngineIdentity: engine, ManifestIdentity: id}, nil
}

func validDigest(s string) bool {
	if len(s) != 64 {
		return false
	}
	_, err := hex.DecodeString(s)
	return err == nil
}
