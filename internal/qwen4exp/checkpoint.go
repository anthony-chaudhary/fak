package qwen4exp

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// PLEDTypeMetadataKey is the checkpoint-metadata key that declares the PLE
// (per-layer-embedding) tensor's own dtype. It is the fak analog of upstream
// TensorRT-LLM's ple_embedding_dtype (tensorrt_llm/_torch/modules/qwen4_exp/
// ple.py:69): a checkpoint may quantize the PLE table independently of the rest
// of the model, so the table's dtype must be read from metadata rather than
// assumed to be the model default.
const PLEDTypeMetadataKey = "ple_embedding_dtype"

// PLEDTypeMetadata is the PLE dtype a checkpoint declares in its own metadata.
// Declared is false when the metadata omits the key, so a caller can tell a
// declared dtype from an assumed one.
type PLEDTypeMetadata struct {
	DType    string
	Declared bool
}

// PLEDTypeFromManifest reads the PLE tensor's declared dtype from raw checkpoint
// manifest metadata. It returns Declared=false (and no error) when the metadata
// omits the key, which is the only case where a caller may fall back to the
// model default.
func PLEDTypeFromManifest(raw []byte) (PLEDTypeMetadata, error) {
	var probe map[string]json.RawMessage
	if err := json.Unmarshal(raw, &probe); err != nil {
		return PLEDTypeMetadata{}, fmt.Errorf("qwen4exp manifest: decode ple dtype: %w", err)
	}
	rawDType, ok := probe[PLEDTypeMetadataKey]
	if !ok {
		return PLEDTypeMetadata{}, nil
	}
	var dtype string
	if err := json.Unmarshal(rawDType, &dtype); err != nil {
		return PLEDTypeMetadata{}, fmt.Errorf("qwen4exp manifest: %s must be a string: %w", PLEDTypeMetadataKey, err)
	}
	dtype = strings.TrimSpace(dtype)
	if dtype == "" {
		return PLEDTypeMetadata{}, nil
	}
	return PLEDTypeMetadata{DType: dtype, Declared: true}, nil
}

type ArtifactReceipt struct {
	Artifact string `json:"artifact"`
	Revision string `json:"revision"`
	Files    int    `json:"files"`
	Bytes    int64  `json:"bytes"`
	Digest   string `json:"digest"`
	Tensors  int    `json:"tensors"`
	DType    string `json:"dtype"`
	Layers   int    `json:"layers"`
	Engine   string `json:"engine"`
	Fallback string `json:"fallback"`
}

// VerifyArtifacts binds every pinned manifest entry to bytes on disk before full-checkpoint execution.
func VerifyArtifacts(root string, m Manifest) (ArtifactReceipt, error) {
	if err := m.Validate(); err != nil {
		return ArtifactReceipt{}, err
	}
	h := sha256.New()
	var total int64
	for _, a := range m.Artifacts {
		path := filepath.Join(root, filepath.FromSlash(a.Path))
		f, err := os.Open(path)
		if err != nil {
			return ArtifactReceipt{}, fmt.Errorf("qwen4exp artifact %s: %w", a.Path, err)
		}
		fh := sha256.New()
		n, copyErr := io.Copy(io.MultiWriter(h, fh), f)
		closeErr := f.Close()
		if copyErr != nil {
			return ArtifactReceipt{}, copyErr
		}
		if closeErr != nil {
			return ArtifactReceipt{}, closeErr
		}
		if n != a.Size {
			return ArtifactReceipt{}, fmt.Errorf("qwen4exp artifact %s: bytes %d want %d", a.Path, n, a.Size)
		}
		if hex.EncodeToString(fh.Sum(nil)) != a.SHA256 {
			return ArtifactReceipt{}, fmt.Errorf("qwen4exp artifact %s: digest mismatch", a.Path)
		}
		total += n
	}
	if total != manifestBytes(m) {
		return ArtifactReceipt{}, errors.New("qwen4exp artifact total mismatch")
	}
	return ArtifactReceipt{Artifact: m.Repository, Revision: m.Revision, Files: len(m.Artifacts), Bytes: total, Digest: hex.EncodeToString(h.Sum(nil)), Tensors: m.TensorInventory.Count, DType: m.DType, Layers: 48, Engine: "fak-native", Fallback: "none"}, nil
}
func (r ArtifactReceipt) ValidateForExecution() error {
	if r.Artifact == "" || r.Revision == "" || r.Files <= 0 || r.Bytes <= 0 || r.Digest == "" || r.Tensors <= 0 || r.DType != "bfloat16" || r.Layers != 48 {
		return errors.New("qwen4exp: incomplete exact-checkpoint receipt")
	}
	if r.Engine != "fak-native" || r.Fallback != "none" {
		return errors.New("qwen4exp: fallback receipt")
	}
	return nil
}

func manifestBytes(m Manifest) int64 {
	var n int64
	for _, a := range m.Artifacts {
		n += a.Size
	}
	return n
}
