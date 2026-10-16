package computebuild

import (
	"bytes"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const (
	vulkanQualifiedActivationChunkStem      = "v41_routed_swiglu_panel_chunk"
	vulkanQualifiedActivationChunkHeader    = "internal/compute/v41_routed_swiglu_chunk_spirv.h"
	vulkanQualifiedActivationChunkSourceSHA = "68d6def81b6e0810b963696da3bdb9259c2fa08e9480f28a7db86d74e6bcc08d"
	vulkanQualifiedActivationChunkHeaderSHA = "d386ddf385e6332cd8722880743eb988b8d57bf43a615d9babd8b6c437c7a1cf"
	vulkanQualifiedActivationChunkModuleSHA = "6bb4c0d03199686cafb9efa5faa37e4ae14b567f6f2308f687ec5e3b926ed72c"
	vulkanQualifiedActivationChunkBytes     = 6052
)

// The immutable C++ attestation is the single canonical byte owner. We do not
// add a second generated artifact or accept a new compiler's equivalent-looking
// output. Both the complete header and decoded module are pinned independently.
func decodeVulkanQualifiedActivationChunk(source, header []byte) ([]byte, error) {
	if !vulkanQualifiedHash(source, vulkanQualifiedActivationChunkSourceSHA) {
		return nil, fmt.Errorf("qualified activation-chunk GLSL source identity mismatch")
	}
	if !vulkanQualifiedHash(header, vulkanQualifiedActivationChunkHeaderSHA) {
		return nil, fmt.Errorf("qualified activation-chunk header identity mismatch")
	}
	const prefix = "static const unsigned char qualified[] = {"
	if bytes.Count(header, []byte(prefix)) != 1 {
		return nil, fmt.Errorf("qualified activation-chunk array framing mismatch")
	}
	body := bytes.SplitN(header, []byte(prefix), 2)[1]
	end := bytes.Index(body, []byte("};"))
	if end < 0 {
		return nil, fmt.Errorf("qualified activation-chunk array terminator missing")
	}
	tokens := strings.Split(string(body[:end]), ",")
	if len(tokens) != vulkanQualifiedActivationChunkBytes+1 || strings.TrimSpace(tokens[len(tokens)-1]) != "" {
		return nil, fmt.Errorf("qualified activation-chunk byte count mismatch")
	}
	out := make([]byte, vulkanQualifiedActivationChunkBytes)
	for i, t := range tokens[:len(tokens)-1] {
		t = strings.TrimSpace(t)
		if len(t) != 4 || !strings.HasPrefix(t, "0x") {
			return nil, fmt.Errorf("qualified activation-chunk invalid byte %d", i)
		}
		b, err := hex.DecodeString(t[2:])
		if err != nil {
			return nil, fmt.Errorf("qualified activation-chunk invalid byte %d", i)
		}
		out[i] = b[0]
	}
	if !vulkanQualifiedHash(out, vulkanQualifiedActivationChunkModuleSHA) {
		return nil, fmt.Errorf("qualified activation-chunk module identity mismatch")
	}
	return out, nil
}

func readVulkanQualifiedActivationChunk(root string) ([]byte, error) {
	source, err := readVulkanQualifiedFile(filepath.Join(root, "internal", "compute", "shaders", vulkanQualifiedActivationChunkStem+".comp"))
	if err != nil {
		return nil, err
	}
	header, err := readVulkanQualifiedFile(filepath.Join(root, filepath.FromSlash(vulkanQualifiedActivationChunkHeader)))
	if err != nil {
		return nil, err
	}
	return decodeVulkanQualifiedActivationChunk(source, header)
}

func writeVulkanQualifiedActivationChunk(dst string, data []byte) error {
	if len(data) != vulkanQualifiedActivationChunkBytes || !vulkanQualifiedHash(data, vulkanQualifiedActivationChunkModuleSHA) {
		return fmt.Errorf("refuse unqualified activation-chunk output")
	}
	f, err := os.CreateTemp(filepath.Dir(dst), ".qualified-activation-chunk-*")
	if err != nil {
		return err
	}
	name := f.Name()
	defer os.Remove(name)
	if _, err = f.Write(data); err != nil {
		f.Close()
		return err
	}
	if err = f.Chmod(0644); err != nil {
		f.Close()
		return err
	}
	if err = f.Close(); err != nil {
		return err
	}
	return os.Rename(name, dst)
}

func verifyVulkanQualifiedActivationChunk(root, spvRoot string) error {
	expected, err := readVulkanQualifiedActivationChunk(root)
	if err != nil {
		return err
	}
	actual, err := readVulkanQualifiedFile(filepath.Join(spvRoot, vulkanQualifiedActivationChunkStem+".spv"))
	if err != nil {
		return err
	}
	if !bytes.Equal(expected, actual) {
		return fmt.Errorf("packaged activation-chunk module differs from qualified artifact")
	}
	return nil
}

func (r *vulkanSuccessorRegistry) usesQualifiedActivationChunk() bool {
	for ; r != nil; r = r.parent {
		if r == vulkanRegistryV24 {
			return true
		}
	}
	return false
}
