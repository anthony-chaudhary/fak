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
	vulkanQualifiedDownChunkStem      = "v41_routed_down_panel_chunk"
	vulkanQualifiedDownChunkHeader    = "internal/compute/v41_routed_down_chunk_spirv.h"
	vulkanQualifiedDownChunkSourceSHA = "df5e3ab83cce948aa7d246218a7c12eb843f72ffb28be41e00f97b4cb279d8e7"
	vulkanQualifiedDownChunkHeaderSHA = "8ede430565aa4d9c9d1ca7a8c2cad35df89b3bbc09c16808e9b6087305ac9837"
	vulkanQualifiedDownChunkModuleSHA = "a73eb3056a058e058556825254ce9de65e92eba7f3aec946718e272d978ec674"
	vulkanQualifiedDownChunkBytes     = 2768
)

// The immutable C++ attestation is the single canonical byte owner. We do not
// add a second generated artifact or accept a new compiler's equivalent-looking
// output. Both the complete header and decoded module are pinned independently.
func decodeVulkanQualifiedDownChunk(source, header []byte) ([]byte, error) {
	if !vulkanQualifiedHash(source, vulkanQualifiedDownChunkSourceSHA) {
		return nil, fmt.Errorf("qualified down-chunk GLSL source identity mismatch")
	}
	if !vulkanQualifiedHash(header, vulkanQualifiedDownChunkHeaderSHA) {
		return nil, fmt.Errorf("qualified down-chunk header identity mismatch")
	}
	const prefix = "static const unsigned char qualified[] = {"
	if bytes.Count(header, []byte(prefix)) != 1 {
		return nil, fmt.Errorf("qualified down-chunk array framing mismatch")
	}
	body := bytes.SplitN(header, []byte(prefix), 2)[1]
	end := bytes.Index(body, []byte("};"))
	if end < 0 {
		return nil, fmt.Errorf("qualified down-chunk array terminator missing")
	}
	tokens := strings.Split(string(body[:end]), ",")
	if len(tokens) != vulkanQualifiedDownChunkBytes+1 || strings.TrimSpace(tokens[len(tokens)-1]) != "" {
		return nil, fmt.Errorf("qualified down-chunk byte count mismatch")
	}
	out := make([]byte, vulkanQualifiedDownChunkBytes)
	for i, t := range tokens[:len(tokens)-1] {
		t = strings.TrimSpace(t)
		if len(t) != 4 || !strings.HasPrefix(t, "0x") {
			return nil, fmt.Errorf("qualified down-chunk invalid byte %d", i)
		}
		b, err := hex.DecodeString(t[2:])
		if err != nil {
			return nil, fmt.Errorf("qualified down-chunk invalid byte %d", i)
		}
		out[i] = b[0]
	}
	if !vulkanQualifiedHash(out, vulkanQualifiedDownChunkModuleSHA) {
		return nil, fmt.Errorf("qualified down-chunk module identity mismatch")
	}
	return out, nil
}

func readVulkanQualifiedDownChunk(root string) ([]byte, error) {
	source, err := readVulkanQualifiedFile(filepath.Join(root, "internal", "compute", "shaders", vulkanQualifiedDownChunkStem+".comp"))
	if err != nil {
		return nil, err
	}
	header, err := readVulkanQualifiedFile(filepath.Join(root, filepath.FromSlash(vulkanQualifiedDownChunkHeader)))
	if err != nil {
		return nil, err
	}
	return decodeVulkanQualifiedDownChunk(source, header)
}

func writeVulkanQualifiedDownChunk(dst string, data []byte) error {
	if len(data) != vulkanQualifiedDownChunkBytes || !vulkanQualifiedHash(data, vulkanQualifiedDownChunkModuleSHA) {
		return fmt.Errorf("refuse unqualified down-chunk output")
	}
	f, err := os.CreateTemp(filepath.Dir(dst), ".qualified-down-chunk-*")
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

func verifyVulkanQualifiedDownChunk(root, spvRoot string) error {
	expected, err := readVulkanQualifiedDownChunk(root)
	if err != nil {
		return err
	}
	actual, err := readVulkanQualifiedFile(filepath.Join(spvRoot, vulkanQualifiedDownChunkStem+".spv"))
	if err != nil {
		return err
	}
	if !bytes.Equal(expected, actual) {
		return fmt.Errorf("packaged down-chunk module differs from qualified artifact")
	}
	return nil
}

func (r *vulkanSuccessorRegistry) usesQualifiedDownChunk() bool {
	for ; r != nil; r = r.parent {
		if r == vulkanRegistryV25 {
			return true
		}
	}
	return false
}
