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
	vulkanQualifiedQ2TokenPairStem      = "v41_q2k_token_pair"
	vulkanQualifiedQ2TokenPairHeader    = "internal/compute/v41_q2k_token_pair_spirv.h"
	vulkanQualifiedQ2TokenPairSourceSHA = "e1a3fd8e3c1f8ce268e29e0f75a89ee2abcc4ecddce2d0b76f2854abd1b42442"
	vulkanQualifiedQ2TokenPairHeaderSHA = "f9e11a9f4d35830356a070800fb8ac87c0c7b35a30e2d79cc40e4aef968d7e12"
	vulkanQualifiedQ2TokenPairModuleSHA = "106f42d56d87f0d2ce44704a1e944efefc7a67320cf1a44cb492c4bacf07acbf"
	vulkanQualifiedQ2TokenPairBytes     = 11796
)

// The immutable C++ attestation is the single canonical byte owner. We do not
// add a second generated artifact or accept a new compiler's equivalent-looking
// output. Both the complete header and decoded module are pinned independently.
func decodeVulkanQualifiedQ2TokenPair(source, header []byte) ([]byte, error) {
	if !vulkanQualifiedHash(source, vulkanQualifiedQ2TokenPairSourceSHA) {
		return nil, fmt.Errorf("qualified Q2 token-pair GLSL source identity mismatch")
	}
	if !vulkanQualifiedHash(header, vulkanQualifiedQ2TokenPairHeaderSHA) {
		return nil, fmt.Errorf("qualified Q2 token-pair header identity mismatch")
	}
	const prefix = "static const unsigned char qualified[] = {"
	if bytes.Count(header, []byte(prefix)) != 1 {
		return nil, fmt.Errorf("qualified Q2 token-pair array framing mismatch")
	}
	body := bytes.SplitN(header, []byte(prefix), 2)[1]
	end := bytes.Index(body, []byte("};"))
	if end < 0 {
		return nil, fmt.Errorf("qualified Q2 token-pair array terminator missing")
	}
	tokens := strings.Split(string(body[:end]), ",")
	if len(tokens) != vulkanQualifiedQ2TokenPairBytes+1 || strings.TrimSpace(tokens[len(tokens)-1]) != "" {
		return nil, fmt.Errorf("qualified Q2 token-pair byte count mismatch")
	}
	out := make([]byte, vulkanQualifiedQ2TokenPairBytes)
	for i, t := range tokens[:len(tokens)-1] {
		t = strings.TrimSpace(t)
		if len(t) != 4 || !strings.HasPrefix(t, "0x") {
			return nil, fmt.Errorf("qualified Q2 token-pair invalid byte %d", i)
		}
		b, err := hex.DecodeString(t[2:])
		if err != nil {
			return nil, fmt.Errorf("qualified Q2 token-pair invalid byte %d", i)
		}
		out[i] = b[0]
	}
	if !vulkanQualifiedHash(out, vulkanQualifiedQ2TokenPairModuleSHA) {
		return nil, fmt.Errorf("qualified Q2 token-pair module identity mismatch")
	}
	return out, nil
}

func readVulkanQualifiedQ2TokenPair(root string) ([]byte, error) {
	source, err := readVulkanQualifiedFile(filepath.Join(root, "internal", "compute", "shaders", vulkanQualifiedQ2TokenPairStem+".comp"))
	if err != nil {
		return nil, err
	}
	header, err := readVulkanQualifiedFile(filepath.Join(root, filepath.FromSlash(vulkanQualifiedQ2TokenPairHeader)))
	if err != nil {
		return nil, err
	}
	return decodeVulkanQualifiedQ2TokenPair(source, header)
}

func writeVulkanQualifiedQ2TokenPair(dst string, data []byte) error {
	if len(data) != vulkanQualifiedQ2TokenPairBytes || !vulkanQualifiedHash(data, vulkanQualifiedQ2TokenPairModuleSHA) {
		return fmt.Errorf("refuse unqualified Q2 token-pair output")
	}
	f, err := os.CreateTemp(filepath.Dir(dst), ".qualified-Q2 token-pair-*")
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

func verifyVulkanQualifiedQ2TokenPair(root, spvRoot string) error {
	expected, err := readVulkanQualifiedQ2TokenPair(root)
	if err != nil {
		return err
	}
	actual, err := readVulkanQualifiedFile(filepath.Join(spvRoot, vulkanQualifiedQ2TokenPairStem+".spv"))
	if err != nil {
		return err
	}
	if !bytes.Equal(expected, actual) {
		return fmt.Errorf("packaged Q2 token-pair module differs from qualified artifact")
	}
	return nil
}

func (r *vulkanSuccessorRegistry) usesQualifiedQ2TokenPair() bool {
	for ; r != nil; r = r.parent {
		if r == vulkanRegistryV26 {
			return true
		}
	}
	return false
}
