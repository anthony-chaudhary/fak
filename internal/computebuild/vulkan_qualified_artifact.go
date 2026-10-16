package computebuild

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

const (
	vulkanQualifiedIndexScaleStem      = "v41_index_scale_bf16"
	vulkanQualifiedIndexScaleHeader    = "internal/compute/v41_index_scale_bf16_spirv.h"
	vulkanQualifiedIndexScaleSourceSHA = "24cb0eb5c32b545c4c3c495f1ac9529a9013500baad0f0d7dc3f473c28b7d0ec"
	vulkanQualifiedIndexScaleHeaderSHA = "92cdb499bd6909a14488334ff1baa741e092211ab880a26d3c687f2cb18c3fcf"
	vulkanQualifiedIndexScaleModuleSHA = "b67ed132187bde41131d355d9d493f56e5063ab87aa0a6a53ea66c8743136500"
	vulkanQualifiedIndexScaleBytes     = 2836
)

// readVulkanQualifiedFile bounds every read, including files that grow after stat.
// Hash checks below bind the observed bytes; a mutable source cannot supply an
// unreviewed module. Finalization rechecks both inputs and installed output.
func readVulkanQualifiedFile(path string) ([]byte, error) {
	clean, before, err := strictAbsoluteRegularFile(path, "qualified shader input")
	if err != nil {
		return nil, err
	}
	if before.Size() > 64<<10 {
		return nil, fmt.Errorf("qualified shader input exceeds bound")
	}
	f, err := os.Open(clean)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	opened, err := f.Stat()
	if err != nil || !os.SameFile(before, opened) {
		return nil, fmt.Errorf("qualified shader input changed before read")
	}
	data, err := io.ReadAll(io.LimitReader(f, (64<<10)+1))
	if err != nil {
		return nil, err
	}
	after, err := os.Lstat(clean)
	if err != nil || !os.SameFile(opened, after) || !sameFileMetadata(opened, after) || len(data) > 64<<10 {
		return nil, fmt.Errorf("qualified shader input changed or exceeded bound")
	}
	return data, nil
}

func vulkanQualifiedHash(data []byte, want string) bool {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]) == want
}

// The immutable C++ attestation is the single canonical byte owner. We do not
// add a second generated artifact or accept a new compiler's equivalent-looking
// output. Both the complete header and decoded module are pinned independently.
func decodeVulkanQualifiedIndexScale(source, header []byte) ([]byte, error) {
	if !vulkanQualifiedHash(source, vulkanQualifiedIndexScaleSourceSHA) {
		return nil, fmt.Errorf("qualified index-scale GLSL source identity mismatch")
	}
	if !vulkanQualifiedHash(header, vulkanQualifiedIndexScaleHeaderSHA) {
		return nil, fmt.Errorf("qualified index-scale header identity mismatch")
	}
	const prefix = "static const unsigned char qualified[] = {"
	if bytes.Count(header, []byte(prefix)) != 1 {
		return nil, fmt.Errorf("qualified index-scale array framing mismatch")
	}
	body := bytes.SplitN(header, []byte(prefix), 2)[1]
	end := bytes.Index(body, []byte("};"))
	if end < 0 {
		return nil, fmt.Errorf("qualified index-scale array terminator missing")
	}
	tokens := strings.Split(string(body[:end]), ",")
	if len(tokens) != vulkanQualifiedIndexScaleBytes+1 || strings.TrimSpace(tokens[len(tokens)-1]) != "" {
		return nil, fmt.Errorf("qualified index-scale byte count mismatch")
	}
	out := make([]byte, vulkanQualifiedIndexScaleBytes)
	for i, t := range tokens[:len(tokens)-1] {
		t = strings.TrimSpace(t)
		if len(t) != 4 || !strings.HasPrefix(t, "0x") {
			return nil, fmt.Errorf("qualified index-scale invalid byte %d", i)
		}
		b, err := hex.DecodeString(t[2:])
		if err != nil {
			return nil, fmt.Errorf("qualified index-scale invalid byte %d", i)
		}
		out[i] = b[0]
	}
	if !vulkanQualifiedHash(out, vulkanQualifiedIndexScaleModuleSHA) {
		return nil, fmt.Errorf("qualified index-scale module identity mismatch")
	}
	return out, nil
}

func readVulkanQualifiedIndexScale(root string) ([]byte, error) {
	source, err := readVulkanQualifiedFile(filepath.Join(root, "internal", "compute", "shaders", vulkanQualifiedIndexScaleStem+".comp"))
	if err != nil {
		return nil, err
	}
	header, err := readVulkanQualifiedFile(filepath.Join(root, filepath.FromSlash(vulkanQualifiedIndexScaleHeader)))
	if err != nil {
		return nil, err
	}
	return decodeVulkanQualifiedIndexScale(source, header)
}

func writeVulkanQualifiedModule(dst string, data []byte) error {
	if len(data) != vulkanQualifiedIndexScaleBytes || !vulkanQualifiedHash(data, vulkanQualifiedIndexScaleModuleSHA) {
		return fmt.Errorf("refuse unqualified index-scale output")
	}
	f, err := os.CreateTemp(filepath.Dir(dst), ".qualified-index-scale-*")
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

func verifyVulkanQualifiedIndexScale(root, spvRoot string) error {
	expected, err := readVulkanQualifiedIndexScale(root)
	if err != nil {
		return err
	}
	actual, err := readVulkanQualifiedFile(filepath.Join(spvRoot, vulkanQualifiedIndexScaleStem+".spv"))
	if err != nil {
		return err
	}
	if !bytes.Equal(expected, actual) {
		return fmt.Errorf("packaged index-scale module differs from qualified artifact")
	}
	return nil
}

func (r *vulkanSuccessorRegistry) usesQualifiedIndexScale() bool {
	for ; r != nil; r = r.parent {
		if r == vulkanRegistryV23 {
			return true
		}
	}
	return false
}

// Historical schemas keep their original glslc recipe. V23 binds the materialized
// module, source, and header in the existing command/stable-identity hashes.
// This policy applies to the native Go builder on Unix and Windows. The legacy
// internal/compute/build_vulkan.ps1 still recompiles shaders and cannot establish
// this V23 materialization claim or guarantee the optional capability.
func (r *vulkanSuccessorRegistry) normalizedBuildCommand(cfg *VulkanConfig) ([]string, string, error) {
	commands, hash, err := normalizedVulkanBuildCommand(cfg)
	if err != nil || !r.usesQualifiedIndexScale() {
		return commands, hash, err
	}
	commands[0] = "glslc|-O|--target-env=vulkan1.2|-fshader-stage=comp|$REPO/internal/compute/shaders/{registered-except-v41_index_scale_bf16}.comp|-o|$REPO/internal/compute/spirv/{registered-except-v41_index_scale_bf16}.spv"
	materialize := "qualified-header|" + vulkanQualifiedIndexScaleStem + "|source_sha256=" + vulkanQualifiedIndexScaleSourceSHA + "|header=$REPO/" + vulkanQualifiedIndexScaleHeader + "|header_sha256=" + vulkanQualifiedIndexScaleHeaderSHA + "|module_sha256=" + vulkanQualifiedIndexScaleModuleSHA + "|bytes=2836|output=$REPO/internal/compute/spirv/" + vulkanQualifiedIndexScaleStem + ".spv"
	commands = append([]string{commands[0], materialize}, commands[1:]...)
	hash, err = hashJSON(commands)
	return commands, hash, err
}
