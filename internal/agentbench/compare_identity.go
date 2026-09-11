package agentbench

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

var errComparisonInconclusive = errors.New("comparison evidence incomplete")

func canonicalComparisonDirectory(path string) (string, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	real, err := filepath.EvalSymlinks(abs)
	if err != nil {
		return "", err
	}
	info, err := os.Stat(real)
	if err != nil {
		return "", err
	}
	if !info.IsDir() {
		return "", errors.New("comparison operand is not a directory")
	}
	return real, nil
}
func containedComparisonPath(root, rel string) (string, error) {
	if rel == "" || filepath.IsAbs(rel) {
		return "", errors.New("artifact path must be relative")
	}
	real, err := filepath.EvalSymlinks(filepath.Join(root, filepath.Clean(rel)))
	if err != nil {
		return "", err
	}
	within, err := filepath.Rel(root, real)
	if err != nil || within == ".." || strings.HasPrefix(within, ".."+string(filepath.Separator)) {
		return "", errors.New("artifact path escapes operand")
	}
	return real, nil
}
func validateComparisonArmIdentity(a comparisonArmArtifact) error {
	if a.Schema != comparisonArmSchema || a.ArmID == "" || a.ModelID == "" || a.ContextTokens < 32768 || a.QualifiedConcurrency <= 0 || len(a.Replicates) != 5 {
		return fmt.Errorf("%w: comparison arm identity is incomplete", errComparisonInconclusive)
	}
	for _, value := range []string{a.ManifestSHA256, a.TokenizerSHA256, a.RendererSHA256, a.WeightSHA256, a.Quantization, a.SamplingSHA256, a.OutputPolicySHA256, a.ToolContractSHA256, a.TaskWitnessSHA256, a.ResourceSHA256, a.LaunchSHA256, a.CachePolicySHA256, a.LoadScheduleSHA256} {
		if value == "" {
			return fmt.Errorf("%w: comparison arm envelope identity is unknown", errComparisonInconclusive)
		}
	}
	return nil
}
func comparisonEnvelopeMatches(a, b comparisonArmArtifact) bool {
	return a.ManifestSHA256 == b.ManifestSHA256 && a.ModelID == b.ModelID && a.TokenizerSHA256 == b.TokenizerSHA256 && a.RendererSHA256 == b.RendererSHA256 && a.WeightSHA256 == b.WeightSHA256 && a.Quantization == b.Quantization && a.ContextTokens == b.ContextTokens && a.SamplingSHA256 == b.SamplingSHA256 && a.OutputPolicySHA256 == b.OutputPolicySHA256 && a.ToolContractSHA256 == b.ToolContractSHA256 && a.TaskWitnessSHA256 == b.TaskWitnessSHA256 && a.ResourceSHA256 == b.ResourceSHA256 && a.LaunchSHA256 == b.LaunchSHA256 && a.CachePolicySHA256 == b.CachePolicySHA256 && a.LoadScheduleSHA256 == b.LoadScheduleSHA256
}
