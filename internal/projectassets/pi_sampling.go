package projectassets

import "github.com/anthony-chaudhary/fak/pkg/harnesskit"

// PiSamplingParams returns the samplingParams the fak Pi writers own for a model,
// or nil when fak leaves the model's sampling to Pi and the server. The profile
// itself is harnesskit.SamplingParams, shared with the private OpenCode writer.
func PiSamplingParams(modelID string) map[string]interface{} {
	return harnesskit.SamplingParams(modelID)
}
