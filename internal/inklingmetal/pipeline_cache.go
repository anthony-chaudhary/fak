package inklingmetal

import (
	"os"
	"path/filepath"
	"strings"
)

// PipelineCacheEnv overrides where compiled Metal pipelines persist across
// processes; off, 0, false or none keeps none.
const PipelineCacheEnv = "FAK_INKLING_METAL_PIPELINE_CACHE"

// PipelineStats is a model's Metal pipeline archive, read by --timings so a
// cold start (Compiled > 0) reads apart from an archive hit.
type PipelineStats struct {
	// Compiled counts pipelines Metal compiled in this process; ArchiveHits
	// those created from the archive without compiling.
	Compiled, ArchiveHits int
	// Archived is the pipelines the archive listed when the model loaded, and
	// ArchiveWarmMS the load time spent creating them.
	Archived      int
	ArchiveWarmMS float64
	// LoadWarmupMS is the prefill and decode run at load when the archive
	// listed none.
	LoadWarmupMS float64
	Attached     bool
}

// pipelineCacheDir is empty when no archive is kept.
func pipelineCacheDir() string {
	if v, ok := os.LookupEnv(PipelineCacheEnv); ok {
		switch strings.ToLower(strings.TrimSpace(v)) {
		case "", "off", "0", "false", "none":
			return ""
		}
		return v
	}
	base, err := os.UserCacheDir()
	if err != nil {
		return ""
	}
	return filepath.Join(base, "fak", "inkling-metal-pipelines")
}
