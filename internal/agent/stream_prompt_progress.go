package agent

import (
	"encoding/json"
	"math"
)

// streamPromptProgress recognizes llama.cpp's opt-in prompt_progress wire.
// Only the resolved request's explicit return_progress:true enables it. Counts
// are evidence from the selected upstream, not a new deadline or output token;
// all existing context, idle and absolute-duration bounds still apply.
type streamPromptProgress struct {
	enabled, observed       bool
	total, cache, processed int64
}

func newStreamPromptProgress(request []byte) streamPromptProgress {
	var options struct {
		ReturnProgress bool `json:"return_progress"`
	}
	return streamPromptProgress{enabled: json.Unmarshal(request, &options) == nil && options.ReturnProgress}
}

func (s *streamPromptProgress) advance(raw json.RawMessage) bool {
	if !s.enabled {
		return false
	}
	var progress struct {
		Total     *int64 `json:"total"`
		Cache     *int64 `json:"cache"`
		Processed *int64 `json:"processed"`
		TimeMS    *int64 `json:"time_ms"`
	}
	if json.Unmarshal(raw, &progress) != nil || progress.Total == nil || progress.Cache == nil || progress.Processed == nil || progress.TimeMS == nil {
		return false
	}
	total, cache, processed := *progress.Total, *progress.Cache, *progress.Processed
	if total <= 0 || total > math.MaxInt32 || cache < 0 || cache > processed || processed > total || *progress.TimeMS < 0 {
		return false
	}
	if s.observed && (total != s.total || cache != s.cache) {
		// A different tuple cannot establish a new baseline and replay old work.
		s.enabled = false
		return false
	}
	if !s.observed {
		s.observed, s.total, s.cache, s.processed = true, total, cache, cache
	}
	if processed <= s.processed {
		return false
	}
	// time_ms is schema-checked only. A heartbeat with increasing elapsed time,
	// or an initial cache-only observation, never re-arms the progress deadline.
	s.processed = processed
	return true
}
