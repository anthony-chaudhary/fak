package model

// V4ExpertStats is the session-local witness surface for DeepSeek V4 routed
// expert streaming. All counters are cumulative for the lifetime of the live
// expert runtime. Misses count staged tensor page-ins; Hits count resident
// tensor accesses. Hits+Misses is therefore the cache-access denominator, not
// a token or routed-expert denominator.
type V4ExpertStats struct {
	ExpertOpenCount     int   `json:"expert_open_count"`
	ExpertReadCount     int   `json:"expert_read_count"`
	HashOpenCount       int   `json:"hash_open_count"`
	HashReadCount       int   `json:"hash_read_count"`
	HashReadBytes       int64 `json:"hash_read_bytes"`
	SourceReads         int64 `json:"source_reads"`
	SourceBytes         int64 `json:"source_bytes"`
	PageIns             int   `json:"page_ins"`
	Hits                int   `json:"hits"`
	Misses              int   `json:"misses"`
	Evictions           int   `json:"evictions"`
	ResidentBytes       int64 `json:"resident_bytes"`
	PeakResident        int64 `json:"peak_resident_bytes"`
	RingBudget          int64 `json:"ring_budget_bytes"`
	WorldSize           int   `json:"world_size"`
	Rank                int   `json:"rank"`
	LocalSelected       int64 `json:"local_selected"`
	RemoteSelected      int64 `json:"remote_selected"`
	TransportDispatches int64 `json:"transport_dispatches"`
	TransportPartials   int64 `json:"transport_partials"`
}

// CacheAccesses returns the staged tensor cache-access denominator.
func (s V4ExpertStats) CacheAccesses() int { return s.Hits + s.Misses }

// HitRate returns resident hits divided by staged tensor cache accesses. A zero
// denominator returns zero rather than NaN.
func (s V4ExpertStats) HitRate() float64 {
	if s.CacheAccesses() == 0 {
		return 0
	}
	return float64(s.Hits) / float64(s.CacheAccesses())
}

// V4ExpertStats returns a copy of the current live V4 expert runtime counters.
// It never initializes the runtime; nil and uninitialized sessions return zero.
func (s *Session) V4ExpertStats() V4ExpertStats {
	if s == nil || s.v4Expert == nil {
		return V4ExpertStats{}
	}
	raw := s.v4Expert.Stats()
	return V4ExpertStats{
		ExpertOpenCount:     raw.ExpertOpenCount,
		ExpertReadCount:     raw.ExpertReadCount,
		HashOpenCount:       raw.HashOpenCount,
		HashReadCount:       raw.HashReadCount,
		HashReadBytes:       raw.HashReadBytes,
		SourceReads:         raw.SourceReads,
		SourceBytes:         raw.SourceBytes,
		PageIns:             raw.PageIns,
		Hits:                raw.Hits,
		Misses:              raw.PageIns,
		Evictions:           raw.Evictions,
		ResidentBytes:       raw.ResidentBytes,
		PeakResident:        raw.PeakResident,
		RingBudget:          raw.RingBudget,
		WorldSize:           raw.WorldSize,
		Rank:                raw.Rank,
		LocalSelected:       raw.LocalSelected,
		RemoteSelected:      raw.RemoteSelected,
		TransportDispatches: raw.TransportDispatches,
		TransportPartials:   raw.TransportPartials,
	}
}
