package metrics

import "log"

type TuningAdvisor struct {
	loggedHighHitRate     map[uint64]bool
	loggedHighHitRatePeak int
}

func NewTuningAdvisor() *TuningAdvisor {
	return &TuningAdvisor{
		loggedHighHitRate: make(map[uint64]bool),
	}
}

func (ta *TuningAdvisor) Evaluate(cs *ClientStats) {
	ta.checkCacheHitRate(cs)
	ta.checkEvictionPressure(cs)
	ta.checkPrefetchBandwidth(cs)
}

func (ta *TuningAdvisor) checkCacheHitRate(cs *ClientStats) {
	if cs.CacheHitRate == 0 {
		return
	}
	if cs.CacheHitRate < 0.3 {
		log.Printf("[TuningAdvisor] WARN conn=%d: Low SGLang cache hit rate (%.1f%%). "+
			"Consider increasing L3 memory or reviewing eviction policy.",
			cs.ConnID, cs.CacheHitRate*100)
	}
	if cs.CacheHitRate > 0.8 {
		if !ta.loggedHighHitRate[cs.ConnID] {
			log.Printf("[TuningAdvisor] INFO conn=%d: High cache hit rate (%.1f%%). "+
				"Current config is effective.",
				cs.ConnID, cs.CacheHitRate*100)
			ta.loggedHighHitRate[cs.ConnID] = true
			if n := len(ta.loggedHighHitRate); n > ta.loggedHighHitRatePeak {
				ta.loggedHighHitRatePeak = n
			}
		}
	} else {
		delete(ta.loggedHighHitRate, cs.ConnID)
	}
}

func (ta *TuningAdvisor) checkEvictionPressure(cs *ClientStats) {
	if cs.TokenUsage > 0.9 && cs.EvictableRatio < 0.1 {
		log.Printf("[TuningAdvisor] WARN conn=%d: GPU cache under pressure "+
			"(%.0f%% used, %.0f%% evictable). Prefetch/backup throughput may be critical.",
			cs.ConnID, cs.TokenUsage*100, cs.EvictableRatio*100)
	}
}

func (ta *TuningAdvisor) checkPrefetchBandwidth(cs *ClientStats) {
	if cs.PrefetchBandwidthGbps > 0 && cs.PrefetchBandwidthGbps < 1.0 && cs.CacheHitRate < 0.5 {
		log.Printf("[TuningAdvisor] WARN conn=%d: Low prefetch bandwidth (%.2f GB/s) "+
			"with low hit rate. Check RDMA NIC health or io_workers count.",
			cs.ConnID, cs.PrefetchBandwidthGbps)
	}
}

func (ta *TuningAdvisor) Remove(connID uint64) {
	delete(ta.loggedHighHitRate, connID)
	ta.loggedHighHitRate, ta.loggedHighHitRatePeak = compactMap(ta.loggedHighHitRate, ta.loggedHighHitRatePeak)
}
