package perfledger

import "math"

// summaryFold carries the running state of one Summarize pass: the Summary
// being filled plus the raw samples (latency series, native phase
// accumulators, kernel shares, per-regime and per-client buckets) that
// finish later turns into quantiles and weighted rates.
type summaryFold struct {
	s                     Summary
	ttft                  []float64
	prefill               []float64
	decode                []float64
	e2e                   []float64
	gpuBusy               []float64
	prompt                int64
	cached                int64
	prefillTok            int64
	prefillMS             float64
	nativePrefillMS       []float64
	nativeDecodeMS        []float64
	nativePrefillTPS      []float64
	nativeDecodeTPS       []float64
	nativeCPUMSPerToken   []float64
	nativeHostShare       []float64
	nativePrefillTokens   int64
	nativeDecodeTokens    int64
	nativePrefillTotalMS  float64
	nativeDecodeTotalMS   float64
	kernels               kernelFold
	nativePrefillWeighted bool
	nativeDecodeWeighted  bool
	regimeTTFT            map[string][]float64
	regimeE2E             map[string][]float64
	regimeCount           map[string]int
	servedBy              map[ServedBy]struct{}
	clients               map[string]*clientFold
	anyClient             bool
}

// newSummaryFold returns a fold ready to consume records. The weighted-rate
// flags start true (no overflow seen yet) and the lookup maps start empty but
// non-nil, matching the plain map literals the inline code used.
func newSummaryFold() *summaryFold {
	return &summaryFold{
		nativePrefillWeighted: true,
		nativeDecodeWeighted:  true,
		regimeTTFT:            map[string][]float64{},
		regimeE2E:             map[string][]float64{},
		regimeCount:           map[string]int{},
		servedBy:              map[ServedBy]struct{}{},
		clients:               map[string]*clientFold{},
	}
}

// foldRecord folds one record into the running totals, in the same statement
// order the inline loop used. Probe rows only bump the probe counter.
func (f *summaryFold) foldRecord(r Record) {
	if r.IsProbe() {
		f.s.Probes++
		return
	}
	f.s.Count++
	if r.Error != "" {
		f.s.Errors++
		if f.s.ByError == nil {
			f.s.ByError = map[string]int{}
		}
		f.s.ByError[r.Error]++
	}
	if sb := r.servedBy(); sb != (ServedBy{}) {
		f.servedBy[sb] = struct{}{}
		f.s.ServedBy = &sb
	}
	regime := r.regime()
	f.regimeCount[regime]++
	f.anyClient = f.anyClient || r.Client != ""
	f.clients[r.clientKey()] = f.clients[r.clientKey()].add(r, regime)
	if r.TTFTMS > 0 {
		f.regimeTTFT[regime] = append(f.regimeTTFT[regime], r.TTFTMS)
	}
	if r.E2EMS > 0 {
		f.regimeE2E[regime] = append(f.regimeE2E[regime], r.E2EMS)
	}
	if r.TTFTMS > 0 {
		f.ttft = append(f.ttft, r.TTFTMS)
	}
	// The token floor also filters rows written before NewRecord applied it.
	if r.PrefillTPS > 0 && r.PromptTokens >= MinPrefillRateTokens {
		f.prefill = append(f.prefill, r.PrefillTPS)
		f.prefillTok += int64(r.PromptTokens)
		f.prefillMS += r.TTFTMS
	}
	if r.UpstreamSpecDraftTokens > 0 {
		f.s.SpecDraftTokens += r.UpstreamSpecDraftTokens
		f.s.SpecAcceptedTokens += min(max(r.UpstreamSpecAcceptedTokens, 0), r.UpstreamSpecDraftTokens)
	}
	if r.DecodeTPS > 0 {
		f.decode = append(f.decode, r.DecodeTPS)
	}
	if r.E2EMS > 0 {
		f.e2e = append(f.e2e, r.E2EMS)
		if f.s.Slowest == nil || r.E2EMS > f.s.Slowest.E2EMS {
			f.s.Slowest = &SlowestTurn{RequestID: r.RequestID, InboundRequestID: r.InboundRequestID, E2EMS: r.E2EMS}
		}
	}
	if g := r.GPUBusyPercent; g != nil {
		f.gpuBusy = append(f.gpuBusy, float64(*g))
	}
	f.prompt += int64(r.PromptTokens)
	f.cached += int64(r.CachedTokens)
	if stored := r.NativeTiming; stored != nil {
		n := NewNativeTiming(stored.PrefillTokens, stored.DecodeTokens, stored.PrefillMS, stored.DecodeMS)
		if n != nil {
			if f.s.NativeTiming == nil {
				f.s.NativeTiming = &NativeTimingSummary{}
			}
			f.s.NativeTiming.Count++
			if f.kernels.add(stored.Kernels) {
				f.s.NativeTiming.KernelsMeasured++
			}
			if c, h := stored.CPUMSPerToken, stored.HostShare; c != nil && h != nil {
				if cv, ok := finiteNonnegativeHostWork(*c, true); ok {
					if hv, ok := finiteNonnegativeHostWork(*h, true); ok {
						f.s.NativeTiming.HostWorkMeasured++
						f.nativeCPUMSPerToken = append(f.nativeCPUMSPerToken, cv)
						f.nativeHostShare = append(f.nativeHostShare, hv)
					}
				}
			}
			if n.PrefillMS > 0 {
				f.s.NativeTiming.PrefillMeasured++
				f.nativePrefillMS = append(f.nativePrefillMS, n.PrefillMS)
				if n.PrefillTokens >= MinPrefillRateTokens {
					if n.PrefillTPS > 0 {
						f.nativePrefillTPS = append(f.nativePrefillTPS, n.PrefillTPS)
					}
					if total := f.nativePrefillTotalMS + n.PrefillMS; math.IsInf(total, 0) || math.IsNaN(total) {
						f.nativePrefillWeighted = false
					} else {
						f.nativePrefillTotalMS = total
					}
					if total, ok := addNativeTokens(f.nativePrefillTokens, n.PrefillTokens); ok {
						f.nativePrefillTokens = total
					} else {
						f.nativePrefillWeighted = false
					}
				}
			}
			if n.DecodeMS > 0 {
				f.s.NativeTiming.DecodeMeasured++
				f.nativeDecodeMS = append(f.nativeDecodeMS, n.DecodeMS)
				if n.DecodeTokens >= 2 {
					if n.DecodeTPS > 0 {
						f.nativeDecodeTPS = append(f.nativeDecodeTPS, n.DecodeTPS)
					}
					if total := f.nativeDecodeTotalMS + n.DecodeMS; math.IsInf(total, 0) || math.IsNaN(total) {
						f.nativeDecodeWeighted = false
					} else {
						f.nativeDecodeTotalMS = total
					}
					if total, ok := addNativeTokens(f.nativeDecodeTokens, n.DecodeTokens-1); ok {
						f.nativeDecodeTokens = total
					} else {
						f.nativeDecodeWeighted = false
					}
				}
			}
		}
	}
	if e := r.Engine; e != nil {
		f.s.Native++
		if f.s.ByPath == nil {
			f.s.ByPath = make(map[string]int, len(Paths))
		}
		f.s.ByPath[e.Path]++
		f.s.SpecRounds += e.SpecRounds
		f.s.SpecDraftTokens += e.SpecDraftTokens
		f.s.SpecAcceptedTokens += e.SpecAcceptedTokens
	}
}

// finish applies the post-loop folds: quantiles over the accumulated sample
// slices, weighted prefill/decode rates when their totals stayed finite, and
// the per-regime and per-client rollups. It returns the completed Summary.
func (f *summaryFold) finish(recs []Record) Summary {
	s := &f.s
	if s.SpecDraftTokens > 0 {
		s.SpecAcceptRate = roundTo(float64(s.SpecAcceptedTokens)/float64(s.SpecDraftTokens), 10000)
	}
	s.Identities = len(f.servedBy)
	s.TTFTMeasured = len(f.ttft)
	s.TTFTP50MS = quantile(f.ttft, 0.50)
	s.TTFTP99MS = quantile(f.ttft, 0.99)
	s.PrefillTPSP50 = quantile(f.prefill, 0.50)
	if f.prefillMS > 0 {
		s.PrefillTPSWeighted = roundTo(float64(f.prefillTok)/(f.prefillMS/1000), 100)
	}
	s.DecodeTPSP50 = quantile(f.decode, 0.50)
	s.E2EP50MS = quantile(f.e2e, 0.50)
	s.E2EP99MS = quantile(f.e2e, 0.99)
	s.GPUBusyMeasured = len(f.gpuBusy)
	s.GPUBusyP50 = quantile(f.gpuBusy, 0.50)
	if total := f.prompt + f.cached; total > 0 {
		s.CacheHitShare = roundTo(float64(f.cached)/float64(total), 10000)
	}
	if n := s.NativeTiming; n != nil {
		n.PrefillP50MS = quantile(f.nativePrefillMS, 0.50)
		n.DecodeP50MS = quantile(f.nativeDecodeMS, 0.50)
		n.PrefillTPSP50 = quantile(f.nativePrefillTPS, 0.50)
		n.DecodeTPSP50 = quantile(f.nativeDecodeTPS, 0.50)
		n.CPUMSPerTokenP50 = quantile(f.nativeCPUMSPerToken, 0.50)
		n.HostShareP50 = quantile(f.nativeHostShare, 0.50)
		if n.KernelsMeasured > 0 {
			n.KernelsComplete = f.kernels.complete
			n.GPUMSPerTokenP50 = quantile(f.kernels.gpuMSPerToken, 0.50)
			n.CoverageP50 = quantile(f.kernels.coverage, 0.50)
			n.DeviceBusyP50 = quantile(f.kernels.deviceBusy, 0.50)
			n.TopKernels = f.kernels.top(n.KernelsMeasured)
		}
		if f.nativePrefillWeighted && f.nativePrefillTotalMS > 0 && f.nativePrefillTokens > 0 {
			n.PrefillTPSWeighted = roundTo(float64(f.nativePrefillTokens)/(f.nativePrefillTotalMS/1000), 100)
		}
		if f.nativeDecodeWeighted && f.nativeDecodeTotalMS > 0 && f.nativeDecodeTokens > 0 {
			n.DecodeTPSWeighted = roundTo(float64(f.nativeDecodeTokens)/(f.nativeDecodeTotalMS/1000), 100)
		}
	}
	for regime, n := range f.regimeCount {
		if s.ByRegime == nil {
			s.ByRegime = make(map[string]RegimeSummary, len(f.regimeCount))
		}
		s.ByRegime[regime] = RegimeSummary{
			Count:        n,
			TTFTMeasured: len(f.regimeTTFT[regime]),
			TTFTP50MS:    quantile(f.regimeTTFT[regime], 0.50),
			TTFTP99MS:    quantile(f.regimeTTFT[regime], 0.99),
			E2EP50MS:     quantile(f.regimeE2E[regime], 0.50),
		}
	}
	if f.anyClient {
		s.ByClient = make(map[string]ClientSummary, len(f.clients))
		for c, cf := range f.clients {
			s.ByClient[c] = cf.summary()
		}
	}
	s.ByPathStats = summarizeNativePaths(recs)
	return *s
}
