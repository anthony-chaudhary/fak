package resume

// backtest.go — the deterministic VALIDATION of the resume-cache projection against billed
// reality. Plan (resume.go) PROJECTS a cache posture from idle-vs-TTL and prices the cold
// re-prefill; the honest fence is that those are projections, never a witnessed bill. This
// file closes the gap one rung: given the provider's OWN per-turn usage records from real
// sessions — the cache_read / cache_creation token counts and timestamps it actually billed —
// it back-tests whether the projection's posture call agrees with what the provider did, and
// whether the cold-cost model (a cold first turn re-writes the whole resident) holds.
//
// It is the same OBSERVED-vs-WITNESSED discipline the rest of the package keeps, run as a
// measurement: the projection is the model, the usage records are the ground truth, and the
// BacktestReport is the residual. Pure tier-1 leaf, stdlib-only, no clock and no I/O — the
// corpus scan that produces the ObservedTurns lives in the cmd/fak/resume.go shell, exactly
// as Plan's transcript grounding does.
//
// # What a posture means in billed reality
//
// The projection predicts whether the PRIOR prefix is still served on the next turn. The
// provider answers that directly: cache_read_input_tokens on a turn is how much of the prompt
// it served from cache. So the observed posture of a turn is the RECOVERY fraction — the
// cache_read it billed divided by the EARLIER turn's prompt size (how much of the prior prefix
// actually came back warm). A recovery near zero is an observed COLD (the prefix aged out and
// had to be re-written); a recovery near one is an observed WARM (the prefix survived). The
// band between is a partial re-serve we refuse to score, exactly as Plan refuses to shed on an
// ambiguous span.

// ObservedTurn is one assistant turn's billed reality, lifted from a real transcript usage
// record: the wall-clock time it happened and the provider's three input-token axes. It
// carries NO transcript content — only the counts and the timestamp the back-test needs, the
// same content-free discipline the closed reason vocabulary keeps.
type ObservedTurn struct {
	UnixSeconds         int64 `json:"unix_seconds"`
	InputTokens         int   `json:"input_tokens"`
	CacheCreationTokens int   `json:"cache_creation_tokens"`
	CacheReadTokens     int   `json:"cache_read_tokens"`
}

// Prompt is the whole prompt the model read this turn — the uncached remainder plus whatever
// was written to and read from the cache. It is the resident context size AT this turn, the
// same quantity Plan's --transcript grounding derives for the most recent turn.
func (t ObservedTurn) Prompt() int {
	return t.InputTokens + t.CacheCreationTokens + t.CacheReadTokens
}

// RecoveryBand maps a prefix-recovery fraction to an observed posture. Recovery is the later
// turn's cache_read divided by the EARLIER turn's prompt size: the fraction of the prior
// prefix the provider actually re-served. Below ColdMax the prefix was not served (observed
// cold); above WarmMin it was (observed warm); the band between is an ambiguous partial
// re-serve, excluded from the accuracy denominator rather than scored on a guess.
type RecoveryBand struct {
	ColdMax float64 `json:"cold_max"`
	WarmMin float64 `json:"warm_min"`
}

// DefaultRecoveryBand is the conservative scoring band: a turn that re-served under 5% of the
// prior prefix is cold, over 50% is warm, and the gap between is unscored. The wide dead-zone
// keeps the accuracy number honest — only unambiguous re-serves count for or against the
// projection.
func DefaultRecoveryBand() RecoveryBand { return RecoveryBand{ColdMax: 0.05, WarmMin: 0.50} }

// FirstTurnResumeThresholdTokens is the prompt size at or above which a session's FIRST
// assistant turn counts as a genuine RESUME re-prefill (a carried-over transcript) rather than
// a small fresh start. A multi-hour resume opens a NEW transcript file and re-prefills the
// prior conversation on turn one; a brand-new session opens small. ~20k separates the two: it
// is the ~48k shed budget's neighborhood, comfortably above an opening prompt but below a real
// resumed transcript. The cross-file cold case within-file idle gaps under-sample lives here.
const FirstTurnResumeThresholdTokens = 20000

// classify returns the observed posture of a turn whose prior prompt was prevPrompt and whose
// own cache_read was cacheRead, plus the recovery fraction, plus whether it is scorable (false
// for a zero prior prompt or an ambiguous partial re-serve).
func (b RecoveryBand) classify(prevPrompt, cacheRead int) (Posture, float64, bool) {
	if prevPrompt <= 0 {
		return PostureUnknown, 0, false
	}
	rec := float64(cacheRead) / float64(prevPrompt)
	switch {
	case rec < b.ColdMax:
		return PostureCold, rec, true
	case rec > b.WarmMin:
		return PostureWarm, rec, true
	default:
		return PostureUnknown, rec, false // ambiguous partial re-serve: not scored
	}
}

// GapBucket tallies the observed postures over one wall-clock gap range, so the report shows
// WHERE on the idle axis the projection agrees with reality — the empirical cache-survival
// curve the single TTL cutoff approximates.
type GapBucket struct {
	LoSeconds    int64   `json:"lo_seconds"`
	HiSeconds    int64   `json:"hi_seconds"`
	N            int     `json:"n"`
	WarmN        int     `json:"warm_n"`
	ColdN        int     `json:"cold_n"`
	AmbiguousN   int     `json:"ambiguous_n"`
	MeanRecovery float64 `json:"mean_recovery"`
}

// DefaultGapBuckets is the fixed gap ladder the report buckets boundaries into: sub-minute,
// up to the 5m TTL, then widening windows out past a day. It straddles the TTL cutoff (300s)
// so the table reveals the empirical cache cliff the projection's single threshold stands in
// for.
func DefaultGapBuckets() []GapBucket {
	edges := []int64{0, 60, 300, 900, 1800, 3600, 18000, 86400, 1 << 62}
	out := make([]GapBucket, 0, len(edges)-1)
	for i := 0; i+1 < len(edges); i++ {
		out = append(out, GapBucket{LoSeconds: edges[i], HiSeconds: edges[i+1]})
	}
	return out
}

// BacktestReport is the deterministic residual of the projection against billed reality: how
// often the posture call agreed, in which direction it erred, the per-gap survival curve, and
// how exactly the cold-cost model held. Every field is a count or a ratio over observed usage
// — no dollar here is a fak figure, they are the provider's own token records scored against
// the projection.
type BacktestReport struct {
	TTL        CacheTTL     `json:"ttl"`
	TTLSeconds int64        `json:"ttl_seconds"`
	Band       RecoveryBand `json:"band"`

	Pairs     int `json:"pairs"`     // adjacent turn pairs with a positive prior prompt
	Scored    int `json:"scored"`    // non-ambiguous pairs — the accuracy denominator
	Ambiguous int `json:"ambiguous"` // partial re-serves, excluded from accuracy
	Agree     int `json:"agree"`
	Disagree  int `json:"disagree"`
	// ProjColdObsWarm: the projection called cold (gap >= TTL) but the prefix was still served
	// — the TTL cutoff is SHORTER than the provider's effective reuse window here (the
	// projection would burst a still-warm cache). ProjWarmObsCold is the opposite miss.
	ProjColdObsWarm int     `json:"proj_cold_obs_warm"`
	ProjWarmObsCold int     `json:"proj_warm_obs_cold"`
	Accuracy        float64 `json:"accuracy"` // Agree / Scored (0 when nothing scored)

	Buckets []GapBucket `json:"buckets"`

	// Cold-cost validation: over boundaries the back-test CONFIRMS cold (gap >= TTL and an
	// observed-cold recovery), how much of the turn's prompt was a fresh cache WRITE
	// (cache_creation / prompt). A value near 1.0 confirms the projection's premise that a
	// cold first turn re-writes essentially the whole resident at the write premium.
	ConfirmedCold      int     `json:"confirmed_cold"`
	ColdWriteRatioMean float64 `json:"cold_write_ratio_mean"`

	// Cross-file resume instrument: a genuine multi-hour resume starts a NEW transcript file
	// whose FIRST assistant turn re-prefills the carried-over transcript — the cold case the
	// within-file gap pairs above under-sample. A first turn whose prompt is large (>=
	// FirstTurnResumeThresholdTokens) is a resume re-prefill; if its cache_read is ~0 it
	// re-prefilled COLD (FirstTurnCold), and if its cache_read is high the prior session's
	// prefix was still provider-warm on re-open — a cross-SESSION cache hit (FirstTurnWarmHit),
	// a reuse the within-session model does not account for.
	FirstTurnResumes int `json:"first_turn_resumes"`  // large first turns (resume re-prefills)
	FirstTurnCold    int `json:"first_turn_cold"`     // ...that re-prefilled cold (cache_read ~0)
	FirstTurnWarmHit int `json:"first_turn_warm_hit"` // ...that hit a still-warm cross-session prefix
	// ColdWriteShareMean is cache_creation / prompt on the cold first-turn resumes: the share
	// of the cold re-prefill that actually paid the WRITE premium. Below 1.0 means the resume
	// re-cached only PART of the transcript and sent the rest as plain input, so the projection
	// (which prices the whole resident at the write premium) OVER-states the cold cost by the
	// shortfall.
	FirstTurnColdWriteShareMean   float64 `json:"first_turn_cold_write_share_mean"`
	FirstTurnColdReprefillTokMean float64 `json:"first_turn_cold_reprefill_tok_mean"`
}

// Backtest is THE deterministic validation: same observed sessions in, same residual out — no
// clock, no I/O. Each inner slice is one session's turns (any order; sorted by time here on a
// copy). For every adjacent pair it compares the projected posture (projectPosture over the
// gap, the exact call Plan makes) against the observed posture (the prior-prefix recovery),
// tallies agreement and the per-gap survival curve, and on confirmed-cold boundaries measures
// how completely the prompt was re-written. It is total: empty input yields a zeroed report.
func Backtest(sessions [][]ObservedTurn, ttl CacheTTL, band RecoveryBand) BacktestReport {
	if ttl != TTL1h {
		ttl = TTL5m
	}
	if band.ColdMax <= 0 && band.WarmMin <= 0 {
		band = DefaultRecoveryBand()
	}
	rep := BacktestReport{
		TTL:        ttl,
		TTLSeconds: ttl.Seconds(),
		Band:       band,
		Buckets:    DefaultGapBuckets(),
	}
	var coldWriteSum float64
	var ftColdWriteShareSum, ftColdReprefillSum float64
	for _, sess := range sessions {
		turns := sortedByTime(sess)
		// Cross-file resume instrument: classify the FIRST assistant turn of the session. A
		// large first turn is a resume re-prefill; whether it came back cold or hit a still-warm
		// cross-session prefix is the provider's call, read from its own cache_read.
		if len(turns) > 0 {
			first := turns[0]
			if p := first.Prompt(); p >= FirstTurnResumeThresholdTokens {
				rep.FirstTurnResumes++
				coldness := float64(first.CacheReadTokens) / float64(p)
				switch {
				case coldness < band.ColdMax:
					rep.FirstTurnCold++
					ftColdWriteShareSum += float64(first.CacheCreationTokens) / float64(p)
					ftColdReprefillSum += float64(p)
				case coldness > band.WarmMin:
					rep.FirstTurnWarmHit++
				}
			}
		}
		for i := 1; i < len(turns); i++ {
			prev, cur := turns[i-1], turns[i]
			gap := cur.UnixSeconds - prev.UnixSeconds
			if gap < 0 {
				continue
			}
			prevPrompt := prev.Prompt()
			// A pair is only scorable when BOTH turns carry a real prompt: a zero-token usage
			// record (a non-substantive continuation row) has no posture — its cache_read of 0
			// would otherwise read as a false COLD and pollute both the confirmed-cold set and
			// the proj-warm/obs-cold misses.
			if prevPrompt <= 0 || cur.Prompt() <= 0 {
				continue
			}
			obs, rec, scorable := band.classify(prevPrompt, cur.CacheReadTokens)
			rep.Pairs++
			addToBucket(rep.Buckets, gap, obs, rec, scorable)

			if !scorable {
				rep.Ambiguous++
				continue
			}
			rep.Scored++
			// Score the projection against the EFFECTIVE reuse cutoff (#940), not the bare
			// billing TTL: the provider keeps the prefix warm past the billing window, so the
			// effective window is what the live Plan now compares idle time against. This is the
			// change whose effect the ProjColdObsWarm miss rate measures.
			proj, _ := projectPosture(gap, ttl, ttl.EffectiveReuseSeconds(), false) // gap >= 0, so cold or warm, never unknown
			switch {
			case proj == obs:
				rep.Agree++
			default:
				rep.Disagree++
				if proj == PostureCold && obs == PostureWarm {
					rep.ProjColdObsWarm++
				} else if proj == PostureWarm && obs == PostureCold {
					rep.ProjWarmObsCold++
				}
			}
			if proj == PostureCold && obs == PostureCold {
				rep.ConfirmedCold++
				if p := cur.Prompt(); p > 0 {
					coldWriteSum += float64(cur.CacheCreationTokens) / float64(p)
				}
			}
		}
	}
	if rep.Scored > 0 {
		rep.Accuracy = float64(rep.Agree) / float64(rep.Scored)
	}
	if rep.ConfirmedCold > 0 {
		rep.ColdWriteRatioMean = coldWriteSum / float64(rep.ConfirmedCold)
	}
	if rep.FirstTurnCold > 0 {
		rep.FirstTurnColdWriteShareMean = ftColdWriteShareSum / float64(rep.FirstTurnCold)
		rep.FirstTurnColdReprefillTokMean = ftColdReprefillSum / float64(rep.FirstTurnCold)
	}
	finalizeBuckets(rep.Buckets)
	return rep
}

// addToBucket folds one boundary into its gap bucket: counts it, accumulates the recovery for
// the bucket mean, and tallies the observed posture (an unscorable partial re-serve counts as
// ambiguous).
func addToBucket(buckets []GapBucket, gap int64, obs Posture, rec float64, scorable bool) {
	for i := range buckets {
		if gap >= buckets[i].LoSeconds && gap < buckets[i].HiSeconds {
			buckets[i].N++
			buckets[i].MeanRecovery += rec // sum here; finalizeBuckets divides
			switch {
			case !scorable:
				buckets[i].AmbiguousN++
			case obs == PostureWarm:
				buckets[i].WarmN++
			case obs == PostureCold:
				buckets[i].ColdN++
			}
			return
		}
	}
}

// finalizeBuckets converts each bucket's accumulated recovery SUM into a mean in place, after
// all boundaries are folded in.
func finalizeBuckets(buckets []GapBucket) {
	for i := range buckets {
		if buckets[i].N > 0 {
			buckets[i].MeanRecovery /= float64(buckets[i].N)
		}
	}
}

// sortedByTime returns a time-ordered copy of a session's turns, so Backtest stays pure (it
// never reorders the caller's slice) and total (any input order yields the same residual). A
// simple insertion sort: sessions are short and already nearly ordered, so this avoids pulling
// in sort just for the leaf.
func sortedByTime(in []ObservedTurn) []ObservedTurn {
	out := make([]ObservedTurn, len(in))
	copy(out, in)
	for i := 1; i < len(out); i++ {
		for j := i; j > 0 && out[j-1].UnixSeconds > out[j].UnixSeconds; j-- {
			out[j-1], out[j] = out[j], out[j-1]
		}
	}
	return out
}
