package stopgate

// EvaluateBoundary unifies turn-boundary lifecycle adjudication across harness architectures.
func EvaluateBoundary(ladder LadderConfig, witnessCfg WitnessGateConfig, in BoundaryInput) Decision {
	// 1. Clean wrap-up: if agent explicitly noted "no allowed path", that is a sanctioned clean stop.
	if in.NotedNoAllowedPath {
		return Decision{
			Action:      ActionAllow,
			Stage:       StageAllow,
			Disposition: DispCleanWrapup,
			Kind:        KindClean,
			ExitCode:    0,
			Signal:      "clean",
		}
	}

	// 2. Tool feedback check: if consecutive deny-all is 0 but tool feedback consecutive > 0.
	if in.ConsecutiveDenyAll <= 0 && in.ConsecutiveToolFeedback > 0 {
		return EvaluateToolFeedback(ladder, in.ConsecutiveToolFeedback)
	}

	// 3. Deny-all check: evaluate graduated back-off ladder.
	if in.ConsecutiveDenyAll > 0 {
		return EvaluateDenyAll(ladder, in.ConsecutiveDenyAll, in.ConsecutiveSameIssue, in.UseSameIssue)
	}

	// 4. Witness check: either via explicit WitnessClaim or via FinalGate check callback.
	if in.WitnessClaim != nil && in.WitnessClaim.Claimed {
		dec := EvaluateWitness(witnessCfg, *in.WitnessClaim, in.WitnessBlockCount)
		if dec.ShouldContinue() {
			return dec
		}
		if dec.Disposition != DispCleanCompletion {
			return dec
		}
	} else if in.FinalGate != nil {
		satisfied, missing := in.FinalGate()
		if !satisfied {
			claim := WitnessClaim{
				Claimed:   true,
				Witnessed: false,
				Reason:    "STOP_UNWITNESSED",
				Detail:    missing,
			}
			fgWitnessCfg := witnessCfg
			if fgWitnessCfg.Mode == "" || fgWitnessCfg.Mode == ModeShadow {
				fgWitnessCfg.Mode = ModeEnforce
			}
			dec := EvaluateWitness(fgWitnessCfg, claim, in.WitnessBlockCount)
			if dec.ShouldContinue() {
				return dec
			}
			if dec.Disposition != DispCleanCompletion {
				return dec
			}
		}
	}

	// 5. Default clean completion.
	return Decision{
		Action:      ActionAllow,
		Stage:       StageAllow,
		Disposition: DispCleanCompletion,
		Kind:        KindClean,
		ExitCode:    0,
		Signal:      "clean",
	}
}
