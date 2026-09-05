package stopgate

// EvaluateBoundary unifies turn-boundary lifecycle adjudication across harness architectures.
func EvaluateBoundary(ladder LadderConfig, witnessCfg WitnessGateConfig, in BoundaryInput) Decision {
	// 1. Witness check gate: if FinalGate is unsatisfied or WitnessClaim is unwitnessed,
	// NotedNoAllowedPath must NOT bypass the witness gate.
	if in.FinalGate != nil {
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
			if dec.Disposition != DispCleanCompletion && dec.Disposition != DispClaimWitnessed {
				return dec
			}
		}
	} else if in.WitnessClaim != nil && in.WitnessClaim.Claimed {
		dec := EvaluateWitness(witnessCfg, *in.WitnessClaim, in.WitnessBlockCount)
		if dec.ShouldContinue() {
			return dec
		}
		if dec.Disposition != DispCleanCompletion && dec.Disposition != DispClaimWitnessed {
			return dec
		}
	} else if in.NotedNoAllowedPath && (witnessCfg.Mode == ModeEnforce || (in.WitnessClaim != nil && !in.WitnessClaim.Witnessed)) {
		claim := WitnessClaim{
			Claimed:   true,
			Witnessed: false,
			Reason:    "STOP_UNWITNESSED",
			Detail:    "missing witness claim",
		}
		if in.WitnessClaim != nil {
			claim = *in.WitnessClaim
			claim.Claimed = true
		}
		dec := EvaluateWitness(witnessCfg, claim, in.WitnessBlockCount)
		if dec.ShouldContinue() {
			return dec
		}
		if dec.Disposition != DispCleanCompletion && dec.Disposition != DispClaimWitnessed {
			return dec
		}
	}

	// 2. Clean wrap-up: if agent explicitly noted "no allowed path", that is a sanctioned clean stop
	// only when witness requirements are satisfied or not in enforce mode.
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

	// 3. Tool feedback check: if consecutive deny-all is 0 but tool feedback consecutive > 0.
	if in.ConsecutiveDenyAll <= 0 && in.ConsecutiveToolFeedback > 0 {
		return EvaluateToolFeedback(ladder, in.ConsecutiveToolFeedback)
	}

	// 4. Deny-all check: evaluate graduated back-off ladder.
	if in.ConsecutiveDenyAll > 0 {
		return EvaluateDenyAll(ladder, in.ConsecutiveDenyAll, in.ConsecutiveSameIssue, in.UseSameIssue)
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
