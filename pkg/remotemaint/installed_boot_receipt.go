package remotemaint

import "fmt"

func cloneInstalledBootScope(in *InstalledBootScope) *InstalledBootScope {
	if in == nil {
		return nil
	}
	b := *in
	b.ReadPaths = append([]string(nil), in.ReadPaths...)
	b.WritePaths = append([]string(nil), in.WritePaths...)
	b.RequiredNodeIDs = append([]string(nil), in.RequiredNodeIDs...)
	b.LocalScopes = append([]InstalledBootLocalScope(nil), in.LocalScopes...)
	for i := range b.LocalScopes {
		b.LocalScopes[i].Paths = append([]string(nil), in.LocalScopes[i].Paths...)
		b.LocalScopes[i].PathPrefixes = append([]PathPrefix(nil), in.LocalScopes[i].PathPrefixes...)
	}
	return &b
}

func cloneInstalledBootObservation(in *InstalledBootObservation) *InstalledBootObservation {
	if in == nil {
		return nil
	}
	b := *in
	b.Files = append([]InstalledBootFileState(nil), in.Files...)
	return &b
}

func validateInstalledBootReceipt(r Receipt, g Grant) error {
	b, scope := r.InstalledBoot, g.InstalledBoot
	if b == nil || scope == nil || r.Readback != nil || len(r.ReservationLeases) != 0 {
		return fmt.Errorf("%w: missing or cross-kind installed boot observations", ErrAuthority)
	}
	kind := map[string]string{"preflight": "preflight", "stage-plan": "staged_plan", "apply": "applied", "readback": "readback", "rollback": "rolled_back"}[scope.Phase]
	if r.Outcome == OutcomeRolledBack && scope.Phase == "apply" {
		kind = "rolled_back"
	}
	if kind == "" || b.Kind != kind || b.ArtifactSHA256 != scope.ArtifactSHA256 {
		return fmt.Errorf("%w: installed boot observation identity", ErrChanged)
	}
	for _, pair := range [][2]string{{b.StageDigest, scope.StageDigest}, {b.PlanDigest, scope.PlanDigest},
		{b.IdentitySHA256, scope.IdentitySHA256}, {b.ServiceIdentitySHA256, scope.ServiceIdentitySHA256}} {
		if pair[0] != "" && !isDigest(pair[0]) || pair[1] != "" && pair[0] != pair[1] || scope.Phase != "preflight" && !isDigest(pair[0]) {
			return fmt.Errorf("%w: installed boot observation binding", ErrChanged)
		}
	}
	if (scope.Phase == "preflight" || scope.Phase == "readback") && r.Outcome != OutcomeUnchanged ||
		scope.Phase == "stage-plan" && r.Outcome != OutcomeApplied && r.Outcome != OutcomeUnchanged ||
		scope.Phase == "rollback" && r.Outcome != OutcomeRolledBack {
		return fmt.Errorf("%w: installed boot observation outcome", ErrAuthority)
	}
	for _, digest := range []string{b.RootImageSHA256, b.ESPImageSHA256} {
		if digest != "" && !isDigest(digest) {
			return fmt.Errorf("%w: installed boot image hash", ErrAuthority)
		}
	}
	if len(b.Files) > 64 {
		return fmt.Errorf("%w: installed boot metadata bounds", ErrAuthority)
	}
	last := ""
	for _, f := range b.Files {
		allowed := false
		for _, p := range g.Paths {
			if p == f.Path {
				allowed = true
				break
			}
		}
		if !allowed || f.Path <= last || !isDigest(f.SHA256) || len(f.Mode) != 4 {
			return fmt.Errorf("%w: installed boot file scope", ErrAuthority)
		}
		for _, digit := range f.Mode {
			if digit < '0' || digit > '7' {
				return fmt.Errorf("%w: installed boot file mode", ErrAuthority)
			}
		}
		last = f.Path
	}
	return nil
}
