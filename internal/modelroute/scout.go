package modelroute

import "github.com/anthony-chaudhary/fak/pkg/routepolicy"

// The scout classify-first step (Manifest.ScoutRoute) lives in pkg/routepolicy.
type (
	ScoutLabel     = routepolicy.ScoutLabel
	Classifier     = routepolicy.Classifier
	ClassifierFunc = routepolicy.ClassifierFunc
	ScoutOutcome   = routepolicy.ScoutOutcome
)

func ApplyScoutLabel(s Subject, l ScoutLabel) Subject { return routepolicy.ApplyScoutLabel(s, l) }
