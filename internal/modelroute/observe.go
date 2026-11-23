package modelroute

import (
	"time"

	"github.com/anthony-chaudhary/fak/pkg/routepolicy"
)

// The decision journal and rollup live in pkg/routepolicy.
type (
	Strategy        = routepolicy.Strategy
	DecisionRecord  = routepolicy.DecisionRecord
	DecisionJournal = routepolicy.DecisionJournal
	Counts          = routepolicy.Counts
)

const (
	StrategyDefault  = routepolicy.StrategyDefault
	StrategySingle   = routepolicy.StrategySingle
	StrategyEnsemble = routepolicy.StrategyEnsemble
)

func StrategyOf(d Decision) Strategy { return routepolicy.StrategyOf(d) }

func RecordDecision(version string, d Decision, overhead time.Duration) DecisionRecord {
	return routepolicy.RecordDecision(version, d, overhead)
}
