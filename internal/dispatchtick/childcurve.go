package dispatchtick

import (
	"fmt"
	"path/filepath"
	"strconv"

	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

// ChildCurve is parent-authored read-back from the trajectory ledger. Objective
// ids are accepted in the stable issue forms used by dispatch integrations;
// absence stays explicit rather than being replaced by worker narration.
func ChildCurve(root string, issue int) map[string]any {
	st := trajctl.Fold(trajctl.ReadLedgerFile(filepath.Join(root, trajctl.DefaultLedgerRel)))
	for _, id := range []string{fmt.Sprintf("issue-%d", issue), "#" + strconv.Itoa(issue)} {
		if curve, ok := st.CurveFor(id); ok {
			return map[string]any{
				"present":      true,
				"objective_id": curve.ObjectiveID,
				"signal":       curve.Signal,
				"latest":       curve.Latest,
				"delta":        curve.Delta,
				"detail":       curve.Detail,
			}
		}
	}
	return map[string]any{"present": false, "reason": "no witnessed trajectory curve for issue objective"}
}
