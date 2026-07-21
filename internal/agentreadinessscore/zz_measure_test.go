package agentreadinessscore

import (
	"fmt"
	"testing"
)

func TestZZMeasureRepo(t *testing.T) {
	p := Build(`C:/work/fak`)
	fmt.Printf("MEASURE friction_debt=%v score=%v grade=%v\n", p.Corpus["friction_debt"], p.Corpus["score"], p.Corpus["grade"])
	for _, k := range p.KPIs {
		if len(k.Defects) > 0 {
			fmt.Printf("MEASURE_KPI %s debt=%d\n", k.Kpi, len(k.Defects))
			for _, d := range k.Defects {
				fmt.Printf("   DEFECT %s\n", d)
			}
		}
	}
}
