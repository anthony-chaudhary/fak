package experiments

const (
	ExperimentSchema    = "fak-experiment/1"
	ExperimentLedgerRel = "experiments/registry.jsonl"
)

type Experiment struct {
	ID           string   `json:"id"`
	Owner        string   `json:"owner"`
	Host         string   `json:"host"`
	Models       []string `json:"models"`
	Backends     []string `json:"backends"`
	Started      string   `json:"started"` // RFC3339
	ArtifactPath string   `json:"artifact_path"`
}

type Cell struct {
	Model   string
	Backend string
}

func (e Experiment) Cells() []Cell {
	var cells []Cell
	for _, m := range e.Models {
		for _, b := range e.Backends {
			cells = append(cells, Cell{Model: m, Backend: b})
		}
	}
	return cells
}

func (e Experiment) Overlaps(other Experiment) bool {
	for _, c1 := range e.Cells() {
		for _, c2 := range other.Cells() {
			if c1.Model == c2.Model && c1.Backend == c2.Backend {
				return true
			}
		}
	}
	return false
}
