package trajctl

const FleetSchema = "fak-trajctl-fleet/1"

type FleetObjective struct {
	ObjectiveID     string  `json:"objective_id"`
	Title           string  `json:"title"`
	Progress        float64 `json:"progress"`
	Signal          Signal  `json:"signal"`
	Sessions        int     `json:"sessions"`
	OpenDescendants int     `json:"open_descendants"`
}
type FleetReport struct {
	Schema     string           `json:"schema"`
	Objectives []FleetObjective `json:"objectives"`
}

func (s State) Fleet() FleetReport {
	rep := FleetReport{Schema: FleetSchema, Objectives: []FleetObjective{}}
	roll := s.Rollup()
	byID := map[string]ObjectiveRollup{}
	for _, n := range roll.Nodes {
		byID[n.ObjectiveID] = n
	}
	for _, rootID := range roll.Roots {
		root := byID[rootID]
		sessions := map[string]bool{}
		ids := s.subtreeIDs(rootID)
		for _, row := range s.Scores {
			if ids[row.ObjectiveID] && row.SessionID != "" {
				sessions[row.SessionID] = true
			}
		}
		for _, d := range s.Steers {
			if ids[d.ObjectiveID] && d.SessionID != "" {
				sessions[d.SessionID] = true
			}
		}
		rep.Objectives = append(rep.Objectives, FleetObjective{ObjectiveID: rootID, Title: s.Objectives[rootID].Statement, Progress: root.AggProgress, Signal: root.Signal, Sessions: len(sessions), OpenDescendants: countOpenRollup(rootID, byID)})
	}
	return rep
}

func (s State) subtreeIDs(id string) map[string]bool {
	out := map[string]bool{id: true}
	for _, child := range s.childrenOf(id) {
		for x := range s.subtreeIDs(child) {
			out[x] = true
		}
	}
	return out
}
func countOpenRollup(id string, nodes map[string]ObjectiveRollup) int {
	v := 0
	for _, childID := range nodes[id].Children {
		child := nodes[childID]
		if objectiveOpen(child.Status) {
			v++
		}
		v += countOpenRollup(childID, nodes)
	}
	return v
}
