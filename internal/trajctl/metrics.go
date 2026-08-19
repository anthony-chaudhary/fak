package trajctl

import (
	"os"
	"sort"
	"sync"
	"time"
)

// MetricsSnapshot is the bounded-cardinality projection of a trajectory ledger.
type MetricsSnapshot struct {
	Objectives map[ObjectiveStatus]int
	Scores     map[string]float64
	Signals    map[Signal]int
	Nudges     map[string]int
}

// Metrics folds objective health without carrying objective IDs into metric labels.
func (s State) Metrics() MetricsSnapshot {
	m := MetricsSnapshot{
		Objectives: map[ObjectiveStatus]int{StatusActive: 0, StatusPaused: 0, StatusMet: 0, StatusAbandoned: 0},
		Scores:     map[string]float64{"root": 0, "child": 0, "scorer": 0},
		Signals:    map[Signal]int{SignalHealthy: 0, SignalStall: 0, SignalDrift: 0},
		Nudges:     map[string]int{"delivered": 0, "failed": 0},
	}
	for _, o := range s.Objectives {
		m.Objectives[o.Status]++
	}
	curves := s.OpenCurves().Objectives
	sums := map[string]float64{}
	counts := map[string]int{}
	for _, c := range curves {
		kind := objectiveKind(s.Objectives[c.ObjectiveID])
		sums[kind] += c.Latest
		counts[kind]++
		m.Signals[c.Signal]++
	}
	for kind, n := range counts {
		if n > 0 {
			m.Scores[kind] = sums[kind] / float64(n)
		}
	}
	for _, d := range s.Steers {
		if d.Action != ActionNudge {
			continue
		}
		if d.Delivered {
			m.Nudges["delivered"]++
		} else {
			m.Nudges["failed"]++
		}
	}
	return m
}

func objectiveKind(o Objective) string {
	if o.Meta != nil {
		return "scorer"
	}
	if o.ParentID != "" {
		return "child"
	}
	return "root"
}

// MetricKinds documents the fixed score label vocabulary for adapters.
func MetricKinds() []string { return []string{"child", "root", "scorer"} }

// MetricStatuses documents the fixed lifecycle label vocabulary for adapters.
func MetricStatuses() []ObjectiveStatus {
	out := []ObjectiveStatus{StatusAbandoned, StatusActive, StatusMet, StatusPaused}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// MetricsFile caches a ledger projection until the file identity changes.
type MetricsFile struct {
	path  string
	mu    sync.Mutex
	mod   time.Time
	size  int64
	value MetricsSnapshot
}

func NewMetricsFile(path string) *MetricsFile { return &MetricsFile{path: path} }
func (f *MetricsFile) Snapshot() MetricsSnapshot {
	if f == nil {
		return State{}.Metrics()
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	info, err := os.Stat(f.path)
	if err != nil {
		return State{}.Metrics()
	}
	if info.ModTime().Equal(f.mod) && info.Size() == f.size {
		return f.value
	}
	f.value = Fold(ReadLedgerFile(f.path)).Metrics()
	f.mod = info.ModTime()
	f.size = info.Size()
	return f.value
}

type SessionObjective struct {
	ObjectiveID     string  `json:"objective_id"`
	Title           string  `json:"title"`
	Score           float64 `json:"score"`
	Signal          Signal  `json:"signal"`
	Nudges          int     `json:"nudges"`
	NudgesDelivered int     `json:"nudges_delivered"`
}
type SessionPanel struct {
	Objectives []SessionObjective `json:"objectives"`
}

func (s State) SessionPanel(sessionID string) *SessionPanel {
	ids := map[string]bool{}
	for _, row := range s.Scores {
		if row.SessionID == sessionID {
			ids[row.ObjectiveID] = true
		}
	}
	for _, d := range s.Steers {
		if d.SessionID == sessionID {
			ids[d.ObjectiveID] = true
		}
	}
	if len(ids) == 0 {
		return nil
	}
	p := &SessionPanel{Objectives: []SessionObjective{}}
	for _, id := range s.ObjectiveIDs() {
		if !ids[id] {
			continue
		}
		c, ok := s.CurveFor(id)
		if !ok {
			continue
		}
		item := SessionObjective{ObjectiveID: id, Title: s.Objectives[id].Statement, Score: c.Latest, Signal: c.Signal}
		for _, d := range s.SteersFor(id) {
			if d.SessionID == sessionID && d.Action == ActionNudge {
				item.Nudges++
				if d.Delivered {
					item.NudgesDelivered++
				}
			}
		}
		p.Objectives = append(p.Objectives, item)
	}
	return p
}
