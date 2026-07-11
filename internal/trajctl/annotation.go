package trajctl

import "sort"

const KindAnnotation = "annotation"

type Annotation struct {
	ObjectiveID string `json:"objective_id"`
	Signal      Signal `json:"signal"`
	Detail      string `json:"detail,omitempty"`
	UnixMillis  int64  `json:"unix_millis"`
}

func AnnotationEntry(a Annotation) Row {
	return Row{Schema: Schema, Kind: KindAnnotation, Annotation: &a}
}

// SignalAnnotations derives append-only, non-interacting observations for every
// non-healthy open curve. These rows alter no session behavior.
func (s State) SignalAnnotations(nowMillis int64) []Row {
	report := s.OpenCurves()
	rows := make([]Row, 0, len(report.Objectives))
	for _, c := range report.Objectives {
		if c.Signal == SignalHealthy {
			continue
		}
		rows = append(rows, AnnotationEntry(Annotation{ObjectiveID: c.ObjectiveID, Signal: c.Signal, Detail: c.Detail, UnixMillis: nowMillis}))
	}
	return rows
}

func annotationsFor(rows []Annotation, objectiveID string) []Annotation {
	out := make([]Annotation, 0)
	for _, a := range rows {
		if a.ObjectiveID == objectiveID {
			out = append(out, a)
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].UnixMillis < out[j].UnixMillis })
	return out
}
