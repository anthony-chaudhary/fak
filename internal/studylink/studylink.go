// Package studylink joins stable upstream mechanism clusters to witnessed FAK artifacts.
package studylink

import (
	"errors"
	"sort"
)

type Disposition string

const (
	Landed    Disposition = "landed"
	OpenExact Disposition = "open_exact"
	Partial   Disposition = "partial"
	Conflict  Disposition = "conflict"
	Obsolete  Disposition = "obsolete"
	Uncovered Disposition = "uncovered"
)

type Artifact struct{ Kind, ID, Revision, Path, State string }
type Evidence struct {
	Query   string
	Matches []string
	Digest  string
}
type Join struct {
	ClusterID    string
	Actionable   bool
	Disposition  Disposition
	Artifacts    []Artifact
	Confidence   float64
	Evidence     Evidence
	ManualReview bool
}
type Ledger struct {
	Schema, Cutoff, SourceRevision string
	Joins                          []Join
}
type Summary struct {
	Counts       map[Disposition]int
	ManualReview []string
}

var ErrInvalid = errors.New("studylink: invalid join ledger")

func Validate(l Ledger) error {
	if l.Schema != "fak.study-join-ledger/1" || l.Cutoff == "" || l.SourceRevision == "" {
		return ErrInvalid
	}
	seenCluster := map[string]bool{}
	exact := map[string]string{}
	for _, j := range l.Joins {
		if j.ClusterID == "" || seenCluster[j.ClusterID] || j.Disposition == "" || j.Evidence.Digest == "" {
			return ErrInvalid
		}
		seenCluster[j.ClusterID] = true
		if j.Actionable && j.Disposition == Uncovered && len(j.Artifacts) > 0 {
			return ErrInvalid
		}
		if j.Actionable && j.Disposition == "" {
			return ErrInvalid
		}
		for _, a := range j.Artifacts {
			if a.ID == "" || a.Kind == "" {
				return ErrInvalid
			}
			if j.Disposition == OpenExact && a.Kind == "issue" && a.State != "open" {
				return ErrInvalid
			}
			if j.Disposition == OpenExact {
				key := a.Kind + ":" + a.ID
				if prior := exact[key]; prior != "" && prior != j.ClusterID {
					return ErrInvalid
				}
				exact[key] = j.ClusterID
			}
		}
	}
	return nil
}
func Summarize(l Ledger) (Summary, error) {
	if e := Validate(l); e != nil {
		return Summary{}, e
	}
	s := Summary{Counts: map[Disposition]int{}}
	for _, j := range l.Joins {
		s.Counts[j.Disposition]++
		if j.ManualReview || j.Disposition == Partial || j.Disposition == Conflict {
			s.ManualReview = append(s.ManualReview, j.ClusterID)
		}
	}
	sort.Strings(s.ManualReview)
	return s, nil
}
