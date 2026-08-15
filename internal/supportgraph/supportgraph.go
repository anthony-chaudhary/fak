// Package supportgraph queries provenance-bearing hardware and quantization support facts.
package supportgraph

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"
)

const Schema = "fak-support-graph/1"

type State string

const (
	Supported   State = "supported"
	Unsupported State = "unsupported"
	Unknown     State = "unknown"
	Stale       State = "stale"
	Conflict    State = "conflict"
)

type Tier string

const (
	Vendor    Tier = "vendor"
	Observed  Tier = "observed"
	Witnessed Tier = "witnessed"
)

var rank = map[Tier]int{Vendor: 1, Observed: 2, Witnessed: 3}

type Tuple struct {
	Artifact     string `json:"artifact"`
	Architecture string `json:"architecture"`
	Quant        string `json:"quant"`
	Layout       string `json:"layout"`
	Backend      string `json:"backend"`
	Kernel       string `json:"kernel"`
	Runtime      string `json:"runtime"`
	Hardware     string `json:"hardware"`
}
type Evidence struct {
	ID             string    `json:"id"`
	State          State     `json:"state"`
	Tier           Tier      `json:"tier"`
	Authority      string    `json:"authority"`
	Source         string    `json:"source"`
	Expires        time.Time `json:"expires,omitempty"`
	Fallback       string    `json:"fallback,omitempty"`
	Penalty        string    `json:"penalty,omitempty"`
	ArtifactDigest string    `json:"artifact_digest,omitempty"`
	Environment    string    `json:"environment,omitempty"`
	Reproduce      string    `json:"reproduce,omitempty"`
}
type Edge struct {
	Tuple       Tuple      `json:"tuple"`
	Required    []string   `json:"required_baseline"`
	Recommended []string   `json:"recommended_baseline,omitempty"`
	Evidence    []Evidence `json:"evidence"`
}
type Graph struct {
	Schema string `json:"schema"`
	Edges  []Edge `json:"edges"`
}
type Result struct {
	Schema      string     `json:"schema"`
	State       State      `json:"state"`
	Tuple       Tuple      `json:"tuple"`
	Required    []string   `json:"required_baseline,omitempty"`
	Recommended []string   `json:"recommended_baseline,omitempty"`
	Decisive    []Evidence `json:"decisive_evidence,omitempty"`
	Reason      string     `json:"reason"`
	Fallback    string     `json:"fallback,omitempty"`
	Penalty     string     `json:"penalty,omitempty"`
}

func Parse(raw []byte) (Graph, error) {
	var g Graph
	d := json.NewDecoder(strings.NewReader(string(raw)))
	d.DisallowUnknownFields()
	if err := d.Decode(&g); err != nil {
		return g, err
	}
	if g.Schema != Schema {
		return g, fmt.Errorf("schema %q, want %q", g.Schema, Schema)
	}
	return g, nil
}
func Query(g Graph, q Tuple, asOf time.Time) Result {
	r := Result{Schema: Schema, State: Unknown, Tuple: q, Reason: "no exact support edge; not evaluated"}
	var edge *Edge
	for i := range g.Edges {
		if g.Edges[i].Tuple == q {
			edge = &g.Edges[i]
			break
		}
	}
	if edge == nil {
		return r
	}
	r.Required = append([]string(nil), edge.Required...)
	r.Recommended = append([]string(nil), edge.Recommended...)
	current := []Evidence{}
	stale := []Evidence{}
	for _, e := range edge.Evidence {
		if !e.Expires.IsZero() && !asOf.Before(e.Expires) {
			stale = append(stale, e)
		} else {
			current = append(current, e)
		}
	}
	if len(current) == 0 {
		r.State = Stale
		r.Reason = "all matching evidence expired"
		r.Decisive = stale
		return r
	}
	max := 0
	for _, e := range current {
		if rank[e.Tier] > max {
			max = rank[e.Tier]
		}
	}
	for _, e := range current {
		if rank[e.Tier] == max {
			r.Decisive = append(r.Decisive, e)
		}
	}
	sort.Slice(r.Decisive, func(i, j int) bool { return r.Decisive[i].ID < r.Decisive[j].ID })
	states := map[State]bool{}
	for _, e := range r.Decisive {
		states[e.State] = true
		if e.Fallback != "" {
			r.Fallback = e.Fallback
			r.Penalty = e.Penalty
		}
	}
	if len(states) > 1 {
		r.State = Conflict
		r.Reason = "equal-authority evidence conflicts"
		return r
	}
	for s := range states {
		r.State = s
	}
	r.Reason = "highest-tier current evidence decides exact tuple"
	return r
}

// ArtifactRuntimeRequest is the narrow bridge consumed from #6224's adjudicator output.
type ArtifactRuntimeRequest struct{ Artifact, Architecture, Quant, Layout, Backend, Kernel, Runtime, Hardware string }

func (q ArtifactRuntimeRequest) Tuple() Tuple {
	return Tuple{q.Artifact, q.Architecture, q.Quant, q.Layout, q.Backend, q.Kernel, q.Runtime, q.Hardware}
}
