package workshape

import (
	"fmt"
	"slices"
	"strings"
)

type Kind string

const (
	Bounded         Kind = "BOUNDED"
	Broad           Kind = "BROAD"
	BlockedExternal Kind = "BLOCKED_EXTERNAL"
)

type Packet struct {
	ID             string   `json:"id"`
	Role           string   `json:"role"`
	Tree           []string `json:"tree"`
	Witness        string   `json:"witness"`
	Inputs         []string `json:"inputs,omitempty"`
	Outputs        []string `json:"outputs,omitempty"`
	Acceptance     []string `json:"acceptance,omitempty"`
	DependsOn      []string `json:"depends_on,omitempty"`
	SharedContract []string `json:"shared_contracts,omitempty"`
}

type Contract struct {
	DeclaredKind     Kind     `json:"kind"`
	Evidence         []string `json:"evidence"`
	RootSpine        string   `json:"root_spine,omitempty"`
	Packets          []Packet `json:"packets,omitempty"`
	IntegrationOwner string   `json:"integration_owner,omitempty"`
	ExternalBlocker  string   `json:"external_blocker,omitempty"`
	FitsDeadline     bool     `json:"fits_deadline"`
}

type Result struct {
	Kind             Kind       `json:"kind"`
	Evidence         []string   `json:"evidence"`
	RootSpine        string     `json:"root_spine,omitempty"`
	Packets          []Packet   `json:"packets,omitempty"`
	IntegrationOwner string     `json:"integration_owner,omitempty"`
	ParentCapacity   int        `json:"parent_capacity"`
	ChildCapacity    int        `json:"child_capacity"`
	TotalCapacity    int        `json:"total_capacity"`
	ParallelGroups   [][]string `json:"parallel_groups,omitempty"`
	Serialized       []string   `json:"serialized,omitempty"`
	SemanticVerdict  string     `json:"semantic_verdict,omitempty"`
	SemanticReasons  []string   `json:"semantic_reasons,omitempty"`
	Verdict          string     `json:"verdict"`
	Reason           string     `json:"reason"`
}

func Evaluate(c Contract) Result {
	r := Result{Kind: c.DeclaredKind, Evidence: slices.Clone(c.Evidence), RootSpine: c.RootSpine, Packets: slices.Clone(c.Packets), IntegrationOwner: c.IntegrationOwner, ParentCapacity: 1}
	switch c.DeclaredKind {
	case BlockedExternal:
		r.TotalCapacity = 0
		r.Verdict = "REFUSE_BLOCKED_EXTERNAL"
		r.Reason = first(c.ExternalBlocker, "external blocker was not named")
		return r
	case Bounded:
		r.TotalCapacity = 1
		if !c.FitsDeadline || len(c.Packets) > 1 || hasDependencies(c.Packets) {
			r.Verdict = "REFUSE_FALSE_BOUNDED"
			r.Reason = "bounded work must fit one worker and contain no unresolved child packets or dependencies"
			return r
		}
		r.Verdict, r.Reason = "ADMIT_BOUNDED", "one issue owner fits the declared execution envelope"
		return r
	case Broad:
		if strings.TrimSpace(c.RootSpine) == "" || strings.TrimSpace(c.IntegrationOwner) == "" || len(c.Packets) == 0 {
			r.Verdict = "REFUSE_MALFORMED_BROAD"
			r.Reason = "broad work requires a root spine, integration owner, and executable child packets"
			return r
		}
		for _, p := range c.Packets {
			if p.ID == "" || len(p.Tree) == 0 || p.Witness == "" {
				r.Verdict = "REFUSE_MALFORMED_BROAD"
				r.Reason = fmt.Sprintf("packet %q lacks id, exact tree, or witness", p.ID)
				return r
			}
		}
		r.ChildCapacity = len(c.Packets)
		r.TotalCapacity = 1 + r.ChildCapacity
		r.ParallelGroups, r.Serialized, r.SemanticVerdict, r.SemanticReasons = schedule(c.Packets)
		if strings.HasPrefix(r.SemanticVerdict, "REFUSE_") {
			r.Verdict = r.SemanticVerdict
			r.Reason = strings.Join(r.SemanticReasons, "; ")
			return r
		}
		r.Verdict, r.Reason = "ADMIT_BROAD", "reserve one issue owner plus independently executable child capacity"
		return r
	default:
		r.Kind = BlockedExternal
		r.TotalCapacity = 0
		r.Verdict = "REFUSE_UNSUPPORTED_WORK_SHAPE"
		r.Reason = "work shape is absent or unsupported"
		return r
	}
}

func hasDependencies(ps []Packet) bool {
	for _, p := range ps {
		if len(p.DependsOn) > 0 || len(p.SharedContract) > 0 {
			return true
		}
	}
	return false
}

func schedule(ps []Packet) ([][]string, []string, string, []string) {
	byID := map[string]Packet{}
	acceptanceOwner := map[string]string{}
	var reasons []string
	for _, p := range ps {
		if _, exists := byID[p.ID]; exists {
			reasons = append(reasons, "duplicate packet id "+p.ID)
		}
		byID[p.ID] = p
		for _, item := range p.Acceptance {
			if owner, exists := acceptanceOwner[item]; exists {
				reasons = append(reasons, "acceptance item "+item+" claimed by "+owner+" and "+p.ID)
			} else {
				acceptanceOwner[item] = p.ID
			}
		}
	}
	for _, p := range ps {
		for _, dep := range p.DependsOn {
			if _, ok := byID[dep]; !ok {
				reasons = append(reasons, "packet "+p.ID+" depends on unknown packet "+dep)
			}
		}
	}
	if cycle := dependencyCycle(ps); len(cycle) > 0 {
		reasons = append(reasons, "dependency cycle: "+strings.Join(cycle, " -> "))
	}
	if len(reasons) > 0 {
		return nil, nil, "REFUSE_SEMANTIC_DEPENDENCY", reasons
	}

	shared := map[string][]string{}
	for _, p := range ps {
		for _, contract := range p.SharedContract {
			shared[contract] = append(shared[contract], p.ID)
		}
	}
	serialize := map[string]bool{}
	for contract, owners := range shared {
		if len(owners) > 1 {
			for _, id := range owners {
				serialize[id] = true
			}
			reasons = append(reasons, "shared contract "+contract+" serializes "+strings.Join(owners, ","))
		}
	}
	var parallel, serial []string
	for _, p := range ps {
		if len(p.DependsOn) == 0 && !serialize[p.ID] {
			parallel = append(parallel, p.ID)
		} else {
			serial = append(serial, p.ID)
		}
	}
	verdict := "SEMANTIC_PARALLEL"
	if len(serial) > 0 {
		verdict = "SEMANTIC_SERIALIZED"
	}
	if len(parallel) == 0 {
		return nil, serial, verdict, reasons
	}
	return [][]string{parallel}, serial, verdict, reasons
}

func dependencyCycle(ps []Packet) []string {
	graph := map[string][]string{}
	for _, p := range ps {
		graph[p.ID] = append([]string(nil), p.DependsOn...)
	}
	state := map[string]int{}
	var stack []string
	var visit func(string) []string
	visit = func(id string) []string {
		if state[id] == 1 {
			for i, s := range stack {
				if s == id {
					return append(append([]string(nil), stack[i:]...), id)
				}
			}
		}
		if state[id] == 2 {
			return nil
		}
		state[id] = 1
		stack = append(stack, id)
		for _, dep := range graph[id] {
			if cycle := visit(dep); len(cycle) > 0 {
				return cycle
			}
		}
		stack = stack[:len(stack)-1]
		state[id] = 2
		return nil
	}
	for id := range graph {
		if cycle := visit(id); len(cycle) > 0 {
			return cycle
		}
	}
	return nil
}

func first(v, fallback string) string {
	if strings.TrimSpace(v) != "" {
		return v
	}
	return fallback
}
