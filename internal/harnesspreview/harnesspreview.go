package harnesspreview

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/harnessclassify"
	"github.com/anthony-chaudhary/fak/internal/harnesscompose"
	"github.com/anthony-chaudhary/fak/internal/harnessresolve"
)

// Schema identifies the harness preview schema format.
const Schema = "fak.harness-preview/v1alpha1"

const (
	// VerdictQuiet indicates no operator decision is required before execution.
	VerdictQuiet = "quiet"
	// VerdictDecision indicates changes require explicit operator choice or confirmation.
	VerdictDecision = "decision-required"
)

// Input captures current and candidate lock state along with classification diagnostics.
type Input struct {
	Current         *harnessresolve.Lock
	Candidate       *harnessresolve.Lock
	CurrentDomain   string
	CandidateDomain string
	Classification  *harnessclassify.Result
	Conflict        string
}

// Change describes an individual risk or behavioral difference between locks.
type Change struct {
	Reason           string `json:"reason"`
	Layer            string `json:"layer"`
	Capability       string `json:"capability"`
	Consequence      string `json:"consequence"`
	ReversibleChoice string `json:"reversible_choice"`
}

// Action represents an operator option to resolve a harness decision requirement.
type Action struct {
	ID          string `json:"id"`
	Description string `json:"description"`
}

// Preview holds the evaluation verdict, delta details, and recovery choices.
type Preview struct {
	Schema           string   `json:"schema"`
	Verdict          string   `json:"verdict"`
	RequiresDecision bool     `json:"requires_decision"`
	CurrentLockID    string   `json:"current_lock_id,omitempty"`
	CandidateLockID  string   `json:"candidate_lock_id,omitempty"`
	Changes          []Change `json:"changes,omitempty"`
	Recovery         []Action `json:"recovery,omitempty"`
}

// Compare assesses current and proposed harness configurations to detect authority widening,
// conflicts, novel domains, and low-confidence classifications.
func Compare(in Input) Preview {
	p := Preview{Schema: Schema, Verdict: VerdictQuiet}
	if in.Current != nil {
		p.CurrentLockID = in.Current.ID
	}
	if in.Candidate != nil {
		p.CandidateLockID = in.Candidate.ID
	}
	if in.Current != nil && in.Candidate != nil && in.Current.ID != "" && in.Current.ID == in.Candidate.ID && in.Conflict == "" && !lowConfidence(in.Classification) && !novelDomain(in) {
		return p
	}
	if strings.TrimSpace(in.Conflict) != "" {
		p.Changes = append(p.Changes, Change{Reason: "conflict", Layer: "resolver", Capability: "effective-product", Consequence: strings.TrimSpace(in.Conflict), ReversibleChoice: "keep the current lock"})
	}
	if lowConfidence(in.Classification) {
		layer := "classifier"
		consequence := "domain inference is not confident enough to select a contextual harness"
		if in.Classification != nil && in.Classification.DecisionRequest != nil {
			if in.Classification.DecisionRequest.Scope != "" {
				layer = in.Classification.DecisionRequest.Scope
			}
			if in.Classification.DecisionRequest.Reason != "" {
				consequence = in.Classification.DecisionRequest.Reason
			}
		}
		p.Changes = append(p.Changes, Change{Reason: "low-confidence", Layer: layer, Capability: "domain-selection", Consequence: consequence, ReversibleChoice: "choose a domain for this scope or keep the current lock"})
	}
	if novelDomain(in) {
		layer := domainLayer(in.Candidate)
		p.Changes = append(p.Changes, Change{Reason: "novel-domain", Layer: layer, Capability: "domain:" + in.CandidateDomain, Consequence: fmt.Sprintf("switch contextual defaults from %s to %s", displayDomain(in.CurrentDomain), in.CandidateDomain), ReversibleChoice: "keep the current lock"})
	}
	if in.Candidate != nil {
		current := harnessresolve.Lock{}
		if in.Current != nil {
			current = *in.Current
		}
		p.Changes = append(p.Changes, wideningChanges(current, *in.Candidate)...)
	}
	sort.SliceStable(p.Changes, func(i, j int) bool {
		a, b := p.Changes[i], p.Changes[j]
		if reasonRank(a.Reason) != reasonRank(b.Reason) {
			return reasonRank(a.Reason) < reasonRank(b.Reason)
		}
		if a.Layer != b.Layer {
			return a.Layer < b.Layer
		}
		return a.Capability < b.Capability
	})
	if len(p.Changes) == 0 {
		return p
	}
	p.Verdict, p.RequiresDecision = VerdictDecision, true
	p.Recovery = []Action{
		{ID: "approve-once", Description: "launch this lock once without changing the remembered lock"},
		{ID: "remember", Description: "admit this lock for the current scope with an expiry"},
		{ID: "keep-current", Description: "continue with the current lock; no state changes"},
	}
	return p
}

func lowConfidence(r *harnessclassify.Result) bool {
	return r != nil && (r.NeedsDecision || r.Confidence < 0.75)
}

func novelDomain(in Input) bool {
	next := strings.TrimSpace(in.CandidateDomain)
	if next == "" {
		return false
	}
	current := strings.TrimSpace(in.CurrentDomain)
	return current == "" || current != next
}

func displayDomain(domain string) string {
	if strings.TrimSpace(domain) == "" {
		return "stock"
	}
	return domain
}

func domainLayer(lock *harnessresolve.Lock) string {
	if lock != nil {
		for i := len(lock.Assets) - 1; i >= 0; i-- {
			if lock.Assets[i].Source != "" {
				return lock.Assets[i].Source
			}
		}
	}
	return "domain"
}

func wideningChanges(old, next harnessresolve.Lock) []Change {
	before := assetMap(old.Assets)
	after := assetMap(next.Assets)
	var changes []Change
	for key, n := range after {
		o, existed := before[key]
		switch n.Kind {
		case "policy":
			for _, grant := range difference(n.Grants, o.Grants) {
				changes = append(changes, risk(n, "policy:"+n.ID+":grant:"+grant, "grants "+grant+" authority"))
			}
			if existed {
				for _, deny := range difference(o.Denies, n.Denies) {
					changes = append(changes, risk(n, "policy:"+n.ID+":deny:"+deny, "removes the "+deny+" denial"))
				}
				if o.Locked && !n.Locked {
					changes = append(changes, risk(n, "policy:"+n.ID+":lock", "removes a locked policy floor"))
				}
			}
		case "tool":
			if !existed {
				changes = append(changes, risk(n, "tool:"+n.ID, "makes the "+n.ID+" tool callable"))
			}
		case "secret":
			if !existed || o.Ref != n.Ref {
				changes = append(changes, risk(n, "secret:"+n.ID, "changes credential access for "+n.ID))
			}
		case "workflow":
			if existed && o.Mandatory && !n.Mandatory {
				changes = append(changes, risk(n, "workflow:"+n.ID+":mandatory", "makes a mandatory workflow optional"))
			} else if existed && assetBehaviorChanged(o, n) {
				changes = append(changes, behaviorChange(n))
			}
		case "instruction", "memory", "route", "ui":
			if existed && assetBehaviorChanged(o, n) {
				changes = append(changes, behaviorChange(n))
			}
		}
	}
	for key, o := range before {
		if _, ok := after[key]; ok {
			continue
		}
		if o.Kind == "policy" && (o.Locked || len(o.Denies) > 0) {
			changes = append(changes, risk(o, "policy:"+o.ID, "removes a policy boundary"))
		}
		if o.Kind == "workflow" && o.Mandatory {
			changes = append(changes, risk(o, "workflow:"+o.ID, "removes a mandatory workflow"))
		}
	}
	return changes
}

func behaviorChange(a harnesscompose.EffectiveAsset) Change {
	layer := a.Source
	if layer == "" {
		layer = "unknown"
	}
	return Change{Reason: "behavior-change", Layer: layer, Capability: a.Kind + ":" + a.ID, Consequence: "changes effective " + a.Kind + " behavior", ReversibleChoice: "keep the current lock"}
}

func assetBehaviorChanged(current, candidate harnesscompose.EffectiveAsset) bool {
	return current.Value != candidate.Value || current.Ref != candidate.Ref || current.Boundary != candidate.Boundary
}
func risk(a harnesscompose.EffectiveAsset, capability, consequence string) Change {
	layer := a.Source
	if layer == "" {
		layer = "unknown"
	}
	return Change{Reason: "privilege-widening", Layer: layer, Capability: capability, Consequence: consequence, ReversibleChoice: "keep the current lock"}
}

func assetMap(assets []harnesscompose.EffectiveAsset) map[string]harnesscompose.EffectiveAsset {
	out := make(map[string]harnesscompose.EffectiveAsset, len(assets))
	for _, a := range assets {
		out[a.Kind+"\x00"+a.ID] = a
	}
	return out
}

func difference(a, b []string) []string {
	seen := make(map[string]bool, len(b))
	for _, v := range b {
		seen[v] = true
	}
	out := make([]string, 0)
	for _, v := range a {
		if !seen[v] {
			out = append(out, v)
		}
	}
	sort.Strings(out)
	return out
}

func reasonRank(reason string) int {
	switch reason {
	case "conflict":
		return 0
	case "privilege-widening":
		return 1
	case "novel-domain":
		return 2
	default:
		return 3
	}
}

// RenderCLI formats the preview as concise plain text suitable for terminal output.
func RenderCLI(p Preview) string { return render(p, false) }

// RenderTUI formats the preview for text user interfaces without ANSI sequences.
func RenderTUI(p Preview) string { return render(p, true) }

func render(p Preview, tui bool) string {
	if !p.RequiresDecision {
		return ""
	}
	var b strings.Builder
	title := "contextual harness decision required"
	if tui {
		title = "HARNESS PREVIEW | decision required"
	}
	b.WriteString(title + "\n")
	for _, c := range p.Changes {
		fmt.Fprintf(&b, "- %s | %s | %s\n  %s; choice: %s\n", c.Reason, c.Layer, c.Capability, c.Consequence, c.ReversibleChoice)
	}
	b.WriteString("choices: approve-once | remember | keep-current\n")
	return b.String()
}
