package reachdelta

import (
	"fmt"
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/knownbad"
)

type Category string

const (
	NewToolPermitted         Category = "new_tool_permitted"
	NewToolPrefixPermitted   Category = "new_tool_prefix_permitted"
	NewEgressHostPermitted   Category = "new_egress_host_permitted"
	DefaultPostureWidened    Category = "default_posture_widened"
	ExplicitDenyRemoved      Category = "explicit_deny_removed"
	SelfModifyProtectionLost Category = "self_modify_protection_lost"
	ArgumentProtectionLost   Category = "argument_protection_lost"
	RedactionProtectionLost  Category = "redaction_protection_lost"
	EgressDenyRemoved        Category = "egress_deny_removed"
	ComplainModeAdded        Category = "complain_mode_added"
	AdvisoryModeAdded        Category = "advisory_mode_added"
	SecretPostureWidened     Category = "secret_posture_widened"
	SecretProtectionLost     Category = "secret_protection_lost"
	WriteLintDisabled        Category = "write_lint_disabled"
)

type Finding struct {
	Category Category `json:"category"`
	Path     string   `json:"path"`
}

func (f Finding) Signature() string { return fmt.Sprintf("reachdelta:%s:%s", f.Category, f.Path) }

func Delta(base, proposed adjudicator.Policy) []Finding {
	var out []Finding
	for k, ok := range proposed.Allow {
		// A tool already in complain mode was admitted-and-logged by base. Moving it
		// to Allow changes observability, not reach, so promotion is an empty delta.
		if ok && !base.Allow[k] && !base.Complain[k] && !prefixCovers(base.AllowPrefix, k) {
			out = append(out, Finding{NewToolPermitted, k})
		}
	}
	for _, prefix := range proposed.AllowPrefix {
		if prefix = strings.TrimSpace(prefix); prefix != "" && !prefixCovers(base.AllowPrefix, prefix) {
			out = append(out, Finding{NewToolPrefixPermitted, prefix})
		}
	}
	for _, host := range proposed.ResearchEgressAllowHosts {
		if host = normalizeHost(host); host != "" && !hostCovered(base.ResearchEgressAllowHosts, host) {
			out = append(out, Finding{NewEgressHostPermitted, host})
		}
	}
	if base.Posture != adjudicator.PostureAdmitAndLog && proposed.Posture == adjudicator.PostureAdmitAndLog {
		out = append(out, Finding{DefaultPostureWidened, "default"})
	}
	for k, reason := range base.Deny {
		if proposedReason, ok := proposed.Deny[k]; !ok || proposedReason != reason {
			out = append(out, Finding{ExplicitDenyRemoved, k})
		}
	}
	out = appendRemoved(out, SelfModifyProtectionLost, base.SelfModifyGlobs, proposed.SelfModifyGlobs)
	out = appendRemoved(out, ArgumentProtectionLost, argRuleIDs(base), argRuleIDs(proposed))
	out = appendRemoved(out, RedactionProtectionLost, base.RedactFields, proposed.RedactFields)
	out = appendRemoved(out, EgressDenyRemoved, base.EgressExtraDenyHosts, proposed.EgressExtraDenyHosts)
	for tool, on := range proposed.Complain {
		if on && !base.Complain[tool] && !base.Allow[tool] && !prefixCovers(base.AllowPrefix, tool) {
			out = append(out, Finding{ComplainModeAdded, tool})
		}
	}
	for reason, on := range proposed.AdvisoryReasons {
		if on && !base.AdvisoryReasons[reason] {
			out = append(out, Finding{AdvisoryModeAdded, abi.ReasonName(reason)})
		}
	}
	if secretPostureRank(proposed.SecretPosture) > secretPostureRank(base.SecretPosture) {
		out = append(out, Finding{SecretPostureWidened, proposed.SecretPosture.String()})
	}
	out = appendRemoved(out, SecretProtectionLost, regexIDs(base.SecretPatterns), regexIDs(proposed.SecretPatterns))
	if base.LintWrites && !proposed.LintWrites {
		out = append(out, Finding{WriteLintDisabled, "lint_writes"})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Category != out[j].Category {
			return out[i].Category < out[j].Category
		}
		return out[i].Path < out[j].Path
	})
	return out
}
func prefixCovers(base []string, value string) bool {
	for _, prefix := range base {
		if strings.HasPrefix(value, strings.TrimSpace(prefix)) {
			return true
		}
	}
	return false
}
func normalizeHost(host string) string {
	return strings.ToLower(strings.TrimSuffix(strings.TrimSpace(host), "."))
}
func hostCovered(base []string, host string) bool {
	host = normalizeHost(host)
	for _, raw := range base {
		b := normalizeHost(raw)
		if host == b || strings.HasSuffix(host, "."+b) {
			return true
		}
	}
	return false
}

func appendRemoved(out []Finding, cat Category, base, proposed []string) []Finding {
	seen := stringSet(proposed)
	for _, v := range base {
		v = strings.TrimSpace(v)
		if v != "" && !seen[v] {
			out = append(out, Finding{cat, v})
		}
	}
	return out
}
func stringSet(in []string) map[string]bool {
	m := map[string]bool{}
	for _, v := range in {
		if v = strings.TrimSpace(v); v != "" {
			m[v] = true
		}
	}
	return m
}
func regexIDs(in []*regexp.Regexp) []string {
	out := make([]string, 0, len(in))
	for _, r := range in {
		if r != nil {
			out = append(out, r.String())
		}
	}
	return out
}
func secretPostureRank(p adjudicator.SecretPosture) int {
	switch p {
	case adjudicator.SecretAdmitAndLog:
		return 2
	case adjudicator.SecretQuarantine:
		return 1
	default:
		return 0
	}
}
func argRuleIDs(p adjudicator.Policy) []string {
	out := make([]string, 0, len(p.ArgPredicates))
	for _, r := range p.ArgPredicates {
		matcher := r.Glob
		if r.Re != nil {
			matcher = r.Re.String()
		}
		if r.N != 0 {
			matcher = fmt.Sprint(r.N)
		}
		out = append(out, fmt.Sprintf("%s.%s:%d:%s:%s:%t", r.Tool, r.Arg, r.Kind, matcher, abi.ReasonName(r.Reason), r.Advisory))
	}
	return out
}

func SuppressAccepted(findings []Finding, records []knownbad.Record, nowUnix int64) []Finding {
	latest := map[string]knownbad.Record{}
	for _, r := range records {
		latest[strings.TrimSpace(r.Signature)] = r
	}
	out := make([]Finding, 0, len(findings))
	for _, f := range findings {
		if r, ok := latest[f.Signature()]; ok && r.Live(nowUnix) && acceptedRiskClass(r.ReasonClass) {
			continue
		}
		out = append(out, f)
	}
	return out
}

func acceptedRiskClass(s string) bool {
	s = strings.ToLower(strings.TrimSpace(s))
	return s == "accepted-risk" || s == "reach-delta-accepted-risk"
}

type RatifyVerdict struct {
	Approved bool      `json:"approved"`
	Review   []Finding `json:"review,omitempty"`
	Reason   string    `json:"reason"`
}

func Ratify(enabled bool, base, proposed adjudicator.Policy, accepted []knownbad.Record, nowUnix int64) RatifyVerdict {
	findings := SuppressAccepted(Delta(base, proposed), accepted, nowUnix)
	if !enabled {
		return RatifyVerdict{false, findings, "disabled"}
	}
	if len(findings) > 0 {
		return RatifyVerdict{false, findings, "reach_expansion"}
	}
	return RatifyVerdict{true, nil, "empty_delta"}
}

// RatifyPromotion composes the existing complain-mode promotion offer with the
// deterministic reach referee. It returns a proposed Policy but never mutates base.
// The enabled bit is the default-off operator gate.
func RatifyPromotion(enabled bool, base adjudicator.Policy, offer adjudicator.PromotionOffer, accepted []knownbad.Record, nowUnix int64) (adjudicator.Policy, RatifyVerdict) {
	proposed := clonePolicy(base)
	if proposed.Allow == nil {
		proposed.Allow = map[string]bool{}
	}
	proposed.Allow[offer.Tool] = true
	delete(proposed.Complain, offer.Tool)
	return proposed, Ratify(enabled, base, proposed, accepted, nowUnix)
}

func clonePolicy(p adjudicator.Policy) adjudicator.Policy {
	out := p
	out.Allow = cloneBoolMap(p.Allow)
	out.Complain = cloneBoolMap(p.Complain)
	out.Deny = make(map[string]abi.ReasonCode, len(p.Deny))
	for k, v := range p.Deny {
		out.Deny[k] = v
	}
	out.AdvisoryReasons = make(map[abi.ReasonCode]bool, len(p.AdvisoryReasons))
	for k, v := range p.AdvisoryReasons {
		out.AdvisoryReasons[k] = v
	}
	out.AllowPrefix = append([]string(nil), p.AllowPrefix...)
	out.SelfModifyGlobs = append([]string(nil), p.SelfModifyGlobs...)
	out.RedactFields = append([]string(nil), p.RedactFields...)
	out.ArgPredicates = append([]adjudicator.ArgPredicate(nil), p.ArgPredicates...)
	out.EgressExtraDenyHosts = append([]string(nil), p.EgressExtraDenyHosts...)
	out.ResearchEgressAllowHosts = append([]string(nil), p.ResearchEgressAllowHosts...)
	out.SecretPatterns = append([]*regexp.Regexp(nil), p.SecretPatterns...)
	return out
}
func cloneBoolMap(in map[string]bool) map[string]bool {
	out := make(map[string]bool, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}
