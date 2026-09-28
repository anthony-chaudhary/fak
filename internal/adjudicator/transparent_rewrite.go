package adjudicator

import (
	"context"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// transparent_rewrite.go — the in-syscall host-tool → fak_* rewrite: Read →
// fak_read (#11150) and the grep / glob families → fak_grep / fak_glob (#11499).
//
// The rewrite is a DISPATCH-SHAPE repair, never an admission: it changes which
// engine serves a call the floor ALREADY admitted. Adjudicate selects it right after
// decoding the args but applies it only to an Allow verdict of the ORIGINAL call,
// after every rung — self-modify, arg predicates, egress, the affirmative allow
// list, default-deny — has judged the host tool. Returning the TRANSFORM before
// those rungs let a Read the floor confines by arg_rules (or never lists) dispatch
// as fak_read: the kernel treats TRANSFORM as admitted, so the rewrite laundered a
// refused call into a dispatched one.

// rewriteArg is one normalized argument: the canonical key the fak_* tool reads
// and the host-dialect aliases it is taken from, first non-nil wins.
type rewriteArg struct {
	out  string
	keys []string
}

// rewriteRule is one host-tool family's rewrite onto its fak_* engine tool.
type rewriteRule struct {
	newTool   string
	name      string // the reversibility_autorepair tag
	by        string // "monitor/" + name
	primary   rewriteArg
	secondary rewriteArg // optional; zero value = none
}

var (
	readRewrite = rewriteRule{
		newTool: "fak_read", name: "read_to_fak_read", by: "monitor/read_to_fak_read",
		primary: rewriteArg{"file_path", []string{"file_path", "filePath", "path"}},
	}
	grepRewrite = rewriteRule{
		newTool: "fak_grep", name: "grep_to_fak_grep", by: "monitor/grep_to_fak_grep",
		primary:   rewriteArg{"pattern", []string{"pattern", "regex", "query"}},
		secondary: rewriteArg{"path", []string{"path", "filePath", "file_path", "dir", "directory"}},
	}
	globRewrite = rewriteRule{
		newTool: "fak_glob", name: "glob_to_fak_glob", by: "monitor/glob_to_fak_glob",
		primary:   rewriteArg{"pattern", []string{"pattern", "glob", "query"}},
		secondary: rewriteArg{"path", []string{"path", "directory", "dir", "filePath", "file_path"}},
	}
)

// transparentRewrite is a pending rewrite: the rule plus the ORIGINAL decoded args
// it normalizes if (and only if) the call is admitted. tool and preds are the
// original call's name and arg predicates, set when the arg-predicate rung runs.
type transparentRewrite struct {
	rule  *rewriteRule
	args  map[string]any
	tool  string
	preds []ArgPredicate
}

// transparentRewriteFor reports whether this call is a rewrite candidate: a
// host Read / grep-family / glob-family tool carrying its primary argument. It
// never decides; see admit.
func transparentRewriteFor(lowerTool string, args map[string]any) (transparentRewrite, bool) {
	var r *rewriteRule
	switch lowerTool {
	case "read":
		r = &readRewrite
	case "grep", "rg", "ripgrep", "search":
		r = &grepRewrite
	case "glob", "find":
		r = &globRewrite
	default:
		return transparentRewrite{}, false
	}
	if _, ok := firstArg(args, r.primary.keys); !ok {
		return transparentRewrite{}, false
	}
	return transparentRewrite{rule: r, args: args}, true
}

// admit upgrades an Allow of the ORIGINAL call to the fak_* TRANSFORM, carrying
// the admitting rung's Meta (posture, advisory notes, would-deny record) forward.
// Every other verdict — a deny, a hold, an args-only TRANSFORM (redaction, a spent
// confirmation) — is returned unchanged, so the host tool is dispatched or refused
// exactly as the floor decided. A floor that denies the fak_* tool BY NAME keeps
// the host tool: the rewrite never dispatches a tool the operator refused.
func (rw transparentRewrite) admit(ctx context.Context, p Policy, v abi.Verdict) abi.Verdict {
	if rw.rule == nil || v.Kind != abi.VerdictAllow {
		return v
	}
	if _, denied := p.Deny[rw.rule.newTool]; denied {
		return v
	}
	args := rw.rule.normalize(rw.args)
	// normalize PROMOTES an alias onto the canonical key (Read's filePath → file_path)
	// after the arg predicates judged only the raw keys. Judge the promoted shape by
	// the same rules, so a rule on file_path cannot be dodged by spelling it filePath
	// and letting the rewrite canonicalize it. An advisory rule stays advisory.
	if dv, denied, _ := evalArgPredicates(rw.preds, rw.tool, args); denied {
		if dv = p.soften(dv, nil); dv.Kind != abi.VerdictAllow {
			return dv
		}
	}
	ref, ok := putJSON(ctx, args)
	if !ok {
		return v
	}
	meta := make(map[string]string, len(v.Meta)+1)
	for k, val := range v.Meta {
		meta[k] = val
	}
	meta["reversibility_autorepair"] = rw.rule.name
	return abi.Verdict{
		Kind:    abi.VerdictTransform,
		By:      rw.rule.by,
		Payload: abi.TransformPayload{NewTool: rw.rule.newTool, NewArgs: ref},
		Meta:    meta,
	}
}

// normalize maps the host-dialect args onto the fak_* tool's canonical keys:
// every alias is dropped, the canonical keys are set from the first non-nil
// alias, and all other args pass through unchanged.
func (r *rewriteRule) normalize(args map[string]any) map[string]any {
	out := make(map[string]any, len(args))
	for k, v := range args {
		if !r.alias(k) {
			out[k] = v
		}
	}
	if v, ok := firstArg(args, r.primary.keys); ok {
		out[r.primary.out] = v
	}
	if v, ok := firstArg(args, r.secondary.keys); ok {
		out[r.secondary.out] = v
	}
	return out
}

func (r *rewriteRule) alias(k string) bool {
	for _, keys := range [][]string{r.primary.keys, r.secondary.keys} {
		for _, a := range keys {
			if k == a {
				return true
			}
		}
	}
	return false
}

func firstArg(args map[string]any, keys []string) (any, bool) {
	for _, k := range keys {
		if v, ok := args[k]; ok && v != nil {
			return v, true
		}
	}
	return nil, false
}
