package gateway

import (
	"fmt"
	"os"
	"regexp"
	"strings"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/negframe"
)

var positiveComplementSubstitutions atomic.Uint64
var negframeRewriteSubstitutions atomic.Uint64
var negationOperatorSubstitutions atomic.Uint64

// positiveComplementEmitEnabled is default-off. Only an explicit truthy value enables mutation.
// FAK_POSITIVE_COMPLEMENT is the #4445 soak flag; #4448 owns the broader lexical rewrite flag.
func positiveComplementEmitEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("FAK_POSITIVE_COMPLEMENT"))) {
	case "1", "true", "on", "yes":
		return true
	default:
		return false
	}
}

func negframeRewriteEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("FAK_NEGFRAME_REWRITE"))) {
	case "1", "true", "on", "yes":
		return true
	default:
		return false
	}
}

// negationOperatorEnabled controls the unified L1/L2 operator. It is default-on so managed
// turns receive bounded positive-state normalization unless the benchmark ablation says off.
func negationOperatorEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("FAK_NEGATION_OP"))) {
	case "off", "0", "false", "no":
		return false
	default:
		return true
	}
}

// applyNegframeRequestPass is the common managed-turn request seam. FAK_NEGATION_OP is the
// master ablation: off means byte-identical input even if a legacy soak flag is set.
func applyNegframeRequestPass(messages []agent.Message, traceID string) []agent.Message {
	enabled := negationOperatorEnabled()
	result := negframe.ReframeResult{}
	if enabled {
		messages, result = applyEmitNegationPass(messages)
		messages = applyPositiveComplementEmit(messages, positiveComplementEmitEnabled())
		messages = applyMechanicalNegframeRewrite(messages, negframeRewriteEnabled())
	}
	if path := reframeJournalPath(); path != "" {
		arm := "treatment"
		if !enabled {
			arm = "control"
			result.ResidualNegatives = countMessageNegatives(messages)
		}
		_ = negframe.AppendReframeJournal(path, negframe.NewReframeJournalSiteRow(traceID, "gateway.negation_operator", arm, result, time.Now()), negframe.DefaultJournalMaxRows)
	}
	return messages
}

// applyEmitNegationPass applies equivalence-gated L1 NNF followed by exact L2 finite-domain
// complements. Refused and residual counts are content-free journal inputs.
func applyEmitNegationPass(messages []agent.Message) ([]agent.Message, negframe.ReframeResult) {
	result := negframe.ReframeResult{}
	for i := range messages {
		text := messages[i].Content
		text, admitted, refused := applyRegisteredL1(text)
		text, exact := substituteRegisteredComplements(text)
		if admitted+exact > 0 {
			messages[i].Content = text
		}
		result.Applied += admitted + exact
		result.VerbatimFallback += refused
	}
	result.ResidualNegatives = countMessageNegatives(messages)
	if result.Applied > 0 {
		negationOperatorSubstitutions.Add(uint64(result.Applied))
	}
	return messages, result
}

func applyRegisteredL1(text string) (string, int, int) {
	for _, domain := range negframe.Domains() {
		r := negframe.RewriteL1(text, domain)
		if r.Admitted > 0 {
			return r.Text, r.Admitted, r.Refused
		}
	}
	// A structurally recognized but unprovable clause is one refusal, not one per registry domain.
	for _, domain := range negframe.Domains() {
		if r := negframe.RewriteL1(text, domain); r.Refused > 0 {
			return text, 0, 1
		}
	}
	return text, 0, 0
}

func countMessageNegatives(messages []agent.Message) int {
	total := 0
	for _, message := range messages {
		total += len(negframe.Classify("gateway-managed-turn", message.Content))
	}
	return total
}

func applyMechanicalNegframeRewrite(messages []agent.Message, enabled bool) []agent.Message {
	if !enabled {
		return messages
	}
	for i := range messages {
		result := negframe.ReframePass(messages[i].Content)
		if result.Applied == 0 {
			continue
		}
		messages[i].Content = result.Text
		negframeRewriteSubstitutions.Add(uint64(result.Applied))
	}
	return messages
}

// applyPositiveComplementEmit is the request-side L2 hook. It walks model-visible message prose,
// substitutes only exact finite-domain complements, and leaves the slice byte-for-byte equivalent
// when disabled. The registry and ResolvePositive remain owned by negframe.
func applyPositiveComplementEmit(messages []agent.Message, enabled bool) []agent.Message {
	if !enabled {
		return messages
	}
	for i := range messages {
		text, count := substituteRegisteredComplements(messages[i].Content)
		if count == 0 {
			continue
		}
		messages[i].Content = text
		positiveComplementSubstitutions.Add(uint64(count))
	}
	return messages
}

func substituteRegisteredComplements(text string) (string, int) {
	if text == "" {
		return text, 0
	}
	out := text
	count := 0
	for _, domain := range negframe.Domains() {
		for _, member := range domain.Members {
			pattern := regexp.MustCompile(`(?i)\bnot\s+` + regexp.QuoteMeta(member) + `\b`)
			out = rewriteProseMatches(out, pattern, func(match string) string {
				resolution := negframe.ResolvePositive(match, domain)
				if !resolution.Exact {
					return match
				}
				count++
				return strings.Join(resolution.Members, " or ")
			})
		}
	}
	return out, count
}

// rewriteProseMatches preserves fenced and inline code as opaque request spans.
func rewriteProseMatches(text string, pattern *regexp.Regexp, replace func(string) string) string {
	lines := strings.SplitAfter(text, "\n")
	inFence := false
	for i, raw := range lines {
		line, suffix := strings.TrimSuffix(raw, "\n"), ""
		if strings.HasSuffix(raw, "\n") {
			suffix = "\n"
		}
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "```") || strings.HasPrefix(trimmed, "~~~") {
			inFence = !inFence
			continue
		}
		if inFence {
			continue
		}
		parts := strings.Split(line, "`")
		for j := 0; j < len(parts); j += 2 {
			parts[j] = pattern.ReplaceAllStringFunc(parts[j], replace)
		}
		lines[i] = strings.Join(parts, "`") + suffix
	}
	return strings.Join(lines, "")
}

func writePositiveComplementMetrics(b *strings.Builder) {
	writeHelpType(b, "fak_positive_complement_substitutions_total",
		"Exact finite-domain positive complements substituted on the model request path.", "counter")
	fmt.Fprintf(b, "fak_positive_complement_substitutions_total %d\n", positiveComplementSubstitutions.Load())
	writeHelpType(b, "fak_negframe_rewrite_substitutions_total",
		"Mechanical negframe substitutions applied on the model request path.", "counter")
	fmt.Fprintf(b, "fak_negframe_rewrite_substitutions_total %d\n", negframeRewriteSubstitutions.Load())
	writeHelpType(b, "fak_negation_operator_substitutions_total",
		"Positive-state substitutions applied by the default-on managed-turn negation operator.", "counter")
	fmt.Fprintf(b, "fak_negation_operator_substitutions_total %d\n", negationOperatorSubstitutions.Load())
}
