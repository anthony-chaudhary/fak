package gateway

import (
	"fmt"
	"os"
	"regexp"
	"strings"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/negframe"
)

var positiveComplementSubstitutions atomic.Uint64

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
}
