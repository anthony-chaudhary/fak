// Package taskidentity derives the canonical task identity for an agent session.
package taskidentity

import (
	"crypto/sha256"
	"fmt"
	"regexp"
	"strings"
)

const directiveHashCap = 400

var (
	wrapperRE = regexp.MustCompile(`(?i)^\s*(?:Caveat:|<system-reminder|<command-name>|<command-message>|<command-args>|<local-command-|<user-memory|Codebase and user instructions)`)
	taskCmdRE = regexp.MustCompile(`(?i)<command-name>\s*/(goal|loop|dispatch|fanout|next-up)\b`)
	cmdArgsRE = regexp.MustCompile(`(?is)<command-args>(.*?)</command-args>`)
)

const ResumePromptPrefix = "Resume where you left off"

// TextOf flattens a transcript message content field to human-readable text.
func TextOf(content any) string {
	switch v := content.(type) {
	case string:
		return v
	case []any:
		var out []string
		for _, item := range v {
			block, ok := item.(map[string]any)
			if !ok {
				continue
			}
			switch block["type"] {
			case "text":
				if s, ok := block["text"].(string); ok {
					out = append(out, s)
				}
			case "tool_result":
				switch c := block["content"].(type) {
				case string:
					out = append(out, c)
				default:
					if s := TextOf(c); s != "" {
						out = append(out, s)
					}
				}
			}
		}
		return strings.Join(out, " ")
	default:
		return ""
	}
}

// CanonicalDirective returns the real first task directive, skipping harness boilerplate.
func CanonicalDirective(headRecords []map[string]any) string {
	var cmdArgs string
	for _, ho := range headRecords {
		typ, _ := ho["type"].(string)
		if typ != "user" && typ != "system" {
			continue
		}
		content := ho["content"]
		if msg, ok := ho["message"].(map[string]any); ok {
			if c, ok := msg["content"]; ok {
				content = c
			}
		}
		txt := TextOf(content)
		if strings.TrimSpace(txt) == "" {
			continue
		}
		if taskCmdRE.MatchString(txt) {
			if m := cmdArgsRE.FindStringSubmatch(txt); m != nil && strings.TrimSpace(m[1]) != "" {
				cmdArgs = strings.Join(strings.Fields(m[1]), " ")
			}
		}
		stripped := strings.TrimSpace(txt)
		if wrapperRE.MatchString(stripped) {
			continue
		}
		if strings.HasPrefix(stripped, ResumePromptPrefix) {
			continue
		}
		directive := strings.Join(strings.Fields(stripped), " ")
		if cmdArgs != "" {
			return strings.Join(strings.Fields(cmdArgs+" "+directive), " ")
		}
		return directive
	}
	return cmdArgs
}

// Signature returns the stable 16-hex signature of a task identity.
func Signature(project, cwd, directive string) string {
	if directive == "" {
		return ""
	}
	if len(directive) > directiveHashCap {
		directive = directive[:directiveHashCap]
	}
	sum := sha256.Sum256([]byte(project + "\x00" + cwd + "\x00" + directive))
	return fmt.Sprintf("%x", sum[:])[:16]
}

// Identity returns the canonical directive and signature in one call.
func Identity(project, cwd string, headRecords []map[string]any) (string, string) {
	directive := CanonicalDirective(headRecords)
	return directive, Signature(project, cwd, directive)
}
