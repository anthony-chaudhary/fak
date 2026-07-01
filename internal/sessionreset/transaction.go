package sessionreset

import (
	"crypto/sha256"
	"encoding/hex"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/session"
)

const seedDigestDomain = "fak-sessionreset-seed-v1"
const spanDigestDomain = "fak-sessionreset-span-v1"

// BuildResetTransaction turns a freshly built carryover seed into the replayable
// transaction row a child session stores. It records only payload-free handles:
// seed digest, contributor names, omitted span digests, and the fresh budget re-arm.
func BuildResetTransaction(in Input, newTrace string, seed Seed) session.ResetTransaction {
	tx := session.NewResetTransaction(in.Trace, newTrace, session.Budget{
		TurnsLeft:         session.Unbounded,
		TokensLeft:        session.Unbounded,
		ContextTokensLeft: in.FreshBudgetTok,
	})
	tx.SeedDigest = DigestSeed(seed)
	tx.Contributors = seedContributorNames(seed)
	tx.OmittedSpans = omittedSpans(in, seed)
	if seed.WarmPrefix != nil {
		tx.WarmPrefixDigest = seed.WarmPrefix.Digest
	}
	return tx
}

// DigestSeed returns the content address for the rendered reset seed.
func DigestSeed(seed Seed) string {
	return digestFields(seedDigestDomain, seed.Recap)
}

func seedContributorNames(seed Seed) []string {
	out := make([]string, 0, len(seed.Parts))
	for _, p := range seed.Parts {
		if name := strings.TrimSpace(p.Name); name != "" {
			out = append(out, name)
		}
	}
	return out
}

func omittedSpans(in Input, seed Seed) []session.ResetOmittedSpan {
	recap := seed.Recap
	out := make([]session.ResetOmittedSpan, 0)
	for i, m := range in.Messages {
		content := strings.TrimSpace(m.Content)
		if content == "" || strings.Contains(recap, content) {
			continue
		}
		out = append(out, session.ResetOmittedSpan{
			Index:  i,
			Role:   m.Role,
			Digest: digestFields(spanDigestDomain, strconv.Itoa(i), m.Role, content),
			Reason: omittedReason(m.Role, content, seed),
		})
	}
	return out
}

func omittedReason(role, content string, seed Seed) string {
	switch role {
	case "system":
		if seed.WarmPrefix != nil {
			return "stable_prefix_descriptor"
		}
		return "system_preamble_not_carried"
	case "tool":
		return "tool_result_turn_scoped"
	}
	if ctxmmu.ClassifyText(role, content) == ctxmmu.DurabilityDurable {
		return "summarized_or_clipped"
	}
	return "ephemeral_or_turn_scoped"
}

func digestFields(fields ...string) string {
	h := sha256.New()
	for _, f := range fields {
		_, _ = h.Write([]byte(strconv.Itoa(len(f))))
		_, _ = h.Write([]byte{0})
		_, _ = h.Write([]byte(f))
		_, _ = h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil))
}
