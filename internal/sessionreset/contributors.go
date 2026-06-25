package sessionreset

import (
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/vcachechain"
)

// The four built-in contributors register at package init, so importing sessionreset
// gives the default human-like reset out of the box. Order values space them so a
// third-party Register can slot between (e.g. Order 25 lands between durabilityFacts
// and taskDistill). Each is independently testable and pure.
func init() {
	Register(durabilityFacts{})
	Register(taskDistill{})
	Register(warmPrefix{})
	Register(verbatimTail{DefaultTailTurns})
}

// DefaultTailTurns is how many trailing transcript turns verbatimTail keeps by
// default — the "last thing on screen" a human carries across a reset.
const DefaultTailTurns = 2

// --- Contributor 1: durabilityFacts -----------------------------------------
//
// Keeps the DURABLE lines (preferences, identity, conventions) and DROPS the
// turn/session ephemera, by reusing the shipped ctxmmu durability prior verbatim
// (ctxmmu.ClassifyText is the chat-shaped entry to the same classifyDurability the
// admit gate runs). This is the core "what a human keeps on reset" move from
// CONTEXT-IS-NOT-MEMORY.md: "it's 3pm" evaporates, "I prefer afternoons" survives.

type durabilityFacts struct{}

func (durabilityFacts) Name() string { return "durability_facts" }

func (durabilityFacts) Contribute(in Input) (Part, bool) {
	var kept []string
	kept, durable, dropped := classifyTranscript(in.Messages)
	if len(kept) == 0 {
		return Part{Name: "durability_facts", Order: 10,
			Meta: map[string]string{"durable": "0", "dropped": strconv.Itoa(dropped)}}, false
	}
	text := "Durable facts to keep:\n- " + strings.Join(kept, "\n- ")
	return Part{
		Name:  "durability_facts",
		Order: 10,
		Text:  text,
		Meta:  map[string]string{"durable": strconv.Itoa(durable), "dropped": strconv.Itoa(dropped)},
	}, true
}

// classifyTranscript runs each non-empty user/assistant line through the shipped
// durability classifier and returns the durable lines kept, plus the durable/dropped
// counts (for the audit Meta). System lines are skipped (they are framing, not facts);
// tool lines are skipped (results, not standing facts) — matching the admit gate's
// posture that a tool result defaults turn-class.
func classifyTranscript(msgs []Msg) (kept []string, durable, dropped int) {
	seen := map[string]bool{}
	for _, m := range msgs {
		content := strings.TrimSpace(m.Content)
		if content == "" || m.Role == "system" || m.Role == "tool" {
			continue
		}
		switch ctxmmu.ClassifyText(m.Role, content) {
		case ctxmmu.DurabilityDurable:
			if !seen[content] { // dedup repeated standing facts
				seen[content] = true
				kept = append(kept, content)
				durable++
			}
		default:
			dropped++
		}
	}
	return kept, durable, dropped
}

// --- Contributor 2: taskDistill ---------------------------------------------
//
// A compact "where we are" recap: the standing objective (the first substantive user
// line, the task framing) and the latest step (the last user ask). v1 is deterministic
// and extractive — NO model call; a model-call distiller that summarizes the middle is
// a named follow-on.

type taskDistill struct{}

func (taskDistill) Name() string { return "task_distill" }

func (taskDistill) Contribute(in Input) (Part, bool) {
	objective := firstUserLine(in.Messages)
	latest := lastUserLine(in.Messages)
	if objective == "" && latest == "" {
		return Part{Name: "task_distill", Order: 20}, false
	}
	var b strings.Builder
	b.WriteString("Where we are:")
	if objective != "" {
		b.WriteString("\n- Objective: ")
		b.WriteString(clip(objective, 280))
	}
	if latest != "" && latest != objective {
		b.WriteString("\n- Latest request: ")
		b.WriteString(clip(latest, 280))
	}
	return Part{Name: "task_distill", Order: 20, Text: b.String()}, true
}

// --- Contributor 3: warmPrefix ----------------------------------------------
//
// Emits a DESCRIPTOR for replaying the stable prefix (system preamble + durable
// preamble) from the vCache prefix-DAG, so the fresh window does not re-pay to
// prefill the part that never changed. HONEST SCOPE: vCache is a decision layer
// (vcachechain is "NOT registered; gated OFF by default"), so v1 emits the recall
// PLAN, not a live KV splice — live same-model KV reuse is the named follow-on. The
// part is Order 0 so the prefix sits at the very top of the seed.

type warmPrefix struct{}

func (warmPrefix) Name() string { return "warm_prefix" }

func (warmPrefix) Contribute(in Input) (Part, bool) {
	sys := systemPreamble(in.Messages)
	if sys == "" {
		return Part{Name: "warm_prefix", Order: 0}, false
	}
	// Cost-gate the warm-prefix reuse the same way vcachechain prices a chain replay:
	// only advertise a warm-prefix recall when the stable prefix is large enough that
	// re-prefilling it is worth avoiding. ReplayCost is a pure, deterministic price.
	prefixTokens := int64(approxTokens(sys))
	cost := vcachechain.ReplayCost(prefixTokens, 1.0)
	meta := map[string]string{
		"prefix_tokens": strconv.FormatInt(prefixTokens, 10),
		"replay_cost":   strconv.FormatFloat(cost, 'f', 2, 64),
		"live_kv_reuse": "deferred", // honest: decision layer only, no live splice in v1
	}
	text := "Stable prefix retained for warm-cache replay (" +
		strconv.FormatInt(prefixTokens, 10) + " approx tokens; live KV reuse is a follow-on)."
	return Part{Name: "warm_prefix", Order: 0, Text: text, Meta: meta}, true
}

// --- Contributor 4: verbatimTail --------------------------------------------
//
// Keeps the last N turns verbatim — the immediate working context a human glances at
// before continuing. It is the highest Order so it renders LAST, closest to the fresh
// request, where recency belongs.

type verbatimTail struct{ N int }

func (v verbatimTail) Name() string { return "verbatim_tail" }

func (v verbatimTail) Contribute(in Input) (Part, bool) {
	n := v.N
	if n <= 0 {
		n = DefaultTailTurns
	}
	tail := lastTurns(in.Messages, n)
	if len(tail) == 0 {
		return Part{Name: "verbatim_tail", Order: 90}, false
	}
	var b strings.Builder
	b.WriteString("Most recent exchange (verbatim):")
	for _, m := range tail {
		b.WriteString("\n")
		b.WriteString(m.Role)
		b.WriteString(": ")
		b.WriteString(clip(strings.TrimSpace(m.Content), 600))
	}
	return Part{Name: "verbatim_tail", Order: 90, Text: b.String(),
		Meta: map[string]string{"turns": strconv.Itoa(len(tail))}}, true
}
