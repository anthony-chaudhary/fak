package agent

// syspromptmmu_seam.go — #1322 (epic #1258): the owned loop's FIRST non-test importer of
// the system-prompt MMU spine. internal/syspromptmmu Rungs 1-5 are committed with tests
// but have ZERO importers outside the package; this file is the request-path spine where
// fak authors and queries its OWN system block from a loop it owns:
//
//   - fak-concepts (the spine) are pinned FIRST and are byte-identical every turn
//     (BaseContextPlan leads with TierSpine, then the TierPolicy floor);
//   - the harness/overlay items are dynamically authored through the witness-gated
//     ApplyEdit and appended AFTER the single cache breakpoint, so the resident prefix is
//     never re-serialized — the cache-stability win (a head mutation busts the prefix
//     cache, so the overlay rides past the breakpoint, masked not mutated);
//   - AuditRealizedPrefix re-derives the realized resident prefix and proves it still
//     equals the planned spine, so the loop can witness the cache hit before it sends.
//
// SCOPE FENCE (matches internal/syspromptmmu/splice.go's own note): this rung BUILDS fak's
// owned system block and proves its prefix is cache-stable across overlay authorship.
// Promoting it to REPLACE the harness-authored head on the live Anthropic wire (today a
// passthrough where the harness authors the system prompt) is the gateway-wiring rung,
// deferred there — this file is the builder that rung adopts, witnessed today.

import (
	"os"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/syspromptmmu"
)

// SystemBlock is the owned loop's realized system block plus the proof it stayed
// cache-stable across overlay authorship.
type SystemBlock struct {
	// Value is the Anthropic `system[]` JSON value: the resident spine+policy prefix (the
	// last resident block carrying the single cache_control breakpoint) followed by the
	// admitted overlay cards. It is the value the owned loop places under a request body's
	// `system` field (see RequestBody).
	Value []byte
	// Audit is the Rung-6 re-derivation over the RESIDENT prefix. Status == AuditOK iff the
	// realized spine is byte-identical to the plan — the cache hit holds.
	Audit syspromptmmu.PrefixAudit
	// Overlays is how many authored items the witness gate admitted past the breakpoint.
	Overlays int
	// Refused carries the verdict for each authored item the gate rejected (a nil witness,
	// empty content, ...), so a refusal is auditable, never a silent drop.
	Refused []syspromptmmu.EditVerdict
	// Steering is the terseness level (#5047) this block was built at: SteeringOff when the
	// opt-in knob is unset or out of range (no steering block appended), else the applied
	// 1..4 level. The steering block, when present, rides strictly AFTER the cache
	// breakpoint alongside the queried overlay, so it never re-serializes the resident
	// prefix (CacheStable still holds).
	Steering int
	// Style is the canonical opt-in response profile, including family/intensity aliases.
	Style string
	// StyleFamily and StyleIntensity make mixed profile captures reproducible.
	StyleFamily    string
	StyleIntensity string
}

// CacheStable is the one-bit verdict the owned loop checks before sending: the realized
// resident prefix equals the planned spine, so the cached prefix still hits. True iff the
// Rung-6 audit found a fak-shaped base context whose every resident block is unchanged.
func (b SystemBlock) CacheStable() bool {
	return b.Audit.Status == syspromptmmu.AuditOK
}

// BuildOwnedSystemBlock builds the agent loop's system block from fak's OWN authored base
// context (the spine pinned first), then dynamically authors each overlay item through the
// witness-gated ApplyEdit and appends the admitted ones after the cache breakpoint. The
// resident spine+policy plan is the SAME bytes regardless of the overlay, so the realized
// prefix is cache-stable — proven by the returned Audit (Status AuditOK).
//
// witness is the INJECTED success predicate ApplyEdit gates each authored item on — the
// agent never grades its own edit, so a nil witness is fail-closed: every item is refused
// and the block carries the bare spine (still AuditOK, because the spine is untouched).
// ApplyEdit never mutates its input, so the resident plan can never be corrupted by an
// authored overlay item.
func BuildOwnedSystemBlock(items [][]byte, witness func(syspromptmmu.BaseEdit) bool) SystemBlock {
	return buildOwnedSystemBlockWithStyle(items, witness, styleReadoutFromEnv())
}

// styleReadoutFromEnv resolves the named response profile first, then the legacy numeric
// steering knob. Unknown named profiles fail safe to full/off and never fall through to a
// second knob: an explicit but malformed selection cannot accidentally enable another mode.
func styleReadoutFromEnv() syspromptmmu.StyleReadout {
	if raw := strings.TrimSpace(os.Getenv(syspromptmmu.StyleEnvVar)); raw != "" {
		return syspromptmmu.DescribeStyle(raw)
	}
	level := steeringLevelFromEnv()
	if level == syspromptmmu.SteeringOff {
		return syspromptmmu.DescribeStyle(syspromptmmu.StyleFull)
	}
	for _, name := range syspromptmmu.StyleNames() {
		if strings.Contains(name, ":") {
			continue
		}
		if candidate, ok := syspromptmmu.StyleLevel(name); ok && candidate == level {
			return syspromptmmu.DescribeStyle(name)
		}
	}
	return syspromptmmu.DescribeStyle("")
}

// steeringLevelFromEnv reads the opt-in FAK_STEERING_LEVEL knob (#5047). Unset, empty, or
// unparseable ⇒ SteeringOff, so steering NEVER self-enables — the forward path only moves
// when a level is deliberately configured. An out-of-range numeric value is returned as-is
// and refused fail-safe downstream (SteeringSegment returns ok false), so it too collapses
// to SteeringOff in the built block.
func steeringLevelFromEnv() int {
	raw := strings.TrimSpace(os.Getenv(syspromptmmu.SteeringEnvVar))
	if raw == "" {
		return syspromptmmu.SteeringOff
	}
	level, err := strconv.Atoi(raw)
	if err != nil {
		return syspromptmmu.SteeringOff
	}
	return level
}

// buildOwnedSystemBlockAt is BuildOwnedSystemBlock with the steering level passed
// EXPLICITLY (BuildOwnedSystemBlock reads it from the env knob). It builds the witness-gated
// capability overlay exactly as before, then — when `level` selects a valid steering level
// (1..4) — appends the sentinel-wrapped terseness block as the LAST overlay segment, strictly
// after the cache breakpoint and after the queried capability cards. A SteeringOff or
// out-of-range level appends nothing, so the returned Value is byte-identical to the
// pre-#5047 forward path. The steering block is NOT counted in Overlays (it is fak's own
// appended segment, not a witness-gated authored item); its presence is reported by Steering.
//
// Cache-stability is preserved by construction: the resident spine+policy plan and the
// breakpoint that sits on its last block are untouched, so AuditRealizedPrefix over the
// resident prefix still reads AuditOK and the resident prefix bytes are byte-identical to
// the off build — only after-breakpoint tail bytes are added.
func buildOwnedSystemBlockAt(items [][]byte, witness func(syspromptmmu.BaseEdit) bool, level int) SystemBlock {
	style := syspromptmmu.DescribeStyle("")
	for _, name := range syspromptmmu.StyleNames() {
		if strings.Contains(name, ":") {
			continue
		}
		if candidate, ok := syspromptmmu.StyleLevel(name); ok && candidate == level {
			style = syspromptmmu.DescribeStyle(name)
			break
		}
	}
	return buildOwnedSystemBlockWithStyle(items, witness, style)
}

func buildOwnedSystemBlockWithStyle(items [][]byte, witness func(syspromptmmu.BaseEdit) bool, style syspromptmmu.StyleReadout) SystemBlock {
	residentPlan := syspromptmmu.BaseContextPlan() // spine + policy floor, fak-concepts first

	var overlayBase []syspromptmmu.Segment // dynamically authored overlay layer, starts empty
	var refused []syspromptmmu.EditVerdict
	for _, content := range items {
		edit := syspromptmmu.BaseEdit{Op: syspromptmmu.EditAdd, Tier: syspromptmmu.TierOverlay, Content: content}
		next, v := syspromptmmu.ApplyEdit(overlayBase, edit, witness)
		if v.Applied {
			overlayBase = next
			continue
		}
		refused = append(refused, v)
	}

	overlayPlan := syspromptmmu.PlanOf(overlayBase)
	steering := syspromptmmu.SteeringOff
	if style.Known {
		if seg, ok := syspromptmmu.StyleSegment(style.Style); ok {
			overlayPlan = append(overlayPlan, seg) // strictly after the queried cards, past the breakpoint
			steering = style.Level
		}
	}

	value := syspromptmmu.BuildSystemValue(residentPlan, overlayPlan)
	return SystemBlock{
		Value:          value,
		Audit:          syspromptmmu.AuditRealizedPrefix(systemRequestBody(value), residentPlan),
		Overlays:       len(overlayBase),
		Refused:        refused,
		Steering:       steering,
		Style:          style.Style,
		StyleFamily:    style.Family,
		StyleIntensity: style.Intensity,
	}
}

// systemRequestBody wraps a `system[]` array value into the minimal request body the
// Rung-6 auditor (AuditRealizedPrefix) and the Rung-2 splicer read — both key off the
// body's `system` field, never a bare array. Building the body here keeps the audit honest:
// it re-derives the realized prefix from the SAME shape the wire carries.
func systemRequestBody(value []byte) []byte {
	body := make([]byte, 0, len(value)+12)
	body = append(body, `{"system":`...)
	body = append(body, value...)
	body = append(body, '}')
	return body
}

// RequestBody wraps this block's Value into the minimal `{"system": …}` request body shape
// the wire (and the auditor) consume, so a caller can audit/splice the exact bytes it sends.
func (b SystemBlock) RequestBody() []byte { return systemRequestBody(b.Value) }

// OwnedResidentHead is the byte-identical resident prefix the owned loop sends every turn:
// fak's spine+policy plan realized with NO overlay. The cache-stability contract is that
// BuildOwnedSystemBlock's Value carries THIS exact sequence of resident blocks as its
// head regardless of which overlay items were authored — the head is never re-serialized
// per turn. Exposed so a caller (and the test) can assert the prefix invariant directly.
func OwnedResidentHead() []byte {
	plan := syspromptmmu.BaseContextPlan()
	return syspromptmmu.BuildSystemValue(plan, nil)
}
