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

	value := syspromptmmu.BuildSystemValue(residentPlan, syspromptmmu.PlanOf(overlayBase))
	return SystemBlock{
		Value:    value,
		Audit:    syspromptmmu.AuditRealizedPrefix(systemRequestBody(value), residentPlan),
		Overlays: len(overlayBase),
		Refused:  refused,
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
