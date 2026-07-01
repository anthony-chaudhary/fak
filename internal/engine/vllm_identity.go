package engine

import (
	"encoding/json"
	"os"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// fak-owned request-control meta keys. The gateway/session layer stamps these on
// a ToolCall so the vLLM adapter can lower fak cache identity and scheduling
// intent into the request instead of trying to rediscover both below the request
// boundary. Meta is OPEN, so an unstamped call simply omits the control and the
// adapter degrades to vLLM's defaults.
const (
	// MetaCacheTenant, MetaCacheAuthority, and MetaCacheFamily are the three fak
	// identity axes folded into the vLLM per-request cache_salt. Tenant is the
	// trust boundary: two tenants with a byte-identical prefix must not share a
	// prefix-cache slot, so tenant is always part of the salt input and a
	// different tenant yields a different salt by construction.
	MetaCacheTenant    = "fak_cache_tenant"
	MetaCacheAuthority = "fak_cache_authority"
	MetaCacheFamily    = "fak_cache_family"

	// MetaTurnPriority is the read-only TurnIntent/session priority. Lower value
	// = earlier, matching vLLM's priority-scheduling convention. It is lowered to
	// the request only when the served engine advertises priority scheduling.
	MetaTurnPriority = "fak_turn_priority"
)

// vllmControls is the fak-derived cache identity and scheduling intent to lower
// onto one vLLM request. It carries a cache_salt that is a DIGEST family label
// (never the raw tenant/authority/family identity), so it is safe to place on
// the wire and to record in the trace for cache attribution.
type vllmControls struct {
	cacheSalt   string // "fak-<hex>" family token; empty when no fak identity was stamped
	priority    string // decimal vLLM priority; empty when not emitted
	hasPriority bool
}

// deriveVLLMControls reads the fak cache identity and TurnIntent priority off the
// call meta and folds them into the request controls. Priority is only derived
// when the served engine advertises priority scheduling; otherwise it is dropped
// and the request degrades to the engine's FCFS default.
func (e *VLLMEngine) deriveVLLMControls(c *abi.ToolCall) vllmControls {
	var ctrl vllmControls
	if c == nil || c.Meta == nil {
		return ctrl
	}
	ctrl.cacheSalt = deriveCacheSalt(c.Meta[MetaCacheTenant], c.Meta[MetaCacheAuthority], c.Meta[MetaCacheFamily])
	if e.cfg.PriorityScheduling {
		if p, ok := derivePriority(c.Meta[MetaTurnPriority]); ok {
			ctrl.priority = strconv.Itoa(p)
			ctrl.hasPriority = true
		}
	}
	return ctrl
}

// deriveCacheSalt folds the fak tenant/authority/cache-family identity into a
// stable, non-secret per-request cache_salt. The tenant axis makes cross-tenant
// prefix-cache reuse impossible by construction: a different tenant produces a
// different salt even for a byte-identical prefix, so vLLM's APC cannot alias the
// two. The returned value is a short hex family label derived by SHA-256, NOT the
// raw identity, so it never carries a secret onto the wire or into a trace. An
// empty identity yields an empty salt (degrade to vLLM's default, no isolation).
func deriveCacheSalt(tenant, authority, family string) string {
	tenant = strings.TrimSpace(tenant)
	authority = strings.TrimSpace(authority)
	family = strings.TrimSpace(family)
	if tenant == "" && authority == "" && family == "" {
		return ""
	}
	// Null-separate the axes so distinct identities cannot collide by
	// concatenation (e.g. tenant "ab"+authority "c" vs tenant "a"+authority "bc").
	digest := cachemeta.DigestBytes([]byte(tenant + "\x00" + authority + "\x00" + family))
	return "fak-" + digest[:16]
}

// derivePriority parses the read-only TurnIntent/session priority. ok is false
// when no priority was stamped or the value is not an integer, so the caller
// emits NO vLLM priority field and the request keeps the engine's default order.
func derivePriority(raw string) (priority int, ok bool) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return 0, false
	}
	v, err := strconv.Atoi(raw)
	if err != nil {
		return 0, false
	}
	return v, true
}

// applyVLLMControls lowers the derived cache identity and scheduling intent onto
// an already-built vLLM request body. cache_salt is set whenever a fak identity
// was stamped, so cross-tenant reuse is refused by construction; priority is set
// only when the served engine advertises priority scheduling. Caller-provided
// fields are never overwritten. Both are performance controls only: neither is
// consulted on any correctness path, and a cache hit is never trusted on their
// basis. On any decode error the original body is returned unchanged so a control
// can never break an otherwise valid request.
func applyVLLMControls(body []byte, ctrl vllmControls) []byte {
	if ctrl.cacheSalt == "" && !ctrl.hasPriority {
		return body
	}
	obj := map[string]json.RawMessage{}
	if len(body) == 0 || json.Unmarshal(body, &obj) != nil {
		return body
	}
	if ctrl.cacheSalt != "" {
		if _, ok := obj["cache_salt"]; !ok {
			obj["cache_salt"] = mustJSON(ctrl.cacheSalt)
		}
	}
	if ctrl.hasPriority {
		if _, ok := obj["priority"]; !ok {
			obj["priority"] = json.RawMessage(ctrl.priority)
		}
	}
	out, err := json.Marshal(obj)
	if err != nil {
		return body
	}
	return out
}

// envBool reports whether an environment variable names an enabled boolean.
func envBool(key string) bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(key))) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}
