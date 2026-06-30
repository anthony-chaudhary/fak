// Package fleetaccounts is the Go port of the READ-ONLY roster/resolve/probe fold
// from tools/fleet_accounts.py — the single source of truth for "what is an account,
// and is it offered?" across both product families (Claude Code and opencode).
//
// The fleet resume/switcher layer discovers accounts by globbing config dirs:
// Claude dirs under the user home (<home>/.claude*) and opencode dirs under the XDG
// config home (<config_home>/opencode*). Each discovered dir is classified into ONE
// Kind — worker | excluded | non-account — by an operator-editable POLICY
// (accounts_policy.json), then folded with live runtime status (usage throttle /
// auth block / live sessions) read from the watchdog's sessions.json registry.
//
// This package preserves the Python contract byte-compatibly for the high-frequency,
// read-only operators the standalone operators + resume/watchdog paths use:
// roster (Discover/Annotate), list/json rendering, Available, Resolve (pin + route),
// and the seat pool. The mutating ops (relogin/top-up/launch) and the ACTIVE network
// probe (account_probe.py) are out of scope here — see the package doc note in
// cmd/fak/fleetaccounts.go for the documented follow-on.
//
// Discovery + classification are pure functions of (home, config_home, policy, registry),
// so a test drives the whole fold from a fixture tree with no global state.
package fleetaccounts

import (
	"path/filepath"
	"regexp"
	"strings"
)

// Kind classifies a discovered config dir.
type Kind string

const (
	// KindWorker is a real, offered account (a resume/switcher target).
	KindWorker Kind = "worker"
	// KindExcluded is tombstoned by policy or a .DELETED marker — present on disk but
	// never offered as a resume target (the backup account, by default).
	KindExcluded Kind = "excluded"
	// KindNonAccount is not an account dir at all: no account marker (Claude: no
	// projects/ subdir; opencode: no opencode.json) or a plain file.
	KindNonAccount Kind = "non-account"
)

// OpencodeMarkerFiles are the config files whose presence makes a dir an opencode account
// (the opencode.json/jsonc is the switch seam, the way projects/ is for Claude).
var OpencodeMarkerFiles = []string{"opencode.json", "opencode.jsonc"}

// Policy is the operator-editable account policy (accounts_policy.json), applying
// uniformly to BOTH products. Exclude substrings tombstone accounts; IncludeOnly (when
// non-empty) is an allowlist. AccountProfiles overrides model-tier inference; RouteWeights
// biases the routing tie-break. The JSON keys match the Python policy file.
type Policy struct {
	Exclude         []string                   `json:"exclude"`
	IncludeOnly     []string                   `json:"include_only"`
	Notes           map[string]string          `json:"notes"`
	AccountProfiles map[string]ProfileOverride `json:"account_profiles"`
	RouteWeights    map[string]int             `json:"route_weights"`
	Routing         Routing                    `json:"routing"`
}

// ProfileOverride is one operator account-profile override (model tier/model/effort/agent).
type ProfileOverride struct {
	ModelTier   int    `json:"model_tier"`
	Tier        int    `json:"tier"`
	Model       string `json:"model"`
	SmallModel  string `json:"small_model"`
	Effort      string `json:"effort"`
	ModelEffort string `json:"model_effort"`
	Agent       string `json:"agent"`
}

// Routing carries the v1 routing knobs.
type Routing struct {
	LightConfidence   float64 `json:"light_confidence"`
	HardTier1Fallback string  `json:"hard_tier1_fallback"`
}

// DefaultPolicy mirrors fleet_accounts.DEFAULT_POLICY: backup/breakglass off the
// auto-resume roster, conservative tier inference, no profile/weight overrides.
func DefaultPolicy() Policy {
	return Policy{
		Exclude:     []string{"backup", "breakglass"},
		IncludeOnly: []string{},
		Notes: map[string]string{
			"backup": "break-glass backup account; never auto-resume",
		},
		AccountProfiles: map[string]ProfileOverride{},
		RouteWeights:    map[string]int{},
		Routing: Routing{
			LightConfidence:   0.999,
			HardTier1Fallback: "stop",
		},
	}
}

// AccountProduct classifies a discovered dir basename to its product family.
// .claude* -> "claude" (the default under the user home); opencode* -> "opencode"
// (the config under ~/.config). Anything else defaults to "claude" so historical call
// sites keep working.
func AccountProduct(account string) string {
	if strings.HasPrefix(strings.ToLower(account), "opencode") {
		return "opencode"
	}
	return "claude"
}

// AccountTag normalizes a config-dir basename to its short tag, matching the resume
// layer convention. Claude: ".claude-gem8-acct" -> "gem8"; ".claude" -> "default".
// opencode: "opencode-glm" -> "glm"; "opencode" -> "default". The trailing "-acct" org
// suffix is stripped if present.
func AccountTag(account string) string {
	product := AccountProduct(account)
	var tag string
	if product == "opencode" {
		tag = strings.ReplaceAll(account, "opencode-", "")
		tag = strings.ReplaceAll(tag, "opencode", "")
	} else {
		tag = strings.ReplaceAll(account, ".claude-", "")
		tag = strings.ReplaceAll(tag, ".claude", "")
	}
	if strings.HasSuffix(tag, "-acct") {
		tag = tag[:len(tag)-len("-acct")]
	}
	if tag == "" {
		return "default"
	}
	return tag
}

// excludedMatch returns the matching exclude substring (for the reason text), or "".
func excludedMatch(tag, account string, exclude []string, identityValues ...string) string {
	haystacks := append([]string{tag, account}, identityValues...)
	for _, sub := range exclude {
		if sub == "" {
			continue
		}
		sl := strings.ToLower(sub)
		for _, value := range haystacks {
			if value != "" && strings.Contains(strings.ToLower(value), sl) {
				return sub
			}
		}
	}
	return ""
}

var nonAlnum = regexp.MustCompile(`[^a-z0-9]+`)

// modelTierFromName is the small v1 model taxonomy. Tier 1 is the max-quality frontier
// set; tier 2 is GLM-5.2 for lightweight work; everything else is tier 3.
func modelTierFromName(model string) int {
	text := strings.ToLower(model)
	text = strings.ReplaceAll(text, "_", "-")
	text = strings.ReplaceAll(text, " ", "-")
	compact := nonAlnum.ReplaceAllString(text, "")
	if strings.Contains(text, "gpt-5.5") || strings.Contains(compact, "gpt55") {
		return 1
	}
	if strings.Contains(text, "opus-4.6") || strings.Contains(compact, "opus46") ||
		text == "opus" || text == "claude-opus" {
		return 1
	}
	if strings.Contains(text, "glm-5.2") || strings.Contains(compact, "glm52") {
		return 2
	}
	return 3
}

// Profile is an account's model-routing profile (the cleaned shape from account_profile).
type Profile struct {
	ModelTier     int    `json:"model_tier"`
	Model         string `json:"model"`
	SmallModel    string `json:"small_model"`
	ModelEffort   string `json:"model_effort"`
	Agent         string `json:"agent"`
	ProfileSource string `json:"profile_source"`
}

func cleanProfile(raw ProfileOverride, source string) Profile {
	tier := raw.ModelTier
	if tier == 0 {
		tier = raw.Tier
	}
	effort := raw.Effort
	if effort == "" {
		effort = raw.ModelEffort
	}
	p := Profile{
		ModelTier:     tier,
		Model:         raw.Model,
		SmallModel:    raw.SmallModel,
		ModelEffort:   effort,
		Agent:         raw.Agent,
		ProfileSource: source,
	}
	if p.ModelTier != 1 && p.ModelTier != 2 && p.ModelTier != 3 {
		p.ModelTier = modelTierFromName(raw.Model)
	}
	if p.ModelTier != 1 && p.ModelTier != 2 && p.ModelTier != 3 {
		p.ModelTier = 3
	}
	return p
}

// safeOpencodeModels reads only model identifiers from an opencode account's config files.
func safeOpencodeModels(acctDir string) map[string]string {
	for _, marker := range OpencodeMarkerFiles {
		doc, ok := readJSONObject(filepath.Join(acctDir, marker))
		if !ok {
			continue
		}
		out := map[string]string{}
		for _, key := range []string{"model", "small_model"} {
			if v, ok := doc[key].(string); ok && v != "" {
				out[key] = v
			}
		}
		return out
	}
	return map[string]string{}
}

// accountProfile returns the model-routing profile for an account row, honoring policy
// overrides by exact account, product:tag, short tag, or product.
func accountProfile(row Account, pol Policy) Profile {
	product := row.Product
	if product == "" {
		product = AccountProduct(row.Account)
	}
	tag := row.Tag
	if tag == "" {
		tag = AccountTag(row.Account)
	}
	for _, key := range profileKeys(product, row.Account, tag) {
		if ov, ok := pol.AccountProfiles[key]; ok {
			return cleanProfile(ov, "policy:"+key)
		}
	}
	if product == "claude" {
		localish := strings.Contains(strings.ToLower(tag), "local") ||
			strings.Contains(strings.ToLower(row.Account), "faklocal")
		if localish {
			return cleanProfile(ProfileOverride{ModelTier: 3, Model: "local", Agent: "claude"},
				"default:claude-local")
		}
		return cleanProfile(ProfileOverride{ModelTier: 1, Model: "opus", Effort: "xhigh", Agent: "claude"},
			"default:claude-opus")
	}
	if product == "opencode" {
		models := safeOpencodeModels(row.Dir)
		model := models["model"]
		tier := modelTierFromName(model)
		tl, al := strings.ToLower(tag), strings.ToLower(row.Account)
		if tier == 3 && (strings.Contains(tl, "glm") || strings.Contains(tl, "zai") ||
			strings.Contains(al, "glm") || strings.Contains(al, "zai")) {
			if model == "" {
				model = "zai-coding-plan/glm-5.2"
			}
			tier = 2
		}
		return cleanProfile(ProfileOverride{
			ModelTier: tier, Model: model, SmallModel: models["small_model"], Agent: "opencode",
		}, "default:opencode-config")
	}
	return cleanProfile(ProfileOverride{ModelTier: 3, Agent: product}, "default:unknown")
}

// profileKeys is the policy-override key precedence shared by accountProfile and
// accountRouteWeight: exact account, product:account, product:tag, short tag, product.
func profileKeys(product, account, tag string) []string {
	return []string{account, product + ":" + account, product + ":" + tag, tag, product}
}

// accountRouteWeight resolves the operator capacity bias (default 0) from RouteWeights.
func accountRouteWeight(row Account, pol Policy) int {
	if len(pol.RouteWeights) == 0 {
		return 0
	}
	product := row.Product
	if product == "" {
		product = AccountProduct(row.Account)
	}
	tag := row.Tag
	if tag == "" {
		tag = AccountTag(row.Account)
	}
	for _, key := range profileKeys(product, row.Account, tag) {
		if w, ok := pol.RouteWeights[key]; ok {
			return w
		}
	}
	return 0
}
