// Package fleetaccounts is the Go port of the READ-ONLY roster/resolve/probe fold
// from tools/fleet_accounts.py — the single source of truth for "what is an account,
// and is it offered?" across the Claude Code, Codex, and opencode product families.
//
// The fleet resume/switcher layer discovers accounts by globbing config dirs:
// Claude and Codex dirs under the user home (<home>/.claude*, <home>/.codex*) and
// opencode dirs under the XDG config home (<config_home>/opencode*). Each discovered dir
// is classified into ONE
// Kind — worker | excluded | non-account — by an operator-editable POLICY
// (accounts_policy.json), then folded with live runtime status (usage throttle /
// auth block / live sessions) read from the watchdog's sessions.json registry.
//
// This package preserves the Python contract for the high-frequency read-only operators
// the standalone operators + resume/watchdog paths use, while adding the shared
// credential-safe login_status/can_serve fields for Claude switcher rows:
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

	configaccounts "github.com/anthony-chaudhary/fak/internal/accounts"
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

const (
	// Benchmark-ranked NVIDIA NIM coding seats, snapshot 2026-07-06.
	NIMDeepSeekV4ProModel = "deepseek-ai/deepseek-v4-pro"
	NIMKimiK26Model       = "moonshotai/kimi-k2.6"
	NIMGLM52Model         = "z-ai/glm-5.2"
)

var nimCodingSeatProfiles = map[string]ProfileOverride{
	"nim-deepseek-v4-pro": {ModelTier: 1, Model: NIMDeepSeekV4ProModel, Agent: "opencode"},
	"nim-kimi-k26":        {ModelTier: 1, Model: NIMKimiK26Model, Agent: "opencode"},
	"nim-glm52":           {ModelTier: 1, Model: NIMGLM52Model, Agent: "opencode"},
}

var defaultNIMCodingRouteWeights = map[string]int{
	"opencode:nim-deepseek-v4-pro": 30,
	"opencode:nim-kimi-k26":        20,
	"opencode:nim-glm52":           10,
}

// Policy is the operator-editable account policy (accounts_policy.json), applying
// uniformly to every discovered product. Exclude substrings tombstone accounts; IncludeOnly (when
// non-empty) is an allowlist. AccountProfiles overrides model-tier inference; RouteWeights
// biases the routing tie-break. LaneModels pins a dispatch worker's model per LANE (the
// model-switching config seam: a lane can name the model its resolution workers start on,
// independent of the seat's own default). The JSON keys match the Python policy file.
type Policy struct {
	Exclude         []string                   `json:"exclude"`
	IncludeOnly     []string                   `json:"include_only"`
	Notes           map[string]string          `json:"notes"`
	AccountProfiles map[string]ProfileOverride `json:"account_profiles"`
	RouteWeights    map[string]int             `json:"route_weights"`
	LaneModels      map[string]string          `json:"lane_models"`
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
// auto-resume roster, conservative tier inference, and built-in route bias for the
// current NVIDIA NIM coding-seat trio.
func DefaultPolicy() Policy {
	return Policy{
		Exclude:     []string{"backup", "breakglass"},
		IncludeOnly: []string{},
		Notes: map[string]string{
			"backup": "break-glass backup account; never auto-resume",
		},
		AccountProfiles: map[string]ProfileOverride{},
		RouteWeights: map[string]int{
			"opencode:nim-deepseek-v4-pro": defaultNIMCodingRouteWeights["opencode:nim-deepseek-v4-pro"],
			"opencode:nim-kimi-k26":        defaultNIMCodingRouteWeights["opencode:nim-kimi-k26"],
			"opencode:nim-glm52":           defaultNIMCodingRouteWeights["opencode:nim-glm52"],
		},
		LaneModels: map[string]string{},
		Routing: Routing{
			LightConfidence:   0.999,
			HardTier1Fallback: "stop",
		},
	}
}

// AccountProduct classifies a discovered dir basename to its product family.
// .claude* -> "claude" and .codex* -> "codex" under the user home; opencode* ->
// "opencode" under ~/.config. Anything else defaults to "claude" so historical call sites
// keep working.
func AccountProduct(account string) string {
	lower := strings.ToLower(account)
	if strings.HasPrefix(lower, "opencode") {
		return "opencode"
	}
	if strings.HasPrefix(lower, ".codex") {
		return "codex"
	}
	return "claude"
}

// AccountTag normalizes a config-dir basename to its short tag, matching the resume
// layer convention. Claude: ".claude-gem8-acct" -> "gem8"; ".claude" -> "default".
// Codex: ".codex-work" -> "work"; ".codex" -> "default". opencode:
// "opencode-glm" -> "glm"; "opencode" -> "default". The trailing "-acct" org suffix is
// stripped if present.
func AccountTag(account string) string {
	product := AccountProduct(account)
	var tag string
	switch product {
	case "opencode":
		tag = strings.ReplaceAll(account, "opencode-", "")
		tag = strings.ReplaceAll(tag, "opencode", "")
	case "codex":
		tag = strings.ReplaceAll(account, ".codex-", "")
		tag = strings.ReplaceAll(tag, ".codex", "")
	default:
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

// modelTierFromName is the small v1 model taxonomy. Tier 0 is the restricted apex
// model (Fable 5 — see apextier.go); tier 1 is the max-quality frontier set; tier 2 is
// the lightweight-work set (GLM-5.2 and Gemini 3.5 Flash); everything else is tier 3.
func modelTierFromName(model string) int {
	if IsApexModel(model) {
		return TierApex
	}
	text := strings.ToLower(model)
	text = strings.ReplaceAll(text, "_", "-")
	text = strings.ReplaceAll(text, " ", "-")
	compact := nonAlnum.ReplaceAllString(text, "")
	// GPT-5.6 Luna is OpenAI's fast/cheap seat in the 5.6 generation — the
	// lightweight tier, checked before the generation-wide frontier match below so
	// the `gpt-5.6` substring does not sweep it up into tier 1.
	if strings.Contains(text, "gpt-5.6-luna") || strings.Contains(compact, "gpt56luna") {
		return 2
	}
	// GPT-5.6 Sol (the flagship, aliased by the bare `gpt-5.6` id) and GPT-5.6 Terra
	// (≈ GPT-5.5) are the current OpenAI frontier seats; GPT-5.5 stays classified
	// alongside them.
	if strings.Contains(text, "gpt-5.6") || strings.Contains(compact, "gpt56") ||
		strings.Contains(text, "gpt-5.5") || strings.Contains(compact, "gpt55") {
		return 1
	}
	if strings.Contains(text, "opus-4.6") || strings.Contains(compact, "opus46") ||
		text == "opus" || text == "claude-opus" {
		return 1
	}
	if strings.Contains(text, "deepseek-v4-pro") || strings.Contains(compact, "deepseekv4pro") ||
		strings.Contains(text, "kimi-k2.6") || strings.Contains(compact, "kimik26") {
		return 1
	}
	if strings.Contains(text, "glm-5.2") || strings.Contains(compact, "glm52") {
		return 2
	}
	// Gemini 3.5 Flash — Google's fast/lightweight tier, served via GCP Vertex AI on the
	// OpenAI-compatible endpoint as `google/gemini-3.5-flash`. Lightweight work, tier 2.
	if strings.Contains(text, "gemini-3.5-flash") || strings.Contains(compact, "gemini35flash") {
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
	// An explicit profile tier of 0 means "unset" (fall through to name inference),
	// NOT apex: the apex tier is reached only by naming a Fable-5 model, never by
	// setting model_tier:0 in a profile. That keeps the restricted apex tier out of
	// reach of a casual numeric override — see apextier.go.
	if p.ModelTier != TierFrontier && p.ModelTier != TierLight && p.ModelTier != TierOther {
		p.ModelTier = modelTierFromName(raw.Model)
	}
	// An INFERRED apex (tier 0 from a Fable-5 name) is kept; anything still outside the
	// taxonomy falls to tier 3.
	if !validModelTier(p.ModelTier) {
		p.ModelTier = TierOther
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
	product, tag := resolveProductTag(row)
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
	if product == "codex" {
		return cleanProfile(ProfileOverride{
			ModelTier: TierFrontier,
			Model:     configaccounts.CodexDefaultModel,
			Effort:    configaccounts.CodexDefaultReasoningEffort,
			Agent:     "codex",
		}, "default:codex-"+configaccounts.CodexDefaultModel+"-"+configaccounts.CodexDefaultReasoningEffort)
	}
	if product == "opencode" {
		models := safeOpencodeModels(row.Dir)
		if ov, ok := nimCodingSeatProfiles[strings.ToLower(tag)]; ok {
			if ov.SmallModel == "" {
				ov.SmallModel = models["small_model"]
			}
			return cleanProfile(ov, "default:nvidia-nim-coding:"+tag)
		}
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

// resolveProductTag returns the account's product and tag, falling back to deriving each
// from the account name when the row leaves it blank. Shared by accountProfile and
// accountRouteWeight before they consult profileKeys.
func resolveProductTag(row Account) (product, tag string) {
	product = row.Product
	if product == "" {
		product = AccountProduct(row.Account)
	}
	tag = row.Tag
	if tag == "" {
		tag = AccountTag(row.Account)
	}
	return product, tag
}

// profileKeys is the policy-override key precedence shared by accountProfile and
// accountRouteWeight: exact account, product:account, product:tag, short tag, product.
func profileKeys(product, account, tag string) []string {
	return []string{account, product + ":" + account, product + ":" + tag, tag, product}
}

// LaneModel returns the model id an operator pinned for a dispatch LANE via lane_models,
// or "" when the lane has no pin. Match is on the trimmed lane name, exact first, then
// case-insensitive so a "Docs" pin still covers the "docs" lane. Empty/blank lane -> "".
func (p Policy) LaneModel(lane string) string {
	lane = strings.TrimSpace(lane)
	if lane == "" || len(p.LaneModels) == 0 {
		return ""
	}
	if m := strings.TrimSpace(p.LaneModels[lane]); m != "" {
		return m
	}
	for k, v := range p.LaneModels {
		if strings.EqualFold(strings.TrimSpace(k), lane) {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

// ProfileModel returns the model id an account's routing PROFILE resolves to (the
// account_profiles override, else the product default). It is the per-account fallback
// the model resolver consults after a per-lane pin: "" only when the profile itself names
// no model (e.g. a bare non-account row).
func (p Policy) ProfileModel(row Account) string {
	return strings.TrimSpace(accountProfile(row, p).Model)
}

// accountRouteWeight resolves the operator capacity bias (default 0) from RouteWeights.
func accountRouteWeight(row Account, pol Policy) int {
	if len(pol.RouteWeights) == 0 {
		return 0
	}
	product, tag := resolveProductTag(row)
	for _, key := range profileKeys(product, row.Account, tag) {
		if w, ok := pol.RouteWeights[key]; ok {
			return w
		}
	}
	return 0
}
