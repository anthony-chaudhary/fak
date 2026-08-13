package portability

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/url"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// Sensitivity is the adapter-authored data class used by the source-boundary gate.
type Sensitivity string

const (
	SensitivityUnknown             Sensitivity = "unknown"
	SensitivityPublic              Sensitivity = "public"
	SensitivityOrganization        Sensitivity = "organization"
	SensitivityPrivate             Sensitivity = "private"
	SensitivityMachineLocal        Sensitivity = "machine-local"
	SensitivityCredentialReference Sensitivity = "credential-reference"
	SensitivityForbidden           Sensitivity = "forbidden"
)

type Channel string

const (
	ChannelPublic       Channel = "public"
	ChannelOrganization Channel = "organization"
	ChannelPrivate      Channel = "private"
	ChannelMachineLocal Channel = "machine-local"
)

type EgressAction string

const (
	ActionInclude   EgressAction = "include"
	ActionReference EgressAction = "reference"
	ActionRedact    EgressAction = "redact"
	ActionDeny      EgressAction = "deny"
)

type Classification struct {
	Path        string      `json:"path"`
	Sensitivity Sensitivity `json:"sensitivity"`
	ReasonCode  string      `json:"reason_code"`
}
type FieldDisposition struct {
	Path        string       `json:"path"`
	Sensitivity Sensitivity  `json:"sensitivity"`
	Action      EgressAction `json:"action"`
	ReasonCode  string       `json:"reason_code"`
}
type EgressPreview struct {
	Channel   Channel            `json:"channel"`
	Allowed   bool               `json:"allowed"`
	Decisions []FieldDisposition `json:"decisions"`
	Payload   json.RawMessage    `json:"payload,omitempty"`
}

// SensitivityAdapter is deliberately small: new managed-object adapters can type fields
// without changing the egress policy or package format.
type SensitivityAdapter interface {
	Classify(path string, value any) (Sensitivity, string, bool)
}
type SensitivityAdapterFunc func(string, any) (Sensitivity, string, bool)

func (f SensitivityAdapterFunc) Classify(p string, v any) (Sensitivity, string, bool) { return f(p, v) }

var credentialKey = regexp.MustCompile(`(?i)(password|secret|token|api[_-]?key|private[_-]?key|credential)(?:$|[_-])`)
var credentialRefKey = regexp.MustCompile(`(?i)(credential|secret|token|key)[_-]?ref(?:erence)?$`)
var tokenLike = regexp.MustCompile(`(?i)(bearer\s+[a-z0-9._~+/=-]{8,}|(?:gh[pousr]_|sk-|xox[baprs]-)[a-z0-9._-]{8,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)`)
var piiLike = regexp.MustCompile(`(?i)(?:[a-z0-9.!#$%&'*+/=?^_` + "`" + `{|}~-]+@[a-z0-9.-]+\.[a-z]{2,}|\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b)`)
var historyKey = regexp.MustCompile(`(?i)(history|memory|transcript|trajectory|conversation|messages?)`)
var hostLike = regexp.MustCompile(`(?i)(?:^|[^a-z0-9])(?:localhost|[a-z0-9-]+\.(?:internal|local|lan))(?:[^a-z0-9]|$)`)
var privateRepo = regexp.MustCompile(`(?i)(?:git@|ssh://|https?://)[^\s]+(?:/private/|[?&](?:token|access_token)=|\.git(?:$|[#?]))`)
var winPath = regexp.MustCompile(`^[A-Za-z]:[\\/]`)

func validCredentialReference(s string) bool {
	prefixes := []string{"env:", "keychain:", "vault:", "secret-store:", "credential:"}
	for _, p := range prefixes {
		if strings.HasPrefix(s, p) && len(s) > len(p) {
			return !strings.ContainsAny(s, "\r\n")
		}
	}
	return false
}
func looksEncodedSecret(s string) bool {
	if len(s) < 16 || len(s)%4 != 0 {
		return false
	}
	b, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		return false
	}
	x := string(b)
	return tokenLike.MatchString(x) || credentialKey.MatchString(x)
}
func classifyBuiltin(path string, value any) (Sensitivity, string, bool) {
	key := path[strings.LastIndex(path, "/")+1:]
	if credentialRefKey.MatchString(key) {
		s, ok := value.(string)
		if ok && validCredentialReference(s) {
			return SensitivityCredentialReference, "credential-reference", true
		}
		return SensitivityForbidden, "credential-value-not-reference", true
	}
	if credentialKey.MatchString(key) {
		return SensitivityForbidden, "credential-material", true
	}
	if historyKey.MatchString(key) {
		return SensitivityPrivate, "history-or-memory", true
	}
	s, ok := value.(string)
	if !ok {
		return "", "", false
	}
	if tokenLike.MatchString(s) || looksEncodedSecret(s) {
		return SensitivityForbidden, "secret-pattern", true
	}
	if privateRepo.MatchString(s) {
		return SensitivityPrivate, "private-repository-or-url", true
	}
	if hostLike.MatchString(s) {
		return SensitivityMachineLocal, "private-hostname", true
	}
	if filepath.IsAbs(s) || winPath.MatchString(s) {
		return SensitivityMachineLocal, "absolute-path", true
	}
	if piiLike.MatchString(s) {
		return SensitivityPrivate, "pii", true
	}
	if u, err := url.Parse(s); err == nil && u.Hostname() != "" && (u.Scheme == "http" || u.Scheme == "https") {
		return SensitivityPublic, "public-url", true
	}
	return "", "", false
}

func actionFor(ch Channel, s Sensitivity) EgressAction {
	if s == SensitivityForbidden {
		return ActionDeny
	}
	if s == SensitivityCredentialReference {
		return ActionReference
	}
	switch ch {
	case ChannelPublic:
		if s == SensitivityPublic {
			return ActionInclude
		}
		if s == SensitivityUnknown {
			return ActionDeny
		}
		return ActionRedact
	case ChannelOrganization:
		if s == SensitivityPublic || s == SensitivityOrganization {
			return ActionInclude
		}
		if s == SensitivityUnknown {
			return ActionDeny
		}
		return ActionRedact
	case ChannelPrivate:
		if s == SensitivityMachineLocal {
			return ActionRedact
		}
		return ActionInclude
	case ChannelMachineLocal:
		return ActionInclude
	default:
		return ActionDeny
	}
}

// PreviewEgress classifies and transforms payload bytes before package identity/signing.
// Explanations contain only paths, typed classes, and stable reason codes.
func PreviewEgress(ch Channel, payload json.RawMessage, adapters ...SensitivityAdapter) (EgressPreview, error) {
	var root any
	if err := json.Unmarshal(payload, &root); err != nil {
		return EgressPreview{}, fmt.Errorf("egress payload: invalid JSON")
	}
	plan := EgressPreview{Channel: ch, Allowed: true}
	var walk func(string, any) (any, error)
	walk = func(path string, v any) (any, error) {
		switch x := v.(type) {
		case map[string]any:
			// Explicit typed value is the stable cross-adapter extension form.
			if raw, ok := x["sensitivity"].(string); ok {
				if val, has := x["value"]; has {
					return decide(path, val, Sensitivity(raw), "adapter-declared", &plan)
				}
			}
			out := map[string]any{}
			keys := make([]string, 0, len(x))
			for k := range x {
				keys = append(keys, k)
			}
			sort.Strings(keys)
			for _, k := range keys {
				nv, e := walk(path+"/"+k, x[k])
				if e != nil {
					return nil, e
				}
				out[k] = nv
			}
			return out, nil
		case []any:
			out := make([]any, len(x))
			for i := range x {
				nv, e := walk(fmt.Sprintf("%s/%d", path, i), x[i])
				if e != nil {
					return nil, e
				}
				out[i] = nv
			}
			return out, nil
		default:
			// Built-in deny patterns cannot be weakened by an adapter.
			if s, r, ok := classifyBuiltin(path, v); ok {
				return decide(path, v, s, r, &plan)
			}
			for _, a := range adapters {
				if s, r, ok := a.Classify(path, v); ok {
					return decide(path, v, s, r, &plan)
				}
			}
			return decide(path, v, SensitivityUnknown, "unclassified", &plan)
		}
	}
	transformed, err := walk("$", root)
	if err != nil {
		return EgressPreview{}, err
	}
	b, _ := json.Marshal(transformed)
	plan.Payload = b
	sort.SliceStable(plan.Decisions, func(i, j int) bool { return plan.Decisions[i].Path < plan.Decisions[j].Path })
	return plan, nil
}
func decide(path string, v any, s Sensitivity, reason string, p *EgressPreview) (any, error) {
	if s == SensitivityCredentialReference {
		raw, ok := v.(string)
		if !ok || !validCredentialReference(raw) {
			s = SensitivityForbidden
			reason = "credential-value-not-reference"
		}
	}
	switch s {
	case SensitivityUnknown, SensitivityPublic, SensitivityOrganization, SensitivityPrivate, SensitivityMachineLocal, SensitivityCredentialReference, SensitivityForbidden:
	default:
		s = SensitivityUnknown
		reason = "invalid-sensitivity"
	}
	a := actionFor(p.Channel, s)
	p.Decisions = append(p.Decisions, FieldDisposition{path, s, a, reason})
	if a == ActionDeny {
		p.Allowed = false
	}
	if a == ActionRedact {
		return "[redacted:" + string(s) + "]", nil
	}
	if a == ActionDeny {
		return "[denied]", nil
	}
	return v, nil
}
