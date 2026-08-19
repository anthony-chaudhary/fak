// Package conversationprofile binds portable conversation intent to harness-specific adapters.
package conversationprofile

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

const Schema = "fak.conversation-profile/v1"

type Fidelity string

const (
	Required Fidelity = "required"
	Optional Fidelity = "optional"
)

type Setting struct {
	Value    string   `json:"value"`
	Fidelity Fidelity `json:"fidelity"`
}

type Profile struct {
	Schema   string             `json:"schema"`
	ID       string             `json:"id"`
	Settings map[string]Setting `json:"settings"`
}

type Binding struct {
	Key        string `json:"key"`
	Value      string `json:"value"`
	Effect     string `json:"effect"`
	Provenance string `json:"provenance"`
}

type Adapter interface {
	Name() string
	Resolve(key, value string) (Binding, bool)
}

type Gap struct {
	Key      string   `json:"key"`
	Value    string   `json:"value"`
	Fidelity Fidelity `json:"fidelity"`
}

type Receipt struct {
	Schema    string             `json:"schema"`
	ProfileID string             `json:"profile_id"`
	Adapter   string             `json:"adapter"`
	Requested map[string]Setting `json:"requested"`
	Bindings  []Binding          `json:"bindings"`
	Gaps      []Gap              `json:"gaps,omitempty"`
	Outcome   json.RawMessage    `json:"outcome"`
}

type ErrorCode string

const (
	InvalidProfile      ErrorCode = "invalid_profile"
	UnsupportedRequired ErrorCode = "unsupported_required"
	AmbiguousBinding    ErrorCode = "ambiguous_binding"
	AuthorityWidening   ErrorCode = "authority_widening"
	ExecutionFailed     ErrorCode = "execution_failed"
)

type Error struct {
	Code ErrorCode
	Key  string
	Err  error
}

func (e *Error) Error() string {
	return fmt.Sprintf("conversation profile %s (%s): %v", e.Code, e.Key, e.Err)
}
func (e *Error) Unwrap() error { return e.Err }

var vocabulary = map[string]map[string]bool{
	"response.detail":       {"brief": true, "balanced": true},
	"interaction.questions": {"when_blocked": true, "proactive": true},
	"tone":                  {"plain": true, "warm": true},
}

func Parse(data []byte) (Profile, error) {
	var p Profile
	if err := json.Unmarshal(data, &p); err != nil {
		return Profile{}, &Error{Code: InvalidProfile, Err: err}
	}
	if p.Schema != Schema || p.ID == "" || len(p.Settings) == 0 {
		return Profile{}, &Error{Code: InvalidProfile, Err: errors.New("schema, id, and settings are required")}
	}
	for key, s := range p.Settings {
		values, ok := vocabulary[key]
		if !ok || !values[s.Value] || (s.Fidelity != Required && s.Fidelity != Optional) {
			return Profile{}, &Error{Code: InvalidProfile, Key: key, Err: fmt.Errorf("unknown value %q or fidelity %q", s.Value, s.Fidelity)}
		}
	}
	return p, nil
}

func Run(ctx context.Context, p Profile, adapter Adapter, services harnesskit.Services) (Receipt, error) {
	encoded, err := json.Marshal(p)
	if err != nil {
		return Receipt{}, &Error{Code: InvalidProfile, Err: err}
	}
	if _, err := Parse(encoded); err != nil {
		return Receipt{}, err
	}
	if adapter == nil || services == nil {
		return Receipt{}, &Error{Code: InvalidProfile, Err: errors.New("adapter and services are required")}
	}
	keys := make([]string, 0, len(p.Settings))
	for key := range p.Settings {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	r := Receipt{Schema: Schema + "/receipt", ProfileID: p.ID, Adapter: adapter.Name(), Requested: cloneSettings(p.Settings)}
	seen := map[string]bool{}
	for _, key := range keys {
		s := p.Settings[key]
		b, ok := adapter.Resolve(key, s.Value)
		if !ok {
			r.Gaps = append(r.Gaps, Gap{Key: key, Value: s.Value, Fidelity: s.Fidelity})
			if s.Fidelity == Required {
				return Receipt{}, &Error{Code: UnsupportedRequired, Key: key, Err: errors.New("adapter cannot preserve required semantic")}
			}
			continue
		}
		if b.Key != key || b.Value != s.Value {
			return Receipt{}, &Error{Code: AuthorityWidening, Key: key, Err: errors.New("binding changed requested semantic")}
		}
		if b.Effect == "" || b.Provenance == "" {
			return Receipt{}, &Error{Code: InvalidProfile, Key: key, Err: errors.New("binding lacks effect or provenance")}
		}
		if seen[b.Effect] {
			return Receipt{}, &Error{Code: AmbiguousBinding, Key: key, Err: fmt.Errorf("duplicate effect %q", b.Effect)}
		}
		seen[b.Effect] = true
		r.Bindings = append(r.Bindings, b)
	}
	payload, _ := json.Marshal(struct {
		Profile  string    `json:"profile"`
		Bindings []Binding `json:"bindings"`
	}{p.ID, r.Bindings})
	result, err := services.Invoke(ctx, harnesskit.Invocation{Tool: "conversation.apply", Arguments: payload})
	if err != nil {
		return Receipt{}, &Error{Code: ExecutionFailed, Err: err}
	}
	r.Outcome = append(json.RawMessage(nil), result.Content...)
	return r, nil
}

func cloneSettings(in map[string]Setting) map[string]Setting {
	out := make(map[string]Setting, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

// DirectConfig identifies the explicit non-portable escape hatch. It is never
// accepted by Parse or Run and therefore cannot be mistaken for portable intent.
type DirectConfig struct {
	Adapter string          `json:"adapter"`
	Raw     json.RawMessage `json:"raw"`
}
