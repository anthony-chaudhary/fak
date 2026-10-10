package taskrun

import (
	"bytes"
	"encoding/json"
	"errors"
	"math"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// Sampling is the explicit decode configuration for task children. A nil field
// is not sent, so the endpoint default applies; the planner's built-in
// temperature 0 still rides the wire unless Temperature overrides it.
type Sampling struct {
	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"top_p,omitempty"`
	TopK        *int     `json:"top_k,omitempty"`
	MaxTokens   *int     `json:"max_tokens,omitempty"`
	Seed        *int64   `json:"seed,omitempty"`
}

func (s Sampling) Validate() error {
	if s.Temperature != nil && (*s.Temperature < 0 || math.IsNaN(*s.Temperature) || math.IsInf(*s.Temperature, 0)) {
		return errors.New("taskrun: temperature must be a finite value >= 0")
	}
	if s.TopP != nil && (!(*s.TopP > 0) || *s.TopP > 1) {
		return errors.New("taskrun: top_p must be in (0, 1]")
	}
	if s.TopK != nil && *s.TopK < 0 {
		return errors.New("taskrun: top_k must be >= 0")
	}
	if s.MaxTokens != nil && *s.MaxTokens <= 0 {
		return errors.New("taskrun: max_tokens must be positive")
	}
	return nil
}

func (s Sampling) empty() bool {
	return s.Temperature == nil && s.TopP == nil && s.TopK == nil && s.MaxTokens == nil && s.Seed == nil
}

// newTaskClient builds the child's OpenAI-compatible planner. Temperature and
// max_tokens configure the planner; top_p and top_k ride as per-request sample
// options; seed has no first-class field, so it rides the provider extra body.
func newTaskClient(endpoint, model string, s Sampling) (*agent.HTTPPlanner, []agent.SampleOpt, error) {
	if err := s.Validate(); err != nil {
		return nil, nil, err
	}
	p := agent.NewHTTPPlanner(endpoint, model, "")
	var opts []agent.SampleOpt
	if s.Temperature != nil {
		p.Temperature = *s.Temperature
		opts = append(opts, agent.WithTemperature(s.Temperature))
	}
	if s.MaxTokens != nil {
		p.MaxTokens = *s.MaxTokens
		opts = append(opts, agent.WithMaxTokens(*s.MaxTokens))
	}
	if s.TopP != nil {
		opts = append(opts, agent.WithTopP(s.TopP))
	}
	if s.TopK != nil {
		opts = append(opts, agent.WithChatWireTopK(s.TopK))
	}
	if s.Seed != nil {
		extra, err := json.Marshal(map[string]int64{"seed": *s.Seed})
		if err != nil {
			return nil, nil, err
		}
		if err := p.SetExtraBodyJSON(string(extra)); err != nil {
			return nil, nil, err
		}
	}
	return p, opts, nil
}

// sentSampling extracts the sampling fields an OpenAI-compatible request body
// actually carried. It is the witness the observer records per request.
func sentSampling(body []byte) Sampling {
	var v struct {
		Temperature *float64     `json:"temperature"`
		TopP        *float64     `json:"top_p"`
		TopK        *int         `json:"top_k"`
		MaxTokens   *int         `json:"max_tokens"`
		Seed        *json.Number `json:"seed"`
	}
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.UseNumber()
	if dec.Decode(&v) != nil {
		return Sampling{}
	}
	s := Sampling{Temperature: v.Temperature, TopP: v.TopP, TopK: v.TopK, MaxTokens: v.MaxTokens}
	if v.Seed != nil {
		if n, err := v.Seed.Int64(); err == nil {
			s.Seed = &n
		}
	}
	return s
}
