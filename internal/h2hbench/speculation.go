package h2hbench

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

// ParseSpeculativeModes parses requested llama-server draft caps. These controls
// do not prove that a server has a drafter or honors the requested cap.
func ParseSpeculativeModes(raw string) ([]string, error) {
	var modes []string
	seen := map[string]bool{}
	for _, mode := range strings.Split(raw, ",") {
		mode = strings.TrimSpace(mode)
		if mode != "server-default" {
			n, err := strconv.Atoi(mode)
			if err != nil || n < 0 || n > 64 {
				return nil, fmt.Errorf("speculative: want server-default or draft caps 0..64")
			}
			mode = strconv.Itoa(n)
		}
		if seen[mode] {
			return nil, fmt.Errorf("speculative: duplicate mode %q", mode)
		}
		seen[mode] = true
		modes = append(modes, mode)
	}
	return modes, nil
}

func speculativeNonceID(cfg Config) string {
	if cfg.speculativeMode == "" {
		return cfg.RunID
	}
	return cfg.RunID + "/" + cfg.speculativeMode
}

func (s *ServerTimings) UnmarshalJSON(data []byte) error {
	type plain ServerTimings
	var parsed plain
	if err := json.Unmarshal(data, &parsed); err != nil {
		return err
	}
	var counters struct {
		Draft    *int `json:"draft_n"`
		Accepted *int `json:"draft_n_accepted"`
	}
	if err := json.Unmarshal(data, &counters); err != nil {
		return err
	}
	*s = ServerTimings(parsed)
	s.draftPresent = counters.Draft != nil
	s.acceptedPresent = counters.Accepted != nil
	return nil
}

func (s ServerTimings) MarshalJSON() ([]byte, error) {
	type plain ServerTimings
	var draft, accepted *int
	if s.draftPresent || s.DraftN != 0 {
		draft = &s.DraftN
	}
	if s.acceptedPresent || s.DraftAccepted != 0 {
		accepted = &s.DraftAccepted
	}
	return json.Marshal(struct {
		plain
		Draft    *int `json:"draft_n,omitempty"`
		Accepted *int `json:"draft_n_accepted,omitempty"`
	}{plain: plain(s), Draft: draft, Accepted: accepted})
}
