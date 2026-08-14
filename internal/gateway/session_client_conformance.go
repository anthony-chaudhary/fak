package gateway

import "sort"

const SessionCapabilityCorpusSchema = "fak.session-capability-corpus.v1"

var SessionRecoveryScenarios = []string{"disconnect_replay", "duplicate_delivery", "stale_epoch", "lease_transfer", "pending_approval", "source_process_loss", "interrupted_cutover"}

type SessionClientActionSpec struct {
	ID                string `json:"id"`
	Kind              string `json:"kind"`
	Capability        string `json:"capability"`
	Label             string `json:"label"`
	Method            string `json:"method,omitempty"`
	Route             string `json:"route,omitempty"`
	Available         bool   `json:"available"`
	UnavailableCode   string `json:"unavailable_code,omitempty"`
	UnavailableReason string `json:"unavailable_reason,omitempty"`
	Handoff           string `json:"handoff,omitempty"`
}

type sessionCapabilityDefinition struct {
	ID, Capability, Label, Kind, Method, Route, Handoff string
}

var sessionCapabilityCorpus = []sessionCapabilityDefinition{
	{"observe", "observe", "Observe live state", "link", "GET", "/client", ""},
	{"terminal", "terminal_transcript", "Render the exact terminal transcript", "link", "GET", "/client", ""},
	{"replay", "replay", "Replay from cursor", "button", "POST", "/attach", ""},
	{"input", "text_input", "Send terminal input", "text", "POST", "/input", ""},
	{"approve", "approve", "Approve pending interaction", "button", "POST", "/decision", ""},
	{"deny", "deny", "Deny pending interaction", "button", "POST", "/decision", ""},
	{"steer", "text_input", "Steer the running session", "text", "POST", "/input", ""},
	{"pause", "pause", "Pause execution", "button", "", "", "Open the terminal client on the execution host."},
	{"resume", "resume", "Resume execution", "button", "", "", "Open the terminal client on the execution host."},
	{"drain", "drain", "Drain at the next safe point", "button", "", "", "Open the terminal client on the execution host."},
	{"terminate", "close", "Terminate session", "button", "POST", "/close", ""},
	{"checkpoint", "checkpoint", "Create portable checkpoint", "button", "", "", "Run `fak session checkpoint` on an admitted control point."},
	{"move", "move", "Move execution placement", "form", "POST", "/move", ""},
	{"detach", "detach", "Detach this client", "button", "POST", "/detach", ""},
	{"effect_recovery", "effect_recovery", "Resolve uncertain effects", "button", "POST", "/decision", ""},
}

func SessionCapabilityCorpus(capabilities []string) []SessionClientActionSpec {
	have := make(map[string]bool, len(capabilities))
	for _, capability := range capabilities {
		have[capability] = true
	}
	out := make([]SessionClientActionSpec, 0, len(sessionCapabilityCorpus))
	for _, definition := range sessionCapabilityCorpus {
		action := SessionClientActionSpec{ID: definition.ID, Capability: definition.Capability, Label: definition.Label, Kind: definition.Kind, Method: definition.Method, Route: definition.Route, Available: have[definition.Capability], Handoff: definition.Handoff}
		if !action.Available {
			action.UnavailableCode = "CAPABILITY_NOT_ADVERTISED"
			action.UnavailableReason = "current execution epoch does not advertise " + definition.Capability
			if action.Handoff == "" {
				action.Handoff = "Rediscover after placement or policy changes."
			}
		}
		out = append(out, action)
	}
	return out
}

func sessionCorpusCapabilities() []string {
	set := map[string]bool{}
	for _, d := range sessionCapabilityCorpus {
		set[d.Capability] = true
	}
	out := make([]string, 0, len(set))
	for capability := range set {
		out = append(out, capability)
	}
	sort.Strings(out)
	return out
}
