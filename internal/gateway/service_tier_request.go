package gateway

import (
	"errors"
	"net/http"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

const invalidServiceTierCode = "invalid_service_tier"

var errInvalidServiceTier = errors.New("service_tier must be one of auto, default, standard, scale, priority, fast, flex")

// clientServiceTier is a parsed client service_tier. Present is false when the
// request omitted the field, which keeps the response free of an echo.
type clientServiceTier struct {
	Present bool
	Mode    modelroute.ServiceMode
	// RequestedFlex records a flex request; flex is accepted but served standard.
	RequestedFlex bool
}

// parseClientServiceTier maps the client wire vocabulary onto the portable
// service mode the upstream adapters already understand.
func parseClientServiceTier(raw string) (clientServiceTier, error) {
	if raw == "" {
		return clientServiceTier{Mode: modelroute.ServiceModeStandard}, nil
	}
	tier := clientServiceTier{Present: true, Mode: modelroute.ServiceModeStandard}
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "auto", "default", "standard", "scale":
	case "priority", "fast":
		tier.Mode = modelroute.ServiceModeFast
	case "flex":
		tier.RequestedFlex = true
	default:
		return clientServiceTier{}, errInvalidServiceTier
	}
	return tier, nil
}

// rejectInvalidServiceTier writes the typed 400 for an unrecognized tier and
// reports whether it did.
func rejectInvalidServiceTier(w http.ResponseWriter, raw string) bool {
	if _, err := parseClientServiceTier(raw); err != nil {
		writeErrCode(w, http.StatusBadRequest, invalidServiceTierCode, err.Error())
		return true
	}
	return false
}

// appendSampleOpt forwards the tier only when the client sent one, so a request
// without service_tier reaches the upstream with unchanged sample options.
func (t clientServiceTier) appendSampleOpt(opts []agent.SampleOpt) []agent.SampleOpt {
	if !t.Present {
		return opts
	}
	return append(opts, agent.WithServiceTier(t.Mode))
}

// echo renders the realized tier in the OpenAI response vocabulary. A mode the
// upstream reported wins over the requested one; nothing is echoed when the
// client did not send the field.
func (t clientServiceTier) echo(realized modelroute.ServiceMode) string {
	if !t.Present {
		return ""
	}
	mode := t.Mode
	if realized == modelroute.ServiceModeStandard || realized == modelroute.ServiceModeFast {
		mode = realized
	}
	if mode == modelroute.ServiceModeFast {
		return "priority"
	}
	return "default"
}
