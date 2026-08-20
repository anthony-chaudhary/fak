package main

import (
	"fmt"
	"strconv"
	"strings"
)

type denyAllRetrySettings struct {
	Mode     string
	Warn     int
	Final    int
	Max      int
	SameStop int
}

func newDenyAllRetrySettings() denyAllRetrySettings {
	return denyAllRetrySettings{
		Mode:     guardPreCompactModeEnforce,
		Warn:     guardStopHookDefaultWarn,
		Final:    guardStopHookDefaultFinal,
		Max:      guardStopHookDefaultMax,
		SameStop: guardStopHookSameStopFromEnv(),
	}
}

func (p *denyAllRetrySettings) String() string {
	return fmt.Sprintf("%s,warn=%d,final=%d,max=%d,same-stop=%d", p.Mode, p.Warn, p.Final, p.Max, p.SameStop)
}

func (p *denyAllRetrySettings) Set(raw string) error {
	for _, item := range strings.Split(raw, ",") {
		item = strings.TrimSpace(item)
		if item == "" {
			return fmt.Errorf("deny-all policy contains an empty setting")
		}
		if !strings.Contains(item, "=") {
			switch item {
			case guardPreCompactModeOff, guardPreCompactModeShadow, guardPreCompactModeEnforce:
				p.Mode = item
				continue
			default:
				return fmt.Errorf("deny-all mode %q must be off, shadow, or enforce", item)
			}
		}
		key, value, _ := strings.Cut(item, "=")
		n, err := strconv.Atoi(value)
		if err != nil {
			return fmt.Errorf("deny-all %s must be an integer: %q", key, value)
		}
		switch key {
		case "warn":
			p.Warn = n
		case "final":
			p.Final = n
		case "max":
			p.Max = n
		case "same-stop":
			p.SameStop = n
		default:
			return fmt.Errorf("unknown deny-all setting %q", key)
		}
	}
	return nil
}

// rewriteLegacyDenyAllArgs preserves the four pre-consolidation spellings without
// advertising four independent controls at the guard front door. Child arguments after --
// remain byte-for-byte untouched.
func rewriteLegacyDenyAllArgs(argv []string) []string {
	keys := map[string]string{
		"deny-all-warn":  "warn",
		"deny-all-final": "final",
		"deny-all-max":   "max",
		"same-stop":      "same-stop",
	}
	out := make([]string, 0, len(argv))
	for i := 0; i < len(argv); i++ {
		arg := argv[i]
		if arg == "--" {
			return append(out, argv[i:]...)
		}
		trimmed := strings.TrimLeft(arg, "-")
		name, value, hasValue := strings.Cut(trimmed, "=")
		key, legacy := keys[name]
		if !legacy {
			out = append(out, arg)
			continue
		}
		if !hasValue && i+1 < len(argv) {
			i++
			value = argv[i]
		}
		out = append(out, "--deny-all-continue="+key+"="+value)
	}
	return out
}
