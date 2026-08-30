package main

import (
	"fmt"
	"strings"
	"time"
)

// guardFleetBusFlag composes fleet-control membership into one operator setting.
// Directory, identity, and cadence are configuration of that single concern, not
// independent front-door capabilities.
type guardFleetBusFlag struct {
	enabled  bool
	dir      string
	id       string
	interval time.Duration
}

func newGuardFleetBusFlag() guardFleetBusFlag {
	return guardFleetBusFlag{enabled: true, interval: DefaultFleetBusInterval}
}

func (f *guardFleetBusFlag) String() string {
	if f == nil {
		return ""
	}
	state := "off"
	if f.enabled {
		state = "on"
	}
	parts := []string{state}
	if f.dir != "" {
		parts = append(parts, "dir="+f.dir)
	}
	if f.id != "" {
		parts = append(parts, "id="+f.id)
	}
	if f.interval != DefaultFleetBusInterval {
		parts = append(parts, "interval="+f.interval.String())
	}
	return strings.Join(parts, ",")
}

// IsBoolFlag preserves bare --fleet-bus as the spelling for enabling membership.
func (*guardFleetBusFlag) IsBoolFlag() bool { return true }

func (f *guardFleetBusFlag) Set(raw string) error {
	for i, part := range strings.Split(raw, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			return fmt.Errorf("empty fleet-bus setting")
		}
		if !strings.Contains(part, "=") {
			if i != 0 {
				return fmt.Errorf("fleet-bus state %q must be first", part)
			}
			switch strings.ToLower(part) {
			case "true", "on":
				f.enabled = true
			case "false", "off":
				f.enabled = false
			default:
				return fmt.Errorf("fleet-bus state %q must be on or off", part)
			}
			continue
		}
		key, value, _ := strings.Cut(part, "=")
		key, value = strings.TrimSpace(key), strings.TrimSpace(value)
		if value == "" {
			return fmt.Errorf("fleet-bus %s value is empty", key)
		}
		switch key {
		case "dir":
			f.dir = value
		case "id":
			f.id = value
		case "interval":
			interval, err := time.ParseDuration(value)
			if err != nil {
				return fmt.Errorf("fleet-bus interval: %w", err)
			}
			f.interval = interval
		default:
			return fmt.Errorf("unknown fleet-bus setting %q", key)
		}
	}
	return nil
}

// rewriteLegacyGuardFleetBusArgs preserves the three pre-composition spellings while
// making --fleet-bus the single documented setting. These are compatibility aliases,
// not independent controls: each is normalized in place so ordering and last-write
// precedence remain exactly what flag parsing previously provided.
func rewriteLegacyGuardFleetBusArgs(argv []string) ([]string, error) {
	out := make([]string, 0, len(argv))
	for i := 0; i < len(argv); i++ {
		arg := argv[i]
		if arg == "--" {
			return append(out, argv[i:]...), nil
		}
		key := ""
		switch arg {
		case "--fleet-bus-dir":
			key = "dir"
		case "--fleet-bus-id":
			key = "id"
		case "--fleet-bus-interval":
			key = "interval"
		}
		if key != "" {
			if i+1 >= len(argv) || argv[i+1] == "--" {
				return nil, fmt.Errorf("%s requires a value", arg)
			}
			i++
			out = append(out, "--fleet-bus="+key+"="+argv[i])
			continue
		}
		matched := false
		for _, legacy := range []struct{ prefix, key string }{
			{"--fleet-bus-dir=", "dir"},
			{"--fleet-bus-id=", "id"},
			{"--fleet-bus-interval=", "interval"},
		} {
			if strings.HasPrefix(arg, legacy.prefix) {
				out = append(out, "--fleet-bus="+legacy.key+"="+strings.TrimPrefix(arg, legacy.prefix))
				matched = true
				break
			}
		}
		if !matched {
			out = append(out, arg)
		}
	}
	return out, nil
}
