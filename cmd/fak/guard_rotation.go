package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/harnessprofile"
	"github.com/anthony-chaudhary/fak/internal/journal"
)

const (
	guardRotateAuto = "auto"
	guardRotateOff  = "off"
)

type guardRotation struct{ Seat, Dir, Reason, EnvKey string }

type guardRotationRuntime struct {
	Mode        string
	CurrentSeat string
	Registry    accounts.Registry
	EnvKey      string
	Headroom    accounts.RotationHeadroom
}

func guardRotationRuntimeFor(command []string, mode string) guardRotationRuntime {
	r := guardRotationRuntime{Mode: mode}
	if mode == guardRotateOff || len(command) == 0 {
		return r
	}
	profile, ok := harnessprofile.Lookup(command[0])
	if !ok || strings.TrimSpace(profile.ConfigHomeGlob) == "" {
		return r
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return r
	}
	homes, err := accounts.DiscoverProfile(home, profile)
	if err != nil {
		return r
	}
	r.Registry = accounts.Registry{Homes: homes}
	switch profile.Identity {
	case harnessprofile.IdentityClaude:
		r.EnvKey = "CLAUDE_CONFIG_DIR"
	case harnessprofile.IdentityCodex:
		r.EnvKey = "CODEX_HOME"
	}
	for _, h := range homes {
		if current := os.Getenv(r.EnvKey); current != "" && strings.EqualFold(current, h.Dir) {
			r.CurrentSeat = h.Name
			break
		}
	}
	return r
}

func (rt *guardRotationRuntime) rotate(command []string, injected [][2]string, reason string, audit *journal.Journal, traceID string, stderr *os.File) ([]string, [][2]string, bool) {
	if rt == nil || rt.Mode == guardRotateOff {
		return command, injected, false
	}
	r, ok := guardNextRotationWithHeadroom(rt.Registry, rt.CurrentSeat, rt.Mode, reason, rt.EnvKey, rt.Headroom)
	if !ok {
		return command, injected, false
	}
	command, injected = guardApplyRotation(command, injected, r)
	rt.CurrentSeat = r.Seat
	if stderr != nil {
		fmt.Fprint(stderr, guardRotationBanner(r))
	}
	if audit != nil {
		audit.AppendAgentEvent("ACCOUNT_ROTATION", traceID, r.Seat+":"+r.Reason)
	}
	return command, injected, true
}

func normalizeGuardRotateMode(raw string, explicitlySet, interactive bool) (string, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		if !explicitlySet && interactive {
			return guardRotateOff, nil
		}
		return guardRotateAuto, nil
	}
	if raw == guardRotateAuto || raw == guardRotateOff {
		return raw, nil
	}
	if strings.ContainsAny(raw, "/\\") {
		return "", fmt.Errorf("guard --rotate: seat must be a name, not a path")
	}
	return raw, nil
}

func guardNextRotation(reg accounts.Registry, currentSeat, requested, reason, envKey string) (guardRotation, bool) {
	return guardNextRotationWithHeadroom(reg, currentSeat, requested, reason, envKey, nil)
}

func guardNextRotationWithHeadroom(reg accounts.Registry, currentSeat, requested, reason, envKey string, headroom accounts.RotationHeadroom) (guardRotation, bool) {
	if requested == guardRotateOff {
		return guardRotation{}, false
	}
	var next accounts.RotationSeat
	var ok bool
	if requested != guardRotateAuto {
		for _, s := range reg.RotationPlan().Pool {
			if s.Name == requested {
				next = s
				ok = s.Status == accounts.RotationIncluded
				break
			}
		}
	} else {
		next, ok = reg.NextInRotationWithHeadroom(currentSeat, headroom)
	}
	if !ok || next.Name == currentSeat {
		return guardRotation{}, false
	}
	return guardRotation{Seat: next.Name, Dir: next.Dir, Reason: reason, EnvKey: envKey}, true
}

func guardApplyRotation(command []string, injected [][2]string, r guardRotation) ([]string, [][2]string) {
	if strings.TrimSpace(r.EnvKey) == "" || strings.TrimSpace(r.Dir) == "" {
		return command, injected
	}
	out := append([][2]string(nil), injected...)
	for i := range out {
		if strings.EqualFold(out[i][0], r.EnvKey) {
			out[i][1] = r.Dir
			return command, out
		}
	}
	return command, append(out, [2]string{r.EnvKey, r.Dir})
}

func guardRotationBanner(r guardRotation) string {
	return fmt.Sprintf("fak guard: rotated to account %s (%s)\n", r.Seat, r.Reason)
}
