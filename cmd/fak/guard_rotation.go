package main

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

const (
	guardRotateAuto = "auto"
	guardRotateOff  = "off"
)

type guardRotation struct{ Seat, Dir, Reason, EnvKey string }

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
		next, ok = reg.NextInRotation(currentSeat)
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
