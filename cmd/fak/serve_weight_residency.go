package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

// admitWeightResidency records the GGUF weight payload on the model, puts the
// resident-vs-file receipt on the load profile, and refuses a degraded load on a Strix
// Halo unless an admitted bringup policy lifts it.
func (rt *serveRuntime) admitWeightResidency(m *fakmodel.Model, ggufPath string) error {
	if m == nil || rt.chatBackend == nil {
		return nil
	}
	fileBytes, ok := serveFileWeightBytes(ggufPath)
	if !ok {
		return nil
	}
	m.SetFileWeightBytes(fileBytes)
	r := m.WeightResidency(rt.chatBackend)
	level := "info"
	if r.Degraded {
		level = "warning"
	}
	msg := newServeStartupMessage("model-load", "weight-residency", level, weightResidencyLine(r))
	rt.addStartupMessage(msg)
	rt.loadProfile = withServeStartupMessages(rt.loadProfile, msg)
	if r.Degraded {
		fmt.Fprintln(os.Stderr, "fak serve: "+msg.Text)
	}
	appliance := rt.strixPreflight.Detected
	admitted := rt.executionPolicy != nil && rt.executionPolicy.Admitted
	if err := r.Admit(appliance, admitted); err != nil {
		return err
	}
	if r.Degraded && appliance {
		// Not recorded in Bypassed: that list also lifts the device-only session policy.
		fmt.Fprintf(os.Stderr, "fak serve: --%s=%s admitted %s for model %q\n", serveExecutionPolicyFlag, rt.executionPolicy.Name, fakmodel.RequantizedWeightsDegradedToken, rt.executionPolicy.Model)
	}
	return nil
}

func weightResidencyLine(r fakmodel.WeightResidencyReceipt) string {
	types := strings.Join(r.RequantizedTypes, ",")
	if types == "" {
		types = "none"
	}
	line := fmt.Sprintf("weight residency file_weight_bytes=%d resident_weight_bytes=%d ratio=%.2fx requantized=%s",
		r.FileWeightBytes, r.ResidentWeightBytes, r.Ratio, types)
	if r.Degraded {
		line += fmt.Sprintf(" %s (above %.1fx)", fakmodel.RequantizedWeightsDegradedToken, fakmodel.ResidentBytesRatioAlarm)
	}
	return line
}

// serveFileWeightBytes is the GGUF header's summed tensor payload; false when path is not
// a readable GGUF.
func serveFileWeightBytes(path string) (int64, bool) {
	path = strings.TrimSpace(path)
	if info, err := os.Stat(path); path == "" || err != nil || info.IsDir() {
		return 0, false
	}
	ws, err := ggufload.OpenWeights(path)
	if err != nil {
		return 0, false
	}
	defer ws.Close()
	n, err := ws.EstimateLoadBytes()
	return n, err == nil && n > 0
}
