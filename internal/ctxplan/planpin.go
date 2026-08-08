package ctxplan

import "strings"

// StepPin is one stable step in an ordered plan. StepID is assigned once by
// the caller and is part of the plan's content address alongside Text.
type StepPin struct {
	StepID string `json:"step_id"`
	Text   string `json:"text"`
}

// PlanPin content-addresses an ordered multi-step plan. Steps remain in caller
// order: changing a step ID, changing its text, adding or dropping a step, or
// reordering steps changes Digest.
type PlanPin struct {
	Steps  []StepPin `json:"steps"`
	Digest string    `json:"digest"`
}

// NewPlanPin snapshots steps and computes their canonical, order-sensitive
// digest. The copy prevents later mutations of the caller's slice from
// silently changing the pin's content.
func NewPlanPin(steps []StepPin) PlanPin {
	p := PlanPin{Steps: append([]StepPin(nil), steps...)}
	p.Digest = p.contentDigest()
	return p
}

// Verify reports whether the stored digest still matches the ordered steps.
func (p PlanPin) Verify() bool {
	return p.Digest != "" && p.Digest == p.contentDigest()
}

// IsZero reports whether the pin carries neither steps nor a digest.
func (p PlanPin) IsZero() bool {
	return len(p.Steps) == 0 && p.Digest == ""
}

func (p PlanPin) contentDigest() string {
	var canonical strings.Builder
	// Bind the step count so an empty plan cannot collide with a different
	// field shape if this representation is extended later. Each StepID/Text
	// pair reuses ObjectivePin's NUL-separated field encoding.
	canonical.WriteString(itoa(len(p.Steps)))
	canonical.WriteByte(0)
	for _, step := range p.Steps {
		canonical.Write(digestFields(step.StepID, step.Text))
		canonical.WriteByte(0)
	}
	return Digest([]byte(canonical.String()))
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}
