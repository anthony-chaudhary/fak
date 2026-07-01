package taskmgr

// BeatWithEvidence records a task heartbeat and grades the emitted output bytes
// in the same origin call. The returned WitnessRecord is also stored on the task
// snapshot, beside the claimed state.
func (t *Task) BeatWithEvidence(output []byte) (WitnessRecord, error) {
	return t.manager.BeatTaskWithEvidence(t.id, output)
}

// BeatWithEvidence records a step heartbeat and grades the emitted output bytes
// in the same origin call. The returned WitnessRecord is also stored on the step
// snapshot, beside the claimed state.
func (s *Step) BeatWithEvidence(output []byte) (WitnessRecord, error) {
	return s.manager.BeatStepWithEvidence(s.taskID, s.stepID, output)
}

// BeatTaskWithEvidence records a task heartbeat and immediately grades the output
// shape for that beat. This is the at-origin form of "beat, then inspect the
// transcript later": the task can be live while its output witness refuses.
func (m *Manager) BeatTaskWithEvidence(taskID string, output []byte) (WitnessRecord, error) {
	if err := m.BeatTask(taskID); err != nil {
		return WitnessRecord{}, err
	}
	return m.WitnessTask(taskID, ShapeWitness{}, []EvidenceRef{beatOutputRef(output)})
}

// BeatStepWithEvidence records a step heartbeat and immediately grades the output
// shape for that beat. The parent task heartbeat is updated by BeatStep, while the
// output verdict is stored on the step witness rung.
func (m *Manager) BeatStepWithEvidence(taskID, stepID string, output []byte) (WitnessRecord, error) {
	if err := m.BeatStep(taskID, stepID); err != nil {
		return WitnessRecord{}, err
	}
	return m.WitnessStep(taskID, stepID, ShapeWitness{}, []EvidenceRef{beatOutputRef(output)})
}

func beatOutputRef(output []byte) EvidenceRef {
	return EvidenceRef{Kind: OutputRefKind, Ref: string(output)}
}
