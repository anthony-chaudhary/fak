package polymodel

// AdmissionVictims previews the ordered victims and refusal that Admit would
// produce for m at the current pool snapshot. It changes no live membership,
// descriptors, pin state, recency or accounting. The returned slice belongs to
// the caller. In particular, previewing an existing ID does not Touch it.
//
// Like Admit, callers must serialize access. This is not a reservation: any
// intervening Admit, Touch, Evict or Resize can change the result, and a caller
// must not use a stale preview to authorize overwriting a resident allocation.
//
// The private snapshot executes the incumbent policy instead of maintaining a
// second victim-selection algorithm. Its temporary storage is O(resident count).
func (p *Pool) AdmissionVictims(m Model) ([]ModelID, error) {
	if err := p.CheckAdmission(m); err != nil {
		return nil, err
	}
	if _, exists := p.models[m.ID]; exists {
		return nil, nil
	}
	preview := *p
	preview.models = make(map[ModelID]*entry, len(p.models))
	for id, e := range p.models {
		value := *e
		preview.models[id] = &value
	}
	return preview.Admit(m)
}
