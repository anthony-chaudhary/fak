package model

// v41PublicationPrefix owns sealed host publication rows. Only cache admission
// creates it, by moving independently owned snapshot maps out of writable state.
// Readers never mutate these maps or rows; Go reachability retains them until
// the cache and every restored branch have released their references.
type v41PublicationPrefix struct {
	kv, index           map[v41AttentionPublicationKey][]float32
	kvBytes, indexBytes int64
}

// sealPublications changes representation without changing visible rows. The
// caller exclusively owns this state; a snapshot must seal before cache
// publication, never lazily during a hit or a live session's forward.
func (s *V41AttentionState) sealPublications() {
	if s == nil || s.kvPublications == nil || s.indexPublications == nil ||
		s.kvPublishedEnd == nil || s.indexPublishedEnd == nil {
		return // Preserve malformed/absent stores and their existing refusals.
	}
	if len(s.kvPublications) == 0 && len(s.indexPublications) == 0 {
		return
	}
	kv, index := s.kvPublications, s.indexPublications
	if s.publicationPrefix != nil {
		kv = mergeV41PublicationMaps(s.publicationPrefix.kv, kv)
		index = mergeV41PublicationMaps(s.publicationPrefix.index, index)
	}
	next := &v41PublicationPrefix{
		kv: kv, index: index,
		kvBytes: v41PublicationBytes(kv), indexBytes: v41PublicationBytes(index),
	}
	// Allocate replacement metadata before moving ownership. No writable map
	// retains an alias to the maps the immutable owner receives.
	mutableKV := make(map[v41AttentionPublicationKey][]float32)
	mutableIndex := make(map[v41AttentionPublicationKey][]float32)
	s.publicationPrefix = next
	s.kvPublications, s.indexPublications = mutableKV, mutableIndex
}

// mergeV41PublicationMaps merges descriptors only. Base rows are already sealed;
// overlay rows transfer from the exclusively owned snapshot. Overlay precedence
// matches reads and rollback, and flattening prevents unbounded prefix chains.
func mergeV41PublicationMaps(base, overlay map[v41AttentionPublicationKey][]float32) map[v41AttentionPublicationKey][]float32 {
	out := make(map[v41AttentionPublicationKey][]float32, len(base)+len(overlay))
	for key, row := range base {
		out[key] = row
	}
	for key, row := range overlay {
		out[key] = row
	}
	return out
}

// Publication lookups return read-only rows. Public readers copy them before
// returning; arithmetic must never receive these slices for in-place writes.
func (s *V41AttentionState) kvPublication(key v41AttentionPublicationKey) ([]float32, bool) {
	if s == nil || s.kvPublications == nil {
		return nil, false
	}
	if row, ok := s.kvPublications[key]; ok {
		return row, true
	}
	if s.publicationPrefix != nil {
		row, ok := s.publicationPrefix.kv[key]
		return row, ok
	}
	return nil, false
}

func (s *V41AttentionState) indexPublication(key v41AttentionPublicationKey) ([]float32, bool) {
	if s == nil || s.indexPublications == nil {
		return nil, false
	}
	if row, ok := s.indexPublications[key]; ok {
		return row, true
	}
	if s.publicationPrefix != nil {
		row, ok := s.publicationPrefix.index[key]
		return row, ok
	}
	return nil, false
}

// cloneV41PublicationView materializes a fully writable independent map. A nil
// overlay remains nil instead of repairing a malformed store with base rows.
func cloneV41PublicationView(base, overlay map[v41AttentionPublicationKey][]float32) map[v41AttentionPublicationKey][]float32 {
	if overlay == nil {
		return nil
	}
	out := cloneV41PublicationRows(base)
	if out == nil {
		out = make(map[v41AttentionPublicationKey][]float32, len(overlay))
	}
	for key, row := range overlay {
		out[key] = append([]float32(nil), row...)
	}
	return out
}

// v41PublicationViewBytes keeps the established per-snapshot logical payload
// charge. A shadow replaces a key's charge, so sealing cannot change the budget.
func v41PublicationViewBytes(base map[v41AttentionPublicationKey][]float32, baseBytes int64, overlay map[v41AttentionPublicationKey][]float32) int64 {
	if overlay == nil {
		return 0
	}
	bytes := baseBytes
	for key, row := range overlay {
		bytes += int64(len(row)) * 4
		if previous, ok := base[key]; ok {
			bytes -= int64(len(previous)) * 4
		}
	}
	return bytes
}
