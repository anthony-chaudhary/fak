package vdso

import "encoding/json"

// RegisterNonsemanticPathFields declares top-level string path fields that do not
// affect a tool's result. Their values are omitted from that tool's tier-2 cache
// hash, while every undeclared field remains semantic.
func (v *VDSO) RegisterNonsemanticPathFields(tool string, fields ...string) {
	v.regMu.Lock()
	declaration := v.nonsemanticPathFields[tool]
	if declaration == nil {
		if v.nonsemanticPathFields == nil {
			v.nonsemanticPathFields = make(map[string]map[string]struct{})
		}
		declaration = make(map[string]struct{})
		v.nonsemanticPathFields[tool] = declaration
	}
	for _, field := range fields {
		if field != "" {
			declaration[field] = struct{}{}
		}
	}
	v.regMu.Unlock()
}

func (v *VDSO) normalizeNonsemanticPathFields(tool string, args []byte) ([]byte, bool, bool) {
	v.regMu.RLock()
	declaration := v.nonsemanticPathFields[tool]
	fields := make([]string, 0, len(declaration))
	for field := range declaration {
		fields = append(fields, field)
	}
	v.regMu.RUnlock()
	if len(fields) == 0 {
		return nil, false, true
	}

	var object map[string]any
	if err := json.Unmarshal(args, &object); err != nil || object == nil {
		return nil, true, false
	}
	for _, field := range fields {
		value, present := object[field]
		if !present {
			continue
		}
		if _, ok := value.(string); !ok {
			return nil, true, false
		}
		delete(object, field)
	}
	normalized, err := json.Marshal(object)
	if err != nil {
		return nil, true, false
	}
	return normalized, true, true
}
