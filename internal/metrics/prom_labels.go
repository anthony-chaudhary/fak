package metrics

import "strings"

// ParsePromLabels decodes a comma-separated Prometheus label set without braces.
func ParsePromLabels(value string) (map[string]string, bool) {
	labels := map[string]string{}
	for strings.TrimSpace(value) != "" {
		value = strings.TrimLeft(value, " \t,")
		equal := strings.IndexByte(value, '=')
		if equal <= 0 {
			return nil, false
		}
		key := strings.TrimSpace(value[:equal])
		value = strings.TrimLeft(value[equal+1:], " \t")
		if !strings.HasPrefix(value, `"`) {
			return nil, false
		}
		decoded, consumed, ok := ParsePromQuotedLabel(value)
		if !ok {
			return nil, false
		}
		labels[key] = decoded
		value = value[consumed:]
	}
	return labels, true
}
