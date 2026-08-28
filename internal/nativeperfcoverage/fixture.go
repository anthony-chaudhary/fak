package nativeperfcoverage

import (
	"bufio"
	"bytes"
	"fmt"
	"math"
	"strconv"
	"strings"
)

// ParseFixture parses the text exposition subset used by deterministic
// Prometheus fixtures, including escaped labels and optional millisecond times.
func ParseFixture(raw []byte) ([]Series, error) {
	var series []Series
	scanner := bufio.NewScanner(bytes.NewReader(raw))
	scanner.Buffer(make([]byte, 1024), 1024*1024)
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		sample, err := parseSample(line)
		if err != nil {
			return nil, fmt.Errorf("line %d: %w", lineNo, err)
		}
		series = append(series, sample)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if len(series) == 0 {
		return nil, fmt.Errorf("fixture has no samples")
	}
	return series, nil
}

func parseSample(line string) (Series, error) {
	var sample Series
	metricEnd := strings.IndexAny(line, "{ \t")
	if metricEnd < 0 {
		return sample, fmt.Errorf("sample lacks a value")
	}
	sample.Metric = line[:metricEnd]
	if !validMetricName(sample.Metric) {
		return sample, fmt.Errorf("invalid metric name %q", sample.Metric)
	}
	rest := strings.TrimSpace(line[metricEnd:])
	sample.Labels = make(map[string]string)
	if strings.HasPrefix(rest, "{") {
		end, err := closingBrace(rest)
		if err != nil {
			return sample, err
		}
		labels, err := parseLabels(rest[1:end])
		if err != nil {
			return sample, err
		}
		sample.Labels = labels
		rest = strings.TrimSpace(rest[end+1:])
	}
	fields := strings.Fields(rest)
	if len(fields) < 1 || len(fields) > 2 {
		return sample, fmt.Errorf("sample must have value and optional timestamp")
	}
	value, err := parseFloat(fields[0])
	if err != nil {
		return sample, fmt.Errorf("value %q: %w", fields[0], err)
	}
	sample.Value = value
	if len(fields) == 2 {
		timestamp, err := strconv.ParseInt(fields[1], 10, 64)
		if err != nil {
			return sample, fmt.Errorf("timestamp %q: %w", fields[1], err)
		}
		sample.TimestampMS = &timestamp
	}
	return sample, nil
}

func validMetricName(name string) bool {
	if name == "" {
		return false
	}
	for i, r := range name {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || r == '_' || r == ':' || (i > 0 && r >= '0' && r <= '9') {
			continue
		}
		return false
	}
	return true
}

func closingBrace(text string) (int, error) {
	quoted, escaped := false, false
	for i, r := range text {
		if escaped {
			escaped = false
			continue
		}
		if quoted && r == '\\' {
			escaped = true
			continue
		}
		if r == '"' {
			quoted = !quoted
			continue
		}
		if r == '}' && !quoted {
			return i, nil
		}
	}
	return 0, fmt.Errorf("unterminated label set")
}

func parseLabels(text string) (map[string]string, error) {
	labels := make(map[string]string)
	for strings.TrimSpace(text) != "" {
		text = strings.TrimSpace(text)
		eq := strings.IndexByte(text, '=')
		if eq <= 0 {
			return nil, fmt.Errorf("invalid label set near %q", text)
		}
		name := strings.TrimSpace(text[:eq])
		_, duplicate := labels[name]
		if !validLabelName(name) || duplicate {
			return nil, fmt.Errorf("invalid or duplicate label %q", name)
		}
		text = strings.TrimSpace(text[eq+1:])
		if !strings.HasPrefix(text, `"`) {
			return nil, fmt.Errorf("label %q value is not quoted", name)
		}
		value, consumed, err := quotedValue(text)
		if err != nil {
			return nil, fmt.Errorf("label %q: %w", name, err)
		}
		labels[name] = value
		text = strings.TrimSpace(text[consumed:])
		if text == "" {
			break
		}
		if text[0] != ',' {
			return nil, fmt.Errorf("label %q is not comma separated", name)
		}
		text = text[1:]
	}
	return labels, nil
}

func quotedValue(text string) (string, int, error) {
	for i := 1; i < len(text); i++ {
		if text[i] == '\\' {
			i++
			continue
		}
		if text[i] == '"' {
			value, err := strconv.Unquote(text[:i+1])
			return value, i + 1, err
		}
	}
	return "", 0, fmt.Errorf("unterminated quoted value")
}

func validLabelName(name string) bool {
	if name == "" {
		return false
	}
	for i, r := range name {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || r == '_' || (i > 0 && r >= '0' && r <= '9') {
			continue
		}
		return false
	}
	return true
}

func parseFloat(text string) (float64, error) {
	switch text {
	case "+Inf", "Inf":
		return math.Inf(1), nil
	case "-Inf":
		return math.Inf(-1), nil
	case "NaN":
		return math.NaN(), nil
	default:
		return strconv.ParseFloat(text, 64)
	}
}
