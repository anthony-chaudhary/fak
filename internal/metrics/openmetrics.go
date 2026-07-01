package metrics

import (
	"bytes"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

// OpenMetricType is the Prometheus/OpenMetrics family kind.
type OpenMetricType string

const (
	// OpenMetricCounter renders a monotonically increasing counter family.
	OpenMetricCounter OpenMetricType = "counter"
	// OpenMetricGauge renders a point-in-time gauge family.
	OpenMetricGauge OpenMetricType = "gauge"
	// OpenMetricHistogram renders cumulative histogram bucket, sum, and count series.
	OpenMetricHistogram OpenMetricType = "histogram"
)

// OpenMetricLabel is one metric label pair. Labels are rendered in name order.
type OpenMetricLabel struct {
	Name  string
	Value string
}

// OpenMetricSample is one counter or gauge sample within a family.
type OpenMetricSample struct {
	Labels []OpenMetricLabel
	Value  float64
}

// OpenMetricBucket is one cumulative histogram bucket.
type OpenMetricBucket struct {
	UpperBound      float64
	CumulativeCount uint64
}

// OpenMetricHistogramSample is one histogram instance within a histogram family.
type OpenMetricHistogramSample struct {
	Labels  []OpenMetricLabel
	Buckets []OpenMetricBucket
	Count   uint64
	Sum     float64
}

// OpenMetricFamily is a complete metric family ready for text exposition.
type OpenMetricFamily struct {
	Name       string
	Help       string
	Type       OpenMetricType
	Samples    []OpenMetricSample
	Histograms []OpenMetricHistogramSample
}

// RenderOpenMetricsText renders metric families in deterministic OpenMetrics text form.
func RenderOpenMetricsText(families []OpenMetricFamily) ([]byte, error) {
	ordered := append([]OpenMetricFamily(nil), families...)
	sort.SliceStable(ordered, func(i, j int) bool {
		return ordered[i].Name < ordered[j].Name
	})

	seenFamilies := make(map[string]struct{}, len(ordered))
	var out bytes.Buffer
	for _, family := range ordered {
		if err := validateMetricName(family.Name); err != nil {
			return nil, fmt.Errorf("metrics: invalid metric family %q: %w", family.Name, err)
		}
		if _, ok := seenFamilies[family.Name]; ok {
			return nil, fmt.Errorf("metrics: duplicate metric family %q", family.Name)
		}
		seenFamilies[family.Name] = struct{}{}

		if err := validateOpenMetricType(family.Type); err != nil {
			return nil, fmt.Errorf("metrics: family %q: %w", family.Name, err)
		}
		if family.Type == OpenMetricHistogram {
			if len(family.Samples) != 0 {
				return nil, fmt.Errorf("metrics: histogram family %q has scalar samples", family.Name)
			}
		} else if len(family.Histograms) != 0 {
			return nil, fmt.Errorf("metrics: scalar family %q has histogram samples", family.Name)
		}

		out.WriteString("# HELP ")
		out.WriteString(family.Name)
		out.WriteByte(' ')
		out.WriteString(escapeOpenMetricHelp(family.Help))
		out.WriteByte('\n')
		out.WriteString("# TYPE ")
		out.WriteString(family.Name)
		out.WriteByte(' ')
		out.WriteString(string(family.Type))
		out.WriteByte('\n')

		if family.Type == OpenMetricHistogram {
			histograms, err := normalizeOpenMetricHistograms(family.Name, family.Histograms)
			if err != nil {
				return nil, err
			}
			for _, histogram := range histograms {
				for _, bucket := range histogram.buckets {
					out.WriteString(family.Name)
					out.WriteString("_bucket")
					writeOpenMetricBucketLabels(&out, histogram.labels, formatOpenMetricFloat(bucket.UpperBound))
					out.WriteByte(' ')
					out.WriteString(strconv.FormatUint(bucket.CumulativeCount, 10))
					out.WriteByte('\n')
				}
				out.WriteString(family.Name)
				out.WriteString("_sum")
				writeOpenMetricLabels(&out, histogram.labels)
				out.WriteByte(' ')
				out.WriteString(formatOpenMetricFloat(histogram.sum))
				out.WriteByte('\n')

				out.WriteString(family.Name)
				out.WriteString("_count")
				writeOpenMetricLabels(&out, histogram.labels)
				out.WriteByte(' ')
				out.WriteString(strconv.FormatUint(histogram.count, 10))
				out.WriteByte('\n')
			}
			continue
		}

		samples, err := normalizeOpenMetricSamples(family.Name, family.Samples)
		if err != nil {
			return nil, err
		}
		for _, sample := range samples {
			out.WriteString(family.Name)
			writeOpenMetricLabels(&out, sample.labels)
			out.WriteByte(' ')
			out.WriteString(formatOpenMetricFloat(sample.value))
			out.WriteByte('\n')
		}
	}

	out.WriteString("# EOF\n")
	return out.Bytes(), nil
}

type normalizedOpenMetricSample struct {
	labels []OpenMetricLabel
	key    string
	value  float64
}

type normalizedOpenMetricHistogram struct {
	labels  []OpenMetricLabel
	key     string
	buckets []OpenMetricBucket
	count   uint64
	sum     float64
}

func normalizeOpenMetricSamples(familyName string, samples []OpenMetricSample) ([]normalizedOpenMetricSample, error) {
	out := make([]normalizedOpenMetricSample, 0, len(samples))
	for _, sample := range samples {
		labels, key, err := canonicalOpenMetricLabels(sample.Labels)
		if err != nil {
			return nil, fmt.Errorf("metrics: family %q: %w", familyName, err)
		}
		out = append(out, normalizedOpenMetricSample{
			labels: labels,
			key:    key,
			value:  sample.Value,
		})
	}
	sort.SliceStable(out, func(i, j int) bool {
		return out[i].key < out[j].key
	})
	for i := 1; i < len(out); i++ {
		if out[i].key == out[i-1].key {
			return nil, fmt.Errorf("metrics: family %q has duplicate sample label set", familyName)
		}
	}
	return out, nil
}

func normalizeOpenMetricHistograms(familyName string, histograms []OpenMetricHistogramSample) ([]normalizedOpenMetricHistogram, error) {
	out := make([]normalizedOpenMetricHistogram, 0, len(histograms))
	for _, histogram := range histograms {
		labels, key, err := canonicalOpenMetricLabels(histogram.Labels)
		if err != nil {
			return nil, fmt.Errorf("metrics: family %q: %w", familyName, err)
		}
		for _, label := range labels {
			if label.Name == "le" {
				return nil, fmt.Errorf("metrics: histogram family %q uses reserved label %q", familyName, label.Name)
			}
		}
		buckets, err := normalizeOpenMetricBuckets(familyName, histogram.Count, histogram.Buckets)
		if err != nil {
			return nil, err
		}
		out = append(out, normalizedOpenMetricHistogram{
			labels:  labels,
			key:     key,
			buckets: buckets,
			count:   histogram.Count,
			sum:     histogram.Sum,
		})
	}
	sort.SliceStable(out, func(i, j int) bool {
		return out[i].key < out[j].key
	})
	for i := 1; i < len(out); i++ {
		if out[i].key == out[i-1].key {
			return nil, fmt.Errorf("metrics: histogram family %q has duplicate sample label set", familyName)
		}
	}
	return out, nil
}

func normalizeOpenMetricBuckets(familyName string, count uint64, buckets []OpenMetricBucket) ([]OpenMetricBucket, error) {
	out := append([]OpenMetricBucket(nil), buckets...)
	for _, bucket := range out {
		if math.IsNaN(bucket.UpperBound) || math.IsInf(bucket.UpperBound, -1) {
			return nil, fmt.Errorf("metrics: histogram family %q has invalid bucket bound %v", familyName, bucket.UpperBound)
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		if math.IsInf(out[i].UpperBound, 1) {
			return false
		}
		if math.IsInf(out[j].UpperBound, 1) {
			return true
		}
		return out[i].UpperBound < out[j].UpperBound
	})

	var previousCount uint64
	var previousBound float64
	hasPreviousBound := false
	hasPositiveInfinity := false
	for i, bucket := range out {
		if bucket.CumulativeCount < previousCount {
			return nil, fmt.Errorf("metrics: histogram family %q has non-monotonic bucket counts", familyName)
		}
		if bucket.CumulativeCount > count {
			return nil, fmt.Errorf("metrics: histogram family %q bucket count exceeds total count", familyName)
		}
		if hasPreviousBound && bucket.UpperBound == previousBound {
			return nil, fmt.Errorf("metrics: histogram family %q has duplicate bucket bound %s", familyName, formatOpenMetricFloat(bucket.UpperBound))
		}
		if math.IsInf(bucket.UpperBound, 1) {
			hasPositiveInfinity = true
			if bucket.CumulativeCount != count {
				return nil, fmt.Errorf("metrics: histogram family %q +Inf bucket count must equal total count", familyName)
			}
			if i != len(out)-1 {
				return nil, fmt.Errorf("metrics: histogram family %q has bucket after +Inf", familyName)
			}
		}
		previousCount = bucket.CumulativeCount
		previousBound = bucket.UpperBound
		hasPreviousBound = true
	}
	if !hasPositiveInfinity {
		out = append(out, OpenMetricBucket{UpperBound: math.Inf(1), CumulativeCount: count})
	}
	return out, nil
}

func validateOpenMetricType(metricType OpenMetricType) error {
	switch metricType {
	case OpenMetricCounter, OpenMetricGauge, OpenMetricHistogram:
		return nil
	default:
		return fmt.Errorf("unknown metric type %q", metricType)
	}
}

func validateMetricName(name string) error {
	if name == "" {
		return fmt.Errorf("empty name")
	}
	for i := 0; i < len(name); i++ {
		c := name[i]
		if i == 0 {
			if (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_' || c == ':' {
				continue
			}
			return fmt.Errorf("name must start with [A-Za-z_:]")
		}
		if (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == ':' {
			continue
		}
		return fmt.Errorf("name contains invalid byte %q", c)
	}
	return nil
}

func validateLabelName(name string) error {
	if name == "" {
		return fmt.Errorf("empty label name")
	}
	for i := 0; i < len(name); i++ {
		c := name[i]
		if i == 0 {
			if (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_' {
				continue
			}
			return fmt.Errorf("label %q must start with [A-Za-z_]", name)
		}
		if (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' {
			continue
		}
		return fmt.Errorf("label %q contains invalid byte %q", name, c)
	}
	return nil
}

func canonicalOpenMetricLabels(labels []OpenMetricLabel) ([]OpenMetricLabel, string, error) {
	out := append([]OpenMetricLabel(nil), labels...)
	for _, label := range out {
		if err := validateLabelName(label.Name); err != nil {
			return nil, "", err
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Name == out[j].Name {
			return out[i].Value < out[j].Value
		}
		return out[i].Name < out[j].Name
	})
	for i := 1; i < len(out); i++ {
		if out[i].Name == out[i-1].Name {
			return nil, "", fmt.Errorf("duplicate label %q", out[i].Name)
		}
	}
	return out, openMetricLabelKey(out), nil
}

func openMetricLabelKey(labels []OpenMetricLabel) string {
	var key strings.Builder
	for _, label := range labels {
		key.WriteString(label.Name)
		key.WriteByte(0)
		key.WriteString(label.Value)
		key.WriteByte(0)
	}
	return key.String()
}

func writeOpenMetricLabels(out *bytes.Buffer, labels []OpenMetricLabel) {
	if len(labels) == 0 {
		return
	}
	out.WriteByte('{')
	for i, label := range labels {
		if i > 0 {
			out.WriteByte(',')
		}
		out.WriteString(label.Name)
		out.WriteString(`="`)
		out.WriteString(escapeOpenMetricLabelValue(label.Value))
		out.WriteByte('"')
	}
	out.WriteByte('}')
}

func writeOpenMetricBucketLabels(out *bytes.Buffer, labels []OpenMetricLabel, le string) {
	out.WriteByte('{')
	for i, label := range labels {
		if i > 0 {
			out.WriteByte(',')
		}
		out.WriteString(label.Name)
		out.WriteString(`="`)
		out.WriteString(escapeOpenMetricLabelValue(label.Value))
		out.WriteString(`",`)
	}
	out.WriteString(`le="`)
	out.WriteString(escapeOpenMetricLabelValue(le))
	out.WriteString(`"}`)
}

func escapeOpenMetricHelp(s string) string {
	return escapeOpenMetricString(s, false)
}

func escapeOpenMetricLabelValue(s string) string {
	return escapeOpenMetricString(s, true)
}

// escapeOpenMetricString escapes backslash and newline runes for OpenMetrics
// text exposition. When escapeQuote is set the double-quote rune is also
// escaped, as required inside quoted label values; help text leaves it verbatim.
func escapeOpenMetricString(s string, escapeQuote bool) string {
	var out strings.Builder
	for _, r := range s {
		switch {
		case r == '\\':
			out.WriteString(`\\`)
		case r == '\n':
			out.WriteString(`\n`)
		case escapeQuote && r == '"':
			out.WriteString(`\"`)
		default:
			out.WriteRune(r)
		}
	}
	return out.String()
}

func formatOpenMetricFloat(v float64) string {
	return strconv.FormatFloat(v, 'g', -1, 64)
}
