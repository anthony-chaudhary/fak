// Package blastlease projects live or fixture lease records into blast-radius inputs.
// It is runtime-shared because known-bad reporting and the development-only blast
// estimator consume the same lease-set contract.
package blastlease

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/blastradius"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

// Live reads the live lease ledger and projects records into blast-radius lease inputs.
func Live(root string, now time.Time) ([]blastradius.Lease, error) {
	live, _, err := leaseref.NewInDir(root).Live(context.Background(), now)
	if err != nil {
		return nil, err
	}
	out := make([]blastradius.Lease, 0, len(live))
	for _, record := range live {
		out = append(out, blastradius.Lease{Lane: record.ID, TreeGlobs: record.TreeGlobs})
	}
	return out, nil
}

// Read decodes a JSONL fixture of blast-radius lease inputs.
func Read(path string) ([]blastradius.Lease, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	lines := strings.Split(string(data), "\n")
	out := make([]blastradius.Lease, 0, len(lines))
	for i, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		lease, ok := parseLeaseLine(line)
		if !ok {
			if err := json.Unmarshal([]byte(line), &lease); err != nil {
				return nil, fmt.Errorf("%s line %d: %w", path, i+1, err)
			}
		}
		out = append(out, lease)
	}
	return out, nil
}

func parseLeaseLine(s string) (blastradius.Lease, bool) {
	if len(s) < 2 || s[0] != '{' || s[len(s)-1] != '}' {
		return blastradius.Lease{}, false
	}
	i := 1
	skipWS := func() {
		for i < len(s) && (s[i] == ' ' || s[i] == '\t' || s[i] == '\r' || s[i] == '\n') {
			i++
		}
	}

	parseStr := func() (string, bool) {
		if i >= len(s) || s[i] != '"' {
			return "", false
		}
		i++
		start := i
		for i < len(s) {
			c := s[i]
			if c == '\\' {
				return "", false
			}
			if c == '"' {
				val := s[start:i]
				i++
				return val, true
			}
			if c < 0x20 {
				return "", false
			}
			i++
		}
		return "", false
	}

	var lease blastradius.Lease
	skipWS()
	if i < len(s) && s[i] == '}' {
		i++
		skipWS()
		return lease, i == len(s)
	}

	for i < len(s) {
		skipWS()
		key, ok := parseStr()
		if !ok {
			return blastradius.Lease{}, false
		}
		skipWS()
		if i >= len(s) || s[i] != ':' {
			return blastradius.Lease{}, false
		}
		i++
		skipWS()

		switch key {
		case "lane":
			val, ok := parseStr()
			if !ok {
				return blastradius.Lease{}, false
			}
			lease.Lane = val
		case "tree_globs":
			if i >= len(s) || s[i] != '[' {
				return blastradius.Lease{}, false
			}
			i++
			skipWS()
			if i < len(s) && s[i] == ']' {
				i++
				lease.TreeGlobs = []string{}
			} else {
				commaCount := 0
				bracketDepth := 1
				for j := i; j < len(s); j++ {
					if s[j] == '[' {
						bracketDepth++
					} else if s[j] == ']' {
						bracketDepth--
						if bracketDepth == 0 {
							break
						}
					} else if s[j] == ',' && bracketDepth == 1 {
						commaCount++
					}
				}
				globs := make([]string, 0, commaCount+1)
				for {
					skipWS()
					g, ok := parseStr()
					if !ok {
						return blastradius.Lease{}, false
					}
					globs = append(globs, g)
					skipWS()
					if i < len(s) && s[i] == ',' {
						i++
						continue
					}
					if i < len(s) && s[i] == ']' {
						i++
						break
					}
					return blastradius.Lease{}, false
				}
				lease.TreeGlobs = globs
			}
		default:
			return blastradius.Lease{}, false
		}

		skipWS()
		if i < len(s) && s[i] == ',' {
			i++
			continue
		}
		if i < len(s) && s[i] == '}' {
			i++
			break
		}
		return blastradius.Lease{}, false
	}

	skipWS()
	if i != len(s) {
		return blastradius.Lease{}, false
	}
	return lease, true
}
