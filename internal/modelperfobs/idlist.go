package modelperfobs

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
)

func parseIDList(s string) ([]int, error) {
	if strings.TrimSpace(s) == "" {
		return []int{}, nil
	}
	set := map[int]bool{}
	for _, part := range strings.Split(s, ",") {
		part = strings.TrimSpace(part)
		bounds := strings.Split(part, "-")
		if len(bounds) > 2 {
			return nil, fmt.Errorf("invalid id range %q", part)
		}
		lo, err := strconv.Atoi(bounds[0])
		if err != nil || lo < 0 {
			return nil, fmt.Errorf("invalid id %q", part)
		}
		hi := lo
		if len(bounds) == 2 {
			hi, err = strconv.Atoi(bounds[1])
			if err != nil || hi < lo {
				return nil, fmt.Errorf("invalid id range %q", part)
			}
		}
		for i := lo; i <= hi; i++ {
			set[i] = true
		}
	}
	out := make([]int, 0, len(set))
	for id := range set {
		out = append(out, id)
	}
	sort.Ints(out)
	return out, nil
}
