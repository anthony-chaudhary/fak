package deepseekv4moe

import "container/list"

type adaptiveSLRUEntry struct {
	group     ExpertGroup
	protected bool
}

type adaptiveSLRUState struct {
	capacity  int
	requests  int
	resident  map[ExpertGroup]*list.Element
	probation list.List
	protected list.List
}

func adaptiveSLRUProtectedLimit(capacity, requests int) int {
	limit := capacity / 10
	// Division preserves requests > 4*capacity without overflowing at large capacities.
	if requests > 0 && (requests-1)/4 >= capacity {
		limit = (capacity/5)*4 + (capacity%5)*4/5
	}
	if limit < 1 {
		return 1
	}
	return limit
}

func (s *adaptiveSLRUState) access(group ExpertGroup) (hit, evicted bool) {
	s.requests++
	if element, ok := s.resident[group]; ok {
		entry := element.Value.(adaptiveSLRUEntry)
		if entry.protected {
			s.protected.MoveToBack(element)
		} else {
			s.probation.Remove(element)
			entry.protected = true
			s.resident[group] = s.protected.PushBack(entry)
			if s.protected.Len() > adaptiveSLRUProtectedLimit(s.capacity, s.requests) {
				oldest := s.protected.Front()
				demoted := oldest.Value.(adaptiveSLRUEntry)
				s.protected.Remove(oldest)
				demoted.protected = false
				s.resident[demoted.group] = s.probation.PushBack(demoted)
			}
		}
		return true, false
	}
	if len(s.resident) == s.capacity {
		segment := &s.probation
		if segment.Len() == 0 {
			segment = &s.protected
		}
		oldest := segment.Front()
		delete(s.resident, oldest.Value.(adaptiveSLRUEntry).group)
		segment.Remove(oldest)
		evicted = true
	}
	s.resident[group] = s.probation.PushBack(adaptiveSLRUEntry{group: group})
	return false, evicted
}

func replayAdaptiveSLRU(groups []ExpertGroup, capacity int) (hits, misses, evictions int) {
	state := adaptiveSLRUState{capacity: capacity, resident: make(map[ExpertGroup]*list.Element)}
	for _, group := range groups {
		hit, evicted := state.access(group)
		if hit {
			hits++
		} else {
			misses++
		}
		if evicted {
			evictions++
		}
	}
	return hits, misses, evictions
}
