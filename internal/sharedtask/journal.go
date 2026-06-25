package sharedtask

import "fmt"

const (
	SchemaTaskJournal = "fak.shared-task-journal.v1"
	SchemaJournal     = SchemaTaskJournal
)

type Journal struct {
	Schema  string         `json:"schema"`
	TaskID  string         `json:"task_id"`
	Initial TaskRecord     `json:"initial"`
	Entries []JournalEntry `json:"entries"`
	Digest  string         `json:"digest"`
}

type JournalEntry struct {
	Event        Event      `json:"event"`
	Record       TaskRecord `json:"record"`
	RecordDigest string     `json:"record_digest"`
}

func (s *Store) Journal(taskID string) (Journal, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	initial, ok := s.initial[taskID]
	if !ok {
		return Journal{}, false
	}
	j := Journal{
		Schema:  SchemaTaskJournal,
		TaskID:  taskID,
		Initial: cloneRecord(initial),
	}
	for _, event := range s.events[taskID] {
		record, ok := s.history[event.NextRev]
		if !ok {
			return Journal{}, false
		}
		j.Entries = append(j.Entries, JournalEntry{
			Event:        event,
			Record:       cloneRecord(record),
			RecordDigest: digest(record),
		})
	}
	j.Digest = journalDigest(j)
	return j, true
}

func (j Journal) Verify() error {
	if j.Schema != SchemaTaskJournal {
		return fmt.Errorf("sharedtask: journal schema %q", j.Schema)
	}
	if j.TaskID == "" || j.Initial.TaskID != j.TaskID {
		return fmt.Errorf("sharedtask: journal task mismatch")
	}
	seenRev := map[string]bool{j.Initial.Rev: true}
	prevEvent := ""
	for i, entry := range j.Entries {
		if entry.Event.TaskID != j.TaskID || entry.Record.TaskID != j.TaskID {
			return fmt.Errorf("sharedtask: entry %d task mismatch", i)
		}
		if !seenRev[entry.Event.BaseRev] || entry.Event.NextRev != entry.Record.Rev {
			return fmt.Errorf("sharedtask: entry %d revision chain mismatch", i)
		}
		if entry.Event.PrevEvent != prevEvent {
			return fmt.Errorf("sharedtask: entry %d event chain mismatch", i)
		}
		if entry.RecordDigest != digest(entry.Record) {
			return fmt.Errorf("sharedtask: entry %d record digest mismatch", i)
		}
		seenRev[entry.Record.Rev] = true
		prevEvent = digest(entry.Event)
	}
	if j.Digest != "" && j.Digest != journalDigest(j) {
		return fmt.Errorf("sharedtask: journal digest mismatch")
	}
	return nil
}

func LoadJournal(j Journal, policy Policy) (*Store, error) {
	if err := j.Verify(); err != nil {
		return nil, err
	}
	store := NewStore(policy)
	store.tasks[j.TaskID] = cloneRecord(j.Initial)
	store.initial[j.TaskID] = cloneRecord(j.Initial)
	store.history[j.Initial.Rev] = cloneRecord(j.Initial)
	for _, entry := range j.Entries {
		store.tasks[j.TaskID] = cloneRecord(entry.Record)
		store.history[entry.Record.Rev] = cloneRecord(entry.Record)
		store.events[j.TaskID] = append(store.events[j.TaskID], entry.Event)
		store.seq++
	}
	return store, nil
}

func (j Journal) CurrentRecord() (TaskRecord, bool) {
	if len(j.Entries) == 0 {
		return cloneRecord(j.Initial), j.Initial.TaskID != ""
	}
	return cloneRecord(j.Entries[len(j.Entries)-1].Record), true
}

func (j Journal) ComputeDigest() string {
	return journalDigest(j)
}

func journalDigest(j Journal) string {
	j.Digest = ""
	return digest(j)
}
