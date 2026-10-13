package stepspan

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

const DefaultMaxBytes int64 = 8 << 20
const DefaultMaxFiles = 4
const DefaultCapacity = 1024

var ErrSpanTooLarge = errors.New("step span exceeds JSONL file cap")
var ErrClosed = errors.New("step span sink closed")
var ErrInvalidConfig = errors.New("invalid step span sink configuration")

type JSONLConfig struct {
	Directory string
	Service   string
	MaxBytes  int64
	MaxFiles  int
	Now       func() time.Time
}
type JSONLSink struct {
	mu     sync.Mutex
	config JSONLConfig
	file   *os.File
	path   string
	size   int64
	closed bool
}

func NewJSONLSink(c JSONLConfig) (*JSONLSink, error) {
	if c.Directory == "" {
		c.Directory = filepath.Join(".fak", "spans")
	}
	if c.Service == "" {
		c.Service = "fak"
	}
	if c.MaxBytes == 0 {
		c.MaxBytes = DefaultMaxBytes
	}
	if c.MaxFiles == 0 {
		c.MaxFiles = DefaultMaxFiles
	}
	if c.Now == nil {
		c.Now = time.Now
	}
	if c.MaxBytes < 1 || c.MaxFiles < 1 || !safeService(c.Service) {
		return nil, ErrInvalidConfig
	}
	if err := os.MkdirAll(c.Directory, 0700); err != nil {
		return nil, err
	}
	return &JSONLSink{config: c}, nil
}
func safeService(s string) bool {
	if s == "" || s == "." || s == ".." {
		return false
	}
	for _, c := range s {
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '-' || c == '_' || c == '.') {
			return false
		}
	}
	return true
}
func NewDefaultEmitter(service string) (*Emitter, error) {
	sink, err := NewJSONLSink(JSONLConfig{Service: service})
	if err != nil {
		return nil, err
	}
	e, err := NewEmitter(DefaultCapacity, sink)
	if err != nil {
		_ = sink.Close()
	}
	return e, err
}
func (s *JSONLSink) WriteSpan(span Span) error {
	if err := span.Validate(); err != nil {
		return err
	}
	b, err := json.Marshal(span)
	if err != nil {
		return err
	}
	b = append(b, '\n')
	if int64(len(b)) > s.config.MaxBytes {
		return ErrSpanTooLarge
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return ErrClosed
	}
	path := filepath.Join(s.config.Directory, s.config.Service+"-"+s.config.Now().UTC().Format("2006-01-02")+".jsonl")
	if s.path != path || s.file == nil {
		if err := s.closeFile(); err != nil {
			return err
		}
		if err := s.open(path); err != nil {
			return err
		}
	}
	if s.size+int64(len(b)) > s.config.MaxBytes {
		if err := s.closeFile(); err != nil {
			return err
		}
		if err := s.rotate(path); err != nil {
			return err
		}
		if err := s.open(path); err != nil {
			return err
		}
	}
	if err := s.prune(); err != nil {
		return err
	}
	n, err := s.file.Write(b)
	s.size += int64(n)
	return err
}
func (s *JSONLSink) open(path string) error {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	info, err := f.Stat()
	if err != nil {
		_ = f.Close()
		return err
	}
	s.file = f
	s.path = path
	s.size = info.Size()
	return nil
}
func (s *JSONLSink) closeFile() error {
	if s.file == nil {
		return nil
	}
	err := s.file.Close()
	s.file = nil
	return err
}
func (s *JSONLSink) rotate(path string) error {
	if s.config.MaxFiles == 1 {
		return os.Remove(path)
	}
	for i := 1; ; i++ {
		dst := strings.TrimSuffix(path, ".jsonl") + fmt.Sprintf(".%d.jsonl", i)
		if _, err := os.Lstat(dst); errors.Is(err, os.ErrNotExist) {
			return os.Rename(path, dst)
		} else if err != nil {
			return err
		}
	}
}

// Own only this service's date-named files; similarly prefixed services are peers.
func (s *JSONLSink) owns(name string) bool {
	prefix := s.config.Service + "-"
	if !strings.HasPrefix(name, prefix) || !strings.HasSuffix(name, ".jsonl") {
		return false
	}
	tail := strings.TrimSuffix(strings.TrimPrefix(name, prefix), ".jsonl")
	if len(tail) < 10 {
		return false
	}
	if _, err := time.Parse("2006-01-02", tail[:10]); err != nil {
		return false
	}
	if len(tail) == 10 {
		return true
	}
	if tail[10] != '.' || len(tail) == 11 {
		return false
	}
	for _, c := range tail[11:] {
		if c < '0' || c > '9' {
			return false
		}
	}
	return true
}
func (s *JSONLSink) prune() error {
	entries, err := os.ReadDir(s.config.Directory)
	if err != nil {
		return err
	}
	type oldFile struct {
		path string
		mod  time.Time
	}
	var files []oldFile
	for _, entry := range entries {
		if entry.IsDir() || !s.owns(entry.Name()) {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			return err
		}
		path := filepath.Join(s.config.Directory, entry.Name())
		if path == s.path {
			continue
		}
		files = append(files, oldFile{path, info.ModTime()})
	}
	sort.Slice(files, func(i, j int) bool {
		if files[i].mod.Equal(files[j].mod) {
			return files[i].path < files[j].path
		}
		return files[i].mod.Before(files[j].mod)
	})
	for len(files) >= s.config.MaxFiles {
		if err := os.Remove(files[0].path); err != nil {
			return err
		}
		files = files[1:]
	}
	return nil
}
func (s *JSONLSink) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closed = true
	return s.closeFile()
}
