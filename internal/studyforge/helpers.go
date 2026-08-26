package studyforge

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"
)

func linkNext(link string) string {
	for _, p := range strings.Split(link, ",") {
		bits := strings.Split(p, ";")
		if len(bits) < 2 {
			continue
		}
		for _, a := range bits[1:] {
			if strings.TrimSpace(a) == `rel="next"` {
				return strings.Trim(strings.TrimSpace(bits[0]), "<>")
			}
		}
	}
	return ""
}
func fillAPIHeaders(requestID, etag *string, limit, remain *int, reset *int64, h http.Header) {
	*requestID = h.Get("X-GitHub-Request-Id")
	*etag = h.Get("ETag")
	*limit, _ = strconv.Atoi(h.Get("X-RateLimit-Limit"))
	*remain, _ = strconv.Atoi(h.Get("X-RateLimit-Remaining"))
	*reset, _ = strconv.ParseInt(h.Get("X-RateLimit-Reset"), 10, 64)
}
func apiReceipt(purpose, u string, b []byte, h http.Header, status int, now time.Time) APIReceipt {
	r := APIReceipt{Purpose: purpose, URL: u, FetchedAt: now.UTC().Format(time.RFC3339Nano), StatusCode: status, Checksum: digest(b)}
	fillAPIHeaders(&r.RequestID, &r.ETag, &r.RateLimit, &r.RateRemain, &r.RateReset, h)
	return r
}
func sourceByName(ss []SourceReceipt, n string) (SourceReceipt, bool) {
	for _, s := range ss {
		if s.Name == n {
			return s, true
		}
	}
	return SourceReceipt{}, false
}
func upsertSource(ss *[]SourceReceipt, s SourceReceipt) {
	for i := range *ss {
		if (*ss)[i].Name == s.Name {
			(*ss)[i] = s
			return
		}
	}
	*ss = append(*ss, s)
}
func recordsForSource(rs []Record, n string) []Record {
	var o []Record
	for _, r := range rs {
		if r.Source == n {
			o = append(o, r)
		}
	}
	return o
}
func recordsExceptSource(rs []Record, n string) []Record {
	var o []Record
	for _, r := range rs {
		if r.Source != n {
			o = append(o, r)
		}
	}
	return o
}
func completeSourceCount(ss []SourceReceipt) int {
	n := 0
	for _, s := range ss {
		if s.Status == StatusComplete {
			n++
		}
	}
	return n
}
func sourceRank(s string) int {
	for i, n := range SourceNames {
		if s == n {
			return i
		}
	}
	return len(SourceNames)
}
func sortCorpus(c *Corpus) {
	sort.Slice(c.Records, func(i, j int) bool {
		a, b := c.Records[i], c.Records[j]
		if sourceRank(a.Source) != sourceRank(b.Source) {
			return sourceRank(a.Source) < sourceRank(b.Source)
		}
		if a.ID != b.ID {
			return a.ID < b.ID
		}
		if a.Number != b.Number {
			return a.Number < b.Number
		}
		return a.Name < b.Name
	})
	sort.Slice(c.Receipt.Sources, func(i, j int) bool {
		return sourceRank(c.Receipt.Sources[i].Name) < sourceRank(c.Receipt.Sources[j].Name)
	})
}
func recordDigest(rs []Record) string {
	b, _ := json.Marshal(rs)
	s := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(s[:])
}
func pageDigest(ps []PageReceipt) string {
	b, _ := json.Marshal(ps)
	s := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(s[:])
}
func refreshChecksums(c *Corpus) {
	for i := range c.Receipt.Sources {
		rs := recordsForSource(c.Records, c.Receipt.Sources[i].Name)
		c.Receipt.Sources[i].NormalizedCount = len(rs)
		seen := map[int64]bool{}
		for _, r := range rs {
			seen[r.ID] = true
		}
		c.Receipt.Sources[i].UniqueCount = len(seen)
		c.Receipt.Sources[i].PageChecksum = pageDigest(c.Receipt.Sources[i].Pages)
		c.Receipt.Sources[i].Checksum = recordDigest(rs)
	}
	c.Receipt.IndexChecksum = recordDigest(c.Records)
}
func cloneCorpus(c Corpus) Corpus {
	b, _ := json.Marshal(c)
	var out Corpus
	_ = json.Unmarshal(b, &out)
	return out
}
