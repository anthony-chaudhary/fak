// Package studyforge captures reproducible, paginated GitHub repository corpora.
package studyforge

import (
	"context"
	"net/http"
	"time"
)

const (
	// CorpusSchema identifies the normalized index envelope.
	CorpusSchema = "fak-studyforge-corpus/1"
	// ReceiptSchema identifies capture receipts consumed by study inventory tooling.
	ReceiptSchema = "fak-studyforge-receipt/1"
)

const (
	StatusComplete = "complete"
	StatusPartial  = "partial"
	StatusFailed   = "failed"
)

// SourceNames is the stable collection and receipt order.
var SourceNames = []string{"issues", "pulls", "discussions", "releases", "labels", "milestones"}

// CaptureRequest declares one repository snapshot boundary. A partial prior corpus may
// be supplied to resume at its first unfetched page without duplicating records.
// Checkpoint receives canonical snapshots after each accepted page by default;
// CheckpointEvery may raise that bounded interval to reduce write amplification.
type CaptureRequest struct {
	Owner           string
	Repository      string
	Cutoff          time.Time
	Resume          *Corpus
	Checkpoint      func(Corpus) error
	CheckpointEvery int
}

// Collector configures GitHub REST collection. BaseURL defaults to api.github.com.
type Collector struct {
	Client     *http.Client
	BaseURL    string
	MaxRetries int
	Now        func() time.Time
	RetryWait  func(context.Context, time.Duration) error
}

// Corpus is the reviewable normalized index plus its proof receipt.
type Corpus struct {
	Schema  string   `json:"schema"`
	Receipt Receipt  `json:"receipt"`
	Records []Record `json:"records"`
}

// Receipt proves the revision, cutoff, API calls, completeness, and index digest.
type Receipt struct {
	Schema        string          `json:"schema"`
	Repository    string          `json:"repository"`
	Revision      string          `json:"revision"`
	Cutoff        string          `json:"cutoff"`
	APIBase       string          `json:"api_base"`
	StartedAt     string          `json:"started_at"`
	CompletedAt   string          `json:"completed_at,omitempty"`
	Status        string          `json:"status"`
	Sources       []SourceReceipt `json:"sources"`
	IndexChecksum string          `json:"index_checksum"`
	API           []APIReceipt    `json:"api,omitempty"`
}

// SourceReceipt records the complete page chain and count reconciliation for a source.
type SourceReceipt struct {
	Name                string        `json:"name"`
	Endpoint            string        `json:"endpoint"`
	Status              string        `json:"status"`
	Pages               []PageReceipt `json:"pages"`
	FetchedCount        int           `json:"fetched_count"`
	NormalizedCount     int           `json:"normalized_count"`
	UniqueCount         int           `json:"unique_count"`
	ClassifiedPullCount int           `json:"classified_pull_count,omitempty"`
	CutoffExcludedCount int           `json:"cutoff_excluded_count,omitempty"`
	PageChecksum        string        `json:"page_checksum"`
	Checksum            string        `json:"checksum"`
	Failure             string        `json:"failure,omitempty"`
}

// PageReceipt binds a page body digest to its pagination and API evidence.
type PageReceipt struct {
	Number     int    `json:"number"`
	URL        string `json:"url"`
	ItemCount  int    `json:"item_count"`
	Checksum   string `json:"checksum"`
	Next       string `json:"next,omitempty"`
	FetchedAt  string `json:"fetched_at"`
	StatusCode int    `json:"status_code"`
	RequestID  string `json:"request_id,omitempty"`
	ETag       string `json:"etag,omitempty"`
	RateLimit  int    `json:"rate_limit,omitempty"`
	RateRemain int    `json:"rate_remaining,omitempty"`
	RateReset  int64  `json:"rate_reset,omitempty"`
}

// APIReceipt records non-page REST calls used to resolve repository revision.
type APIReceipt struct {
	Purpose    string `json:"purpose"`
	URL        string `json:"url"`
	FetchedAt  string `json:"fetched_at"`
	StatusCode int    `json:"status_code"`
	Checksum   string `json:"checksum"`
	RequestID  string `json:"request_id,omitempty"`
	ETag       string `json:"etag,omitempty"`
	RateLimit  int    `json:"rate_limit,omitempty"`
	RateRemain int    `json:"rate_remaining,omitempty"`
	RateReset  int64  `json:"rate_reset,omitempty"`
}

// Record is a compact normalized top-level GitHub object. Source and Kind are
// deliberately both present so mixed issue/PR rows cannot become ambiguous.
type Record struct {
	Source          string   `json:"source"`
	Kind            string   `json:"kind"`
	ID              int64    `json:"id"`
	NodeID          string   `json:"node_id,omitempty"`
	Number          int      `json:"number,omitempty"`
	Name            string   `json:"name,omitempty"`
	Title           string   `json:"title,omitempty"`
	Body            string   `json:"body,omitempty"`
	State           string   `json:"state,omitempty"`
	URL             string   `json:"url,omitempty"`
	Author          string   `json:"author,omitempty"`
	Labels          []string `json:"labels,omitempty"`
	MilestoneNumber int      `json:"milestone_number,omitempty"`
	Category        string   `json:"category,omitempty"`
	AnswerID        int64    `json:"answer_id,omitempty"`
	Draft           bool     `json:"draft,omitempty"`
	Locked          bool     `json:"locked,omitempty"`
	Merged          bool     `json:"merged,omitempty"`
	BaseRef         string   `json:"base_ref,omitempty"`
	BaseSHA         string   `json:"base_sha,omitempty"`
	HeadRef         string   `json:"head_ref,omitempty"`
	HeadSHA         string   `json:"head_sha,omitempty"`
	TagName         string   `json:"tag_name,omitempty"`
	TargetCommitish string   `json:"target_commitish,omitempty"`
	CreatedAt       string   `json:"created_at,omitempty"`
	UpdatedAt       string   `json:"updated_at,omitempty"`
	ClosedAt        string   `json:"closed_at,omitempty"`
	MergedAt        string   `json:"merged_at,omitempty"`
	PublishedAt     string   `json:"published_at,omitempty"`
	DueOn           string   `json:"due_on,omitempty"`
}
