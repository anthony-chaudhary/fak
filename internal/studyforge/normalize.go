package studyforge

import (
	"encoding/json"
	"fmt"
	"sort"
	"time"
)

type rawRecord struct {
	ID              int64           `json:"id"`
	NodeID          string          `json:"node_id"`
	Number          int             `json:"number"`
	Name            string          `json:"name"`
	Title           string          `json:"title"`
	Body            string          `json:"body"`
	State           string          `json:"state"`
	HTMLURL         string          `json:"html_url"`
	TagName         string          `json:"tag_name"`
	TargetCommitish string          `json:"target_commitish"`
	CreatedAt       string          `json:"created_at"`
	UpdatedAt       string          `json:"updated_at"`
	ClosedAt        string          `json:"closed_at"`
	MergedAt        string          `json:"merged_at"`
	PublishedAt     string          `json:"published_at"`
	DueOn           string          `json:"due_on"`
	Draft           bool            `json:"draft"`
	Locked          bool            `json:"locked"`
	Merged          bool            `json:"merged"`
	PullRequest     json.RawMessage `json:"pull_request"`
	User            struct {
		Login string `json:"login"`
	} `json:"user"`
	Labels []struct {
		Name string `json:"name"`
	} `json:"labels"`
	Milestone *struct {
		Number int `json:"number"`
	} `json:"milestone"`
	Category *struct {
		Name string `json:"name"`
	} `json:"category"`
	AnswerChosenBy *struct {
		ID int64 `json:"id"`
	} `json:"answer_chosen_by"`
	Base *struct {
		Ref string `json:"ref"`
		SHA string `json:"sha"`
	} `json:"base"`
	Head *struct {
		Ref string `json:"ref"`
		SHA string `json:"sha"`
	} `json:"head"`
}

func normalize(source string, data json.RawMessage, cutoff time.Time) (Record, bool, bool, error) {
	var raw rawRecord
	if err := json.Unmarshal(data, &raw); err != nil {
		return Record{}, false, false, fmt.Errorf("decode %s record: %w", source, err)
	}
	if raw.ID == 0 {
		return Record{}, false, false, fmt.Errorf("%s record missing id", source)
	}
	if source == "issues" && len(raw.PullRequest) > 0 && string(raw.PullRequest) != "null" {
		return Record{}, true, false, nil
	}
	if raw.CreatedAt != "" {
		if created, err := time.Parse(time.RFC3339, raw.CreatedAt); err != nil {
			return Record{}, false, false, fmt.Errorf("%s record %d invalid created_at", source, raw.ID)
		} else if created.After(cutoff) {
			return Record{}, false, true, nil
		}
	}
	kind := map[string]string{"issues": "issue", "pulls": "pull", "discussions": "discussion", "releases": "release", "labels": "label", "milestones": "milestone"}[source]
	if kind == "" {
		return Record{}, false, false, fmt.Errorf("unsupported source %q", source)
	}
	r := Record{Source: source, Kind: kind, ID: raw.ID, NodeID: raw.NodeID, Number: raw.Number, Name: raw.Name, Title: raw.Title, Body: raw.Body, State: raw.State, URL: raw.HTMLURL, Author: raw.User.Login, Draft: raw.Draft, Locked: raw.Locked, Merged: raw.Merged, TagName: raw.TagName, TargetCommitish: raw.TargetCommitish, CreatedAt: raw.CreatedAt, UpdatedAt: raw.UpdatedAt, ClosedAt: raw.ClosedAt, MergedAt: raw.MergedAt, PublishedAt: raw.PublishedAt, DueOn: raw.DueOn}
	for _, label := range raw.Labels {
		r.Labels = append(r.Labels, label.Name)
	}
	sort.Strings(r.Labels)
	if raw.Milestone != nil {
		r.MilestoneNumber = raw.Milestone.Number
	}
	if raw.Category != nil {
		r.Category = raw.Category.Name
	}
	if raw.AnswerChosenBy != nil {
		r.AnswerID = raw.AnswerChosenBy.ID
	}
	if raw.Base != nil {
		r.BaseRef, r.BaseSHA = raw.Base.Ref, raw.Base.SHA
	}
	if raw.Head != nil {
		r.HeadRef, r.HeadSHA = raw.Head.Ref, raw.Head.SHA
	}
	return r, false, false, nil
}
