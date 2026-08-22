package main

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

const dispatchProjectNumberEnv = "FAK_DISPATCH_PROJECT_NUMBER"

var dispatchFetchProjectFields = dispatchFetchProjectFieldsGH
var dispatchFetchProjectFieldsContext = dispatchFetchProjectFieldsGHContext

func dispatchFetchProjectFieldsGH(root string) map[int]dispatchtick.ProjectIssueFields {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	return dispatchFetchProjectFieldsGHContext(ctx, root)
}

func dispatchFetchProjectFieldsGHContext(ctx context.Context, root string) map[int]dispatchtick.ProjectIssueFields {
	n, _ := strconv.Atoi(strings.TrimSpace(os.Getenv(dispatchProjectNumberEnv)))
	if n <= 0 {
		return nil
	}
	query := `query($owner:String!,$number:Int!){repositoryOwner(login:$owner){projectV2(number:$number){items(first:100){nodes{content{... on Issue{number}} fieldValues(first:20){nodes{... on ProjectV2ItemFieldSingleSelectValue{name field{... on ProjectV2SingleSelectField{name}}}}}}}}}}`
	owner := "anthony-chaudhary"
	cmd := exec.CommandContext(ctx, "gh", "api", "graphql", "-f", "query="+query, "-F", "owner="+owner, "-F", "number="+strconv.Itoa(n))
	cmd.Dir = root
	configureDispatchSpawn(cmd)
	raw, err := cmd.Output()
	if err != nil {
		return nil
	}
	return parseDispatchProjectFields(raw)
}

func parseDispatchProjectFields(raw []byte) map[int]dispatchtick.ProjectIssueFields {
	var doc struct {
		Data struct {
			RepositoryOwner struct {
				Project struct {
					Items struct {
						Nodes []struct {
							Content struct {
								Number int `json:"number"`
							} `json:"content"`
							FieldValues struct {
								Nodes []struct {
									Name  string `json:"name"`
									Field struct {
										Name string `json:"name"`
									} `json:"field"`
								} `json:"nodes"`
							} `json:"fieldValues"`
						} `json:"nodes"`
					} `json:"items"`
				} `json:"projectV2"`
			} `json:"repositoryOwner"`
		} `json:"data"`
	}
	if json.Unmarshal(raw, &doc) != nil {
		return nil
	}
	out := map[int]dispatchtick.ProjectIssueFields{}
	for _, item := range doc.Data.RepositoryOwner.Project.Items.Nodes {
		if item.Content.Number <= 0 {
			continue
		}
		f := dispatchtick.ProjectIssueFields{Issue: item.Content.Number}
		for _, v := range item.FieldValues.Nodes {
			switch strings.ToLower(strings.TrimSpace(v.Field.Name)) {
			case "priority":
				f.Priority = v.Name
			case "status":
				f.Status = v.Name
			}
		}
		out[f.Issue] = f
	}
	return out
}
