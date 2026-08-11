// This file is what step3 adds. Step2 built the edges the book already had,
// these are inferred: the chat model reads every chunk and reports what it is
// about, and those subjects become nodes any chunk can share. NEXT joins two
// chunks adjacent on a page, MENTIONS joins two that discuss the same thing four
// hundred pages apart. The price is one model call per chunk.

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/ardanlabs/ai-training/foundation/age"
	"github.com/ardanlabs/kronk/sdk/kronk"
	"github.com/ardanlabs/kronk/sdk/kronk/model"
)

// extractPrompt asks for JSON only. The parser below copes rather than trusts.
const extractPrompt = `Extract the technical subjects of the text below and the relationships between them.

Rules:
- Only list subjects the text is actually about. Do not list every noun.
- Name each subject with the shortest natural term, lowercase and singular.
- kind must be one of: concept, type, function, package, tool.
- Only list a relationship the text actually states.
- At most 8 subjects and 8 relationships.
- Reply with JSON and nothing else, in exactly this shape:

{"entities":[{"name":"goroutine","kind":"concept"}],"relationships":[{"from":"scheduler","to":"goroutine","kind":"schedules"}]}

Text:

%s`

// extraction is the JSON the model is asked to produce.
type extraction struct {
	Entities []struct {
		Name string `json:"name"`
		Kind string `json:"kind"`
	} `json:"entities"`
	Relationships []struct {
		From string `json:"from"`
		To   string `json:"to"`
		Kind string `json:"kind"`
	} `json:"relationships"`
}

// =============================================================================

// buildEntities runs the chat model over every chunk and writes what it finds
// into the graph. This is the slow part of ingest and it only happens once.
func buildEntities(ctx context.Context, g *age.Graph, krnChat *kronk.Kronk, chunks []chunk) error {
	fmt.Print("\nExtracting entities, one model call per chunk. This takes a while.\n")
	fmt.Print("\033[s")

	t := time.Now()

	var entityCount, relCount, failed int

	for i, c := range chunks {
		fmt.Print("\033[u\033[K")
		fmt.Printf("Extracting: %d of %d (entities %d, relationships %d, unparsable %d)",
			i+1, len(chunks), entityCount, relCount, failed)

		ext, err := extract(ctx, krnChat, c.text)
		if err != nil {
			// One bad chunk should not throw away an hour of extraction.
			failed++
			continue
		}

		for _, e := range ext.Entities {
			name := normalize(e.Name)
			if name == "" {
				continue
			}

			kind := normalize(e.Kind)
			if !validKinds[kind] {
				kind = "concept"
			}

			q := fmt.Sprintf(`MERGE (:Entity {name: %s, kind: %s})`,
				age.Quote(name), age.Quote(kind))

			if err := g.Exec(ctx, q); err != nil {
				return fmt.Errorf("merge entity %q: %w", name, err)
			}

			q = fmt.Sprintf(`
				MATCH (c:Chunk {id: %d}), (e:Entity {name: %s})
				MERGE (c)-[:MENTIONS]->(e)`,
				c.id, age.Quote(name))

			if err := g.Exec(ctx, q); err != nil {
				return fmt.Errorf("mention edge %d %q: %w", c.id, name, err)
			}

			entityCount++
		}

		// ---------------------------------------------------------------------
		// Only written when both ends exist. The model relates what it never listed.

		for _, r := range ext.Relationships {
			from, to := normalize(r.From), normalize(r.To)
			if from == "" || to == "" || from == to {
				continue
			}

			q := fmt.Sprintf(`
				MATCH (a:Entity {name: %s}), (b:Entity {name: %s})
				MERGE (a)-[:RELATED {kind: %s}]->(b)`,
				age.Quote(from), age.Quote(to), age.Quote(normalize(r.Kind)))

			if err := g.Exec(ctx, q); err != nil {
				return fmt.Errorf("related edge %q -> %q: %w", from, to, err)
			}

			relCount++
		}
	}

	fmt.Printf("\nExtracted %d mentions and %d relationships in %v (%d chunks unparsable)\n",
		entityCount, relCount, time.Since(t), failed)

	return nil
}

// extract asks the model about one chunk and parses the answer.
func extract(ctx context.Context, krn *kronk.Kronk, text string) (extraction, error) {
	ctx, cancel := context.WithTimeout(ctx, 2*time.Minute)
	defer cancel()

	d := model.D{
		"messages": []model.D{
			model.TextMessage("user", fmt.Sprintf(extractPrompt, text)),
		},
		// Thinking off. This model reasons for over a thousand tokens, running past
		// max_tokens before the JSON starts, so every chunk comes back unparsable
		// and costs the full budget to fail.
		"enable_thinking": false,

		"max_tokens":  2048,
		"temperature": 0.1,
		"top_p":       0.1,
	}

	resp, err := krn.Chat(ctx, d)
	if err != nil {
		return extraction{}, fmt.Errorf("chat: %w", err)
	}

	if len(resp.Choices) == 0 || resp.Choices[0].Message == nil {
		return extraction{}, fmt.Errorf("empty response")
	}

	return parseExtraction(resp.Choices[0].Message.Content)
}

// jsonRE finds the outermost JSON object. Reasoning models think out loud and
// fence their code, so the object is located, not assumed to be the message.
var jsonRE = regexp.MustCompile(`(?s)\{.*\}`)

func parseExtraction(content string) (extraction, error) {
	match := jsonRE.FindString(content)
	if match == "" {
		return extraction{}, fmt.Errorf("no json object in reply")
	}

	var ext extraction
	if err := json.Unmarshal([]byte(match), &ext); err != nil {
		return extraction{}, fmt.Errorf("unmarshal: %w", err)
	}

	return ext, nil
}

var validKinds = map[string]bool{
	"concept":  true,
	"type":     true,
	"function": true,
	"package":  true,
	"tool":     true,
}

// nameRE keeps entity names to characters that are safe in a Cypher literal.
var nameRE = regexp.MustCompile(`[^a-z0-9 ._/-]+`)

// normalize turns whatever the model returned into a usable node name. Without
// it "Goroutines", "goroutine" and "a goroutine" become three unconnected nodes.
func normalize(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	s = nameRE.ReplaceAllString(s, "")
	s = strings.Join(strings.Fields(s), " ")

	s = strings.TrimPrefix(s, "the ")
	s = strings.TrimPrefix(s, "a ")
	s = strings.TrimPrefix(s, "an ")

	if len(s) < 2 || len(s) > 40 {
		return ""
	}

	return s
}

// =============================================================================

// entityHop walks from the chunks the vector search found out to the subjects
// they mention and back down to other chunks mentioning the same subject. The
// map says which subject carried each chunk in.
func entityHop(ctx context.Context, g *age.Graph, ids []int, perEntity int) (map[int]string, error) {
	q := fmt.Sprintf(`
		MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(n:Chunk)
		WHERE c.id IN [%s]
		RETURN n.id, e.name`, age.Ints(ids))

	rows, err := g.Query(ctx, q, "id agtype", "name agtype")
	if err != nil {
		return nil, err
	}

	// perEntity keeps one popular subject from filling the whole context. "type"
	// and "value" are mentioned nearly everywhere in this book.
	counts := make(map[string]int)
	out := make(map[int]string)

	for _, r := range rows {
		id, err := age.ParseID(r[0])
		if err != nil {
			return nil, err
		}

		if _, ok := out[id]; ok {
			continue
		}

		name := age.Scalar(r[1])
		if counts[name] >= perEntity {
			continue
		}

		counts[name]++
		out[id] = name
	}

	return out, nil
}
