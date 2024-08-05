package ai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/ardanlabs/ai-training/foundation/mongodb"
	"github.com/google/uuid"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

// Board represents connect 4 board information.
type Board struct {
	ID        string    `bson:"board_id"`
	Board     string    `bson:"board"`
	Text      string    `bson:"text"`
	Embedding []float32 `bson:"embedding"`
}

// SimilarBoard represents connect 4 board found in the similarity search.
type SimilarBoard struct {
	ID        string    `bson:"board_id"`
	Board     string    `bson:"board"`
	Text      string    `bson:"text"`
	Embedding []float32 `bson:"embedding"`
	Score     float64   `bson:"score"`
}

// AI provides support to process connect 4 boards.
type AI struct {
	filePath string
	client   *mongo.Client
	col      *mongo.Collection
	embed    *ollama.LLM
	chat     *ollama.LLM
}

// New construct the AI api for use.
func New(client *mongo.Client, embed *ollama.LLM, chat *ollama.LLM) (*AI, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// -------------------------------------------------------------------------
	// Create database and collection

	const dbName = "connect4"
	const collectionName = "boards"

	db := client.Database(dbName)

	col, err := mongodb.CreateCollection(ctx, db, collectionName)
	if err != nil {
		return nil, fmt.Errorf("createCollection: %w", err)
	}

	// -------------------------------------------------------------------------
	// Create indexes

	unique := true
	indexModel := mongo.IndexModel{
		Keys:    bson.D{{Key: "board_id", Value: 1}},
		Options: &options.IndexOptions{Unique: &unique},
	}
	col.Indexes().CreateOne(ctx, indexModel)

	const indexName = "vector_index"
	settings := mongodb.VectorIndexSettings{
		NumDimensions: 1024,
		Path:          "embedding",
		Similarity:    "cosine",
	}

	if err := mongodb.CreateVectorIndex(ctx, col, indexName, settings); err != nil {
		return nil, fmt.Errorf("createVectorIndex: %w", err)
	}

	// -------------------------------------------------------------------------
	// Return the api

	ai := AI{
		filePath: "cmd/connect/training-data/",
		client:   client,
		col:      col,
		embed:    embed,
		chat:     chat,
	}

	return &ai, nil
}

// CalculateEmbedding takes a given board data and produces the vector embedding.
func (ai *AI) CalculateEmbedding(boardData string) ([]float32, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	embData := strings.ReplaceAll(boardData, "🔵", "blue")
	embData = strings.ReplaceAll(embData, "🔴", "red")
	embData = strings.ReplaceAll(embData, "🟢", "green")

	embedding, err := ai.embed.CreateEmbedding(ctx, []string{embData})
	if err != nil {
		return nil, fmt.Errorf("create embedding: %w", err)
	}

	return embedding[0], nil
}

// PickResponse provides the LLM's choice for the next move.
type PickResponse struct {
	Column   int
	Reason   string
	Attmepts int
}

// LLMPick perform a review of the game board and makes a choice.
func (ai *AI) LLMPick(boardData string, board SimilarBoard) (PickResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Second)
	defer cancel()

	// We need the board to look as the LLM expects it to look.
	// It knows about Red and Yellow disks, so Blue will be Yellow.
	// | . | . | Y | Y | . | . | . |
	// | . | . | . | R | . | . | . |
	// | . | . | . | . | . | . | . |

	boardData = strings.ReplaceAll(boardData, "🟢", " . ")
	boardData = strings.ReplaceAll(boardData, "🔵", " Y ")
	boardData = strings.ReplaceAll(boardData, "🔴", " R ")

	rows := strings.Split(boardData, "\n")

	// We have to reverse the board so the rows are flipped.
	var grid string
	for i := 5; i >= 0; i-- {
		grid = fmt.Sprintf("%s%s\n", rows[i], grid)
	}

	// Generate the prompt to use to ask the LLM to pick a column.
	m := ParseBoardText(board)
	redMoves := m["Red-Moves"]
	prompt := fmt.Sprintf(promptPick, redMoves, grid)

	var pick PickResponse

	// The LLM sometimes doesn't pick a column from the list, so we may
	// need to tell the LLM it didn't listen and try again.
	attempts := 1
	for ; attempts <= 2; attempts++ {

		// Ask the LLM to choose a column from the training data.
		response, err := ai.chat.Call(ctx, prompt, llms.WithMaxTokens(5000))
		if err != nil {
			return PickResponse{}, fmt.Errorf("call: %w", err)
		}

		if err := json.Unmarshal([]byte(response), &pick); err != nil {
			return PickResponse{}, fmt.Errorf("unmarshal: %w", err)
		}

		// Did the LLM listen and pick a column from the redMoves list?
		if strings.Contains(redMoves, strconv.Itoa(pick.Column)) {
			break
		}

		// Tell the LLM they didn't listen and try again.
		prompt = fmt.Sprintf(promptPickAgain, prompt, response)
	}

	pick.Attmepts = attempts

	return pick, nil
}

// FindSimilarBoard performs a vector search to find the most similar board.
func (ai *AI) FindSimilarBoard(boardData string) ([]SimilarBoard, error) {
	embedding, err := ai.CalculateEmbedding(boardData)
	if err != nil {
		return nil, err
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// We want to find the nearest neighbors from the question vector embedding.
	pipeline := mongo.Pipeline{
		{{
			Key: "$vectorSearch",
			Value: bson.M{
				"index":       "vector_index",
				"exact":       true,
				"path":        "embedding",
				"queryVector": embedding,
				"limit":       2,
			}},
		},
		{{
			Key: "$project",
			Value: bson.M{
				"board_id":  1,
				"board":     1,
				"text":      1,
				"embedding": 1,
				"score": bson.M{
					"$meta": "vectorSearchScore",
				},
			}},
		},
	}

	// TODO: We only red boards.

	cur, err := ai.col.Aggregate(ctx, pipeline)
	if err != nil {
		return nil, fmt.Errorf("aggregate: %w", err)
	}
	defer cur.Close(ctx)

	var results []SimilarBoard
	if err := cur.All(ctx, &results); err != nil {
		return nil, fmt.Errorf("all: %w", err)
	}

	return results, nil
}

// CreateAIResponse is a blocking call that sends the prompt to the LLM for a
// game remark. This should be called by a Goroutine during game play.
func (ai *AI) CreateAIResponse(board SimilarBoard, currentTurnColor string, lastMove int) (string, error) {
	var prompt string

	m := ParseBoardText(board)

	feedBack := m["Red-Feedback"]
	blueMarkerCount := m["Blue-Markers"]
	redMarkerCounted := m["Red-Markers"]

	var nextTurn = "Red"
	if currentTurnColor == "Red" {
		nextTurn = "Blue"
	}

	switch feedBack {
	case "Normal-GamePlay":
		prompt = fmt.Sprintf(promptNormalGamePlay, blueMarkerCount, redMarkerCounted, nextTurn, currentTurnColor, lastMove)
	case "Will-Win":
		if currentTurnColor == "Red" {
			prompt = fmt.Sprintf(promptRedWonGame, blueMarkerCount, redMarkerCounted, nextTurn, currentTurnColor, lastMove)
		} else {
			prompt = fmt.Sprintf(promptBlueWonGame, blueMarkerCount, redMarkerCounted, nextTurn, currentTurnColor, lastMove)
		}
	case "Won-Game":
		if currentTurnColor == "Red" {
			prompt = fmt.Sprintf(promptRedWonGame, blueMarkerCount, redMarkerCounted, nextTurn, currentTurnColor, lastMove)
		} else {
			prompt = fmt.Sprintf(promptBlueWonGame, blueMarkerCount, redMarkerCounted, nextTurn, currentTurnColor, lastMove)
		}
	case "Lost-Game":
		prompt = fmt.Sprintf(promptLostGame, blueMarkerCount, redMarkerCounted, nextTurn, currentTurnColor, lastMove)
	case "Tie-Game":
		prompt = fmt.Sprintf(promptTieGame, blueMarkerCount, redMarkerCounted, nextTurn, currentTurnColor, lastMove)
	default:
		return "", fmt.Errorf("unknown feedback: %s", feedBack)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Second)
	defer cancel()

	response, err := ai.chat.Call(ctx, prompt, llms.WithMaxTokens(5000))
	if err != nil {
		return "", fmt.Errorf("call: %w", err)
	}

	return response, nil
}

// SaveBoardData knows how to write a board file with the following information.
func (ai *AI) SaveBoardData(boardData string, blue int, red int, gameOver bool, lastWinner string) error {

	// -------------------------------------------------------------------------
	// Check if we have captured this board alread.

	var foundMatch bool

	fsys := os.DirFS(ai.filePath)

	fn := func(fileName string, dirEntry fs.DirEntry, err error) error {
		if foundMatch {
			return errors.New("found match")
		}

		if err != nil {
			return fmt.Errorf("walkdir failure: %w", err)
		}

		file, err := fsys.Open(fileName)
		if err != nil {
			return fmt.Errorf("opening key file: %w", err)
		}
		defer file.Close()

		var board strings.Builder
		var lineCount int

		scanner := bufio.NewScanner(file)
		for scanner.Scan() {
			board.WriteString(scanner.Text())
			board.WriteString("\n")

			lineCount++

			// Capture just the game board.
			if lineCount == 6 {
				break
			}
		}

		if strings.Compare(boardData, board.String()) == 0 {
			foundMatch = true
		}

		return nil
	}

	fs.WalkDir(fsys, ".", fn)

	if foundMatch {
		return nil
	}

	// -------------------------------------------------------------------------
	// Save a copy of this board and extra information.

	fileID := uuid.NewString()

	f, _ := os.Create(ai.filePath + fileID + ".txt")
	defer f.Close()

	template := `%s
-- State
Winner: %s
Turn: %s

-- Blue
Markers: %d
Moves: ()
Feedback: Normal-GamePlay, Blocked-Win, Will-Win, Won-Game, Lost-Game, Tie-Game

-- Red
Markers: %d
Moves: ()
Feedback: Normal-GamePlay, Blocked-Win, Will-Win, Won-Game, Lost-Game, Tie-Game
`

	winner := "None"
	var turn string

	switch gameOver {
	case true:
		winner = lastWinner
	default:
		switch {
		case blue > red:
			turn = "Red"
		case red > blue:
			turn = "Blue"
		case red == blue:
			turn = "Blue or Red"
		}
	}

	_, err := fmt.Fprintf(f, template, boardData, winner, turn, blue, red)
	if err != nil {
		return err
	}

	return nil
}

// ProcessBoardFiles reads the training data directory and saved the AI data needed
// for the AI to play connect 4.
func (ai *AI) ProcessBoardFiles() error {
	files, err := os.ReadDir(ai.filePath)
	if err != nil {
		return fmt.Errorf("read directory: %w", err)
	}

	var count int
	total := len(files)

	fmt.Printf("Found %d documents to process\n", total)

	for _, file := range files {
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()

		fmt.Println("-------------------------------------------")

		count++
		fmt.Printf("Processing file %d of %d\n", count, total)

		boardID := strings.TrimSuffix(file.Name(), ".txt")

		fmt.Printf("Checking if board exists in DB: %s\n", boardID)

		if _, err := ai.findBoard(ctx, boardID); err == nil {
			fmt.Printf("Board EXISTS in DB: %s\n", boardID)
			continue
		}

		fmt.Printf("Creating board: %s\n", boardID)

		board, err := ai.newBoard(boardID)
		if err != nil {
			return fmt.Errorf("new board: %s: %w", boardID, err)
		}

		if !strings.Contains(board.Text, "Turn: Red") && !strings.Contains(board.Text, "Turn: Blue or Red") {
			fmt.Printf("Blue Turn Only Board: %s\n", boardID)
			continue
		}

		fmt.Printf("Create board embedding: %s\n", boardID)

		board, err = ai.createEmbedding(board)
		if err != nil {
			return fmt.Errorf("create embedding: %s: %w", boardID, err)
		}

		fmt.Printf("Saving board data: %s\n", boardID)

		if err := ai.saveBoard(ctx, board); err != nil {
			return fmt.Errorf("saving board: %s: %w", boardID, err)
		}
	}

	return nil
}

// =============================================================================

func (ai *AI) findBoard(ctx context.Context, boardID string) (Board, error) {
	filter := bson.D{{Key: "board_id", Value: boardID}}
	res := ai.col.FindOne(ctx, filter)
	if res.Err() != nil {
		return Board{}, res.Err()
	}

	var b Board
	if err := res.Decode(&b); err != nil {
		return Board{}, fmt.Errorf("decode: %w", err)
	}

	return b, nil
}

func (ai *AI) newBoard(boardID string) (Board, error) {
	fileName := fmt.Sprintf("%s%s.txt", ai.filePath, boardID)

	f, err := os.Open(fileName)
	if err != nil {
		return Board{}, fmt.Errorf("open file: %s: %w", boardID, err)
	}
	defer f.Close()

	data, err := io.ReadAll(f)
	if err != nil {
		return Board{}, fmt.Errorf("read file: %s: %w", boardID, err)
	}

	var board strings.Builder
	var text strings.Builder
	var lineCount int

	scanner := bufio.NewScanner(bytes.NewReader(data))
	for scanner.Scan() {
		lineCount++

		// Capture the game board.
		if lineCount <= 6 {
			board.WriteString(scanner.Text())
			board.WriteString("\n")
			continue
		}

		// Capture the game board content.
		text.WriteString(scanner.Text())
		text.WriteString("\n")
	}

	boardData := board.String()

	b := Board{
		ID:    boardID,
		Board: boardData,
		Text:  text.String(),
	}

	return b, nil
}

func (ai *AI) createEmbedding(board Board) (Board, error) {
	embedding, err := ai.CalculateEmbedding(board.Board)
	if err != nil {
		return Board{}, fmt.Errorf("calculate embedding: %s: %w", board.ID, err)
	}

	board.Embedding = embedding

	return board, nil
}

func (ai *AI) saveBoard(ctx context.Context, board Board) error {
	d := Board{
		ID:        board.ID,
		Board:     board.Board,
		Text:      board.Text,
		Embedding: board.Embedding,
	}

	if _, err := ai.col.InsertOne(ctx, d); err != nil {
		return fmt.Errorf("insert: %w", err)
	}

	return nil
}
