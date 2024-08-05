package game

import (
	"crypto/rand"
	"fmt"
	"math/big"
	"strings"

	"github.com/ardanlabs/ai-training/cmd/connect/ai"
)

const (
	rows = 6
	cols = 7
)

const (
	colorBlue = "Blue"
	colorRed  = "Red"
)

type Cell struct {
	HasPiece bool
	Color    string
}

type LastMove struct {
	Column int
	Row    int
	Color  string
}

// Board represents the game board and all its state.
type Board struct {
	ai          *ai.AI
	cells       [cols][rows]Cell
	lastMove    LastMove
	aiMessage   string
	gameMessage string
	gameOver    bool
	winner      string
}

// New contructs a game board and renders the board.
func New(ai *ai.AI) (*Board, error) {
	currentTurn := colorBlue
	nBig, err := rand.Int(rand.Reader, big.NewInt(100))
	if err != nil {
		return nil, fmt.Errorf("random number: %w", err)
	}

	if n := nBig.Int64(); n%2 == 0 {
		currentTurn = colorRed
	}

	board := Board{
		ai: ai,
		lastMove: LastMove{
			Column: 4,
			Row:    0,
			Color:  currentTurn,
		},
	}

	return &board, nil
}

// AITurn plays for the AI.
func (b *Board) AITurn() BoardState {
	b.gameMessage = ""
	b.aiMessage = ""

	if b.gameOver {
		b.gameMessage = "game is over"
		return b.ToBoardState()
	}

	// -------------------------------------------------------------------------
	// Check if we have a new game board

	boardData, blue, red := b.BoardData()
	b.ai.SaveBoardData(boardData, blue, red, b.gameOver, b.winner)

	defer func() {
		boardData, blue, red := b.BoardData()
		b.ai.SaveBoardData(boardData, blue, red, b.gameOver, b.winner)
	}()

	// -------------------------------------------------------------------------
	// Find a similar boards from the training data

	boards, err := b.ai.FindSimilarBoard(boardData)
	if err != nil {
		b.gameMessage = err.Error()
		return b.ToBoardState()
	}

	board := boards[0]

	// -------------------------------------------------------------------------
	// Use the LLM to Pick

	pick, err := b.ai.LLMPick(boardData, board)
	if err != nil {
		b.gameMessage = err.Error()
		return b.ToBoardState()
	}

	choice := -1

	// Does that column have an open space?
	if !b.cells[pick.Column-1][0].HasPiece {
		choice = pick.Column
	}

	// If we didn't find a valid column, find an open one.
	if choice == -1 {
		for i := range 6 {
			if !b.cells[i][0].HasPiece {
				choice = i + 1
				break
			}
		}
	}

	// Calculate what row (6 - 1) to drop the marker in.
	row := -1
	for i := rows - 1; i >= 0; i-- {
		cell := b.cells[choice-1][i]
		if !cell.HasPiece {
			row = i
			break
		}
	}

	if row == -1 {
		b.gameMessage = fmt.Sprintf("column is full: %d", choice)
		return b.ToBoardState()
	}

	// Set this piece in the cells.
	b.cells[choice-1][row].HasPiece = true
	b.cells[choice-1][row].Color = colorRed

	// Mark this last move.
	b.lastMove.Color = colorRed
	b.lastMove.Column = choice
	b.lastMove.Row = row + 1

	// Provide final state for display.
	m := ai.ParseBoardText(board)
	b.aiMessage = fmt.Sprintf("BOARD: %s CRLF CHOICE: %d - OPTIONS: (%s) - ATTEMPTS: %d CRLF SCORE: %.2f%% CRLF %s", board.ID, choice, m["Red-Moves"], pick.Attmepts, board.Score*100, pick.Reason)

	// Check if this move allowed the AI player to win the game.
	b.checkForWinner(choice, row+1)

	return b.ToBoardState()
}

// UserTurn plays the user's choice.
func (b *Board) UserTurn(column int) BoardState {
	b.gameMessage = ""
	b.aiMessage = ""

	if b.gameOver {
		b.gameMessage = "game is over"
		return b.ToBoardState()
	}

	// -------------------------------------------------------------------------
	// Check if we have a new game board

	boardData, blue, red := b.BoardData()
	b.ai.SaveBoardData(boardData, blue, red, b.gameOver, b.winner)

	defer func() {
		boardData, blue, red := b.BoardData()
		b.ai.SaveBoardData(boardData, blue, red, b.gameOver, b.winner)
	}()

	// -------------------------------------------------------------------------
	// Apply the user's column choice

	column--

	// Calculate what row (6 - 1) to drop the marker in.
	row := -1
	for i := rows - 1; i >= 0; i-- {
		cell := b.cells[column][i]
		if !cell.HasPiece {
			row = i
			break
		}
	}

	if row == -1 {
		b.gameMessage = fmt.Sprintf("column is full: %d", column)
		return b.ToBoardState()
	}

	// Set this piece in the cells.
	b.cells[column][row].HasPiece = true
	b.cells[column][row].Color = colorBlue

	// Mark this last move.
	b.lastMove.Color = colorBlue
	b.lastMove.Column = column + 1
	b.lastMove.Row = row + 1

	// Check if this move allowed the player to win the game.
	b.checkForWinner(column+1, row+1)

	return b.ToBoardState()
}

// CreateAIMessage produces an AI message for the opponent.
func (b *Board) CreateAIMessage(column int, currentTurn string, board ai.SimilarBoard) {
	b.gameMessage = ""
	b.aiMessage = ""

	boardData, _, _ := b.BoardData()

	boards, err := b.ai.FindSimilarBoard(boardData)
	if err != nil {
		b.gameMessage = err.Error()
		return
	}

	response, err := b.ai.CreateAIResponse(boards[0], currentTurn, column)
	if err != nil {
		b.gameMessage = err.Error()
		return
	}

	b.aiMessage = response
}

// BoardData converts the game board into a text representation.
func (b *Board) BoardData() (boardData string, blue int, red int) {
	var data strings.Builder

	for row := range rows {
		data.WriteString("|")
		for col := range cols {
			cell := b.cells[col][row]
			switch {
			case !cell.HasPiece:
				data.WriteString("🟢|")
			default:
				switch cell.Color {
				case colorBlue:
					data.WriteString("🔵|")
					blue++
				case colorRed:
					data.WriteString("🔴|")
					red++
				}
			}
		}
		data.WriteString("\n")
	}

	return data.String(), blue, red
}

// =============================================================================

func (b *Board) checkForWinner(colInput int, rowInput int) bool {
	defer func() {
		if b.winner != "" {
			b.gameMessage = "there is a winner"
			b.gameOver = true
		}
	}()

	colInput--
	rowInput--

	// -------------------------------------------------------------------------
	// Is there a winner in the specified row.

	var red int
	var blue int

	for col := 0; col < cols; col++ {
		if !b.cells[col][rowInput].HasPiece {
			red = 0
			blue = 0
			continue
		}

		switch b.cells[col][rowInput].Color {
		case colorBlue:
			blue++
			red = 0
		case colorRed:
			red++
			blue = 0
		}

		switch {
		case red == 4:
			b.winner = colorRed
			return true
		case blue == 4:
			b.winner = colorBlue
			return true
		}
	}

	// -------------------------------------------------------------------------
	// Is there a winner in the specified column.

	red = 0
	blue = 0

	for row := 0; row < rows; row++ {
		if !b.cells[colInput][row].HasPiece {
			red = 0
			blue = 0
			continue
		}

		switch b.cells[colInput][row].Color {
		case colorBlue:
			blue++
			red = 0
		case colorRed:
			red++
			blue = 0
		}

		switch {
		case red == 4:
			b.winner = colorRed
			return true
		case blue == 4:
			b.winner = colorBlue
			return true
		}
	}

	// -------------------------------------------------------------------------
	// Is there a winner in the NW to SE line.

	red = 0
	blue = 0

	// Walk up in a diagonal until we hit column 0.
	useRow := rowInput
	useCol := colInput
	for useCol != 0 && useRow != 0 {
		useRow--
		useCol--
	}

	for useCol != cols && useRow != rows {
		if !b.cells[useCol][useRow].HasPiece {
			useCol++
			useRow++
			red = 0
			blue = 0
			continue
		}

		switch b.cells[useCol][useRow].Color {
		case colorBlue:
			blue++
			red = 0
		case colorRed:
			red++
			blue = 0
		}

		switch {
		case red == 4:
			b.winner = colorRed
			return true
		case blue == 4:
			b.winner = colorBlue
			return true
		}

		useCol++
		useRow++
	}

	// -------------------------------------------------------------------------
	// Is there a winner in the SW to NE line.

	red = 0
	blue = 0

	// Walk up in a diagonal until we hit column 0.
	useRow = rowInput
	useCol = colInput
	for useCol != cols-1 && useRow != 0 {
		useRow--
		useCol++
	}

	for useCol >= 0 && useRow != rows {
		if !b.cells[useCol][useRow].HasPiece {
			useCol--
			useRow++
			red = 0
			blue = 0
			continue
		}

		switch b.cells[useCol][useRow].Color {
		case colorBlue:
			blue++
			red = 0
		case colorRed:
			red++
			blue = 0
		}

		switch {
		case red == 4:
			b.winner = colorRed
			return true
		case blue == 4:
			b.winner = colorBlue
			return true
		}

		useCol--
		useRow++
	}

	// No winner, but is there a tie?
	tie := true
stop:
	for col := range b.cells {
		for _, cell := range b.cells[col] {
			if !cell.HasPiece {
				tie = false
				break stop
			}
		}
	}

	if tie {
		b.winner = "Tie Game"
		return true
	}

	return false
}
