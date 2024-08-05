// Package board handles the game board and all interactions.
package board

import (
	"bufio"
	"bytes"
	"fmt"
	"time"

	"github.com/ardanlabs/ai-training/cmd/connect/ai"
	"github.com/ardanlabs/ai-training/cmd/connect/game"
	"github.com/gdamore/tcell/v2"
	"github.com/mattn/go-runewidth"
)

const (
	rows        = 6
	cols        = 7
	cellWidth   = 5
	cellHeight  = 2
	boardWidth  = cols*cellWidth + 1
	boardHeight = rows * cellHeight
	padTop      = 4
	padLeft     = 1
)

const (
	hozTopRune = '━'
	hozBotRune = '▅'
	verRune    = '┃'
	space      = 32
)

const (
	colorBlue = "Blue"
	colorRed  = "Red"
)

const (
	dirLeft  = "left"
	dirRight = "right"
)

type cell struct {
	hasPiece bool
	color    string
}

// Board represents the game board and all its state.
type Board struct {
	ai        *ai.AI
	gameBoard *game.Board
	screen    tcell.Screen
	style     tcell.Style
	inputCol  int
}

// New contructs a game board and renders the board.
func New(ai *ai.AI) (*Board, error) {
	tcell.SetEncodingFallback(tcell.EncodingFallbackASCII)

	screen, err := tcell.NewScreen()
	if err != nil {
		return nil, fmt.Errorf("new screen: %w", err)
	}

	if err := screen.Init(); err != nil {
		return nil, fmt.Errorf("screen init: %w", err)
	}

	style := tcell.StyleDefault
	style = style.Background(tcell.ColorBlack).Foreground(tcell.ColorWhite)

	gameBoard, err := game.New(ai)
	if err != nil {
		return nil, fmt.Errorf("random number: %w", err)
	}

	board := Board{
		ai:        ai,
		gameBoard: gameBoard,
		screen:    screen,
		style:     style,
		inputCol:  4,
	}

	board.drawInit()

	return &board, nil
}

// Shutdown tears down the game board.
func (b *Board) Shutdown() {
	b.screen.Fini()
}

// Run starts a goroutine to handle terminal events. This is a
// blocking call.
func (b *Board) Run() chan struct{} {
	return b.pollEvents()
}

func (b *Board) newGame() {
	gameBoard, _ := game.New(b.ai)

	*b = Board{
		ai:        b.ai,
		gameBoard: gameBoard,
		screen:    b.screen,
		style:     b.style,
		inputCol:  4,
	}

	b.drawInit()
}

func (b *Board) drawInit() {
	b.drawEmptyGameBoard()
	b.appyBoardState()
}

func (b *Board) drawEmptyGameBoard() {
	b.screen.Clear()

	width := boardWidth
	height := boardHeight

	style := b.style
	style = style.Background(tcell.ColorBlack).Foreground(tcell.ColorGrey)

	for h := 0; h <= height; h++ {
		for w := 0; w < width; w++ {

			// Clear the entire line.
			b.screen.SetContent(w+padLeft, h+padTop, space, nil, style)

			if h == 0 || h%cellHeight == 0 {

				// These are the '━' characters creating each row.
				b.screen.SetContent(w+padLeft, h+padTop, hozTopRune, nil, style)

				if h == height {

					// These are the '▅' characters creating the bottom row.
					b.screen.SetContent(w+padLeft, h+padTop, hozBotRune, nil, style)
				}
			}

			if w == 0 || w%cellWidth == 0 {

				// These are the '┃' characters creating each column.
				b.screen.SetContent(w+padLeft, h+padTop, verRune, nil, style)
			}
		}
	}

	b.print(10, 1, "Connect 4 AI Version")
	b.print(0, boardHeight+padTop+1, "   ①    ②    ③    ④    ⑤    ⑥    ⑦")

	b.print(boardWidth+3, padTop-1, "<n> new game   <q> quit game   ")

	screenWidth, _ := b.screen.Size()

	b.drawBox(boardWidth+3, padTop+3, boardWidth+(screenWidth-boardWidth-2), padTop+3+10)
	b.print(boardWidth+4, padTop+3, " AI PLAYER ")
}

func (b *Board) appyBoardState() {
	boardState := b.gameBoard.ToBoardState()

	// Just drop the pieces again, but without animation.
	for col := range boardState.Cells {
		for row := rows - 1; row >= 0; row-- {
			cell := boardState.Cells[col][row]
			if !cell.HasPiece {
				continue
			}

			b.dropPieceInColRow(col, row, cell.Color, false)
		}
	}

	if !boardState.GameOver {
		switch boardState.LastMove.Color {
		case colorBlue:
			b.print(padLeft+2+(cellWidth*(b.inputCol-1)), padTop-1, "🔵")
		default:
			b.print(padLeft+2+(cellWidth*(b.inputCol-1)), padTop-1, "🔴")
		}

		return
	}

	b.printAI(boardState.AIMessage)

	var lastWinnerMsg string
	switch boardState.Winner {
	case colorBlue:
		lastWinnerMsg = "Blue (🔵)"
	case colorRed:
		lastWinnerMsg = "Red (🔴)"
	default:
		lastWinnerMsg = "Tie Game"
	}

	b.print(12, padTop-1, "Winner "+lastWinnerMsg)
	b.screen.Show()
}

func (b *Board) dropPieceInColRow(inputCol int, inputRow int, pieceColor string, animate bool) {

	// Identify where the input marker is located in the board.
	column := padLeft + 2
	if inputCol > 1 {
		column = column + (cellWidth * (inputCol - 1))
	}
	stopRow := padTop + 1

	// We don't use index 0 for the display, so we need to adjust.
	inputRow++

	// Clear the marker.
	b.print(column, padTop-1, " ")

	// Drop the marker into that row.
	for r := 1; r <= inputRow; r++ {
		switch pieceColor {
		case colorBlue:
			b.print(column, stopRow, "🔵")
		case colorRed:
			b.print(column, stopRow, "🔴")
		}

		if r < inputRow {
			if animate {
				time.Sleep(250 * time.Millisecond)
			}
			b.print(column, stopRow, " ")
			stopRow = stopRow + cellHeight
		}
	}
}

func (b *Board) dropPiece(boardState game.BoardState) {
	if boardState.GameOver {
		return
	}

	inputCol := boardState.LastMove.Column

	// Calculate what row to drop the marker in.
	row := -1
	for i := rows - 1; i >= 0; i-- {
		cell := boardState.Cells[inputCol-1][i]
		if !cell.HasPiece {
			row = i
			break
		}
	}

	// Is the column full.
	if row == -1 {
		return
	}

	b.dropPieceInColRow(inputCol, row, boardState.LastMove.Color, true)
}

func (b *Board) movePlayerPiece(direction string) {
	boardState := b.gameBoard.ToBoardState()

	if boardState.GameOver {
		return
	}

	if direction == dirLeft && b.inputCol == 1 {
		return
	}

	if direction == dirRight && b.inputCol == cols {
		return
	}

	// Clear the current marker location.
	column := padLeft + 2
	if b.inputCol > 1 {
		column = column + (cellWidth * (b.inputCol - 1))
	}
	b.print(column, padTop-1, " ")

	// Move the marker column location.
	switch direction {
	case dirLeft:
		b.inputCol--
	case dirRight:
		b.inputCol++
	}

	// Display the marker again in the new location.
	column = padLeft + 2
	if b.inputCol > 1 {
		column = column + (cellWidth * (b.inputCol - 1))
	}

	switch boardState.LastMove.Color {
	case colorBlue:
		b.print(column, padTop-1, "🔵")
	case colorRed:
		b.print(column, padTop-1, "🔴")
	}
}

// drawBox draws an empty box on the screen.
func (b *Board) drawBox(x int, y int, width int, height int) {
	style := b.style
	style = style.Background(tcell.ColorBlack).Foreground(tcell.ColorGray)

	for h := y; h < height; h++ {
		for w := x; w < width; w++ {
			b.screen.SetContent(w, h, ' ', nil, b.style)
		}
	}

	for h := y; h < height; h++ {
		for w := x; w < width; w++ {
			if h == y {
				b.screen.SetContent(w, h, '▀', nil, style)
			}
			if h == height-1 {
				b.screen.SetContent(w, h, '▄', nil, style)
			}
			if w == x || w == width-1 {
				b.screen.SetContent(w, h, '█', nil, style)
			}
		}
	}

	b.screen.Show()
}

func (b *Board) printAI(message string) {
	screenWidth, _ := b.screen.Size()
	actWidth := (screenWidth - boardWidth - 8)

	row := boardWidth + 5
	col := padTop + 4

	for range 8 {
		for range actWidth {
			b.print(row, col, " ")
			row++
		}
		row = boardWidth + 5
		col++
	}

	row = boardWidth + 5
	col = padTop + 4

	scanner := bufio.NewScanner(bytes.NewReader([]byte(message)))
	scanner.Split(bufio.ScanWords)
	for scanner.Scan() {
		word := scanner.Text()
		if word == "CRLF" {
			col++
			row = boardWidth + 5
			continue
		}

		b.print(row, col, word)

		row += len(word) + 1
		if row >= boardWidth+actWidth-4 {
			col++
			row = boardWidth + 5
		}
	}
}

func (b *Board) print(x, y int, str string) {
	for _, c := range str {
		var comb []rune
		w := runewidth.RuneWidth(c)
		if w == 0 {
			comb = []rune{c}
			c = ' '
			w = 1
		}
		b.screen.SetContent(x, y, c, comb, b.style)
		x += w
	}
	b.screen.Show()
}
