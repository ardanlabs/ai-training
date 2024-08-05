package board

import (
	"fmt"
	"runtime/debug"

	"github.com/gdamore/tcell/v2"
)

// pollEvents starts a goroutine to handle terminal events.
func (b *Board) pollEvents() chan struct{} {
	quit := make(chan struct{})

	boardState := b.gameBoard.ToBoardState()

	if boardState.LastMove.Color == colorBlue {
		boardState = b.gameBoard.AITurn()
		b.dropPiece(boardState)
	}

	go func() {
		defer func() {
			if r := recover(); r != nil {
				b.screen.Clear()
				fmt.Println(r)
				debug.PrintStack()
			}
		}()

		for {
			boardState := b.gameBoard.ToBoardState()

			if boardState.LastMove.Color == colorBlue {
				boardState = b.gameBoard.AITurn()
				b.dropPiece(boardState)
			}

			event := b.screen.PollEvent()

			// Check if we received a key event.
			ev, isEventKey := event.(*tcell.EventKey)
			if !isEventKey {
				continue
			}

			keyType := ev.Key()

			// Allow the user to quit the game at any time.
			if keyType == tcell.KeyRune {
				if ev.Rune() == rune('q') {
					close(quit)
					return
				}
			}

			boardState = b.gameBoard.ToBoardState()

			// Only the blue player can control the piece.
			if !boardState.GameOver && boardState.LastMove.Color == colorBlue {
				b.screen.Beep()
				continue
			}

			switch keyType {
			case tcell.KeyRune:
				switch ev.Rune() {
				case rune('n'):
					b.newGame()

				case rune(' '):
					boardState = b.gameBoard.UserTurn(b.inputCol)
					b.dropPiece(boardState)
				}

			case tcell.KeyLeft:
				b.movePlayerPiece(dirLeft)

			case tcell.KeyRight:
				b.movePlayerPiece(dirRight)

			case tcell.KeyEnter, tcell.KeyDown:
				boardState = b.gameBoard.UserTurn(b.inputCol)
				b.dropPiece(boardState)
			}
		}
	}()

	return quit
}
