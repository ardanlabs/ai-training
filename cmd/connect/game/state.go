package game

// BoardState represent the state of the board for any UI to display.
type BoardState struct {
	Cells       [cols][rows]Cell `json:"cells"`
	LastMove    LastMove         `json:"lastMove"`
	AIMessage   string           `json:"aiMessage"`
	GameMessage string           `json:"GameMessage"`
	GameOver    bool             `json:"gameOver"`
	Winner      string           `json:"winner"`
}

// ToBoardState represents what we will get from an API.
func (b *Board) ToBoardState() BoardState {
	return BoardState{
		Cells:       b.cells,
		LastMove:    b.lastMove,
		AIMessage:   b.aiMessage,
		GameMessage: b.gameMessage,
		GameOver:    b.gameOver,
		Winner:      b.winner,
	}
}
