package llamacpp

type RankingDocument struct {
	Document  string
	Embedding []float64
}

type Ranking struct {
	Document string
	Score    float64
}

type ChatMessage struct {
	Role    string
	Content string
}

type ChatResponse struct {
	Response string
	Err      error
}
