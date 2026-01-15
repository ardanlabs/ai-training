package mongodb

// Index represents information about an index.
type Index struct {
	ID   string `bson:"id"`
	Type string `bson:"type"`
}

// VectorIndexSettings represents setting to create a vector index.
type VectorIndexSettings struct {
	NumDimensions int
	Path          string
	Similarity    string
}
