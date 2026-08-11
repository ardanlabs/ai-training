package harness

import "fmt"

// Banner prints a section header. The first line is boxed in rules, the rest
// are printed underneath as description.
func Banner(lines ...string) {
	if len(lines) == 0 {
		return
	}

	fmt.Print("\n============================================================\n")
	fmt.Printf("%s\n", lines[0])
	fmt.Print("============================================================\n")

	for _, l := range lines[1:] {
		fmt.Printf("%s\n", l)
	}
}

// SubBanner prints the title of a subsection framed by rules of '-'.
func SubBanner(title string) {
	fmt.Print("\n------------------------------------------------------------\n")
	fmt.Printf("%s\n", title)
	fmt.Print("------------------------------------------------------------\n")
}

// PrintDocs prints the header followed by one line per retrieved document
// showing the id, similarity score, access level, and the leading 80
// characters of the text.
func PrintDocs(header string, docs []Document) {
	fmt.Printf("\n%s\n", header)

	for _, d := range docs {
		fmt.Printf("  [id=%d sim=%.4f access=%s] %.80s...\n", d.ID, d.Similarity, d.AccessLevel, d.Text)
	}
}
