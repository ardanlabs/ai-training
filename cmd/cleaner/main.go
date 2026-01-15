// This program takes the Ultimate Go Notebook in PDF form and creates chunks
// from the different sections in the book. If these chunks are over 500 words,
// then it breaks those up into 250 word chunks. Each chunk exists on it's own
// line and vectorized.
// NOTE:
// More needs to be done. Code examples are flattened out as an example.
package main

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"os"
	"strings"

	"code.sajari.com/docconv/v2"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	if err := convertPDFtoTxt(); err != nil {
		return fmt.Errorf("convertPDFtoTxt: %w", err)
	}

	if err := findChunks(); err != nil {
		return fmt.Errorf("convertPDFtoTxt: %w", err)
	}

	return nil
}

func convertPDFtoTxt() error {
	if _, err := os.Stat("zarf/data/book.txt"); !os.IsNotExist(err) {
		return nil
	}

	input, err := os.Open("/Users/bill/Documents/book/FE-UGN-41.pdf")
	if err != nil {
		return fmt.Errorf("open file: %w", err)
	}
	defer input.Close()

	doc, _, err := docconv.ConvertPDF(input)
	if err != nil {
		return fmt.Errorf("read file: %w", err)
	}

	output, err := os.Create("zarf/data/book.txt")
	if err != nil {
		return fmt.Errorf("create file: %w", err)
	}

	r := bytes.NewReader([]byte(doc))
	if _, err := io.Copy(output, r); err != nil {
		return fmt.Errorf("write file: %w", err)
	}

	return nil
}

func findChunks() error {

	// This code attempts to find the block of text for each section from
	// the outline in the book. The sections are down below.

	inputB, err := os.ReadFile("zarf/data/book.txt")
	if err != nil {
		return fmt.Errorf("open file: %w", err)
	}

	input := string(inputB)

	var chunks []string

	for i := range sections {
		strSection := sections[i]
		if strSection == "END" {
			break
		}

		endSection := sections[i+1]

		srtIdx := strings.Index(input, strSection+"\n")

		switch {
		case endSection != "END":
			endIdx := strings.Index(input, endSection+"\n")
			chunks = append(chunks, input[srtIdx:endIdx])

		default:
			chunks = append(chunks, input[srtIdx:])
		}
	}

	// -------------------------------------------------------------------------

	// This code takes those chunks we found and cleans them up. It won't
	// save a chunk larger than 500 words. If we have a chunk that is larger,
	// then it's broken up into 250 word sections.

	output, err := os.Create("zarf/data/book.chunks")
	if err != nil {
		return fmt.Errorf("open file: %w", err)
	}
	defer output.Close()

	for _, chunk := range chunks {

		// Clean up the chunk and replace \n with spaces.

		// Figure out how many words we have.
		words := strings.Fields(chunk)

		const min = 200
		const max = 500

		// We have less than or exactly max words.
		if len(words) >= min && len(words) <= max {
			fmt.Println(chunk)
			output.WriteString("<CHUNK>\n")
			output.WriteString(chunk)
			output.WriteString("\n")
			output.WriteString("</CHUNK>\n")
			continue
		}

		var idx int

		for {
			// We have the last section of words.
			if len(words[idx:]) <= max && len(words[idx:]) >= min {
				fmt.Println(strings.Join(words[idx:], " "))
				output.WriteString("<CHUNK>\n")
				output.WriteString(strings.Join(words[idx:], " "))
				output.WriteString("\n")
				output.WriteString("</CHUNK>\n")
				break
			}

			// Throw this out since it's too small.
			if len(words[idx:]) < min {
				break
			}

			// This is a max chunk of words.
			fmt.Println(strings.Join(words[idx:idx+max], " "))
			output.WriteString("<CHUNK>\n")
			output.WriteString(strings.Join(words[idx:idx+max], " "))
			output.WriteString("\n")
			output.WriteString("</CHUNK>\n")

			idx = idx + max
		}
	}

	return nil
}

var sections = []string{
	"Welcome",
	"Intended Audience",
	"Acknowledgements",
	"Table of Contents",
	"Chapter 1: Introduction",
	"1.1 Reading Code",
	"1.2 Legacy Software",
	"1.3 Mental Models",
	"1.4 Productivity vs Performance",
	"1.5 Correctness vs Performance",
	"1.6 Understanding Rules",
	"1.7 Differences Between Senior vs Junior Developers",
	"1.8 Design Philosophy",
	"1.8.1 Integrity",
	"1.8.2 Readability",
	"1.8.3 Simplicity",
	"1.8.4 Performance",
	"1.8.5 Micro-Optimizations",
	"1.8.6 Data-Orientation",
	"1.8.7 Interface And Composition",
	"1.8.8 Writing Concurrent Software",
	"1.8.9 Signaling and Channels",
	"Chapter 2: Language Mechanics",
	"2.1 Built-in Types",
	"2.2 Word Size",
	"2.3 Zero Value Concept",
	"2.4 Declare and Initialize",
	"2.5 Conversion vs Casting",
	"2.6 Struct and Construction Mechanics",
	"2.7 Padding and Alignment",
	"2.8 Assigning Values",
	"2.9 Pointers",
	"2.10 Pass By Value",
	"2.11 Escape Analysis",
	"2.12 Stack Growth",
	"2.13 Garbage Collection",
	"2.14 Constants",
	"2.15 IOTA",
	"Chapter 3: Data Structures",
	"3.1 CPU Caches",
	"3.2 Translation Lookaside Buffer (TLB)",
	"3.3 Declaring and Initializing Values",
	"3.4 String Assignments",
	"3.5 Iterating Over Collections",
	"3.6 Value Semantic Iteration",
	"3.7 Pointer Semantic Iteration",
	"3.8 Data Semantic Guideline For Built-In Types",
	"3.9 Different Type Arrays",
	"3.10 Contiguous Memory Construction",
	"3.11 Constructing Slices",
	"3.12 Slice Length vs Capacity",
	"3.13 Data Semantic Guideline For Slices",
	"3.14 Contiguous Memory Layout",
	"3.15 Appending With Slices",
	"3.16 Slicing Slices",
	"3.17 Mutations To The Backing Array",
	"3.18 Copying Slices Manually",
	"3.19 Slices Use Pointer Semantic Mutation",
	"3.20 Linear Traversal Efficiency",
	"3.21 UTF-8",
	"3.22 Declaring And Constructing Maps",
	"3.23 Lookups and Deleting Map Keys",
	"3.24 Key Map Restrictions",
	"Chapter 4: Decoupling",
	"4.1 Methods",
	"4.2 Method Calls",
	"4.3 Data Semantic Guideline For Internal Types",
	"4.4 Data Semantic Guideline For Struct Types",
	"4.5 Methods Are Just Functions",
	"4.6 Know The Behavior of the Code",
	"4.7 Interfaces",
	"4.8 Interfaces Are Valueless",
	"4.9 Implementing Interfaces",
	"4.10 Polymorphism",
	"4.11 Method Set Rules",
	"4.12 Slice of Interface",
	"4.13 Embedding",
	"4.14 Exporting",
	"Chapter 5: Software Design",
	"5.1 Grouping Different Types of Data",
	"5.2 Don’t Design With Interfaces",
	"5.3 Composition",
	"5.4 Decoupling With Interfaces",
	"5.5 Interface Composition",
	"5.6 Precision Review",
	"5.7 Implicit Interface Conversions",
	"5.8 Type assertions",
	"5.9 Interface Pollution",
	"5.10 Interface Ownership",
	"5.11 Error Handling",
	"5.12 Always Use The Error Interface",
	"5.13 Handling Errors",
	"Chapter 6: Concurrency",
	"6.1 Scheduler Semantics",
	"6.2 Concurrency Basics",
	"6.3 Preemptive Scheduler",
	"6.4 Data Races",
	"6.5 Data Race Example",
	"6.6 Race Detection",
	"6.7 Atomics",
	"6.8 Mutexes",
	"6.9 Read/Write Mutexes",
	"6.10 Channel Semantics",
	"6.11 Channel Patterns",
	"6.11.1 Wait For Result",
	"6.11.2 Fan Out/In",
	"6.11.3 Wait For Task",
	"6.11.4 Pooling",
	"6.11.5 Drop",
	"6.11.6 Cancellation",
	"6.11.7 Fan Out/In Semaphore",
	"6.11.8 Bounded Work Pooling",
	"6.11.9 Retry Timeout",
	"6.11.10 Channel Cancellation",
	"Chapter 7: Testing",
	"7.1 Basic Unit Test",
	"7.2 Table Unit Test",
	"7.3 Web Call Mocking",
	"7.4 Internal Web Endpoints",
	"7.5 Basic Sub-Tests",
	"Chapter 8: Benchmarking",
	"8.1 Basic Benchmark",
	"8.2 Basic Sub-Benchmarks",
	"8.3 Validate Benchmarks",
	"Chapter 9: Generics",
	"9.1 Basic Syntax",
	"9.2 Underlying Types",
	"9.3 Struct Types",
	"9.4 Behavior As Constraint",
	"9.5 Type As Constraint",
	"9.6 Multi-Type Parameters",
	"9.7 Field Access",
	"9.8 Slice Constraints",
	"9.9 Channels",
	"9.10 Hash Tables",
	"10.1 Introduction",
	"10.1.1 The Basics of Profiling",
	"10.1.2 Types of Profiling",
	"10.1.3 Hints to interpret what I see in the profile",
	"10.1.4 Rules of Performance",
	"10.1.5 Go and OS Tooling",
	"10.2 Example Code",
	"10.3 Benchmarking",
	"10.4 Memory Profiling",
	"10.5 Inlining",
	"10.6 Escape Analysis",
	"Chapter 11: Profiling Live Code",
	"11.1 Example Code",
	"11.2 Generating a GC Trace",
	"11.3 Generating Load And Evaluation",
	"11.4 Adding Profile Endpoints",
	"11.5 Viewing Memory Profile",
	"11.6 Removing Allocations",
	"Chapter 12: Tracing",
	"12.1 Example Code",
	"12.2 Generating Traces",
	"12.3 Viewing Traces",
	"12.4 Fan-Out",
	"12.5 Cache Friendly",
	"12.6 Fan-Out Results",
	"12.7 Pooling",
	"12.8 Pooling Results",
	"12.9 GC Percentage",
	"12.10 Tasks And Regions",
	"Chapter 13: Stack Traces / Core Dumps",
	"13.1 ABI Changes In 1.17",
	"13.2 Basic Example",
	"13.3 Word Packing",
	"13.4 Go 1.17 ABI Changes",
	"13.5 Generating Core Dumps",
	"Chapter 14: Blog Posts",
	"14.1 Stacks And Pointer Mechanics",
	"14.2 Escape Analysis Mechanics",
	"14.3 Scheduling In Go: OS Scheduler",
	"14.4 Scheduling In Go: Go Scheduler",
	"14.5 Scheduling In Go: Concurrency",
	"14.6 Garbage Collection Semantics",
	"END",
}
