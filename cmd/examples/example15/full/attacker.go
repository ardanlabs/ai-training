// This program runs a simple HTTP server that accepts POST requests on any path
// and dumps the request body to standard out. It is used to simulate an
// attacker-controlled endpoint for the prompt injection demo.
//
// # Running:
//
//	$ go run attacker.go

//go:build ignore

package main

import (
	"fmt"
	"io"
	"log"
	"net/http"
)

func main() {
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)

		fmt.Println("========================================")
		fmt.Printf("Method: %s\n", r.Method)
		fmt.Printf("Path:   %s\n", r.URL.Path)
		fmt.Printf("Body:\n%s\n", body)
		fmt.Println("========================================")

		w.WriteHeader(http.StatusOK)
	})

	fmt.Println("Attacker server listening on :9999")
	log.Fatal(http.ListenAndServe(":9999", nil))
}
