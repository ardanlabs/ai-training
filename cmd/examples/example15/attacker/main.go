// This program is the attacker's server. It accepts a request on any path and
// dumps the body to standard out, which is how the exfiltrated data becomes
// visible in the room.
//
// All three security steps POST to http://localhost:9999/ when an injection
// succeeds. Run this in a second terminal before running them; without it the
// exfiltration still happens, it just has nowhere to land.
//
// # Running the example:
//
//	$ make example15-attacker
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
