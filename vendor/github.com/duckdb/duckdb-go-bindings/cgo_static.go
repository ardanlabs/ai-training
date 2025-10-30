//go:build duckdb_use_static_lib

package duckdb_go_bindings

/*
#cgo CPPFLAGS: -DDUCKDB_STATIC_BUILD
#include <duckdb.h>
*/
import "C"
