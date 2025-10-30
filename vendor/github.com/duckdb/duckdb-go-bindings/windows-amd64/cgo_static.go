package duckdb_go_bindings

/*
#cgo CPPFLAGS: -DDUCKDB_STATIC_BUILD
#cgo LDFLAGS: -lduckdb_static -lautocomplete_extension -lcore_functions_extension -licu_extension -ljson_extension -lparquet_extension -ltpcds_extension -ltpch_extension -lduckdb_fastpforlib -lduckdb_fmt -lduckdb_fsst -lduckdb_hyperloglog -lduckdb_mbedtls -lduckdb_miniz -lduckdb_pg_query -lduckdb_re2 -lduckdb_skiplistlib -lduckdb_utf8proc -lduckdb_yyjson -lduckdb_zstd -lws2_32 -lwsock32 -lrstrtmgr -lstdc++ -lm --static -L${SRCDIR}/
#include <duckdb.h>
*/
import "C"
