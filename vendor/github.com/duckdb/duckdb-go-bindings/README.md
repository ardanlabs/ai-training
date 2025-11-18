# duckdb-go-bindings

![Tests status](https://github.com/duckdb/duckdb-go-bindings/actions/workflows/run_tests.yml/badge.svg)

This repository wraps DuckDB's C API calls in Go native types and functions.

Minimum Go version: 1.24.

#### 🚧 WORK IN PROGRESS 🚧

> [!IMPORTANT]  
> Some type aliases and function wrappers are still missing.
> 
> Breaking changes can happen.

## Releases

This module's *first* release contains DuckDB's v1.2.0 release.

| duckdb version | main module | darwin amd | darwin arm | linux amd | linux arm | windows amd |
|----------------|-------------|------------|------------|-----------|-----------|-------------|
| v1.4.2         | v0.1.23     | v0.1.23    | v0.1.23    | v0.1.23   | v0.1.23   | v0.1.23     |
| v1.4.1         | v0.1.21     | v0.1.21    | v0.1.21    | v0.1.21   | v0.1.21   | v0.1.21     |
| v1.4.0         | v0.1.19     | v0.1.19    | v0.1.19    | v0.1.19   | v0.1.19   | v0.1.19     |
| v1.3.2         | v0.1.17     | v0.1.12    | v0.1.12    | v0.1.12   | v0.1.12   | v0.1.12     |
| v1.3.1         | v0.1.16     | v0.1.11    | v0.1.11    | v0.1.11   | v0.1.11   | v0.1.11     |
| v1.3.0         | v0.1.15     | v0.1.10    | v0.1.10    | v0.1.10   | v0.1.10   | v0.1.10     |
| v1.2.2         | v0.1.14     | v0.1.9     | v0.1.9     | v0.1.9    | v0.1.9    | v0.1.9      |
| v1.2.1         | v0.1.13     | v0.1.8     | v0.1.8     | v0.1.8    | v0.1.8    | v0.1.8      |
| v1.2.0         | v0.1.10     | v0.1.5     | v0.1.5     | v0.1.5    | v0.1.5    | v0.1.5      |

The main module (`github.com/duckdb/duckdb-go-bindings`) does not link any pre-built static library.

## Releasing a new duckdb version

1. Create a new branch and update the `DUCKDB_VERSION` in the `Makefile`.
2. Invoke the `Fetch and Push Libs` workflow on the new branch.
3. Update the `Releases` table in the `README.md`.
4. If the header (`duckdb.h`) has changes (non-bugfix release), add all changes (new types, functions, etc.) to the bindings.
5. Open a PR.
6. Wait for all tests to pass.
7. Merge the PR into `main`.
8. Publish the tags by incrementing the latest tagged release for the main module, and for each OS+architecture combination.
```
git tag <tagname>
git push origin <tagname>
```

Example PR: https://github.com/duckdb/duckdb-go-bindings/pull/19.

## Using a pre-built static library

A few pre-built static libraries exist for different OS + architecture combinations.
You can import these into your projects without providing additional build flags.
`CGO` must be enabled, and your system needs a compiler available.

Here's a list:
- `github.com/duckdb/duckdb-go-bindings/`...
  - `darwin-amd64`
  - `darwin-arm64`
  - `linux-amd64`
  - `linux-arm64`
  - `windows-amd64`

## Static linking

Note that the lib(s) name must match the name provided in the `CGO_LDFLAGS`.

On Darwin. 
```
CGO_ENABLED=1 CPPFLAGS="-DDUCKDB_STATIC_BUILD" CGO_LDFLAGS="-lduckdb -lc++ -L/path/to/lib" go build -tags=duckdb_use_static_lib
```

On Linux.
```
CGO_ENABLED=1 CPPFLAGS="-DDUCKDB_STATIC_BUILD" CGO_LDFLAGS="-lduckdb -lstdc++ -lm -ldl -L/path/to/lib" go build -tags=duckdb_use_static_lib
```

On Windows.
```
CGO_ENABLED=1 CPPFLAGS="-DDUCKDB_STATIC_BUILD" CGO_LDFLAGS="-lduckdb -lws2_32 -lwsock32 -lrstrtmgr -lstdc++ -lm --static -L/path/to/lib" go build -tags=duckdb_use_static_lib
```

## Dynamic linking

On Darwin.
```
CGO_ENABLED=1 CGO_LDFLAGS="-lduckdb -L/path/to/dir" DYLD_LIBRARY_PATH=/path/to/dir go build -tags=duckdb_use_lib
```

On Linux.
```
CGO_ENABLED=1 CGO_LDFLAGS="-lduckdb -L/path/to/dir" LD_LIBRARY_PATH=/path/to/dir go build -tags=duckdb_use_lib
```

## Arrow functions

Provide the duckdb_arrow build tag if you want to use arrow functions
