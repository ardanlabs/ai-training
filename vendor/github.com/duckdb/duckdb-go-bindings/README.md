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

### Single module (v0.3.0+)

Starting with v0.3.0, the module includes pre-built static libraries for all platforms. Simply import `github.com/duckdb/duckdb-go-bindings`.

| duckdb version | module version |
| -------------- | -------------- |
| v1.4.3         | v0.3.0         |

### Legacy per-platform submodules (v0.1.x)

Older versions require platform-specific imports (e.g., `github.com/duckdb/duckdb-go-bindings/darwin-arm64`). These tags continue to work.

| duckdb version | main    | darwin  | linux   | windows |
| -------------- | ------- | ------- | ------- | ------- |
| v1.4.3         | v0.1.24 | v0.1.24 | v0.1.24 | v0.1.24 |
| v1.4.2         | v0.1.23 | v0.1.23 | v0.1.23 | v0.1.23 |
| v1.4.1         | v0.1.21 | v0.1.21 | v0.1.21 | v0.1.21 |
| v1.4.0         | v0.1.19 | v0.1.19 | v0.1.19 | v0.1.19 |
| v1.3.2         | v0.1.17 | v0.1.12 | v0.1.12 | v0.1.12 |
| v1.3.1         | v0.1.16 | v0.1.11 | v0.1.11 | v0.1.11 |
| v1.3.0         | v0.1.15 | v0.1.10 | v0.1.10 | v0.1.10 |
| v1.2.2         | v0.1.14 | v0.1.9  | v0.1.9  | v0.1.9  |
| v1.2.1         | v0.1.13 | v0.1.8  | v0.1.8  | v0.1.8  |
| v1.2.0         | v0.1.10 | v0.1.5  | v0.1.5  | v0.1.5  |

## Local Development

To develop locally, copy the workspace template file:

```bash
cp go.work.dev go.work
```

This sets up Go workspaces to use the local lib/* submodules instead of fetching from the module proxy.

## Releasing a new duckdb version

1. Create a new branch and update the `DUCKDB_VERSION` in the `Makefile`.
2. Invoke the `Fetch and Push Libs` workflow on the new branch.
3. Update the `Releases` table in the `README.md`.
4. If the header (`duckdb.h`) has changes (non-bugfix release), add all changes (new types, functions, etc.) to the bindings.
5. Open a PR.
6. Wait for all tests to pass.
7. Merge the PR into `main`.
8. Publish tags for the root module and the platform submodules by incrementing the latest tagged release.

```
VERSION=v0.3.0
git tag "${VERSION}"
git tag "lib/darwin-amd64/${VERSION}"
git tag "lib/darwin-arm64/${VERSION}"
git tag "lib/linux-amd64/${VERSION}"
git tag "lib/linux-arm64/${VERSION}"
git tag "lib/windows-amd64/${VERSION}"
git push origin "${VERSION}" "lib/darwin-amd64/${VERSION}" "lib/darwin-arm64/${VERSION}" "lib/linux-amd64/${VERSION}" "lib/linux-arm64/${VERSION}" "lib/windows-amd64/${VERSION}"
```

Example PR: https://github.com/duckdb/duckdb-go-bindings/pull/19.

## Installation

Simply import the module in your Go project:

```go
import "github.com/duckdb/duckdb-go-bindings"
```

The module includes pre-built static libraries for all supported platforms:

- darwin-amd64
- darwin-arm64
- linux-amd64
- linux-arm64
- windows-amd64

Platform detection and linking is handled automatically through cgo directives. `CGO_ENABLED=1` is required, and your system needs a C compiler.

## Build Configuration

### Default: Pre-built Static Libraries

By default (no build tags needed), the module automatically links against the pre-built static libraries for your platform:

```bash
go build  # Just works!
go test   # Just works!
```

The appropriate library and linker flags are selected automatically based on your OS and architecture.

### Alternative: Custom Static Libraries

To use your own DuckDB static library instead of the pre-built ones, use the `duckdb_use_static_lib` build tag and provide library paths via `CGO_LDFLAGS`:

```bash
# Darwin/macOS
CGO_ENABLED=1 \
  CPPFLAGS="-DDUCKDB_STATIC_BUILD" \
  CGO_LDFLAGS="-lduckdb -lc++ -L/path/to/lib" \
  go build -tags=duckdb_use_static_lib

# Linux
CGO_ENABLED=1 \
  CPPFLAGS="-DDUCKDB_STATIC_BUILD" \
  CGO_LDFLAGS="-lduckdb -lstdc++ -lm -ldl -L/path/to/lib" \
  go build -tags=duckdb_use_static_lib

# Windows
CGO_ENABLED=1 \
  CPPFLAGS="-DDUCKDB_STATIC_BUILD" \
  CGO_LDFLAGS="-lduckdb -lws2_32 -lwsock32 -lrstrtmgr -lstdc++ -lm --static -L/path/to/lib" \
  go build -tags=duckdb_use_static_lib
```

### Alternative: Dynamic Libraries

To link against a shared DuckDB library, use the `duckdb_use_lib` build tag:

```bash
# Darwin/macOS
CGO_ENABLED=1 \
  CGO_LDFLAGS="-lduckdb -L/path/to/dir" \
  DYLD_LIBRARY_PATH=/path/to/dir \
  go build -tags=duckdb_use_lib

# Linux
CGO_ENABLED=1 \
  CGO_LDFLAGS="-lduckdb -L/path/to/dir" \
  LD_LIBRARY_PATH=/path/to/dir \
  go build -tags=duckdb_use_lib
```

## Arrow functions

Provide the duckdb_arrow build tag if you want to use arrow functions
