# Check to see if we can use ash, in Alpine images, or default to BASH.
SHELL_PATH = /bin/ash
SHELL = $(if $(wildcard $(SHELL_PATH)),/bin/ash,/bin/bash)

# ==============================================================================
# Go Installation
#
#	You need to have Go version 1.25 to run this code.
#
#	https://go.dev/dl/
#
#	If you are not allowed to update your Go frontend, you can install
#	and use a 1.25 frontend.
#
#	$ go install golang.org/dl/go1.25@latest
#	$ go1.25 download
#
#	This means you need to use `go1.25` instead of `go` for any command
#	using the Go frontend tooling from the makefile.

# ==============================================================================
# Brew Installation
#
#	Having brew installed will simplify the process of installing all the tooling.
#
#	Run this command to install brew on your machine. This works for Linux, Mac and Windows.
#	The script explains what it will do and then pauses before it does it.
#	$ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
#
#	WINDOWS MACHINES
#	These are extra things you will most likely need to do after installing brew
#
# 	Run these three commands in your terminal to add Homebrew to your PATH:
# 	Replace <name> with your username.
#	$ echo '# Set PATH, MANPATH, etc., for Homebrew.' >> /home/<name>/.profile
#	$ echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"' >> /home/<name>/.profile
#	$ eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"
#
# 	Install Homebrew's dependencies:
#	$ sudo apt-get install build-essential
#
# 	Install GCC:
#	$ brew install gcc

# ==============================================================================
# Install Tooling and Dependencies
#
#	This project uses Docker and it is expected to be installed. Please provide
#	Docker at least 4 CPUs. To use Podman instead please alias Docker CLI to
#	Podman CLI or symlink the Docker socket to the Podman socket. More
#	information on migrating from Docker to Podman can be found at
#	https://podman-desktop.io/docs/migrating-from-docker.
#
#	Run these commands to install everything needed.
#	$ make install
#	$ make docker
#	$ make install-python

# ==============================================================================
# Pulling Model Images
#
# Start Kronk and pull down all the images we need for this project.
#
#	Run these commands to download the models we need.
#	$ make install-models

# ==============================================================================
# CLASS NOTES
#
# 	Mongo support
# 		db.book.find({id: 300})
#
# 		db.book.aggregate([
# 		{
# 			"$vectorSearch": {
# 				"index": "vector_index",
# 				"exact": true,
# 				"path": "embedding",
# 				"queryVector": [1.2, 2.2, 3.2, 4.2],
# 				"limit": 10
# 			}
# 		},
# 		{
# 			"$project": {
# 				"text": 1,
# 				"embedding": 1,
# 				"score": {
# 					"$meta": "vectorSearchScore"
# 				}
# 			}
# 		}
# 	}])

# ==============================================================================
# Install dependencies

install:
	brew install mongosh
	brew install mplayer
	brew install pgcli
	brew install uv
	brew install pkgconf
	brew install whisper-cpp
	brew install homebrew-ffmpeg/ffmpeg/ffmpeg --with-whisper-cpp --HEAD
	go install github.com/janpfeifer/gonb@latest
	go install github.com/ardanlabs/kronk/cmd/kronk@latest

docker:
	docker pull mongodb/mongodb-atlas-local:8.2
	docker pull ghcr.io/open-webui/open-webui:v0.7.2
	docker pull postgres:18.1
	docker pull quay.io/docling-project/docling-serve:v1.9.0

install-python:
	rm -rf .venv
	uv venv --python 3.12
	uv lock
	uv sync
	uv pip install vllm
	uv pip install jupyterlab
	uv pip list > pydeps.txt

# Use this to install models. Needed to run examples. You can install a model
# as the example calls for it. Just copy/paste in terminal.
install-models:
	@echo ========== INSTALL MODELS ==========
	kronk model pull --local "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/Qwen2.5-VL-3B-Instruct-Q8_0.gguf" "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf"
	@echo
	kronk model pull --local "mradermacher/Qwen2-Audio-7B-GGUF/Qwen2-Audio-7B.Q8_0.gguf" "mradermacher/Qwen2-Audio-7B-GGUF/Qwen2-Audio-7B.mmproj-Q8_0.gguf"
	@echo
	kronk model pull --local "Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q8_0.gguf"
	@echo
	kronk model pull --local "ggml-org/embeddinggemma-300m-qat-q8_0-GGUF/embeddinggemma-300m-qat-Q8_0.gguf"
	@echo
	kronk model pull --local "bartowski/cerebras_Qwen3-Coder-REAP-25B-A3B-GGUF/cerebras_Qwen3-Coder-REAP-25B-A3B-Q8_0.gguf"
	@echo

# ==============================================================================
# Examples

example01:
	go run cmd/examples/example01/main.go

example02:
	go run cmd/examples/example02/main.go

example03:
	go run cmd/examples/example03/main.go

example04:
	go run cmd/examples/example04/main.go

example05:
	go run cmd/examples/example05/main.go

example06:
	go run cmd/examples/example06/main.go

example07:
	go run cmd/examples/example07/*.go

example08-step1:
	go run cmd/examples/example08/step1/main.go

example08-step2:
	go run cmd/examples/example08/step2/main.go

example08-step3:
	go run cmd/examples/example08/step3/main.go

example08-step4:
	go run cmd/examples/example08/step4/main.go

example08-step5:
	go run cmd/examples/example08/step5/main.go

example09-step1:
	go run cmd/examples/example09/step1/main.go

example09-step2:
	go run cmd/examples/example09/step2/main.go

example09-step3:
	go run cmd/examples/example09/step3/main.go

example09-step4:
	go run cmd/examples/example09/step4/*.go

example10-step1:
	go run cmd/examples/example10/step1/main.go

example10-step2:
	go run cmd/examples/example10/step2/*.go

example11-step1:
	mkdir -p zarf/samples/videos/chunks && \
	mkdir -p zarf/samples/videos/frames && \
	export KRONK_CONTEXT_LENGTH=$(KRONK_CONTEXT_LENGTH) && \
	go run ./cmd/examples/example11/step1/*.go

example11-step2:
	go run cmd/examples/example11/step2/*.go

example12:
	go run cmd/examples/example12/*.go

example13-step1:
	CGO_ENABLED=0 go run cmd/examples/example13/step1/main.go

example13-step2:
	CGO_ENABLED=0 go run cmd/examples/example13/step2/main.go

example13-step3:
	go run cmd/examples/example13/step3/main.go

example13-step4:
	go run cmd/examples/example13/step4/main.go

example13-step4-npm-install:
	cd cmd/examples/example13/step4/react/app && npm install

example13-step4-npm-build:
	cd cmd/examples/example13/step4/react/app && npm run build

example13-step4-npm-run:
	cd cmd/examples/example13/step4/react/app && npm run dev

example13-step4-curl1:
	curl -i -X POST http://0.0.0.0:8080/chat \
     -H "Content-Type: application/json" \
     -d '{ \
		"messages": [ \
			{ \
				"role": "user", \
				"content": "How do you declare an interface in Go?" \
			} \
		] \
    }'

example13-step4-curl2:
	curl -i -X POST http://0.0.0.0:8080/chat \
     -H "Content-Type: application/json" \
     -d '{ \
		"messages": [ \
			{ \
				"role": "user", \
				"content": "What is the weather in London, England?" \
			} \
		] \
    }'

# ==============================================================================
# Run Postgres, MongoDB, and Open WebUI

compose-up:
	docker compose -f zarf/docker/compose.yaml up

compose-down:
	docker compose -f zarf/docker/compose.yaml down

compose-clean-mongo:
	rm -rf zarf/docker/mongodb && \
	mkdir -p zarf/docker/mongodb/db zarf/docker/mongodb/configdb zarf/docker/mongodb/mongot && \
	chmod -R 777 zarf/docker/mongodb

compose-clean-sql:
	rm -rf zarf/docker/sql-data

compose-logs:
	docker compose logs -n 100

# ==============================================================================
# Running Open WebUI only

owu-compose-up:
	docker compose -f zarf/docker/compose.yaml up openwebui

owu-compose-down:
	docker compose -f zarf/docker/compose.yaml down openwebui

owu-browse:
	open -a "Google Chrome" http://localhost:3000/

# ==============================================================================
# Running Docling only

docling-compose-up:
	docker compose -f zarf/docker/compose.yaml up docling

docling-compose-down:
	docker compose -f zarf/docker/compose.yaml down docling

docling-browse:
	open -a "Google Chrome" http://localhost:5001/ui/

# ==============================================================================
# Running Mongo only

mongo-compose-up:
	docker compose -f zarf/docker/compose.yaml up mongodb

mongo-compose-down:
	docker compose -f zarf/docker/compose.yaml down mongodb

# ==============================================================================
# Kronk tooling

kronk-up:
	export KRONK_CACHE_MODEL_CONFIG_FILE=zarf/kms/model_config.yaml && \
	kronk server start

kronk-logs:
	kronk server logs

kronk-list-models:
	kronk model list --local

# ==============================================================================
# Run Tooling

download-data:
	curl -o zarf/data/example3.gz -X GET https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Cell_Phones_and_Accessories_5.json.gz \
	&& gunzip -k -d zarf/data/example3.gz \
	&& mv zarf/data/example3 zarf/data/example3.json

clean-data:
	go run cmd/cleaner/main.go

mongo:
	mongosh -u ardan -p ardan mongodb://localhost:27017

pgcli:
	pgcli postgresql://postgres:postgres@localhost

# ==============================================================================
# VLLM
# You need to add this to your .env file
# 	export VLLM_CPU_KVCACHE_SPACE=26

vllm-run:
	source .env && uv run vllm serve --host 0.0.0.0 --port 8000 --max_num_batched_tokens 131072 "NousResearch/Hermes-3-Llama-3.1-8B"

vllm-test:
	curl -X POST "http://localhost:8000/v1/chat/completions" \
		-H "Content-Type: application/json" \
		--data '{ \
			"model": "NousResearch/Hermes-3-Llama-3.1-8B", \
			"messages": [ \
				{"role": "system", "content": [{"type": "text", "text": "You are an expert developer and you are helping the user with their question."}]}, \
				{"role": "user", "content": [{"type": "text", "text": "How do you declare a variable in Python?"}]} \
			] \
		}'

# ==============================================================================
# Jupyter Notebook using Go

jupyter-run:
	uv run jupyter lab

# ==============================================================================
# Llamacpp support

llama-bench:
	zarf/libraries/llama-bench --list-devices

# ==============================================================================
# Go Modules support

tidy:
	go mod tidy

deps-upgrade:
	go get -u -v ./...
	go mod tidy

# ==============================================================================
# Python Dependencies

deps-python-sync:
	uv sync

deps-python-upgrade:
	uv lock --upgrade && uv sync
	uv pip install vllm
	uv pip install jupyterlab
	uv pip list > pydeps.txt

deps-python-outdated:
	uv pip list --outdated

# ==============================================================================
# FFMpeg test commands

ffmpeg-extract-chunks:
	rm -rf zarf/samples/videos/chunks/*
	ffmpeg -i zarf/samples/videos/test_rag_video.mp4 \
		-c copy -map 0 -f segment -segment_time 15 -reset_timestamps 1 \
		-loglevel error \
		zarf/samples/videos/chunks/output_%05d.mp4

ffmpeg-extract-frames:
	rm -rf zarf/samples/videos/frames/*
	ffmpeg -skip_frame nokey -i zarf/samples/videos/chunks/output_00000.mp4 \
		-frame_pts true -fps_mode vfr \
		-loglevel error \
		zarf/samples/videos/frames/frame-%05d.jpg

ffmpeg-extract-different-frames:
	rm -rf zarf/samples/videos/frames/*
	ffmpeg -i zarf/samples/videos/test_rag_video.mp4 \
		-vf "select='gt(scene,0.05)',setpts=N/FRAME_RATE/TB" \
		-fps_mode vfr \
		-loglevel error \
		zarf/samples/videos/frames/frame-%05d.jpg

ffmpeg-check-chunk-duration:
	ffprobe -v quiet -print_format json -show_entries format=duration zarf/samples/videos/chunks/output_00000.mp4
	ffprobe -v quiet -print_format json -show_entries format=duration zarf/samples/videos/chunks/output_00002.mp4
	ffprobe -v quiet -print_format json -show_entries format=duration zarf/samples/videos/chunks/output_00003.mp4

# ==============================================================================
# curl test commands

curl-tooling:
	curl http://localhost:11434/v1/chat/completions \
	-H "Content-Type: application/json" \
	-d '{ \
	"model": "gpt-oss:latest", \
	"messages": [ \
		{ \
			"role": "user", \
			"content": "What is the weather like in New York, NY?" \
		} \
	], \
	"stream": false, \
	"tools": [ \
		{ \
			"type": "function", \
			"function": { \
				"name": "get_current_weather", \
				"description": "Get the current weather for a location", \
				"parameters": { \
					"type": "object", \
					"properties": { \
						"location": { \
							"type": "string", \
							"description": "The location to get the weather for, e.g. San Francisco, CA" \
						} \
					}, \
					"required": ["location"] \
				} \
			} \
		} \
  	], \
	"tool_selection": "auto", \
	"options": { "num_ctx": 32000 } \
	}'

# ==============================================================================

# This will establish a SSE session and this is where we will get the sessionID
# and the results of the call.
curl-mcp-get-session:
	curl -N -H "Accept: text/event-stream" http://localhost:8080/tool_list_files

# Once we have the sessionID, we can initialize the session.
# Replace the sessionID with the one you get from the SSE session.
curl-mcp-init:
	curl -X POST http://localhost:8080/tool_list_files?sessionid=$(SESSIONID) \
	-H "Content-Type: application/json" \
	-d '{ \
		"jsonrpc": "2.0", \
		"id": 1, \
		"method": "initialize", \
		"params": { \
			"protocolVersion": "2024-11-05", \
			"capabilities": {}, \
			"clientInfo": {"name": "curl-client", "version": "1.0.0"} \
		} \
	}'

# Then we can make the actual tool call. The response will be streamed in the
# session call. Replace the sessionID with the one you get from the SSE session.
curl-mcp-tool-call:
	curl -X POST http://localhost:8080/tool_list_files?sessionid=$(SESSIONID) \
	-H "Content-Type: application/json" \
	-d '{ \
		"jsonrpc": "2.0", \
		"id": 2, \
		"method": "tools/call", \
		"params": { \
			"name": "tool_list_files", \
			"arguments": {"filter": "list any files that have the name example"} \
		} \
	}'

curl-embed-triton:
	curl -i -X POST https://api.predictionguard.com/embeddings \
     -H "Authorization: Bearer $(PG_API_PREDICTIONGUARD_API_KEY)" \
     -H "Content-Type: application/json" \
     -d '{ \
		"model": "bridgetower-large-itm-mlm-itc", \
		"input": [ \
			{ \
				"text": "This is Bill Kennedy, a decent Go developer.", \
				"image": "$(IMAGE)" \
			} \
		] \
	}'

# =============================================================================
# Docling

basic-doc:
	curl -i -X POST "http://0.0.0.0:5001/v1/convert/file" \
		-H "Content-Type: multipart/form-data" \
		-F 'files=@zarf/samples/docs/dinner_menu.pdf;type=application/pdf' \
		-F 'to_formats=md' \
		-F 'include_images=false' \
		-F 'table_mode=accurate' \
		-F 'md_page_break_placeholder=---' \
		-F 'pdf_backend=dlparse_v4' \
		-F 'image_export_mode=placeholder'
