DUCKDB_VERSION=v1.4.3

fetch.static.libs:
	cd ${DIRECTORY} && \
	curl -OL https://github.com/duckdb/duckdb/releases/download/${DUCKDB_VERSION}/${FILENAME}.zip && \
	rm *.a && \
	rm -f duckdb.h && \
	unzip ${FILENAME}.zip && \
	rm -f ${FILENAME}.zip

update.binding:
	rm -f ${DIRECTORY}/bindings*.go && \
	cp bindings*.go ${DIRECTORY}/.

test.dynamic.lib:
	mkdir dynamic-dir && \
	cd dynamic-dir && \
	curl -OL https://github.com/duckdb/duckdb/releases/download/${DUCKDB_VERSION}/${FILENAME}.zip && \
	unzip ${FILENAME}.zip
