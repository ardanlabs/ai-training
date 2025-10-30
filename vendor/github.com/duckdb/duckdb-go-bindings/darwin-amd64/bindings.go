package duckdb_go_bindings

/*
#include <duckdb.h>
#include <stdlib.h>
#include <duckdb_go_bindings.h>
*/
import "C"

import (
	"fmt"
	"log"
	"sync"
	"unsafe"
)

// ------------------------------------------------------------------ //
// Enums
// ------------------------------------------------------------------ //

// Type wraps duckdb_type.
type Type = C.duckdb_type

const (
	TypeInvalid        Type = C.DUCKDB_TYPE_INVALID
	TypeBoolean        Type = C.DUCKDB_TYPE_BOOLEAN
	TypeTinyInt        Type = C.DUCKDB_TYPE_TINYINT
	TypeSmallInt       Type = C.DUCKDB_TYPE_SMALLINT
	TypeInteger        Type = C.DUCKDB_TYPE_INTEGER
	TypeBigInt         Type = C.DUCKDB_TYPE_BIGINT
	TypeUTinyInt       Type = C.DUCKDB_TYPE_UTINYINT
	TypeUSmallInt      Type = C.DUCKDB_TYPE_USMALLINT
	TypeUInteger       Type = C.DUCKDB_TYPE_UINTEGER
	TypeUBigInt        Type = C.DUCKDB_TYPE_UBIGINT
	TypeFloat          Type = C.DUCKDB_TYPE_FLOAT
	TypeDouble         Type = C.DUCKDB_TYPE_DOUBLE
	TypeTimestamp      Type = C.DUCKDB_TYPE_TIMESTAMP
	TypeDate           Type = C.DUCKDB_TYPE_DATE
	TypeTime           Type = C.DUCKDB_TYPE_TIME
	TypeInterval       Type = C.DUCKDB_TYPE_INTERVAL
	TypeHugeInt        Type = C.DUCKDB_TYPE_HUGEINT
	TypeUHugeInt       Type = C.DUCKDB_TYPE_UHUGEINT
	TypeVarchar        Type = C.DUCKDB_TYPE_VARCHAR
	TypeBlob           Type = C.DUCKDB_TYPE_BLOB
	TypeDecimal        Type = C.DUCKDB_TYPE_DECIMAL
	TypeTimestampS     Type = C.DUCKDB_TYPE_TIMESTAMP_S
	TypeTimestampMS    Type = C.DUCKDB_TYPE_TIMESTAMP_MS
	TypeTimestampNS    Type = C.DUCKDB_TYPE_TIMESTAMP_NS
	TypeEnum           Type = C.DUCKDB_TYPE_ENUM
	TypeList           Type = C.DUCKDB_TYPE_LIST
	TypeStruct         Type = C.DUCKDB_TYPE_STRUCT
	TypeMap            Type = C.DUCKDB_TYPE_MAP
	TypeArray          Type = C.DUCKDB_TYPE_ARRAY
	TypeUUID           Type = C.DUCKDB_TYPE_UUID
	TypeUnion          Type = C.DUCKDB_TYPE_UNION
	TypeBit            Type = C.DUCKDB_TYPE_BIT
	TypeTimeTZ         Type = C.DUCKDB_TYPE_TIME_TZ
	TypeTimestampTZ    Type = C.DUCKDB_TYPE_TIMESTAMP_TZ
	TypeAny            Type = C.DUCKDB_TYPE_ANY
	TypeBigNum         Type = C.DUCKDB_TYPE_BIGNUM
	TypeSQLNull        Type = C.DUCKDB_TYPE_SQLNULL
	TypeStringLiteral  Type = C.DUCKDB_TYPE_STRING_LITERAL
	TypeIntegerLiteral Type = C.DUCKDB_TYPE_INTEGER_LITERAL
	TypeTimeNS         Type = C.DUCKDB_TYPE_TIME_NS
)

// State wraps duckdb_state.
type State = C.duckdb_state

const (
	StateSuccess State = C.DuckDBSuccess
	StateError   State = C.DuckDBError
)

// PendingState wraps duckdb_pending_state.
type PendingState = C.duckdb_pending_state

const (
	PendingStateResultReady      PendingState = C.DUCKDB_PENDING_RESULT_READY
	PendingStateResultNotReady   PendingState = C.DUCKDB_PENDING_RESULT_NOT_READY
	PendingStateError            PendingState = C.DUCKDB_PENDING_ERROR
	PendingStateNoTasksAvailable PendingState = C.DUCKDB_PENDING_NO_TASKS_AVAILABLE
)

// ResultType wraps duckdb_result_type.
type ResultType = C.duckdb_result_type

const (
	ResultTypeInvalid     ResultType = C.DUCKDB_RESULT_TYPE_INVALID
	ResultTypeChangedRows ResultType = C.DUCKDB_RESULT_TYPE_CHANGED_ROWS
	ResultTypeNothing     ResultType = C.DUCKDB_RESULT_TYPE_NOTHING
	ResultTypeQueryResult ResultType = C.DUCKDB_RESULT_TYPE_QUERY_RESULT
)

// StatementType wraps duckdb_statement_type.
type StatementType = C.duckdb_statement_type

const (
	StatementTypeInvalid     StatementType = C.DUCKDB_STATEMENT_TYPE_INVALID
	StatementTypeSelect      StatementType = C.DUCKDB_STATEMENT_TYPE_SELECT
	StatementTypeInsert      StatementType = C.DUCKDB_STATEMENT_TYPE_INSERT
	StatementTypeUpdate      StatementType = C.DUCKDB_STATEMENT_TYPE_UPDATE
	StatementTypeExplain     StatementType = C.DUCKDB_STATEMENT_TYPE_EXPLAIN
	StatementTypeDelete      StatementType = C.DUCKDB_STATEMENT_TYPE_DELETE
	StatementTypePrepare     StatementType = C.DUCKDB_STATEMENT_TYPE_PREPARE
	StatementTypeCreate      StatementType = C.DUCKDB_STATEMENT_TYPE_CREATE
	StatementTypeExecute     StatementType = C.DUCKDB_STATEMENT_TYPE_EXECUTE
	StatementTypeAlter       StatementType = C.DUCKDB_STATEMENT_TYPE_ALTER
	StatementTypeTransaction StatementType = C.DUCKDB_STATEMENT_TYPE_TRANSACTION
	StatementTypeCopy        StatementType = C.DUCKDB_STATEMENT_TYPE_COPY
	StatementTypeAnalyze     StatementType = C.DUCKDB_STATEMENT_TYPE_ANALYZE
	StatementTypeVariableSet StatementType = C.DUCKDB_STATEMENT_TYPE_VARIABLE_SET
	StatementTypeCreateFunc  StatementType = C.DUCKDB_STATEMENT_TYPE_CREATE_FUNC
	StatementTypeDrop        StatementType = C.DUCKDB_STATEMENT_TYPE_DROP
	StatementTypeExport      StatementType = C.DUCKDB_STATEMENT_TYPE_EXPORT
	StatementTypePragma      StatementType = C.DUCKDB_STATEMENT_TYPE_PRAGMA
	StatementTypeVacuum      StatementType = C.DUCKDB_STATEMENT_TYPE_VACUUM
	StatementTypeCall        StatementType = C.DUCKDB_STATEMENT_TYPE_CALL
	StatementTypeSet         StatementType = C.DUCKDB_STATEMENT_TYPE_SET
	StatementTypeLoad        StatementType = C.DUCKDB_STATEMENT_TYPE_LOAD
	StatementTypeRelation    StatementType = C.DUCKDB_STATEMENT_TYPE_RELATION
	StatementTypeExtension   StatementType = C.DUCKDB_STATEMENT_TYPE_EXTENSION
	StatementTypeLogicalPlan StatementType = C.DUCKDB_STATEMENT_TYPE_LOGICAL_PLAN
	StatementTypeAttach      StatementType = C.DUCKDB_STATEMENT_TYPE_ATTACH
	StatementTypeDetach      StatementType = C.DUCKDB_STATEMENT_TYPE_DETACH
	StatementTypeMulti       StatementType = C.DUCKDB_STATEMENT_TYPE_MULTI
)

// ErrorType wraps duckdb_error_type.
type ErrorType = C.duckdb_error_type

const (
	ErrorTypeInvalid              ErrorType = C.DUCKDB_ERROR_INVALID
	ErrorTypeOutOfRange           ErrorType = C.DUCKDB_ERROR_OUT_OF_RANGE
	ErrorTypeConversion           ErrorType = C.DUCKDB_ERROR_CONVERSION
	ErrorTypeUnknownType          ErrorType = C.DUCKDB_ERROR_UNKNOWN_TYPE
	ErrorTypeDecimal              ErrorType = C.DUCKDB_ERROR_DECIMAL
	ErrorTypeMismatchType         ErrorType = C.DUCKDB_ERROR_MISMATCH_TYPE
	ErrorTypeDivideByZero         ErrorType = C.DUCKDB_ERROR_DIVIDE_BY_ZERO
	ErrorTypeObjectSize           ErrorType = C.DUCKDB_ERROR_OBJECT_SIZE
	ErrorTypeInvalidType          ErrorType = C.DUCKDB_ERROR_INVALID_TYPE
	ErrorTypeSerialization        ErrorType = C.DUCKDB_ERROR_SERIALIZATION
	ErrorTypeTransaction          ErrorType = C.DUCKDB_ERROR_TRANSACTION
	ErrorTypeNotImplemented       ErrorType = C.DUCKDB_ERROR_NOT_IMPLEMENTED
	ErrorTypeExpression           ErrorType = C.DUCKDB_ERROR_EXPRESSION
	ErrorTypeCatalog              ErrorType = C.DUCKDB_ERROR_CATALOG
	ErrorTypeParser               ErrorType = C.DUCKDB_ERROR_PARSER
	ErrorTypePlanner              ErrorType = C.DUCKDB_ERROR_PLANNER
	ErrorTypeScheduler            ErrorType = C.DUCKDB_ERROR_SCHEDULER
	ErrorTypeExecutor             ErrorType = C.DUCKDB_ERROR_EXECUTOR
	ErrorTypeConstraint           ErrorType = C.DUCKDB_ERROR_CONSTRAINT
	ErrorTypeIndex                ErrorType = C.DUCKDB_ERROR_INDEX
	ErrorTypeStat                 ErrorType = C.DUCKDB_ERROR_STAT
	ErrorTypeConnection           ErrorType = C.DUCKDB_ERROR_CONNECTION
	ErrorTypeSyntax               ErrorType = C.DUCKDB_ERROR_SYNTAX
	ErrorTypeSettings             ErrorType = C.DUCKDB_ERROR_SETTINGS
	ErrorTypeBinder               ErrorType = C.DUCKDB_ERROR_BINDER
	ErrorTypeNetwork              ErrorType = C.DUCKDB_ERROR_NETWORK
	ErrorTypeOptimizer            ErrorType = C.DUCKDB_ERROR_OPTIMIZER
	ErrorTypeNullPointer          ErrorType = C.DUCKDB_ERROR_NULL_POINTER
	ErrorTypeErrorIO              ErrorType = C.DUCKDB_ERROR_IO
	ErrorTypeInterrupt            ErrorType = C.DUCKDB_ERROR_INTERRUPT
	ErrorTypeFatal                ErrorType = C.DUCKDB_ERROR_FATAL
	ErrorTypeInternal             ErrorType = C.DUCKDB_ERROR_INTERNAL
	ErrorTypeInvalidInput         ErrorType = C.DUCKDB_ERROR_INVALID_INPUT
	ErrorTypeOutOfMemory          ErrorType = C.DUCKDB_ERROR_OUT_OF_MEMORY
	ErrorTypePermission           ErrorType = C.DUCKDB_ERROR_PERMISSION
	ErrorTypeParameterNotResolved ErrorType = C.DUCKDB_ERROR_PARAMETER_NOT_RESOLVED
	ErrorTypeParameterNotAllowed  ErrorType = C.DUCKDB_ERROR_PARAMETER_NOT_ALLOWED
	ErrorTypeDependency           ErrorType = C.DUCKDB_ERROR_DEPENDENCY
	ErrorTypeHTTP                 ErrorType = C.DUCKDB_ERROR_HTTP
	ErrorTypeMissingExtension     ErrorType = C.DUCKDB_ERROR_MISSING_EXTENSION
	ErrorTypeAutoload             ErrorType = C.DUCKDB_ERROR_AUTOLOAD
	ErrorTypeSequence             ErrorType = C.DUCKDB_ERROR_SEQUENCE
	ErrorTypeInvalidConfiguration ErrorType = C.DUCKDB_INVALID_CONFIGURATION
)

// CastMode wraps duckdb_cast_mode.
type CastMode = C.duckdb_cast_mode

const (
	CastModeNormal CastMode = C.DUCKDB_CAST_NORMAL
	CastModeTry    CastMode = C.DUCKDB_CAST_TRY
)

// ------------------------------------------------------------------ //
// Types
// ------------------------------------------------------------------ //

type IdxT = C.idx_t

type SelT = C.sel_t

// Types without internal pointers:

type (
	Date              = C.duckdb_date
	DateStruct        = C.duckdb_date_struct
	Time              = C.duckdb_time
	TimeStruct        = C.duckdb_time_struct
	TimeNS            = C.duckdb_time_ns
	TimeTZ            = C.duckdb_time_tz
	TimeTZStruct      = C.duckdb_time_tz_struct
	Timestamp         = C.duckdb_timestamp
	TimestampS        = C.duckdb_timestamp_s
	TimestampMS       = C.duckdb_timestamp_ms
	TimestampNS       = C.duckdb_timestamp_ns
	TimestampStruct   = C.duckdb_timestamp_struct
	Interval          = C.duckdb_interval
	HugeInt           = C.duckdb_hugeint
	UHugeInt          = C.duckdb_uhugeint
	Decimal           = C.duckdb_decimal
	QueryProgressType = C.duckdb_query_progress_type
	// StringT does not export New and Members.
	// Use the respective StringT functions to access / write to this type.
	StringT   = C.duckdb_string_t
	ListEntry = C.duckdb_list_entry
	// Blob does not export New and Members.
	// Use the respective Blob functions to access / write to this type.
	// This type must be destroyed with DestroyBlob.
	Blob = C.duckdb_blob
	// Bit does not export New and Members.
	// Use the respective Bit functions to access / write to this type.
	// This type must be destroyed with DestroyBit.
	Bit = C.duckdb_bit
	// BigNum does not export New and Members.
	// Use the respective BigNum functions to access / write to this type.
	// This type must be destroyed with DestroyBigNum.
	BigNum = C.duckdb_bignum
)

// TODO:
// duckdb_string
// duckdb_extension_access

// Helper functions for types without internal pointers:

// NewDate sets the members of a duckdb_date.
func NewDate(days int32) Date {
	return Date{days: C.int32_t(days)}
}

// DateMembers returns the days of a duckdb_date.
func DateMembers(date *Date) int32 {
	return int32(date.days)
}

// NewDateStruct sets the members of a duckdb_date_struct.
func NewDateStruct(year int32, month int8, day int8) DateStruct {
	return DateStruct{
		year:  C.int32_t(year),
		month: C.int8_t(month),
		day:   C.int8_t(day),
	}
}

// DateStructMembers returns the year, month, and day of a duckdb_date.
func DateStructMembers(date *DateStruct) (int32, int8, int8) {
	return int32(date.year), int8(date.month), int8(date.day)
}

// NewTime sets the members of a duckdb_time.
func NewTime(micros int64) Time {
	return Time{micros: C.int64_t(micros)}
}

// TimeMembers returns the micros of a duckdb_time.
func TimeMembers(ti *Time) int64 {
	return int64(ti.micros)
}

// NewTimeStruct sets the members of a duckdb_time_struct.
func NewTimeStruct(hour int8, min int8, sec int8, micros int32) TimeStruct {
	return TimeStruct{
		hour:   C.int8_t(hour),
		min:    C.int8_t(min),
		sec:    C.int8_t(sec),
		micros: C.int32_t(micros),
	}
}

// TimeStructMembers returns the hour, min, sec, and micros of a duckdb_time_struct.
func TimeStructMembers(ti *TimeStruct) (int8, int8, int8, int32) {
	return int8(ti.hour), int8(ti.min), int8(ti.sec), int32(ti.micros)
}

// NewTimeNS sets the members of a duckdb_time_ns.
func NewTimeNS(nanos int64) TimeNS {
	return TimeNS{nanos: C.int64_t(nanos)}
}

// TimeNSMembers returns the nanos of a duckdb_time_ns.
func TimeNSMembers(ti *TimeNS) int64 {
	return int64(ti.nanos)
}

// NewTimeTZ sets the members of a duckdb_time_tz.
func NewTimeTZ(bits uint64) TimeTZ {
	return TimeTZ{bits: C.uint64_t(bits)}
}

// TimeTZMembers returns the bits of a duckdb_time_tz.
func TimeTZMembers(ti *TimeTZ) uint64 {
	return uint64(ti.bits)
}

// NewTimeTZStruct sets the members of a duckdb_time_tz_struct.
func NewTimeTZStruct(ti TimeStruct, offset int32) TimeTZStruct {
	return TimeTZStruct{
		time:   ti,
		offset: C.int32_t(offset),
	}
}

// TimeTZStructMembers returns the time and offset of a duckdb_time_tz_struct.
func TimeTZStructMembers(ti *TimeTZStruct) (TimeStruct, int32) {
	return ti.time, int32(ti.offset)
}

// NewTimestamp sets the members of a duckdb_timestamp.
func NewTimestamp(micros int64) Timestamp {
	return Timestamp{micros: C.int64_t(micros)}
}

// TimestampMembers returns the micros of a duckdb_timestamp.
func TimestampMembers(ts *Timestamp) int64 {
	return int64(ts.micros)
}

// NewTimestampS sets the members of a duckdb_timestamp_s.
func NewTimestampS(seconds int64) TimestampS {
	return TimestampS{seconds: C.int64_t(seconds)}
}

// TimestampSMembers returns the seconds of a duckdb_timestamp_s.
func TimestampSMembers(ts *TimestampS) int64 {
	return int64(ts.seconds)
}

// NewTimestampMS sets the members of a duckdb_timestamp_ms.
func NewTimestampMS(millis int64) TimestampMS {
	return TimestampMS{millis: C.int64_t(millis)}
}

// TimestampMSMembers returns the millis of a duckdb_timestamp_ms.
func TimestampMSMembers(ts *TimestampMS) int64 {
	return int64(ts.millis)
}

// NewTimestampNS sets the members of a duckdb_timestamp_ns.
func NewTimestampNS(nanos int64) TimestampNS {
	return TimestampNS{nanos: C.int64_t(nanos)}
}

// TimestampNSMembers returns the nanos of a duckdb_timestamp_ns.
func TimestampNSMembers(ts *TimestampNS) int64 {
	return int64(ts.nanos)
}

// NewTimestampStruct sets the members of a duckdb_timestamp_struct.
func NewTimestampStruct(date DateStruct, ti TimeStruct) TimestampStruct {
	return TimestampStruct{
		date: date,
		time: ti,
	}
}

// TimestampStructMembers returns the date and time of a duckdb_timestamp_struct.
func TimestampStructMembers(ts *TimestampStruct) (DateStruct, TimeStruct) {
	return ts.date, ts.time
}

// NewInterval sets the members of a duckdb_interval.
func NewInterval(months int32, days int32, micros int64) Interval {
	return Interval{
		months: C.int32_t(months),
		days:   C.int32_t(days),
		micros: C.int64_t(micros),
	}
}

// IntervalMembers returns the months, days, and micros of a duckdb_interval.
func IntervalMembers(i *Interval) (int32, int32, int64) {
	return int32(i.months), int32(i.days), int64(i.micros)
}

// NewHugeInt sets the members of a duckdb_hugeint.
func NewHugeInt(lower uint64, upper int64) HugeInt {
	return HugeInt{
		lower: C.uint64_t(lower),
		upper: C.int64_t(upper),
	}
}

// HugeIntMembers returns the lower and upper of a duckdb_hugeint.
func HugeIntMembers(hi *HugeInt) (uint64, int64) {
	return uint64(hi.lower), int64(hi.upper)
}

// NewUHugeInt sets the members of a duckdb_uhugeint.
func NewUHugeInt(lower uint64, upper uint64) UHugeInt {
	return UHugeInt{
		lower: C.uint64_t(lower),
		upper: C.uint64_t(upper),
	}
}

// UHugeIntMembers returns the lower and upper of a duckdb_uhugeint.
func UHugeIntMembers(hi *UHugeInt) (uint64, uint64) {
	return uint64(hi.lower), uint64(hi.upper)
}

// NewDecimal sets the members of a duckdb_decimal.
func NewDecimal(width uint8, scale uint8, hi HugeInt) Decimal {
	return Decimal{
		width: C.uint8_t(width),
		scale: C.uint8_t(scale),
		value: hi,
	}
}

// DecimalMembers returns the width, scale, and value of a duckdb_decimal.
func DecimalMembers(d *Decimal) (uint8, uint8, HugeInt) {
	return uint8(d.width), uint8(d.scale), d.value
}

// NewQueryProgressType sets the members of a duckdb_query_progress_type.
func NewQueryProgressType(percentage float64, rowsProcessed uint64, totalRowsToProcess uint64) QueryProgressType {
	return QueryProgressType{
		percentage:            C.double(percentage),
		rows_processed:        C.uint64_t(rowsProcessed),
		total_rows_to_process: C.uint64_t(totalRowsToProcess),
	}
}

// QueryProgressTypeMembers returns the percentage, rows_processed, and total_rows_to_process of a duckdb_query_progress_type.
func QueryProgressTypeMembers(q *QueryProgressType) (float64, uint64, uint64) {
	return float64(q.percentage), uint64(q.rows_processed), uint64(q.total_rows_to_process)
}

// NewListEntry sets the members of a duckdb_list_entry.
func NewListEntry(offset uint64, length uint64) ListEntry {
	return ListEntry{
		offset: C.uint64_t(offset),
		length: C.uint64_t(length),
	}
}

// ListEntryMembers returns the offset and length of a duckdb_list_entry.
func ListEntryMembers(entry *ListEntry) (uint64, uint64) {
	return uint64(entry.offset), uint64(entry.length)
}

// Helper functions for types with internal fields that need freeing:

// DestroyBlob destroys the data field of duckdb_blob.
func DestroyBlob(b *Blob) {
	if b == nil {
		return
	}
	if debugMode {
		decrAllocCount("blob")
	}
	Free(b.data)
}

// DestroyBit destroys the data field of duckdb_bit.
func DestroyBit(b *Bit) {
	if b == nil {
		return
	}
	if debugMode {
		decrAllocCount("bit")
	}
	Free(unsafe.Pointer(b.data))
}

// DestroyBigNum destroys the data field of duckdb_bignum.
func DestroyBigNum(i *BigNum) {
	if i == nil {
		return
	}
	if debugMode {
		decrAllocCount("bigNum")
	}
	Free(unsafe.Pointer(i.data))
}

// Types with internal pointers:

// Column wraps duckdb_column.
// NOTE: Same limitations as Result.
type Column struct {
	data C.duckdb_column
}

// Result wraps duckdb_result.
// NOTE: Using 'type Result = C.duckdb_result' causes a somewhat mysterious
// 'runtime error: cgo argument has Go pointer to unpinned Go pointer'.
// See https://github.com/golang/go/issues/28606#issuecomment-2184269962.
// When using a type alias, duckdb_result itself contains a Go unsafe.Pointer for its 'void *internal_data' field.
type Result struct {
	data C.duckdb_result
}

// ------------------------------------------------------------------ //
// Pointer Types
// ------------------------------------------------------------------ //

// NOTE: No wrappings for function pointers.
// *duckdb_delete_callback_t
// *duckdb_copy_callback_t
// *duckdb_task_state
// *duckdb_scalar_function_bind_t
// *duckdb_scalar_function_t
// *duckdb_aggregate_state_size
// *duckdb_aggregate_init_t
// *duckdb_aggregate_destroy_t
// *duckdb_aggregate_update_t
// *duckdb_aggregate_combine_t
// *duckdb_aggregate_finalize_t
// *duckdb_table_function_bind_t
// *duckdb_table_function_init_t
// *duckdb_table_function_t
// *duckdb_cast_function_t
// *duckdb_replacement_callback_t

// NOTE: We export the Ptr of each wrapped type pointer to allow (void *) typedef's of callback functions.
// See https://golang.org/issue/19837 and https://golang.org/issue/19835.

// NOTE: For some types (e.g., Appender, but not Config) omitting the Ptr causes
// the same somewhat mysterious runtime error as described for Result.
// 'runtime error: cgo argument has Go pointer to unpinned Go pointer'.
// See https://github.com/golang/go/issues/28606#issuecomment-2184269962.
// When using a type alias, duckdb_result itself contains a Go unsafe.Pointer for its 'void *internal_ptr' field.

// TODO:
// *duckdb_task_state

// Vector wraps *duckdb_vector.
type Vector struct {
	Ptr unsafe.Pointer
}

func (vec *Vector) data() C.duckdb_vector {
	return C.duckdb_vector(vec.Ptr)
}

// SelectionVector wraps *duckdb_selection_vector.
type SelectionVector struct {
	Ptr unsafe.Pointer
}

func (sel *SelectionVector) data() C.duckdb_selection_vector {
	return C.duckdb_selection_vector(sel.Ptr)
}

// InstanceCache wraps *duckdb_instance_cache.
type InstanceCache struct {
	Ptr unsafe.Pointer
}

func (cache *InstanceCache) data() C.duckdb_instance_cache {
	return C.duckdb_instance_cache(cache.Ptr)
}

// Database wraps *duckdb_database.
type Database struct {
	Ptr unsafe.Pointer
}

func (db *Database) data() C.duckdb_database {
	return C.duckdb_database(db.Ptr)
}

// Connection wraps *duckdb_connection.
type Connection struct {
	Ptr unsafe.Pointer
}

func (conn *Connection) data() C.duckdb_connection {
	return C.duckdb_connection(conn.Ptr)
}

// ClientContext wraps *duckdb_client_context.
type ClientContext struct {
	Ptr unsafe.Pointer
}

func (ctx *ClientContext) data() C.duckdb_client_context {
	return C.duckdb_client_context(ctx.Ptr)
}

// PreparedStatement wraps *duckdb_prepared_statement.
type PreparedStatement struct {
	Ptr unsafe.Pointer
}

func (preparedStmt *PreparedStatement) data() C.duckdb_prepared_statement {
	return C.duckdb_prepared_statement(preparedStmt.Ptr)
}

// ExtractedStatements wraps *duckdb_extracted_statements.
type ExtractedStatements struct {
	Ptr unsafe.Pointer
}

func (extractedStmts *ExtractedStatements) data() C.duckdb_extracted_statements {
	return C.duckdb_extracted_statements(extractedStmts.Ptr)
}

// PendingResult wraps *duckdb_pending_result.
type PendingResult struct {
	Ptr unsafe.Pointer
}

func (pendingRes *PendingResult) data() C.duckdb_pending_result {
	return C.duckdb_pending_result(pendingRes.Ptr)
}

// Appender wraps *duckdb_appender.
type Appender struct {
	Ptr unsafe.Pointer
}

func (appender *Appender) data() C.duckdb_appender {
	return C.duckdb_appender(appender.Ptr)
}

// TableDescription wraps *duckdb_table_description.
type TableDescription struct {
	Ptr unsafe.Pointer
}

func (description *TableDescription) data() C.duckdb_table_description {
	return C.duckdb_table_description(description.Ptr)
}

// Config wraps *duckdb_config.
type Config struct {
	Ptr unsafe.Pointer
}

func (config *Config) data() C.duckdb_config {
	return C.duckdb_config(config.Ptr)
}

// LogicalType wraps *duckdb_logical_type.
type LogicalType struct {
	Ptr unsafe.Pointer
}

func (logicalType *LogicalType) data() C.duckdb_logical_type {
	return C.duckdb_logical_type(logicalType.Ptr)
}

// CreateTypeInfo wraps *duckdb_create_type_info.
type CreateTypeInfo struct {
	Ptr unsafe.Pointer
}

func (info *CreateTypeInfo) data() C.duckdb_create_type_info {
	return C.duckdb_create_type_info(info.Ptr)
}

// DataChunk wraps *duckdb_data_chunk.
type DataChunk struct {
	Ptr unsafe.Pointer
}

func (chunk *DataChunk) data() C.duckdb_data_chunk {
	return C.duckdb_data_chunk(chunk.Ptr)
}

// Value wraps *duckdb_value.
type Value struct {
	Ptr unsafe.Pointer
}

func (v *Value) data() C.duckdb_value {
	return C.duckdb_value(v.Ptr)
}

// ProfilingInfo wraps *duckdb_profiling_info.
type ProfilingInfo struct {
	Ptr unsafe.Pointer
}

func (info *ProfilingInfo) data() C.duckdb_profiling_info {
	return C.duckdb_profiling_info(info.Ptr)
}

// ErrorData wraps *duckdb_error_data.
type ErrorData struct {
	Ptr unsafe.Pointer
}

func (errorData *ErrorData) data() C.duckdb_error_data {
	return C.duckdb_error_data(errorData.Ptr)
}

// Expression wraps *duckdb_expression.
type Expression struct {
	Ptr unsafe.Pointer
}

func (expr *Expression) data() C.duckdb_expression {
	return C.duckdb_expression(expr.Ptr)
}

// TODO:
// *duckdb_extension_info

// FunctionInfo wraps *duckdb_function_info.
type FunctionInfo struct {
	Ptr unsafe.Pointer
}

func (info *FunctionInfo) data() C.duckdb_function_info {
	return C.duckdb_function_info(info.Ptr)
}

// ScalarFunction wraps *duckdb_scalar_function.
type ScalarFunction struct {
	Ptr unsafe.Pointer
}

func (f *ScalarFunction) data() C.duckdb_scalar_function {
	return C.duckdb_scalar_function(f.Ptr)
}

// ScalarFunctionSet wraps *duckdb_scalar_function_set.
type ScalarFunctionSet struct {
	Ptr unsafe.Pointer
}

func (set *ScalarFunctionSet) data() C.duckdb_scalar_function_set {
	return C.duckdb_scalar_function_set(set.Ptr)
}

// TODO:
// *duckdb_aggregate_function
// *duckdb_aggregate_function_set
// *duckdb_aggregate_state

// TableFunction wraps *duckdb_table_function.
type TableFunction struct {
	Ptr unsafe.Pointer
}

func (f *TableFunction) data() C.duckdb_table_function {
	return C.duckdb_table_function(f.Ptr)
}

// BindInfo wraps *duckdb_bind_info.
type BindInfo struct {
	Ptr unsafe.Pointer
}

func (info *BindInfo) data() C.duckdb_bind_info {
	return C.duckdb_bind_info(info.Ptr)
}

// InitInfo wraps *C.duckdb_init_info.
type InitInfo struct {
	Ptr unsafe.Pointer
}

func (info *InitInfo) data() C.duckdb_init_info {
	return C.duckdb_init_info(info.Ptr)
}

// TODO:
// *duckdb_cast_function

// ReplacementScanInfo wraps *duckdb_replacement_scan.
type ReplacementScanInfo struct {
	Ptr unsafe.Pointer
}

func (info *ReplacementScanInfo) data() C.duckdb_replacement_scan_info {
	return C.duckdb_replacement_scan_info(info.Ptr)
}

// Arrow wraps *duckdb_arrow.
type Arrow struct {
	Ptr unsafe.Pointer
}

func (arrow *Arrow) data() C.duckdb_arrow {
	return C.duckdb_arrow(arrow.Ptr)
}

// ArrowStream wraps *duckdb_arrow_stream.
type ArrowStream struct {
	Ptr unsafe.Pointer
}

func (stream *ArrowStream) data() C.duckdb_arrow_stream {
	return C.duckdb_arrow_stream(stream.Ptr)
}

// ArrowSchema wraps *duckdb_arrow_schema.
type ArrowSchema struct {
	Ptr unsafe.Pointer
}

func (schema *ArrowSchema) data() C.duckdb_arrow_schema {
	return C.duckdb_arrow_schema(schema.Ptr)
}

// ArrowConvertedSchema wraps *duckdb_arrow_converted_schema.
type ArrowConvertedSchema struct {
	Ptr unsafe.Pointer
}

func (schema *ArrowConvertedSchema) data() C.duckdb_arrow_converted_schema {
	return C.duckdb_arrow_converted_schema(schema.Ptr)
}

// ArrowArray wraps *duckdb_arrow_array.
type ArrowArray struct {
	Ptr unsafe.Pointer
}

func (array *ArrowArray) data() C.duckdb_arrow_array {
	return C.duckdb_arrow_array(array.Ptr)
}

// ArrowOptions wraps *duckdb_arrow_options.
type ArrowOptions struct {
	Ptr unsafe.Pointer
}

func (options *ArrowOptions) data() C.duckdb_arrow_options {
	return C.duckdb_arrow_options(options.Ptr)
}

// ------------------------------------------------------------------ //
// Functions
// ------------------------------------------------------------------ //

// ------------------------------------------------------------------ //
// Open Connect
// ------------------------------------------------------------------ //

// CreateInstanceCache wraps duckdb_create_instance_cache.
// The return value must be destroyed with DestroyInstanceCache.
func CreateInstanceCache() InstanceCache {
	cache := C.duckdb_create_instance_cache()
	if debugMode {
		incrAllocCount("cache")
	}
	return InstanceCache{
		Ptr: unsafe.Pointer(cache),
	}
}

// GetOrCreateFromCache wraps duckdb_get_or_create_from_cache.
// outDb must be closed with Close.
func GetOrCreateFromCache(cache InstanceCache, path string, outDb *Database, config Config, errMsg *string) State {
	cPath := C.CString(path)
	defer Free(unsafe.Pointer(cPath))
	var err *C.char
	defer Free(unsafe.Pointer(err))

	var db C.duckdb_database
	state := C.duckdb_get_or_create_from_cache(cache.data(), cPath, &db, config.data(), &err)
	outDb.Ptr = unsafe.Pointer(db)
	*errMsg = C.GoString(err)

	if debugMode {
		incrAllocCount("db")
	}
	return state
}

// DestroyInstanceCache wraps duckdb_destroy_instance_cache.
func DestroyInstanceCache(cache *InstanceCache) {
	if cache.Ptr == nil {
		return
	}
	if debugMode {
		decrAllocCount("cache")
	}
	data := cache.data()
	C.duckdb_destroy_instance_cache(&data)
	cache.Ptr = nil
}

// Open wraps duckdb_open.
// outDb must be closed with Close.
func Open(path string, outDb *Database) State {
	cPath := C.CString(path)
	defer Free(unsafe.Pointer(cPath))

	var db C.duckdb_database
	state := C.duckdb_open(cPath, &db)
	outDb.Ptr = unsafe.Pointer(db)

	if debugMode {
		incrAllocCount("db")
	}
	return state
}

// OpenExt wraps duckdb_open_ext.
// outDb must be closed with Close.
func OpenExt(path string, outDb *Database, config Config, errMsg *string) State {
	cPath := C.CString(path)
	defer Free(unsafe.Pointer(cPath))
	var err *C.char
	defer Free(unsafe.Pointer(err))

	var db C.duckdb_database
	state := C.duckdb_open_ext(cPath, &db, config.data(), &err)
	outDb.Ptr = unsafe.Pointer(db)
	*errMsg = C.GoString(err)

	if debugMode {
		incrAllocCount("db")
	}
	return state
}

// Close wraps duckdb_close.
func Close(db *Database) {
	if db.Ptr == nil {
		return
	}
	if debugMode {
		decrAllocCount("db")
	}
	data := db.data()
	C.duckdb_close(&data)
	db.Ptr = nil
}

// Connect wraps duckdb_connect.
// outConn must be disconnected with Disconnect.
func Connect(db Database, outConn *Connection) State {
	var conn C.duckdb_connection
	state := C.duckdb_connect(db.data(), &conn)
	outConn.Ptr = unsafe.Pointer(conn)
	if debugMode {
		incrAllocCount("conn")
	}
	return state
}

func Interrupt(conn Connection) {
	C.duckdb_interrupt(conn.data())
}

func QueryProgress(conn Connection) QueryProgressType {
	return C.duckdb_query_progress(conn.data())
}

// Disconnect wraps duckdb_disconnect.
func Disconnect(conn *Connection) {
	if conn.Ptr == nil {
		return
	}
	if debugMode {
		decrAllocCount("conn")
	}
	data := conn.data()
	C.duckdb_disconnect(&data)
	conn.Ptr = nil
}

// ConnectionGetClientContext wraps duckdb_connection_get_client_context.
// outCtx must be destroyed with DestroyClientContext.
func ConnectionGetClientContext(conn Connection, outCtx *ClientContext) {
	var ctx C.duckdb_client_context
	C.duckdb_connection_get_client_context(conn.data(), &ctx)
	outCtx.Ptr = unsafe.Pointer(ctx)
	if debugMode {
		incrAllocCount("ctx")
	}
}

// ConnectionGetArrowOptions wraps duckdb_connection_get_arrow_options.
// outOptions must be destroyed with DestroyArrowOptions.
func ConnectionGetArrowOptions(conn Connection, outOptions *ArrowOptions) {
	var options C.duckdb_arrow_options
	C.duckdb_connection_get_arrow_options(conn.data(), &options)
	outOptions.Ptr = unsafe.Pointer(options)
	if debugMode {
		incrAllocCount("arrowOptions")
	}
}

func ClientContextGetConnectionId(ctx ClientContext) IdxT {
	return C.duckdb_client_context_get_connection_id(ctx.data())
}

// DestroyClientContext wraps duckdb_destroy_client_context.
func DestroyClientContext(ctx *ClientContext) {
	if ctx.Ptr == nil {
		return
	}
	if debugMode {
		decrAllocCount("ctx")
	}
	data := ctx.data()
	C.duckdb_destroy_client_context(&data)
	ctx.Ptr = nil
}

// DestroyArrowOptions wraps duckdb_destroy_arrow_options.
func DestroyArrowOptions(options *ArrowOptions) {
	if options.Ptr == nil {
		return
	}
	if debugMode {
		decrAllocCount("arrowOptions")
	}
	data := options.data()
	C.duckdb_destroy_arrow_options(&data)
	options.Ptr = nil
}

func LibraryVersion() string {
	cStr := C.duckdb_library_version()
	defer Free(unsafe.Pointer(cStr))
	return C.GoString(cStr)
}

// GetTableNames wraps duckdb_get_table_names.
// The return value must be destroyed with DestroyValue.
func GetTableNames(conn Connection, query string, qualified bool) Value {
	cQuery := C.CString(query)
	defer Free(unsafe.Pointer(cQuery))
	v := C.duckdb_get_table_names(conn.data(), cQuery, C.bool(qualified))
	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

// ------------------------------------------------------------------ //
// Configuration
// ------------------------------------------------------------------ //

// CreateConfig wraps duckdb_create_config.
// outConfig must be destroyed with DestroyConfig.
func CreateConfig(outConfig *Config) State {
	var config C.duckdb_config
	state := C.duckdb_create_config(&config)
	outConfig.Ptr = unsafe.Pointer(config)
	if debugMode {
		incrAllocCount("config")
	}
	return state
}

func ConfigCount() uint64 {
	return uint64(C.duckdb_config_count())
}

func GetConfigFlag(index uint64, outName *string, outDescription *string) State {
	var name *C.char
	var description *C.char

	state := C.duckdb_get_config_flag(C.size_t(index), &name, &description)
	*outName = C.GoString(name)
	*outDescription = C.GoString(description)
	return state
}

func SetConfig(config Config, name string, option string) State {
	cName := C.CString(name)
	defer Free(unsafe.Pointer(cName))
	cOption := C.CString(option)
	defer Free(unsafe.Pointer(cOption))
	return C.duckdb_set_config(config.data(), cName, cOption)
}

// DestroyConfig wraps duckdb_destroy_config.
func DestroyConfig(config *Config) {
	if config.Ptr == nil {
		return
	}
	if debugMode {
		decrAllocCount("config")
	}
	data := config.data()
	C.duckdb_destroy_config(&data)
	config.Ptr = nil
}

// ------------------------------------------------------------------ //
// Error Data
// ------------------------------------------------------------------ //

// CreateErrorData wraps duckdb_create_error_data.
// The return value must be destroyed with DestroyErrorData.
func CreateErrorData(t ErrorType, msg string) ErrorData {
	cMsg := C.CString(msg)
	defer Free(unsafe.Pointer(cMsg))

	errorData := C.duckdb_create_error_data(t, cMsg)
	if debugMode {
		incrAllocCount("errorData")
	}
	return ErrorData{
		Ptr: unsafe.Pointer(errorData),
	}
}

// DestroyErrorData wraps duckdb_destroy_error_data.
func DestroyErrorData(errorData *ErrorData) {
	if errorData.Ptr == nil {
		return
	}
	if debugMode {
		decrAllocCount("errorData")
	}
	data := errorData.data()
	C.duckdb_destroy_error_data(&data)
	errorData.Ptr = nil
}

func ErrorDataErrorType(errorData ErrorData) ErrorType {
	return C.duckdb_error_data_error_type(errorData.data())
}

func ErrorDataMessage(errorData ErrorData) string {
	msg := C.duckdb_error_data_message(errorData.data())
	return C.GoString(msg)
}

func ErrorDataHasError(errorData ErrorData) bool {
	return bool(C.duckdb_error_data_has_error(errorData.data()))
}

// ------------------------------------------------------------------ //
// Query Execution
// ------------------------------------------------------------------ //

// Query wraps duckdb_query.
// outRes must be destroyed with DestroyResult.
func Query(conn Connection, query string, outRes *Result) State {
	if debugMode {
		incrAllocCount("res")
	}
	cQuery := C.CString(query)
	defer Free(unsafe.Pointer(cQuery))

	return C.duckdb_query(conn.data(), cQuery, &outRes.data)
}

// DestroyResult wraps duckdb_destroy_result.
func DestroyResult(res *Result) {
	if res == nil {
		return
	}
	if debugMode {
		decrAllocCount("res")
	}
	C.duckdb_destroy_result(&res.data)
	res = nil
}

func ColumnName(res *Result, col IdxT) string {
	name := C.duckdb_column_name(&res.data, col)
	return C.GoString(name)
}

func ColumnType(res *Result, col IdxT) Type {
	return C.duckdb_column_type(&res.data, col)
}

func ResultStatementType(res Result) StatementType {
	return C.duckdb_result_statement_type(res.data)
}

// ColumnLogicalType wraps duckdb_column_logical_type.
// The return value must be destroyed with DestroyLogicalType.
func ColumnLogicalType(res *Result, col IdxT) LogicalType {
	logicalType := C.duckdb_column_logical_type(&res.data, col)
	if debugMode {
		incrAllocCount("logicalType")
	}
	return LogicalType{
		Ptr: unsafe.Pointer(logicalType),
	}
}

// ResultGetArrowOptions wraps duckdb_result_get_arrow_options.
// The return value must be destroyed with DestroyArrowOptions.
func ResultGetArrowOptions(res *Result) ArrowOptions {
	options := C.duckdb_result_get_arrow_options(&res.data)
	if debugMode {
		incrAllocCount("arrowOptions")
	}
	return ArrowOptions{
		Ptr: unsafe.Pointer(options),
	}
}

func ColumnCount(res *Result) IdxT {
	return C.duckdb_column_count(&res.data)
}

func RowsChanged(res *Result) IdxT {
	return C.duckdb_rows_changed(&res.data)
}

func ResultError(res *Result) string {
	err := C.duckdb_result_error(&res.data)
	return C.GoString(err)
}

func ResultErrorType(res *Result) ErrorType {
	return C.duckdb_result_error_type(&res.data)
}

// ------------------------------------------------------------------ //
// Result Functions (many are deprecated)
// ------------------------------------------------------------------ //

// ResultGetChunk wraps duckdb_result_get_chunk.
// The return value must be destroyed with DestroyDataChunk.
func ResultGetChunk(res Result, index IdxT) DataChunk {
	chunk := C.duckdb_result_get_chunk(res.data, index)
	if debugMode {
		incrAllocCount("chunk")
	}
	return DataChunk{
		Ptr: unsafe.Pointer(chunk),
	}
}

func ResultChunkCount(res Result) IdxT {
	return C.duckdb_result_chunk_count(res.data)
}

func ResultReturnType(res Result) ResultType {
	return C.duckdb_result_return_type(res.data)
}

// ------------------------------------------------------------------ //
// Safe Fetch Functions (all deprecated)
// ------------------------------------------------------------------ //

func ValueInt64(res *Result, col IdxT, row IdxT) int64 {
	v := C.duckdb_value_int64(&res.data, col, row)
	return int64(v)
}

// ------------------------------------------------------------------ //
// Helpers
// ------------------------------------------------------------------ //

// TODO:
// duckdb_malloc

func Free(ptr unsafe.Pointer) {
	C.duckdb_free(ptr)
}

func VectorSize() IdxT {
	return C.duckdb_vector_size()
}

func StringIsInlined(strT StringT) bool {
	isInlined := C.duckdb_string_is_inlined(strT)
	return bool(isInlined)
}

func StringTLength(strT StringT) uint32 {
	length := C.duckdb_string_t_length(strT)
	return uint32(length)
}

func StringTData(strT *StringT) string {
	length := C.int(StringTLength(*strT))
	ptr := unsafe.Pointer(C.duckdb_string_t_data(strT))
	return string(C.GoBytes(ptr, length))
}

// ------------------------------------------------------------------ //
// Date Time Timestamp Helpers
// ------------------------------------------------------------------ //

func FromDate(date Date) DateStruct {
	return C.duckdb_from_date(date)
}

func ToDate(date DateStruct) Date {
	return C.duckdb_to_date(date)
}

func IsFiniteDate(date Date) bool {
	return bool(C.duckdb_is_finite_date(date))
}

func FromTime(ti Time) TimeStruct {
	return C.duckdb_from_time(ti)
}

func CreateTimeTZ(micros int64, offset int32) TimeTZ {
	return C.duckdb_create_time_tz(C.int64_t(micros), C.int32_t(offset))
}

func FromTimeTZ(ti TimeTZ) TimeTZStruct {
	return C.duckdb_from_time_tz(ti)
}

func ToTime(ti TimeStruct) Time {
	return C.duckdb_to_time(ti)
}

func FromTimestamp(ts Timestamp) TimestampStruct {
	return C.duckdb_from_timestamp(ts)
}

func ToTimestamp(ts TimestampStruct) Timestamp {
	return C.duckdb_to_timestamp(ts)
}

func IsFiniteTimestamp(ts Timestamp) bool {
	return bool(C.duckdb_is_finite_timestamp(ts))
}

func IsFiniteTimestampS(ts TimestampS) bool {
	return bool(C.duckdb_is_finite_timestamp_s(ts))
}

func IsFiniteTimestampMS(ts TimestampMS) bool {
	return bool(C.duckdb_is_finite_timestamp_ms(ts))
}

func IsFiniteTimestampNS(ts TimestampNS) bool {
	return bool(C.duckdb_is_finite_timestamp_ns(ts))
}

// ------------------------------------------------------------------ //
// Hugeint Helpers
// ------------------------------------------------------------------ //

func HugeIntToDouble(hi HugeInt) float64 {
	return float64(C.duckdb_hugeint_to_double(hi))
}

func DoubleToHugeInt(d float64) HugeInt {
	return C.duckdb_double_to_hugeint(C.double(d))
}

// ------------------------------------------------------------------ //
// Unsigned Hugeint Helpers
// ------------------------------------------------------------------ //

func UHugeIntToDouble(hi UHugeInt) float64 {
	return float64(C.duckdb_uhugeint_to_double(hi))
}

func DoubleToUHugeInt(d float64) UHugeInt {
	return C.duckdb_double_to_uhugeint(C.double(d))
}

// ------------------------------------------------------------------ //
// Decimal Helpers
// ------------------------------------------------------------------ //

func DoubleToDecimal(d float64, width uint8, scale uint8) Decimal {
	return C.duckdb_double_to_decimal(C.double(d), C.uint8_t(width), C.uint8_t(scale))
}

func DecimalToDouble(d Decimal) float64 {
	return float64(C.duckdb_decimal_to_double(d))
}

// ------------------------------------------------------------------ //
// Prepared Statements
// ------------------------------------------------------------------ //

// Prepare wraps duckdb_prepare.
// outPreparedStmt must be destroyed with DestroyPrepare.
func Prepare(conn Connection, query string, outPreparedStmt *PreparedStatement) State {
	cQuery := C.CString(query)
	defer Free(unsafe.Pointer(cQuery))

	var preparedStmt C.duckdb_prepared_statement
	state := C.duckdb_prepare(conn.data(), cQuery, &preparedStmt)
	outPreparedStmt.Ptr = unsafe.Pointer(preparedStmt)
	if debugMode {
		incrAllocCount("preparedStmt")
	}
	return state
}

// DestroyPrepare wraps duckdb_destroy_prepare.
func DestroyPrepare(preparedStmt *PreparedStatement) {
	if preparedStmt.Ptr == nil {
		return
	}
	if debugMode {
		decrAllocCount("preparedStmt")
	}
	data := preparedStmt.data()
	C.duckdb_destroy_prepare(&data)
	preparedStmt.Ptr = nil
}

func PrepareError(preparedStmt PreparedStatement) string {
	err := C.duckdb_prepare_error(preparedStmt.data())
	return C.GoString(err)
}

func NParams(preparedStmt PreparedStatement) IdxT {
	return C.duckdb_nparams(preparedStmt.data())
}

func ParameterName(preparedStmt PreparedStatement, index IdxT) string {
	cName := C.duckdb_parameter_name(preparedStmt.data(), index)
	defer Free(unsafe.Pointer(cName))
	return C.GoString(cName)
}

func ParamType(preparedStmt PreparedStatement, index IdxT) Type {
	return C.duckdb_param_type(preparedStmt.data(), index)
}

// ParamLogicalType wraps duckdb_param_logical_type.
// The return value must be destroyed with DestroyLogicalType.
func ParamLogicalType(preparedStmt PreparedStatement, index IdxT) LogicalType {
	logicalType := C.duckdb_param_logical_type(preparedStmt.data(), index)
	if debugMode {
		incrAllocCount("logicalType")
	}
	return LogicalType{
		Ptr: unsafe.Pointer(logicalType),
	}
}

func ClearBindings(preparedStmt PreparedStatement) State {
	return C.duckdb_clear_bindings(preparedStmt.data())
}

func PreparedStatementType(preparedStmt PreparedStatement) StatementType {
	return C.duckdb_prepared_statement_type(preparedStmt.data())
}

func PreparedStatementColumnCount(preparedStmt PreparedStatement) IdxT {
	return C.duckdb_prepared_statement_column_count(preparedStmt.data())
}

func PreparedStatementColumnName(preparedStmt PreparedStatement, index IdxT) string {
	name := C.duckdb_prepared_statement_column_name(preparedStmt.data(), index)
	return C.GoString(name)
}

// PreparedStatementColumnLogicalType wraps duckdb_prepared_statement_column_logical_type.
// The return value must be destroyed with DestroyLogicalType.
func PreparedStatementColumnLogicalType(preparedStmt PreparedStatement, index IdxT) LogicalType {
	logicalType := C.duckdb_prepared_statement_column_logical_type(preparedStmt.data(), index)
	if debugMode {
		incrAllocCount("logicalType")
	}
	return LogicalType{
		Ptr: unsafe.Pointer(logicalType),
	}
}

func PreparedStatementColumnType(preparedStmt PreparedStatement, index IdxT) Type {
	return C.duckdb_prepared_statement_column_type(preparedStmt.data(), index)
}

// ------------------------------------------------------------------ //
// Bind Values To Prepared Statements
// ------------------------------------------------------------------ //

func BindValue(preparedStmt PreparedStatement, index IdxT, v Value) State {
	return C.duckdb_bind_value(preparedStmt.data(), index, v.data())
}

func BindParameterIndex(preparedStmt PreparedStatement, outIndex *IdxT, name string) State {
	cName := C.CString(name)
	defer Free(unsafe.Pointer(cName))
	return C.duckdb_bind_parameter_index(preparedStmt.data(), outIndex, cName)
}

func BindBoolean(preparedStmt PreparedStatement, index IdxT, v bool) State {
	return C.duckdb_bind_boolean(preparedStmt.data(), index, C.bool(v))
}

func BindInt8(preparedStmt PreparedStatement, index IdxT, v int8) State {
	return C.duckdb_bind_int8(preparedStmt.data(), index, C.int8_t(v))
}

func BindInt16(preparedStmt PreparedStatement, index IdxT, v int16) State {
	return C.duckdb_bind_int16(preparedStmt.data(), index, C.int16_t(v))
}

func BindInt32(preparedStmt PreparedStatement, index IdxT, v int32) State {
	return C.duckdb_bind_int32(preparedStmt.data(), index, C.int32_t(v))
}

func BindInt64(preparedStmt PreparedStatement, index IdxT, v int64) State {
	return C.duckdb_bind_int64(preparedStmt.data(), index, C.int64_t(v))
}

func BindHugeInt(preparedStmt PreparedStatement, index IdxT, v HugeInt) State {
	return C.duckdb_bind_hugeint(preparedStmt.data(), index, v)
}

func BindUHugeInt(preparedStmt PreparedStatement, index IdxT, v UHugeInt) State {
	return C.duckdb_bind_uhugeint(preparedStmt.data(), index, v)
}

func BindDecimal(preparedStmt PreparedStatement, index IdxT, v Decimal) State {
	return C.duckdb_bind_decimal(preparedStmt.data(), index, v)
}

func BindUInt8(preparedStmt PreparedStatement, index IdxT, v uint8) State {
	return C.duckdb_bind_uint8(preparedStmt.data(), index, C.uint8_t(v))
}

func BindUInt16(preparedStmt PreparedStatement, index IdxT, v uint16) State {
	return C.duckdb_bind_uint16(preparedStmt.data(), index, C.uint16_t(v))
}

func BindUInt32(preparedStmt PreparedStatement, index IdxT, v uint32) State {
	return C.duckdb_bind_uint32(preparedStmt.data(), index, C.uint32_t(v))
}

func BindUInt64(preparedStmt PreparedStatement, index IdxT, v uint64) State {
	return C.duckdb_bind_uint64(preparedStmt.data(), index, C.uint64_t(v))
}

func BindFloat(preparedStmt PreparedStatement, index IdxT, v float32) State {
	return C.duckdb_bind_float(preparedStmt.data(), index, C.float(v))
}

func BindDouble(preparedStmt PreparedStatement, index IdxT, v float64) State {
	return C.duckdb_bind_double(preparedStmt.data(), index, C.double(v))
}

func BindDate(preparedStmt PreparedStatement, index IdxT, v Date) State {
	return C.duckdb_bind_date(preparedStmt.data(), index, v)
}

func BindTime(preparedStmt PreparedStatement, index IdxT, v Time) State {
	return C.duckdb_bind_time(preparedStmt.data(), index, v)
}

func BindTimestamp(preparedStmt PreparedStatement, index IdxT, v Timestamp) State {
	return C.duckdb_bind_timestamp(preparedStmt.data(), index, v)
}

func BindTimestampTZ(preparedStmt PreparedStatement, index IdxT, v Timestamp) State {
	return C.duckdb_bind_timestamp_tz(preparedStmt.data(), index, v)
}

func BindInterval(preparedStmt PreparedStatement, index IdxT, v Interval) State {
	return C.duckdb_bind_interval(preparedStmt.data(), index, v)
}

func BindVarchar(preparedStmt PreparedStatement, index IdxT, v string) State {
	cStr := C.CString(v)
	defer Free(unsafe.Pointer(cStr))
	return C.duckdb_bind_varchar(preparedStmt.data(), index, cStr)
}

func BindVarcharLength(preparedStmt PreparedStatement, index IdxT, v string, length IdxT) State {
	cStr := C.CString(v)
	defer Free(unsafe.Pointer(cStr))
	return C.duckdb_bind_varchar_length(preparedStmt.data(), index, cStr, length)
}

func BindBlob(preparedStmt PreparedStatement, index IdxT, v []byte) State {
	cBytes := C.CBytes(v)
	defer Free(cBytes)
	return C.duckdb_bind_blob(preparedStmt.data(), index, cBytes, IdxT(len(v)))
}

func BindNull(preparedStmt PreparedStatement, index IdxT) State {
	return C.duckdb_bind_null(preparedStmt.data(), index)
}

// ------------------------------------------------------------------ //
// Execute Prepared Statements (many are deprecated)
// ------------------------------------------------------------------ //

// ExecutePrepared wraps duckdb_execute_prepared.
// outRes must be destroyed with DestroyResult.
func ExecutePrepared(preparedStmt PreparedStatement, outRes *Result) State {
	if debugMode {
		incrAllocCount("res")
	}
	return C.duckdb_execute_prepared(preparedStmt.data(), &outRes.data)
}

// ------------------------------------------------------------------ //
// Extract Statements
// ------------------------------------------------------------------ //

// ExtractStatements wraps duckdb_extract_statements.
// outExtractedStmts must be destroyed with DestroyExtracted.
func ExtractStatements(conn Connection, query string, outExtractedStmts *ExtractedStatements) IdxT {
	cQuery := C.CString(query)
	defer Free(unsafe.Pointer(cQuery))

	var extractedStmts C.duckdb_extracted_statements
	count := C.duckdb_extract_statements(conn.data(), cQuery, &extractedStmts)
	outExtractedStmts.Ptr = unsafe.Pointer(extractedStmts)
	if debugMode {
		incrAllocCount("extractedStmts")
	}
	return count
}

// PrepareExtractedStatement wraps duckdb_prepare_extracted_statement.
// outPreparedStmt must be destroyed with DestroyPrepare.
func PrepareExtractedStatement(conn Connection, extractedStmts ExtractedStatements, index IdxT, outPreparedStmt *PreparedStatement) State {
	var preparedStmt C.duckdb_prepared_statement
	state := C.duckdb_prepare_extracted_statement(conn.data(), extractedStmts.data(), index, &preparedStmt)
	outPreparedStmt.Ptr = unsafe.Pointer(preparedStmt)
	if debugMode {
		incrAllocCount("preparedStmt")
	}
	return state
}

func ExtractStatementsError(extractedStmts ExtractedStatements) string {
	err := C.duckdb_extract_statements_error(extractedStmts.data())
	return C.GoString(err)
}

// DestroyExtracted wraps duckdb_destroy_extracted.
func DestroyExtracted(extractedStmts *ExtractedStatements) {
	if extractedStmts.Ptr == nil {
		return
	}
	if debugMode {
		decrAllocCount("extractedStmts")
	}
	data := extractedStmts.data()
	C.duckdb_destroy_extracted(&data)
	extractedStmts.Ptr = nil
}

// ------------------------------------------------------------------ //
// Pending Result Interface
// ------------------------------------------------------------------ //

// PendingPrepared wraps duckdb_pending_prepared.
// outPendingRes must be destroyed with DestroyPending.
func PendingPrepared(preparedStmt PreparedStatement, outPendingRes *PendingResult) State {
	var pendingRes C.duckdb_pending_result
	state := C.duckdb_pending_prepared(preparedStmt.data(), &pendingRes)
	outPendingRes.Ptr = unsafe.Pointer(pendingRes)
	if debugMode {
		incrAllocCount("pendingRes")
	}
	return state
}

// DestroyPending wraps duckdb_destroy_pending.
func DestroyPending(pendingRes *PendingResult) {
	if pendingRes.Ptr == nil {
		return
	}
	if debugMode {
		decrAllocCount("pendingRes")
	}
	data := pendingRes.data()
	C.duckdb_destroy_pending(&data)
	pendingRes.Ptr = nil
}

func PendingError(pendingRes PendingResult) string {
	err := C.duckdb_pending_error(pendingRes.data())
	return C.GoString(err)
}

func PendingExecuteTask(pendingRes PendingResult) PendingState {
	return C.duckdb_pending_execute_task(pendingRes.data())
}

func PendingExecuteCheckState(pendingRes PendingResult) PendingState {
	return C.duckdb_pending_execute_check_state(pendingRes.data())
}

// ExecutePending wraps duckdb_execute_pending.
// outRes must be destroyed with DestroyResult.
func ExecutePending(res PendingResult, outRes *Result) State {
	if debugMode {
		incrAllocCount("res")
	}
	return C.duckdb_execute_pending(res.data(), &outRes.data)
}

func PendingExecutionIsFinished(state PendingState) bool {
	return bool(C.duckdb_pending_execution_is_finished(state))
}

// ------------------------------------------------------------------ //
// Value Interface
// ------------------------------------------------------------------ //

// DestroyValue wraps duckdb_destroy_value.
func DestroyValue(v *Value) {
	if v.Ptr == nil {
		return
	}
	if debugMode {
		decrAllocCount("v")
	}
	data := v.data()
	C.duckdb_destroy_value(&data)
	v.Ptr = nil
}

// CreateVarchar wraps duckdb_create_varchar.
// The return value must be destroyed with DestroyValue.
func CreateVarchar(str string) Value {
	cStr := C.CString(str)
	defer Free(unsafe.Pointer(cStr))
	v := C.duckdb_create_varchar(cStr)
	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

// CreateVarcharLength wraps duckdb_create_varchar_length.
// The return value must be destroyed with DestroyValue.
func CreateVarcharLength(str string, length IdxT) Value {
	cStr := C.CString(str)
	defer Free(unsafe.Pointer(cStr))
	v := C.duckdb_create_varchar_length(cStr, length)
	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

// CreateBool wraps duckdb_create_bool.
// The return value must be destroyed with DestroyValue.
func CreateBool(val bool) Value {
	v := C.duckdb_create_bool(C.bool(val))
	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

// CreateInt8 wraps duckdb_create_int8.
// The return value must be destroyed with DestroyValue.
func CreateInt8(val int8) Value {
	v := C.duckdb_create_int8(C.int8_t(val))
	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

// CreateUInt8 wraps duckdb_create_uint8.
// The return value must be destroyed with DestroyValue.
func CreateUInt8(val uint8) Value {
	v := C.duckdb_create_uint8(C.uint8_t(val))
	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

// CreateInt16 wraps duckdb_create_int16.
// The return value must be destroyed with DestroyValue.
func CreateInt16(val int16) Value {
	v := C.duckdb_create_int16(C.int16_t(val))
	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

// CreateUInt16 wraps duckdb_create_uint16.
// The return value must be destroyed with DestroyValue.
func CreateUInt16(val uint16) Value {
	v := C.duckdb_create_uint16(C.uint16_t(val))
	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

// CreateInt32 wraps duckdb_create_int32.
// The return value must be destroyed with DestroyValue.
func CreateInt32(val int32) Value {
	v := C.duckdb_create_int32(C.int32_t(val))
	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

// CreateUInt32 wraps duckdb_create_uint32.
// The return value must be destroyed with DestroyValue.
func CreateUInt32(val uint32) Value {
	v := C.duckdb_create_uint32(C.uint32_t(val))
	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

// CreateUInt64 wraps duckdb_create_uint64.
// The return value must be destroyed with DestroyValue.
func CreateUInt64(val uint64) Value {
	v := C.duckdb_create_uint64(C.uint64_t(val))
	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

// CreateInt64 wraps duckdb_create_int64.
// The return value must be destroyed with DestroyValue.
func CreateInt64(val int64) Value {
	v := C.duckdb_create_int64(C.int64_t(val))
	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

// CreateHugeInt wraps duckdb_create_hugeint.
// The return value must be destroyed with DestroyValue.
func CreateHugeInt(val HugeInt) Value {
	v := C.duckdb_create_hugeint(val)
	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

// CreateUHugeInt wraps duckdb_create_uhugeint.
// The return value must be destroyed with DestroyValue.
func CreateUHugeInt(val UHugeInt) Value {
	v := C.duckdb_create_uhugeint(val)
	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

// CreateBigNum wraps duckdb_create_bignum.
// The return value must be destroyed with DestroyValue.
func CreateBigNum(val BigNum) Value {
	v := C.duckdb_create_bignum(val)
	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

// CreateDecimal wraps duckdb_create_decimal.
// The return value must be destroyed with DestroyValue.
func CreateDecimal(val Decimal) Value {
	v := C.duckdb_create_decimal(val)
	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

// CreateFloat wraps duckdb_create_float.
// The return value must be destroyed with DestroyValue.
func CreateFloat(val float32) Value {
	v := C.duckdb_create_float(C.float(val))
	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

// CreateDouble wraps duckdb_create_double.
// The return value must be destroyed with DestroyValue.
func CreateDouble(val float64) Value {
	v := C.duckdb_create_double(C.double(val))
	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

// CreateDate wraps duckdb_create_date.
// The return value must be destroyed with DestroyValue.
func CreateDate(val Date) Value {
	v := C.duckdb_create_date(val)
	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

// CreateTime wraps duckdb_create_time.
// The return value must be destroyed with DestroyValue.
func CreateTime(val Time) Value {
	v := C.duckdb_create_time(val)
	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

// CreateTimeNS wraps duckdb_create_time_ns.
// The return value must be destroyed with DestroyValue.
func CreateTimeNS(val TimeNS) Value {
	v := C.duckdb_create_time_ns(val)
	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

// CreateTimeTZValue wraps duckdb_create_time_tz_value.
// The return value must be destroyed with DestroyValue.
func CreateTimeTZValue(timeTZ TimeTZ) Value {
	v := C.duckdb_create_time_tz_value(timeTZ)
	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

// CreateTimestamp wraps duckdb_create_timestamp.
// The return value must be destroyed with DestroyValue.
func CreateTimestamp(val Timestamp) Value {
	v := C.duckdb_create_timestamp(val)
	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

// CreateTimestampTZ wraps duckdb_create_timestamp_tz.
// The return value must be destroyed with DestroyValue.
func CreateTimestampTZ(val Timestamp) Value {
	v := C.duckdb_create_timestamp_tz(val)
	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

// CreateTimestampS wraps duckdb_create_timestamp_s.
// The return value must be destroyed with DestroyValue.
func CreateTimestampS(val TimestampS) Value {
	v := C.duckdb_create_timestamp_s(val)
	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

// CreateTimestampMS wraps duckdb_create_timestamp_ms.
// The return value must be destroyed with DestroyValue.
func CreateTimestampMS(val TimestampMS) Value {
	v := C.duckdb_create_timestamp_ms(val)
	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

// CreateTimestampNS wraps duckdb_create_timestamp_ns.
// The return value must be destroyed with DestroyValue.
func CreateTimestampNS(val TimestampNS) Value {
	v := C.duckdb_create_timestamp_ns(val)
	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

// CreateInterval wraps duckdb_create_interval.
// The return value must be destroyed with DestroyValue.
func CreateInterval(val Interval) Value {
	v := C.duckdb_create_interval(val)
	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

// CreateBlob wraps duckdb_create_blob.
// The return value must be destroyed with DestroyValue.
func CreateBlob(val []byte) Value {
	cBytes := (*C.uint8_t)(C.CBytes(val))
	defer Free(unsafe.Pointer(cBytes))

	v := C.duckdb_create_blob(cBytes, IdxT(len(val)))
	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

// CreateBit wraps duckdb_create_bit.
// The return value must be destroyed with DestroyValue.
func CreateBit(val Bit) Value {
	v := C.duckdb_create_bit(val)
	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

// CreateUUID wraps duckdb_create_uuid.
// The return value must be destroyed with DestroyValue.
func CreateUUID(val UHugeInt) Value {
	v := C.duckdb_create_uuid(val)
	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

func GetBool(v Value) bool {
	val := C.duckdb_get_bool(v.data())
	return bool(val)
}

func GetInt8(v Value) int8 {
	val := C.duckdb_get_int8(v.data())
	return int8(val)
}

func GetUInt8(v Value) uint8 {
	val := C.duckdb_get_uint8(v.data())
	return uint8(val)
}

func GetInt16(v Value) int16 {
	val := C.duckdb_get_int16(v.data())
	return int16(val)
}

func GetUInt16(v Value) uint16 {
	val := C.duckdb_get_uint16(v.data())
	return uint16(val)
}

func GetInt32(v Value) int32 {
	val := C.duckdb_get_int32(v.data())
	return int32(val)
}

func GetUInt32(v Value) uint32 {
	val := C.duckdb_get_uint32(v.data())
	return uint32(val)
}

func GetInt64(v Value) int64 {
	val := C.duckdb_get_int64(v.data())
	return int64(val)
}

func GetUInt64(v Value) uint64 {
	val := C.duckdb_get_uint64(v.data())
	return uint64(val)
}

func GetHugeInt(v Value) HugeInt {
	return C.duckdb_get_hugeint(v.data())
}

func GetUHugeInt(v Value) UHugeInt {
	return C.duckdb_get_uhugeint(v.data())
}

// GetBigNum wraps duckdb_get_bignum.
// The return value must be destroyed with DestroyBigNum.
func GetBigNum(v Value) BigNum {
	if debugMode {
		incrAllocCount("bigNum")
	}
	return C.duckdb_get_bignum(v.data())
}

func GetDecimal(v Value) Decimal {
	return C.duckdb_get_decimal(v.data())
}

func GetFloat(v Value) float32 {
	val := C.duckdb_get_float(v.data())
	return float32(val)
}

func GetDouble(v Value) float64 {
	val := C.duckdb_get_double(v.data())
	return float64(val)
}

func GetDate(v Value) Date {
	return C.duckdb_get_date(v.data())
}

func GetTime(v Value) Time {
	return C.duckdb_get_time(v.data())
}

func GetTimeNS(v Value) TimeNS {
	return C.duckdb_get_time_ns(v.data())
}

func GetTimeTZ(v Value) TimeTZ {
	return C.duckdb_get_time_tz(v.data())
}

func GetTimestamp(v Value) Timestamp {
	return C.duckdb_get_timestamp(v.data())
}

func GetTimestampTZ(v Value) Timestamp {
	return C.duckdb_get_timestamp_tz(v.data())
}

func GetTimestampS(v Value) TimestampS {
	return C.duckdb_get_timestamp_s(v.data())
}

func GetTimestampMS(v Value) TimestampMS {
	return C.duckdb_get_timestamp_ms(v.data())
}

func GetTimestampNS(v Value) TimestampNS {
	return C.duckdb_get_timestamp_ns(v.data())
}

func GetInterval(v Value) Interval {
	return C.duckdb_get_interval(v.data())
}

// GetValueType wraps duckdb_get_value_type.
// The return value must be destroyed with DestroyLogicalType.
func GetValueType(v Value) LogicalType {
	logicalType := C.duckdb_get_value_type(v.data())
	if debugMode {
		incrAllocCount("logicalType")
	}
	return LogicalType{
		Ptr: unsafe.Pointer(logicalType),
	}
}

// GetBlob wraps duckdb_get_blob.
// The return value must be destroyed with DestroyBlob.
func GetBlob(v Value) Blob {
	if debugMode {
		incrAllocCount("blob")
	}
	return C.duckdb_get_blob(v.data())
}

// GetBit wraps duckdb_get_bit.
// The return value must be destroyed with DestroyBit.
func GetBit(v Value) Bit {
	if debugMode {
		incrAllocCount("bit")
	}
	return C.duckdb_get_bit(v.data())
}

func GetUUID(v Value) UHugeInt {
	return C.duckdb_get_uuid(v.data())
}

func GetVarchar(v Value) string {
	cStr := C.duckdb_get_varchar(v.data())
	defer Free(unsafe.Pointer(cStr))
	return C.GoString(cStr)
}

// CreateStructValue wraps duckdb_create_struct_value.
// The return value must be destroyed with DestroyValue.
func CreateStructValue(logicalType LogicalType, values []Value) Value {
	valuesPtr := allocValues(values)
	defer Free(unsafe.Pointer(valuesPtr))

	v := C.duckdb_create_struct_value(logicalType.data(), valuesPtr)

	if debugMode {
		incrAllocCount("v")
	}

	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

// CreateListValue wraps duckdb_create_list_value.
// The return value must be destroyed with DestroyValue.
func CreateListValue(logicalType LogicalType, values []Value) Value {
	valuesPtr := allocValues(values)
	defer Free(unsafe.Pointer(valuesPtr))

	v := C.duckdb_create_list_value(logicalType.data(), valuesPtr, IdxT(len(values)))

	if debugMode {
		incrAllocCount("v")
	}

	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

// CreateArrayValue wraps duckdb_create_array_value.
// The return value must be destroyed with DestroyValue.
func CreateArrayValue(logicalType LogicalType, values []Value) Value {
	valuesPtr := allocValues(values)
	defer Free(unsafe.Pointer(valuesPtr))

	v := C.duckdb_create_array_value(logicalType.data(), valuesPtr, IdxT(len(values)))

	if debugMode {
		incrAllocCount("v")
	}

	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

// CreateMapValue wraps duckdb_create_map_value.
// The return value must be destroyed with DestroyValue.
func CreateMapValue(logicalType LogicalType, keys []Value, values []Value) Value {
	keyValuesPtr := allocValues(values)
	defer Free(unsafe.Pointer(keyValuesPtr))

	valueValuesPtr := allocValues(values)
	defer Free(unsafe.Pointer(valueValuesPtr))

	m := C.duckdb_create_map_value(logicalType.data(), keyValuesPtr, valueValuesPtr, IdxT(len(keys)))

	if debugMode {
		incrAllocCount("v")
	}

	return Value{
		Ptr: unsafe.Pointer(m),
	}
}

// CreateUnionValue wraps duckdb_create_union_value.
// The return value must be destroyed with DestroyValue.
func CreateUnionValue(logicalType LogicalType, tag IdxT, value Value) Value {
	v := C.duckdb_create_union_value(logicalType.data(), tag, value.data())
	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

func GetMapSize(v Value) IdxT {
	return C.duckdb_get_map_size(v.data())
}

// GetMapKey wraps duckdb_get_map_key.
// The return value must be destroyed with DestroyValue.
func GetMapKey(v Value, index IdxT) Value {
	value := C.duckdb_get_map_key(v.data(), index)
	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(value),
	}
}

// GetMapValue wraps duckdb_get_map_value.
// The return value must be destroyed with DestroyValue.
func GetMapValue(v Value, index IdxT) Value {
	value := C.duckdb_get_map_value(v.data(), index)
	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(value),
	}
}

func IsNullValue(v Value) bool {
	return bool(C.duckdb_is_null_value(v.data()))
}

// CreateNullValue wraps duckdb_create_null_value.
// The return value must be destroyed with DestroyValue.
func CreateNullValue() Value {
	v := C.duckdb_create_null_value()
	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

func GetListSize(v Value) IdxT {
	return C.duckdb_get_list_size(v.data())
}

// GetListChild wraps duckdb_get_list_child.
// The return value must be destroyed with DestroyValue.
func GetListChild(val Value, index IdxT) Value {
	v := C.duckdb_get_list_child(val.data(), index)
	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

// CreateEnumValue wraps duckdb_create_enum_value.
// The return value must be destroyed with DestroyValue.
func CreateEnumValue(logicalType LogicalType, val uint64) Value {
	v := C.duckdb_create_enum_value(logicalType.data(), C.uint64_t(val))
	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

func GetEnumValue(v Value) uint64 {
	return uint64(C.duckdb_get_enum_value(v.data()))
}

// GetStructChild wraps duckdb_get_struct_child.
// The return value must be destroyed with DestroyValue.
func GetStructChild(val Value, index IdxT) Value {
	v := C.duckdb_get_struct_child(val.data(), index)
	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

func ValueToString(val Value) string {
	str := C.duckdb_value_to_string(val.data())
	defer Free(unsafe.Pointer(str))
	return C.GoString(str)
}

// ------------------------------------------------------------------ //
// Logical Type Interface
// ------------------------------------------------------------------ //

// CreateLogicalType wraps duckdb_create_logical_type.
// The return value must be destroyed with DestroyLogicalType.
func CreateLogicalType(t Type) LogicalType {
	logicalType := C.duckdb_create_logical_type(t)
	if debugMode {
		incrAllocCount("logicalType")
	}
	return LogicalType{
		Ptr: unsafe.Pointer(logicalType),
	}
}

func LogicalTypeGetAlias(logicalType LogicalType) string {
	alias := C.duckdb_logical_type_get_alias(logicalType.data())
	defer Free(unsafe.Pointer(alias))
	return C.GoString(alias)
}

func LogicalTypeSetAlias(logicalType LogicalType, alias string) {
	cAlias := C.CString(alias)
	defer Free(unsafe.Pointer(cAlias))
	C.duckdb_logical_type_set_alias(logicalType.data(), cAlias)
}

// CreateListType wraps duckdb_create_list_type.
// The return value must be destroyed with DestroyLogicalType.
func CreateListType(child LogicalType) LogicalType {
	logicalType := C.duckdb_create_list_type(child.data())
	if debugMode {
		incrAllocCount("logicalType")
	}
	return LogicalType{
		Ptr: unsafe.Pointer(logicalType),
	}
}

// CreateArrayType wraps duckdb_create_array_type.
// The return value must be destroyed with DestroyLogicalType.
func CreateArrayType(child LogicalType, size IdxT) LogicalType {
	logicalType := C.duckdb_create_array_type(child.data(), size)
	if debugMode {
		incrAllocCount("logicalType")
	}
	return LogicalType{
		Ptr: unsafe.Pointer(logicalType),
	}
}

// CreateMapType wraps duckdb_create_map_type.
// The return value must be destroyed with DestroyLogicalType.
func CreateMapType(key LogicalType, value LogicalType) LogicalType {
	logicalType := C.duckdb_create_map_type(key.data(), value.data())
	if debugMode {
		incrAllocCount("logicalType")
	}
	return LogicalType{
		Ptr: unsafe.Pointer(logicalType),
	}
}

// CreateUnionType wraps duckdb_create_union_type.
// The return value must be destroyed with DestroyLogicalType.
func CreateUnionType(types []LogicalType, names []string) LogicalType {
	typesPtr := allocLogicalTypes(types)
	defer Free(unsafe.Pointer(typesPtr))

	namesPtr := allocNames(names)
	defer Free(unsafe.Pointer(namesPtr))
	count := IdxT(len(types))
	defer C.duckdb_go_bindings_free_names(namesPtr, count)

	// Create the STRUCT type.
	logicalType := C.duckdb_create_union_type(typesPtr, namesPtr, count)

	if debugMode {
		incrAllocCount("logicalType")
	}
	return LogicalType{
		Ptr: unsafe.Pointer(logicalType),
	}
}

// CreateStructType wraps duckdb_create_struct_type.
// The return value must be destroyed with DestroyLogicalType.
func CreateStructType(types []LogicalType, names []string) LogicalType {
	typesPtr := allocLogicalTypes(types)
	defer Free(unsafe.Pointer(typesPtr))

	namesPtr := allocNames(names)
	defer Free(unsafe.Pointer(namesPtr))
	count := IdxT(len(types))
	defer C.duckdb_go_bindings_free_names(namesPtr, count)

	// Create the STRUCT type.
	logicalType := C.duckdb_create_struct_type(typesPtr, namesPtr, count)

	if debugMode {
		incrAllocCount("logicalType")
	}
	return LogicalType{
		Ptr: unsafe.Pointer(logicalType),
	}
}

// CreateEnumType wraps duckdb_create_enum_type.
// The return value must be destroyed with DestroyLogicalType.
func CreateEnumType(names []string) LogicalType {
	namesPtr := allocNames(names)
	defer Free(unsafe.Pointer(namesPtr))
	count := IdxT(len(names))
	defer C.duckdb_go_bindings_free_names(namesPtr, count)

	// Create the ENUM type.
	logicalType := C.duckdb_create_enum_type(namesPtr, count)

	if debugMode {
		incrAllocCount("logicalType")
	}
	return LogicalType{
		Ptr: unsafe.Pointer(logicalType),
	}
}

// CreateDecimalType wraps duckdb_create_decimal_type.
// The return value must be destroyed with DestroyLogicalType.
func CreateDecimalType(width uint8, scale uint8) LogicalType {
	logicalType := C.duckdb_create_decimal_type(C.uint8_t(width), C.uint8_t(scale))
	if debugMode {
		incrAllocCount("logicalType")
	}
	return LogicalType{
		Ptr: unsafe.Pointer(logicalType),
	}
}

func GetTypeId(logicalType LogicalType) Type {
	return C.duckdb_get_type_id(logicalType.data())
}

func DecimalWidth(logicalType LogicalType) uint8 {
	width := C.duckdb_decimal_width(logicalType.data())
	return uint8(width)
}

func DecimalScale(logicalType LogicalType) uint8 {
	scale := C.duckdb_decimal_scale(logicalType.data())
	return uint8(scale)
}

func DecimalInternalType(logicalType LogicalType) Type {
	return C.duckdb_decimal_internal_type(logicalType.data())
}

func EnumInternalType(logicalType LogicalType) Type {
	return C.duckdb_enum_internal_type(logicalType.data())
}

func EnumDictionarySize(logicalType LogicalType) uint32 {
	size := C.duckdb_enum_dictionary_size(logicalType.data())
	return uint32(size)
}

func EnumDictionaryValue(logicalType LogicalType, index IdxT) string {
	str := C.duckdb_enum_dictionary_value(logicalType.data(), index)
	defer Free(unsafe.Pointer(str))
	return C.GoString(str)
}

// ListTypeChildType wraps duckdb_list_type_child_type.
// The return value must be destroyed with DestroyLogicalType.
func ListTypeChildType(logicalType LogicalType) LogicalType {
	child := C.duckdb_list_type_child_type(logicalType.data())
	if debugMode {
		incrAllocCount("logicalType")
	}
	return LogicalType{
		Ptr: unsafe.Pointer(child),
	}
}

// ArrayTypeChildType wraps duckdb_array_type_child_type.
// The return value must be destroyed with DestroyLogicalType.
func ArrayTypeChildType(logicalType LogicalType) LogicalType {
	child := C.duckdb_array_type_child_type(logicalType.data())
	if debugMode {
		incrAllocCount("logicalType")
	}
	return LogicalType{
		Ptr: unsafe.Pointer(child),
	}
}

func ArrayTypeArraySize(logicalType LogicalType) IdxT {
	return C.duckdb_array_type_array_size(logicalType.data())
}

// MapTypeKeyType wraps duckdb_map_type_key_type.
// The return value must be destroyed with DestroyLogicalType.
func MapTypeKeyType(logicalType LogicalType) LogicalType {
	key := C.duckdb_map_type_key_type(logicalType.data())
	if debugMode {
		incrAllocCount("logicalType")
	}
	return LogicalType{
		Ptr: unsafe.Pointer(key),
	}
}

// MapTypeValueType wraps duckdb_map_type_value_type.
// The return value must be destroyed with DestroyLogicalType.
func MapTypeValueType(logicalType LogicalType) LogicalType {
	value := C.duckdb_map_type_value_type(logicalType.data())
	if debugMode {
		incrAllocCount("logicalType")
	}
	return LogicalType{
		Ptr: unsafe.Pointer(value),
	}
}

func StructTypeChildCount(logicalType LogicalType) IdxT {
	return C.duckdb_struct_type_child_count(logicalType.data())
}

func StructTypeChildName(logicalType LogicalType, index IdxT) string {
	cName := C.duckdb_struct_type_child_name(logicalType.data(), index)
	defer Free(unsafe.Pointer(cName))
	return C.GoString(cName)
}

// StructTypeChildType wraps duckdb_struct_type_child_type.
// The return value must be destroyed with DestroyLogicalType.
func StructTypeChildType(logicalType LogicalType, index IdxT) LogicalType {
	child := C.duckdb_struct_type_child_type(logicalType.data(), index)
	if debugMode {
		incrAllocCount("logicalType")
	}
	return LogicalType{
		Ptr: unsafe.Pointer(child),
	}
}

func UnionTypeMemberCount(logicalType LogicalType) IdxT {
	return C.duckdb_union_type_member_count(logicalType.data())
}

func UnionTypeMemberName(logicalType LogicalType, index IdxT) string {
	cStr := C.duckdb_union_type_member_name(logicalType.data(), index)
	defer Free(unsafe.Pointer(cStr))
	return C.GoString(cStr)
}

// UnionTypeMemberType wraps duckdb_union_type_member_type.
// The return value must be destroyed with DestroyLogicalType.
func UnionTypeMemberType(logicalType LogicalType, index IdxT) LogicalType {
	t := C.duckdb_union_type_member_type(logicalType.data(), index)
	if debugMode {
		incrAllocCount("logicalType")
	}
	return LogicalType{
		Ptr: unsafe.Pointer(t),
	}
}

// DestroyLogicalType wraps duckdb_destroy_logical_type.
func DestroyLogicalType(logicalType *LogicalType) {
	if logicalType.Ptr == nil {
		return
	}
	if debugMode {
		decrAllocCount("logicalType")
	}
	data := logicalType.data()
	C.duckdb_destroy_logical_type(&data)
	logicalType.Ptr = nil
}

func RegisterLogicalType(conn Connection, logicalType LogicalType, info CreateTypeInfo) State {
	return C.duckdb_register_logical_type(conn.data(), logicalType.data(), info.data())
}

// ------------------------------------------------------------------ //
// Data Chunk Interface
// ------------------------------------------------------------------ //

// CreateDataChunk wraps duckdb_create_data_chunk.
// The return value must be destroyed with DestroyDataChunk.
func CreateDataChunk(types []LogicalType) DataChunk {
	typesPtr := allocLogicalTypes(types)
	defer Free(unsafe.Pointer(typesPtr))

	chunk := C.duckdb_create_data_chunk(typesPtr, IdxT(len(types)))
	if debugMode {
		incrAllocCount("chunk")
	}

	return DataChunk{
		Ptr: unsafe.Pointer(chunk),
	}
}

// DestroyDataChunk wraps duckdb_destroy_data_chunk.
func DestroyDataChunk(chunk *DataChunk) {
	if chunk.Ptr == nil {
		return
	}
	if debugMode {
		decrAllocCount("chunk")
	}
	data := chunk.data()
	C.duckdb_destroy_data_chunk(&data)
	chunk.Ptr = nil
}

func DataChunkReset(chunk DataChunk) {
	C.duckdb_data_chunk_reset(chunk.data())
}

func DataChunkGetColumnCount(chunk DataChunk) IdxT {
	return C.duckdb_data_chunk_get_column_count(chunk.data())
}

func DataChunkGetVector(chunk DataChunk, index IdxT) Vector {
	vec := C.duckdb_data_chunk_get_vector(chunk.data(), index)
	return Vector{
		Ptr: unsafe.Pointer(vec),
	}
}

func DataChunkGetSize(chunk DataChunk) IdxT {
	return C.duckdb_data_chunk_get_size(chunk.data())
}

func DataChunkSetSize(chunk DataChunk, size IdxT) {
	C.duckdb_data_chunk_set_size(chunk.data(), size)
}

// ------------------------------------------------------------------ //
// Vector Interface
// ------------------------------------------------------------------ //

// CreateVector wraps duckdb_create_vector.
// The return value must be destroyed with DestroyVector.
func CreateVector(logicalType LogicalType, capacity IdxT) Vector {
	vec := C.duckdb_create_vector(logicalType.data(), capacity)
	if debugMode {
		incrAllocCount("vec")
	}
	return Vector{
		Ptr: unsafe.Pointer(vec),
	}
}

// DestroyVector wraps duckdb_destroy_vector.
func DestroyVector(vec *Vector) {
	if vec.Ptr == nil {
		return
	}
	if debugMode {
		decrAllocCount("vec")
	}
	data := vec.data()
	C.duckdb_destroy_vector(&data)
	vec.Ptr = nil
}

// VectorGetColumnType wraps duckdb_vector_get_column_type.
// The return value must be destroyed with DestroyLogicalType.
func VectorGetColumnType(vec Vector) LogicalType {
	logicalType := C.duckdb_vector_get_column_type(vec.data())
	if debugMode {
		incrAllocCount("logicalType")
	}
	return LogicalType{
		Ptr: unsafe.Pointer(logicalType),
	}
}

func VectorGetData(vec Vector) unsafe.Pointer {
	return C.duckdb_vector_get_data(vec.data())
}

func VectorGetValidity(vec Vector) unsafe.Pointer {
	mask := C.duckdb_vector_get_validity(vec.data())
	return unsafe.Pointer(mask)
}

func VectorEnsureValidityWritable(vec Vector) {
	C.duckdb_vector_ensure_validity_writable(vec.data())
}

func VectorAssignStringElement(vec Vector, index IdxT, str string) {
	cStr := C.CString(str)
	defer Free(unsafe.Pointer(cStr))
	C.duckdb_vector_assign_string_element(vec.data(), index, cStr)
}

func VectorAssignStringElementLen(vec Vector, index IdxT, blob []byte) {
	cBytes := (*C.char)(C.CBytes(blob))
	defer Free(unsafe.Pointer(cBytes))
	C.duckdb_vector_assign_string_element_len(vec.data(), index, cBytes, IdxT(len(blob)))
}

func ListVectorGetChild(vec Vector) Vector {
	child := C.duckdb_list_vector_get_child(vec.data())
	return Vector{
		Ptr: unsafe.Pointer(child),
	}
}

func ListVectorGetSize(vec Vector) IdxT {
	return C.duckdb_list_vector_get_size(vec.data())
}

func ListVectorSetSize(vec Vector, size IdxT) State {
	return C.duckdb_list_vector_set_size(vec.data(), size)
}

func ListVectorReserve(vec Vector, capacity IdxT) State {
	return C.duckdb_list_vector_reserve(vec.data(), capacity)
}

func StructVectorGetChild(vec Vector, index IdxT) Vector {
	child := C.duckdb_struct_vector_get_child(vec.data(), index)
	return Vector{
		Ptr: unsafe.Pointer(child),
	}
}

func ArrayVectorGetChild(vec Vector) Vector {
	child := C.duckdb_array_vector_get_child(vec.data())
	return Vector{
		Ptr: unsafe.Pointer(child),
	}
}

func SliceVector(vec Vector, sel SelectionVector, len IdxT) {
	C.duckdb_slice_vector(vec.data(), sel.data(), len)
}

func VectorCopySel(src Vector, dst Vector, sel SelectionVector, count IdxT, srcOffset IdxT, dstOffset IdxT) {
	C.duckdb_vector_copy_sel(src.data(), dst.data(), sel.data(), count, srcOffset, dstOffset)
}

func VectorReferenceValue(vec Vector, v Value) {
	C.duckdb_vector_reference_value(vec.data(), v.data())
}

func VectorReferenceVector(toVec Vector, fromVec Vector) {
	C.duckdb_vector_reference_vector(toVec.data(), fromVec.data())
}

// ------------------------------------------------------------------ //
// Validity Mask Functions
// ------------------------------------------------------------------ //

func ValidityRowIsValid(maskPtr unsafe.Pointer, row IdxT) bool {
	mask := (*C.uint64_t)(maskPtr)
	return bool(C.duckdb_validity_row_is_valid(mask, row))
}

func ValiditySetRowValidity(maskPtr unsafe.Pointer, row IdxT, valid bool) {
	mask := (*C.uint64_t)(maskPtr)
	C.duckdb_validity_set_row_validity(mask, row, C.bool(valid))
}

func ValiditySetRowInvalid(maskPtr unsafe.Pointer, row IdxT) {
	mask := (*C.uint64_t)(maskPtr)
	C.duckdb_validity_set_row_invalid(mask, row)
}

func ValiditySetRowValid(maskPtr unsafe.Pointer, row IdxT) {
	mask := (*C.uint64_t)(maskPtr)
	C.duckdb_validity_set_row_valid(mask, row)
}

// ------------------------------------------------------------------ //
// Scalar Functions
// ------------------------------------------------------------------ //

// CreateScalarFunction wraps duckdb_create_scalar_function.
// The return value must be destroyed with DestroyScalarFunction.
func CreateScalarFunction() ScalarFunction {
	f := C.duckdb_create_scalar_function()
	if debugMode {
		incrAllocCount("scalarFunc")
	}
	return ScalarFunction{
		Ptr: unsafe.Pointer(f),
	}
}

// DestroyScalarFunction wraps duckdb_destroy_scalar_function.
func DestroyScalarFunction(f *ScalarFunction) {
	if f.Ptr == nil {
		return
	}
	if debugMode {
		decrAllocCount("scalarFunc")
	}
	data := f.data()
	C.duckdb_destroy_scalar_function(&data)
	f.Ptr = nil
}

func ScalarFunctionSetName(f ScalarFunction, name string) {
	cName := C.CString(name)
	defer Free(unsafe.Pointer(cName))
	C.duckdb_scalar_function_set_name(f.data(), cName)
}

func ScalarFunctionSetVarargs(f ScalarFunction, logicalType LogicalType) {
	C.duckdb_scalar_function_set_varargs(f.data(), logicalType.data())
}

func ScalarFunctionSetSpecialHandling(f ScalarFunction) {
	C.duckdb_scalar_function_set_special_handling(f.data())
}

func ScalarFunctionSetVolatile(f ScalarFunction) {
	C.duckdb_scalar_function_set_volatile(f.data())
}

func ScalarFunctionAddParameter(f ScalarFunction, logicalType LogicalType) {
	C.duckdb_scalar_function_add_parameter(f.data(), logicalType.data())
}

func ScalarFunctionSetReturnType(f ScalarFunction, logicalType LogicalType) {
	C.duckdb_scalar_function_set_return_type(f.data(), logicalType.data())
}

func ScalarFunctionSetExtraInfo(f ScalarFunction, extraInfoPtr unsafe.Pointer, callbackPtr unsafe.Pointer) {
	callback := C.duckdb_delete_callback_t(callbackPtr)
	C.duckdb_scalar_function_set_extra_info(f.data(), extraInfoPtr, callback)
}

func ScalarFunctionSetBind(f ScalarFunction, callbackPtr unsafe.Pointer) {
	callback := C.duckdb_scalar_function_bind_t(callbackPtr)
	C.duckdb_scalar_function_set_bind(f.data(), callback)
}

func ScalarFunctionSetBindData(info BindInfo, bindDataPtr unsafe.Pointer, callbackPtr unsafe.Pointer) {
	callback := C.duckdb_delete_callback_t(callbackPtr)
	C.duckdb_scalar_function_set_bind_data(info.data(), bindDataPtr, callback)
}

func ScalarFunctionSetBindDataCopy(info BindInfo, callbackPtr unsafe.Pointer) {
	callback := C.duckdb_copy_callback_t(callbackPtr)
	C.duckdb_scalar_function_set_bind_data_copy(info.data(), callback)
}

func ScalarFunctionBindSetError(info BindInfo, err string) {
	cErr := C.CString(err)
	defer Free(unsafe.Pointer(cErr))
	C.duckdb_scalar_function_bind_set_error(info.data(), cErr)
}

func ScalarFunctionSetFunction(f ScalarFunction, callbackPtr unsafe.Pointer) {
	callback := C.duckdb_scalar_function_t(callbackPtr)
	C.duckdb_scalar_function_set_function(f.data(), callback)
}

func RegisterScalarFunction(conn Connection, f ScalarFunction) State {
	return C.duckdb_register_scalar_function(conn.data(), f.data())
}

func ScalarFunctionGetExtraInfo(info FunctionInfo) unsafe.Pointer {
	return C.duckdb_scalar_function_get_extra_info(info.data())
}

func ScalarFunctionBindGetExtraInfo(info BindInfo) unsafe.Pointer {
	return C.duckdb_scalar_function_bind_get_extra_info(info.data())
}

func ScalarFunctionGetBindData(info FunctionInfo) unsafe.Pointer {
	return C.duckdb_scalar_function_get_bind_data(info.data())
}

// ScalarFunctionGetClientContext wraps duckdb_scalar_function_get_client_context.
// outCtx must be destroyed with DestroyClientContext.
func ScalarFunctionGetClientContext(info BindInfo, outCtx *ClientContext) {
	var ctx C.duckdb_client_context
	C.duckdb_scalar_function_get_client_context(info.data(), &ctx)
	outCtx.Ptr = unsafe.Pointer(ctx)
	if debugMode {
		incrAllocCount("ctx")
	}
}

func ScalarFunctionSetError(info FunctionInfo, err string) {
	cErr := C.CString(err)
	defer Free(unsafe.Pointer(cErr))
	C.duckdb_scalar_function_set_error(info.data(), cErr)
}

// CreateScalarFunctionSet wraps duckdb_create_scalar_function_set.
// The return value must be destroyed with DestroyScalarFunctionSet.
func CreateScalarFunctionSet(name string) ScalarFunctionSet {
	cName := C.CString(name)
	defer Free(unsafe.Pointer(cName))

	set := C.duckdb_create_scalar_function_set(cName)
	if debugMode {
		incrAllocCount("scalarFuncSet")
	}
	return ScalarFunctionSet{
		Ptr: unsafe.Pointer(set),
	}
}

// DestroyScalarFunctionSet wraps duckdb_destroy_scalar_function_set.
func DestroyScalarFunctionSet(set *ScalarFunctionSet) {
	if set.Ptr == nil {
		return
	}
	if debugMode {
		decrAllocCount("scalarFuncSet")
	}
	data := set.data()
	C.duckdb_destroy_scalar_function_set(&data)
	set.Ptr = nil
}

func AddScalarFunctionToSet(set ScalarFunctionSet, f ScalarFunction) State {
	return C.duckdb_add_scalar_function_to_set(set.data(), f.data())
}

func RegisterScalarFunctionSet(conn Connection, f ScalarFunctionSet) State {
	return C.duckdb_register_scalar_function_set(conn.data(), f.data())
}

func ScalarFunctionBindGetArgumentCount(info BindInfo) IdxT {
	return C.duckdb_scalar_function_bind_get_argument_count(info.data())
}

// ScalarFunctionBindGetArgument wraps duckdb_scalar_function_bind_get_argument.
// The return value must be destroyed with DestroyExpression.
func ScalarFunctionBindGetArgument(info BindInfo, index IdxT) Expression {
	expr := C.duckdb_scalar_function_bind_get_argument(info.data(), index)
	if debugMode {
		incrAllocCount("expr")
	}
	return Expression{
		Ptr: unsafe.Pointer(expr),
	}
}

// ------------------------------------------------------------------ //
// Selection Vector Functions
// ------------------------------------------------------------------ //

// CreateSelectionVector wraps duckdb_create_selection_vector.
// The return value must be destroyed with DestroySelectionVector.
func CreateSelectionVector(size IdxT) SelectionVector {
	sel := C.duckdb_create_selection_vector(size)
	if debugMode {
		incrAllocCount("sel")
	}
	return SelectionVector{
		Ptr: unsafe.Pointer(sel),
	}
}

// DestroySelectionVector wraps duckdb_destroy_selection_vector.
func DestroySelectionVector(sel *SelectionVector) {
	if sel.Ptr == nil {
		return
	}
	if debugMode {
		decrAllocCount("sel")
	}
	C.duckdb_destroy_selection_vector(sel.data())
	sel.Ptr = nil
}

func SelectionVectorGetDataPtr(sel SelectionVector) *SelT {
	return C.duckdb_selection_vector_get_data_ptr(sel.data())
}

// ------------------------------------------------------------------ //
// Aggregate Functions
// ------------------------------------------------------------------ //

// TODO:
// duckdb_create_aggregate_function
// duckdb_destroy_aggregate_function
// duckdb_aggregate_function_set_name
// duckdb_aggregate_function_add_parameter
// duckdb_aggregate_function_set_return_type
// duckdb_aggregate_function_set_functions
// duckdb_aggregate_function_set_destructor
// duckdb_register_aggregate_function
// duckdb_aggregate_function_set_special_handling
// duckdb_aggregate_function_set_extra_info
// duckdb_aggregate_function_get_extra_info
// duckdb_aggregate_function_set_error
// duckdb_create_aggregate_function_set
// duckdb_destroy_aggregate_function_set
// duckdb_add_aggregate_function_to_set
// duckdb_register_aggregate_function_set

// ------------------------------------------------------------------ //
// Table Functions
// ------------------------------------------------------------------ //

// CreateTableFunction wraps duckdb_create_table_function.
// The return value must be destroyed with DestroyTableFunction.
func CreateTableFunction() TableFunction {
	f := C.duckdb_create_table_function()
	if debugMode {
		incrAllocCount("tableFunc")
	}
	return TableFunction{
		Ptr: unsafe.Pointer(f),
	}
}

// DestroyTableFunction wraps duckdb_destroy_table_function.
func DestroyTableFunction(f *TableFunction) {
	if f.Ptr == nil {
		return
	}
	if debugMode {
		decrAllocCount("tableFunc")
	}
	data := f.data()
	C.duckdb_destroy_table_function(&data)
	f.Ptr = nil
}

func TableFunctionSetName(f TableFunction, name string) {
	cName := C.CString(name)
	defer Free(unsafe.Pointer(cName))
	C.duckdb_table_function_set_name(f.data(), cName)
}

func TableFunctionAddParameter(f TableFunction, logicalType LogicalType) {
	C.duckdb_table_function_add_parameter(f.data(), logicalType.data())
}

func TableFunctionAddNamedParameter(f TableFunction, name string, logicalType LogicalType) {
	cName := C.CString(name)
	defer Free(unsafe.Pointer(cName))
	C.duckdb_table_function_add_named_parameter(f.data(), cName, logicalType.data())
}

func TableFunctionSetExtraInfo(f TableFunction, extraInfoPtr unsafe.Pointer, callbackPtr unsafe.Pointer) {
	callback := C.duckdb_delete_callback_t(callbackPtr)
	C.duckdb_table_function_set_extra_info(f.data(), extraInfoPtr, callback)
}

func TableFunctionSetBind(f TableFunction, callbackPtr unsafe.Pointer) {
	callback := C.duckdb_table_function_bind_t(callbackPtr)
	C.duckdb_table_function_set_bind(f.data(), callback)
}

func TableFunctionSetInit(f TableFunction, callbackPtr unsafe.Pointer) {
	callback := C.duckdb_table_function_init_t(callbackPtr)
	C.duckdb_table_function_set_init(f.data(), callback)
}

func TableFunctionSetLocalInit(f TableFunction, callbackPtr unsafe.Pointer) {
	callback := C.duckdb_table_function_init_t(callbackPtr)
	C.duckdb_table_function_set_local_init(f.data(), callback)
}

func TableFunctionSetFunction(f TableFunction, callbackPtr unsafe.Pointer) {
	callback := C.duckdb_table_function_t(callbackPtr)
	C.duckdb_table_function_set_function(f.data(), callback)
}

func TableFunctionSupportsProjectionPushdown(f TableFunction, pushdown bool) {
	C.duckdb_table_function_supports_projection_pushdown(f.data(), C.bool(pushdown))
}

func RegisterTableFunction(conn Connection, f TableFunction) State {
	return C.duckdb_register_table_function(conn.data(), f.data())
}

// ------------------------------------------------------------------ //
// Table Function Bind
// ------------------------------------------------------------------ //

func BindGetExtraInfo(info BindInfo) unsafe.Pointer {
	return C.duckdb_bind_get_extra_info(info.data())
}

// TableFunctionGetClientContext wraps duckdb_table_function_get_client_context.
// outCtx must be destroyed with DestroyClientContext.
func TableFunctionGetClientContext(info BindInfo, outCtx *ClientContext) {
	var ctx C.duckdb_client_context
	C.duckdb_table_function_get_client_context(info.data(), &ctx)
	outCtx.Ptr = unsafe.Pointer(ctx)
	if debugMode {
		decrAllocCount("ctx")
	}
}

func BindAddResultColumn(info BindInfo, name string, logicalType LogicalType) {
	cName := C.CString(name)
	defer Free(unsafe.Pointer(cName))
	C.duckdb_bind_add_result_column(info.data(), cName, logicalType.data())
}

func BindGetParameterCount(info BindInfo) IdxT {
	return C.duckdb_bind_get_parameter_count(info.data())
}

// BindGetParameter wraps duckdb_bind_get_parameter.
// The return value must be destroyed with DestroyValue.
func BindGetParameter(info BindInfo, index IdxT) Value {
	v := C.duckdb_bind_get_parameter(info.data(), index)
	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

// BindGetNamedParameter wraps duckdb_bind_get_named_parameter.
// The return value must be destroyed with DestroyValue.
func BindGetNamedParameter(info BindInfo, name string) Value {
	cName := C.CString(name)
	defer Free(unsafe.Pointer(cName))
	v := C.duckdb_bind_get_named_parameter(info.data(), cName)
	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

func BindSetBindData(info BindInfo, bindDataPtr unsafe.Pointer, callbackPtr unsafe.Pointer) {
	callback := C.duckdb_delete_callback_t(callbackPtr)
	C.duckdb_bind_set_bind_data(info.data(), bindDataPtr, callback)
}

func BindSetCardinality(info BindInfo, cardinality IdxT, exact bool) {
	C.duckdb_bind_set_cardinality(info.data(), cardinality, C.bool(exact))
}

func BindSetError(info BindInfo, err string) {
	cErr := C.CString(err)
	defer Free(unsafe.Pointer(cErr))
	C.duckdb_bind_set_error(info.data(), cErr)
}

// ------------------------------------------------------------------ //
// Table Function Init
// ------------------------------------------------------------------ //

func InitGetExtraInfo(info InitInfo) unsafe.Pointer {
	return C.duckdb_init_get_extra_info(info.data())
}

func InitGetBindData(info InitInfo) unsafe.Pointer {
	return C.duckdb_init_get_bind_data(info.data())
}

func InitSetInitData(info InitInfo, initDataPtr unsafe.Pointer, callbackPtr unsafe.Pointer) {
	callback := C.duckdb_delete_callback_t(callbackPtr)
	C.duckdb_init_set_init_data(info.data(), initDataPtr, callback)
}

func InitGetColumnCount(info InitInfo) IdxT {
	return C.duckdb_init_get_column_count(info.data())
}

func InitGetColumnIndex(info InitInfo, index IdxT) IdxT {
	return C.duckdb_init_get_column_index(info.data(), index)
}

func InitSetMaxThreads(info InitInfo, max IdxT) {
	C.duckdb_init_set_max_threads(info.data(), max)
}

func InitSetError(info InitInfo, err string) {
	cStr := C.CString(err)
	defer Free(unsafe.Pointer(cStr))
	C.duckdb_init_set_error(info.data(), cStr)
}

// ------------------------------------------------------------------ //
// Table Function
// ------------------------------------------------------------------ //

func FunctionGetExtraInfo(info FunctionInfo) unsafe.Pointer {
	return C.duckdb_function_get_extra_info(info.data())
}

func FunctionGetBindData(info FunctionInfo) unsafe.Pointer {
	return C.duckdb_function_get_bind_data(info.data())
}

func FunctionGetInitData(info FunctionInfo) unsafe.Pointer {
	return C.duckdb_function_get_init_data(info.data())
}

func FunctionGetLocalInitData(info FunctionInfo) unsafe.Pointer {
	return C.duckdb_function_get_local_init_data(info.data())
}

func FunctionSetError(info FunctionInfo, err string) {
	cErr := C.CString(err)
	defer Free(unsafe.Pointer(cErr))
	C.duckdb_function_set_error(info.data(), cErr)
}

// ------------------------------------------------------------------ //
// Replacement Scans
// ------------------------------------------------------------------ //

func AddReplacementScan(db Database, callbackPtr unsafe.Pointer, extraData unsafe.Pointer, deleteCallbackPtr unsafe.Pointer) {
	callback := C.duckdb_replacement_callback_t(callbackPtr)
	deleteCallback := C.duckdb_delete_callback_t(deleteCallbackPtr)
	C.duckdb_add_replacement_scan(db.data(), callback, extraData, deleteCallback)
}

func ReplacementScanSetFunctionName(info ReplacementScanInfo, name string) {
	cName := C.CString(name)
	defer Free(unsafe.Pointer(cName))
	C.duckdb_replacement_scan_set_function_name(info.data(), cName)
}

func ReplacementScanAddParameter(info ReplacementScanInfo, v Value) {
	C.duckdb_replacement_scan_add_parameter(info.data(), v.data())
}

func ReplacementScanSetError(info ReplacementScanInfo, err string) {
	cErr := C.CString(err)
	defer Free(unsafe.Pointer(cErr))
	C.duckdb_replacement_scan_set_error(info.data(), cErr)
}

// ------------------------------------------------------------------ //
// Profiling Info
// ------------------------------------------------------------------ //

func GetProfilingInfo(conn Connection) ProfilingInfo {
	info := C.duckdb_get_profiling_info(conn.data())
	return ProfilingInfo{
		Ptr: unsafe.Pointer(info),
	}
}

// ProfilingInfoGetValue wraps duckdb_profiling_info_get_value.
// The return value must be destroyed with DestroyValue.
func ProfilingInfoGetValue(info ProfilingInfo, key string) Value {
	cKey := C.CString(key)
	defer Free(unsafe.Pointer(cKey))
	v := C.duckdb_profiling_info_get_value(info.data(), cKey)

	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

// ProfilingInfoGetMetrics wraps duckdb_profiling_info_get_metrics.
// The return value must be destroyed with DestroyValue.
func ProfilingInfoGetMetrics(info ProfilingInfo) Value {
	v := C.duckdb_profiling_info_get_metrics(info.data())
	if debugMode {
		incrAllocCount("v")
	}
	return Value{
		Ptr: unsafe.Pointer(v),
	}
}

func ProfilingInfoGetChildCount(info ProfilingInfo) IdxT {
	return C.duckdb_profiling_info_get_child_count(info.data())
}

func ProfilingInfoGetChild(info ProfilingInfo, index IdxT) ProfilingInfo {
	child := C.duckdb_profiling_info_get_child(info.data(), index)
	return ProfilingInfo{
		Ptr: unsafe.Pointer(child),
	}
}

// ------------------------------------------------------------------ //
// Appender
// ------------------------------------------------------------------ //

// AppenderCreate wraps duckdb_appender_create.
// outAppender must be destroyed with AppenderDestroy.
func AppenderCreate(conn Connection, schema string, table string, outAppender *Appender) State {
	cSchema := C.CString(schema)
	defer Free(unsafe.Pointer(cSchema))
	cTable := C.CString(table)
	defer Free(unsafe.Pointer(cTable))

	var appender C.duckdb_appender
	state := C.duckdb_appender_create(conn.data(), cSchema, cTable, &appender)
	outAppender.Ptr = unsafe.Pointer(appender)
	if debugMode {
		incrAllocCount("appender")
	}
	return state
}

// AppenderCreateExt wraps duckdb_appender_create_ext.
// outAppender must be destroyed with AppenderDestroy.
func AppenderCreateExt(conn Connection, catalog string, schema string, table string, outAppender *Appender) State {
	cCatalog := C.CString(catalog)
	defer Free(unsafe.Pointer(cCatalog))
	cSchema := C.CString(schema)
	defer Free(unsafe.Pointer(cSchema))
	cTable := C.CString(table)
	defer Free(unsafe.Pointer(cTable))

	var appender C.duckdb_appender
	state := C.duckdb_appender_create_ext(conn.data(), cCatalog, cSchema, cTable, &appender)
	outAppender.Ptr = unsafe.Pointer(appender)
	if debugMode {
		incrAllocCount("appender")
	}
	return state
}

// AppenderCreateQuery wraps duckdb_appender_create_query.
// outAppender must be destroyed with AppenderDestroy.
func AppenderCreateQuery(conn Connection, query string, types []LogicalType, tableName string, columnNames []string, outAppender *Appender) State {
	cQuery := C.CString(query)
	defer Free(unsafe.Pointer(cQuery))

	typesPtr := allocLogicalTypes(types)
	defer Free(unsafe.Pointer(typesPtr))

	// The table name is optional.
	cTableName := unsafe.Pointer(nil)
	if tableName != "" {
		cTableName = unsafe.Pointer(C.CString(tableName))
	}
	defer Free(cTableName)

	// Column names are optional.
	namesPtr := unsafe.Pointer(nil)
	countNames := IdxT(len(columnNames))
	if countNames > 0 {
		namesPtr = unsafe.Pointer(allocNames(columnNames))
	}
	defer Free(namesPtr)
	defer C.duckdb_go_bindings_free_names((**C.char)(namesPtr), countNames)

	columnCount := IdxT(len(types))
	var appender C.duckdb_appender
	state := C.duckdb_appender_create_query(conn.data(), cQuery, columnCount, typesPtr, (*C.char)(cTableName), (**C.char)(namesPtr), &appender)
	outAppender.Ptr = unsafe.Pointer(appender)
	if debugMode {
		incrAllocCount("appender")
	}
	return state
}

func AppenderColumnCount(appender Appender) IdxT {
	return C.duckdb_appender_column_count(appender.data())
}

// AppenderColumnType wraps duckdb_appender_column_type.
// The return value must be destroyed with DestroyLogicalType.
func AppenderColumnType(appender Appender, index IdxT) LogicalType {
	logicalType := C.duckdb_appender_column_type(appender.data(), index)
	if debugMode {
		incrAllocCount("logicalType")
	}
	return LogicalType{
		Ptr: unsafe.Pointer(logicalType),
	}
}

func AppenderError(appender Appender) string {
	err := C.duckdb_appender_error(appender.data())
	return C.GoString(err)
}

// AppenderErrorData wraps duckdb_appender_error_data.
// The return value must be destroyed with DestroyErrorData.
func AppenderErrorData(appender Appender) ErrorData {
	errorData := C.duckdb_appender_error_data(appender.data())
	if debugMode {
		incrAllocCount("errorData")
	}
	return ErrorData{
		Ptr: unsafe.Pointer(errorData),
	}
}

func AppenderFlush(appender Appender) State {
	return C.duckdb_appender_flush(appender.data())
}

func AppenderClose(appender Appender) State {
	return C.duckdb_appender_close(appender.data())
}

// AppenderDestroy wraps duckdb_appender_destroy.
func AppenderDestroy(appender *Appender) State {
	if appender.Ptr == nil {
		return StateSuccess
	}
	if debugMode {
		decrAllocCount("appender")
	}
	data := appender.data()
	state := C.duckdb_appender_destroy(&data)
	appender.Ptr = nil
	return state
}

func AppenderAddColumn(appender Appender, name string) State {
	cName := C.CString(name)
	defer Free(unsafe.Pointer(cName))
	return C.duckdb_appender_add_column(appender.data(), cName)
}

func AppenderClearColumns(appender Appender) State {
	return C.duckdb_appender_clear_columns(appender.data())
}

// TODO:
// duckdb_appender_begin_row
// duckdb_appender_end_row
// duckdb_append_default

func AppendDefaultToChunk(appender Appender, chunk DataChunk, col IdxT, row IdxT) State {
	return C.duckdb_append_default_to_chunk(appender.data(), chunk.data(), col, row)
}

// TODO:
// duckdb_append_bool
// duckdb_append_int8
// duckdb_append_int16
// duckdb_append_int32
// duckdb_append_int64
// duckdb_append_hugeint
// duckdb_append_uint8
// duckdb_append_uint16
// duckdb_append_uint32
// duckdb_append_uint64
// duckdb_append_uhugeint
// duckdb_append_float
// duckdb_append_double
// duckdb_append_date
// duckdb_append_time
// duckdb_append_timestamp
// duckdb_append_interval
// duckdb_append_varchar
// duckdb_append_varchar_length
// duckdb_append_blob
// duckdb_append_null
// duckdb_append_value

func AppendDataChunk(appender Appender, chunk DataChunk) State {
	return C.duckdb_append_data_chunk(appender.data(), chunk.data())
}

// ------------------------------------------------------------------ //
// Table Description
// ------------------------------------------------------------------ //

// TableDescriptionCreate wraps duckdb_table_description_create.
// outDesc must be destroyed with TableDescriptionDestroy.
func TableDescriptionCreate(conn Connection, schema string, table string, outDesc *TableDescription) State {
	cSchema := C.CString(schema)
	defer Free(unsafe.Pointer(cSchema))
	cTable := C.CString(table)
	defer Free(unsafe.Pointer(cTable))

	var description C.duckdb_table_description
	state := C.duckdb_table_description_create(conn.data(), cSchema, cTable, &description)
	outDesc.Ptr = unsafe.Pointer(description)
	if debugMode {
		incrAllocCount("tableDesc")
	}
	return state
}

// TableDescriptionCreateExt wraps duckdb_table_description_create_ext.
// outDesc must be destroyed with TableDescriptionDestroy.
func TableDescriptionCreateExt(conn Connection, catalog string, schema string, table string, outDesc *TableDescription) State {
	cCatalog := C.CString(catalog)
	defer Free(unsafe.Pointer(cCatalog))
	cSchema := C.CString(schema)
	defer Free(unsafe.Pointer(cSchema))
	cTable := C.CString(table)
	defer Free(unsafe.Pointer(cTable))

	var description C.duckdb_table_description
	state := C.duckdb_table_description_create_ext(conn.data(), cCatalog, cSchema, cTable, &description)
	outDesc.Ptr = unsafe.Pointer(description)
	if debugMode {
		incrAllocCount("tableDesc")
	}
	return state
}

// TableDescriptionDestroy wraps duckdb_table_description_destroy.
func TableDescriptionDestroy(desc *TableDescription) {
	if desc.Ptr == nil {
		return
	}
	if debugMode {
		decrAllocCount("tableDesc")
	}
	data := desc.data()
	C.duckdb_table_description_destroy(&data)
	desc.Ptr = nil
}

func TableDescriptionError(desc TableDescription) string {
	err := C.duckdb_table_description_error(desc.data())
	return C.GoString(err)
}

func ColumnHasDefault(desc TableDescription, index IdxT, outBool *bool) State {
	var b C.bool
	state := C.duckdb_column_has_default(desc.data(), index, &b)
	*outBool = bool(b)
	return state
}

func TableDescriptionGetColumnName(desc TableDescription, index IdxT) string {
	cName := C.duckdb_table_description_get_column_name(desc.data(), index)
	defer Free(unsafe.Pointer(cName))
	return C.GoString(cName)
}

// ------------------------------------------------------------------ //
// Arrow Interface (entire interface has deprecation notice)
// ------------------------------------------------------------------ //

// TODO:
// duckdb_to_arrow_schema
// duckdb_data_chunk_to_arrow
// duckdb_schema_from_arrow
// duckdb_data_chunk_from_arrow

// DestroyArrowConvertedSchema wraps duckdb_destroy_arrow_converted_schema.
func DestroyArrowConvertedSchema(schema *ArrowConvertedSchema) {
	if schema.Ptr == nil {
		return
	}
	if debugMode {
		decrAllocCount("arrowConvertedSchema")
	}
	data := schema.data()
	C.duckdb_destroy_arrow_converted_schema(&data)
	schema.Ptr = nil
}

// TODO:
// duckdb_query_arrow

func QueryArrowSchema(arrow Arrow, outSchema *ArrowSchema) State {
	return C.duckdb_query_arrow_schema(arrow.data(), (*C.duckdb_arrow_schema)(outSchema.Ptr))
}

// TODO:
// duckdb_prepared_arrow_schema
// duckdb_result_arrow_array

func QueryArrowArray(arrow Arrow, outArray *ArrowArray) State {
	return C.duckdb_query_arrow_array(arrow.data(), (*C.duckdb_arrow_array)(outArray.Ptr))
}

// TODO:
// duckdb_arrow_column_count

func ArrowRowCount(arrow Arrow) IdxT {
	return C.duckdb_arrow_row_count(arrow.data())
}

// TODO:
// duckdb_arrow_rows_changed

func QueryArrowError(arrow Arrow) string {
	err := C.duckdb_query_arrow_error(arrow.data())
	return C.GoString(err)
}

// DestroyArrow wraps duckdb_destroy_arrow.
func DestroyArrow(arrow *Arrow) {
	if arrow.Ptr == nil {
		return
	}
	if debugMode {
		decrAllocCount("arrow")
	}
	data := arrow.data()
	C.duckdb_destroy_arrow(&data)
	arrow.Ptr = nil
}

// TODO:
// duckdb_destroy_arrow_stream

// ExecutePreparedArrow wraps duckdb_execute_prepared_arrow.
// outArrow must be destroyed with DestroyArrow.
func ExecutePreparedArrow(preparedStmt PreparedStatement, outArrow *Arrow) State {
	var arrow C.duckdb_arrow
	state := C.duckdb_execute_prepared_arrow(preparedStmt.data(), &arrow)
	outArrow.Ptr = unsafe.Pointer(arrow)
	if debugMode {
		incrAllocCount("arrow")
	}
	return state
}

func ArrowScan(conn Connection, table string, stream ArrowStream) State {
	cTable := C.CString(table)
	defer Free(unsafe.Pointer(cTable))
	return C.duckdb_arrow_scan(conn.data(), cTable, stream.data())
}

// TODO:
// duckdb_arrow_array_scan

//===--------------------------------------------------------------------===//
// Threading Information
//===--------------------------------------------------------------------===//

// TODO:
// duckdb_execute_tasks
// duckdb_create_task_state
// duckdb_execute_tasks_state
// duckdb_execute_n_tasks_state
// duckdb_finish_execution
// duckdb_task_state_is_finished
// duckdb_destroy_task_state
// duckdb_execution_is_finished

//===--------------------------------------------------------------------===//
// Streaming Result Interface
//===--------------------------------------------------------------------===//

// TODO:
// duckdb_stream_fetch_chunk (deprecation notice)
// duckdb_fetch_chunk

//===--------------------------------------------------------------------===//
// Cast Functions
//===--------------------------------------------------------------------===//

// TODO:
// duckdb_create_cast_function
// duckdb_cast_function_set_source_type
// duckdb_cast_function_set_target_type
// duckdb_cast_function_set_implicit_cast_cost
// duckdb_cast_function_set_function
// duckdb_cast_function_set_extra_info
// duckdb_cast_function_get_extra_info
// duckdb_cast_function_get_cast_mode
// duckdb_cast_function_set_error
// duckdb_cast_function_set_row_error
// duckdb_register_cast_function
// duckdb_destroy_cast_function

// ------------------------------------------------------------------ //
// Expression Interface
// ------------------------------------------------------------------ //

// DestroyExpression wraps duckdb_destroy_expression.
func DestroyExpression(expr *Expression) {
	if expr.Ptr == nil {
		return
	}
	if debugMode {
		decrAllocCount("expr")
	}
	data := expr.data()
	C.duckdb_destroy_expression(&data)
	expr.Ptr = nil
}

// ExpressionReturnType wraps duckdb_expression_return_type.
// The return value must be destroyed with DestroyLogicalType.
func ExpressionReturnType(expr Expression) LogicalType {
	logicalType := C.duckdb_expression_return_type(expr.data())
	if debugMode {
		incrAllocCount("logicalType")
	}
	return LogicalType{
		Ptr: unsafe.Pointer(logicalType),
	}
}

func ExpressionIsFoldable(expr Expression) bool {
	return bool(C.duckdb_expression_is_foldable(expr.data()))
}

// ExpressionFold wraps duckdb_expression_fold.
// outValue must be destroyed with DestroyValue.
// The return value must be destroyed with DestroyErrorData.
func ExpressionFold(ctx ClientContext, expr Expression, outValue *Value) ErrorData {
	var value C.duckdb_value
	errorData := C.duckdb_expression_fold(ctx.data(), expr.data(), &value)
	outValue.Ptr = unsafe.Pointer(value)
	if debugMode {
		incrAllocCount("v")
		incrAllocCount("errorData")
	}
	return ErrorData{
		Ptr: unsafe.Pointer(errorData),
	}
}

// ------------------------------------------------------------------ //
// Go Bindings Helper
// ------------------------------------------------------------------ //

func ValidityMaskValueIsValid(maskPtr unsafe.Pointer, index IdxT) bool {
	castMaskPtr := (*C.uint64_t)(maskPtr)
	return bool(C.duckdb_go_bindings_is_valid(castMaskPtr, index))
}

const (
	logicalTypeSize = C.size_t(unsafe.Sizeof((C.duckdb_logical_type)(nil)))
	valueSize       = C.size_t(unsafe.Sizeof((C.duckdb_value)(nil)))
	charSize        = C.size_t(unsafe.Sizeof((*C.char)(nil)))
)

// The return value must be freed with Free.
func allocLogicalTypes(types []LogicalType) *C.duckdb_logical_type {
	count := len(types)
	typesPtr := (*C.duckdb_logical_type)(C.calloc(C.size_t(count), logicalTypeSize))

	for i, t := range types {
		C.duckdb_go_bindings_set_logical_type(typesPtr, t.data(), IdxT(i))
	}

	return typesPtr
}

// The return value must be freed with Free.
func allocValues(values []Value) *C.duckdb_value {
	count := len(values)
	valuesPtr := (*C.duckdb_value)(C.calloc(C.size_t(count), valueSize))

	for i, val := range values {
		C.duckdb_go_bindings_set_value(valuesPtr, val.data(), IdxT(i))
	}

	return valuesPtr
}

// The return value must be freed with Free.
// The names must also be freed.
func allocNames(names []string) **C.char {
	count := len(names)
	namesPtr := (**C.char)(C.calloc(C.size_t(count), charSize))

	for i, name := range names {
		C.duckdb_go_bindings_set_name(namesPtr, C.CString(name), IdxT(i))
	}

	return namesPtr
}

// ------------------------------------------------------------------ //
// Memory Safety
// ------------------------------------------------------------------ //

var allocCounts syncMap

type syncMap struct {
	lock sync.Mutex
	m    map[string]int
}

func incrAllocCount(k string) {
	allocCounts.lock.Lock()
	defer allocCounts.lock.Unlock()

	if allocCounts.m == nil {
		allocCounts.m = make(map[string]int)
	}

	allocCounts.m[k]++
}

func decrAllocCount(k string) {
	allocCounts.lock.Lock()
	defer allocCounts.lock.Unlock()

	if allocCounts.m == nil {
		allocCounts.m = make(map[string]int)
	}

	if v, ok := allocCounts.m[k]; ok {
		if v == 1 {
			delete(allocCounts.m, k)
			return
		}
		allocCounts.m[k]--
	}
}

// VerifyAllocationCounters verifies all allocation counters.
// This includes the instance cache, which should be kept alive as long as the application is kept alive,
// causing this verification to fail.
// If you're using the instance cache, use VerifyAllocationCounter instead.
func VerifyAllocationCounters() {
	msg := GetAllocationCounts()
	if msg != "" {
		log.Panic(msg)
	}
}

// GetAllocationCount returns the value of an allocation count, and true,
// if it exists, otherwise zero, and false.
func GetAllocationCount(k string) (int, bool) {
	allocCounts.lock.Lock()
	defer allocCounts.lock.Unlock()

	if allocCounts.m == nil {
		return 0, false
	}

	v, ok := allocCounts.m[k]
	return v, ok
}

// GetAllocationCounts returns the value of each non-zero allocation count.
func GetAllocationCounts() string {
	allocCounts.lock.Lock()
	defer allocCounts.lock.Unlock()

	if allocCounts.m == nil {
		return ""
	}

	msg := ""
	if len(allocCounts.m) != 0 {
		for k, v := range allocCounts.m {
			msg += fmt.Sprintf("%s count is %d\n", k, v)
		}
	}

	return msg
}
