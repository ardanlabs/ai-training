//lint:file-ignore ST1020 The documentation of an exported function should start with the function’s name
//lint:file-ignore ST1003 Poorly chosen identifier

package whisper

import (
	"errors"
	"unsafe"
)

// /////////////////////////////////////////////////////////////////////////////
// CGO

/*
#cgo LDFLAGS: -lwhisper -lm -lstdc++ -fopenmp
#cgo darwin LDFLAGS: -framework Accelerate -framework Metal -framework Foundation -framework CoreGraphics
#include <../../../../../zarf/whisper/whisper.h>
#include <stdlib.h>

extern void callNewSegment(void* user_data, int new);
extern void callProgress(void* user_data, int progress);
extern bool callEncoderBegin(void* user_data);

// Text segment callback
// Called on every newly generated text segment
// Use the whisper_full_...() functions to obtain the text segments
static void whisper_new_segment_cb(struct whisper_context* ctx, struct whisper_state* state, int n_new, void* user_data) {
    if(user_data != NULL && ctx != NULL) {
        callNewSegment(user_data, n_new);
    }
}

// Progress callback
// Called on every newly generated text segment
// Use the whisper_full_...() functions to obtain the text segments
static void whisper_progress_cb(struct whisper_context* ctx, struct whisper_state* state, int progress, void* user_data) {
    if(user_data != NULL && ctx != NULL) {
        callProgress(user_data, progress);
    }
}

// Encoder begin callback
// If not NULL, called before the encoder starts
// If it returns false, the computation is aborted
static bool whisper_encoder_begin_cb(struct whisper_context* ctx, struct whisper_state* state, void* user_data) {
    if(user_data != NULL && ctx != NULL) {
        return callEncoderBegin(user_data);
    }
    return false;
}

// Get default parameters and set callbacks
static struct whisper_full_params whisper_full_default_params_cb(struct whisper_context* ctx, enum whisper_sampling_strategy strategy) {
	struct whisper_full_params params = whisper_full_default_params(strategy);
	params.new_segment_callback = whisper_new_segment_cb;
	params.new_segment_callback_user_data = (void*)(ctx);
	params.encoder_begin_callback = whisper_encoder_begin_cb;
	params.encoder_begin_callback_user_data = (void*)(ctx);
	params.progress_callback = whisper_progress_cb;
	params.progress_callback_user_data = (void*)(ctx);
	return params;
}
*/
import "C"

// /////////////////////////////////////////////////////////////////////////////
// TYPES

type (
	Context          C.struct_whisper_context
	Token            C.whisper_token
	TokenData        C.struct_whisper_token_data
	SamplingStrategy C.enum_whisper_sampling_strategy
	Params           C.struct_whisper_full_params
)

// /////////////////////////////////////////////////////////////////////////////
// GLOBALS

const (
	SAMPLING_GREEDY      SamplingStrategy = C.WHISPER_SAMPLING_GREEDY
	SAMPLING_BEAM_SEARCH SamplingStrategy = C.WHISPER_SAMPLING_BEAM_SEARCH
)

const (
	SampleRate = C.WHISPER_SAMPLE_RATE                 // Expected sample rate, samples per second
	SampleBits = uint16(unsafe.Sizeof(C.float(0))) * 8 // Sample size in bits
	NumFFT     = C.WHISPER_N_FFT
	HopLength  = C.WHISPER_HOP_LENGTH
	ChunkSize  = C.WHISPER_CHUNK_SIZE
)

var (
	ErrTokenizerFailed  = errors.New("whisper_tokenize failed")
	ErrAutoDetectFailed = errors.New("whisper_lang_auto_detect failed")
	ErrConversionFailed = errors.New("whisper_convert failed")
	ErrInvalidLanguage  = errors.New("invalid language")
)

// Go enum for alignment heads preset
type WhisperAlignmentHeadsPreset int

const (
	WHISPER_AHEADS_NONE WhisperAlignmentHeadsPreset = iota
	WHISPER_AHEADS_N_TOP_MOST
	WHISPER_AHEADS_CUSTOM
	WHISPER_AHEADS_TINY_EN
	WHISPER_AHEADS_TINY
	WHISPER_AHEADS_BASE_EN
	WHISPER_AHEADS_BASE
	WHISPER_AHEADS_SMALL_EN
	WHISPER_AHEADS_SMALL
	WHISPER_AHEADS_MEDIUM_EN
	WHISPER_AHEADS_MEDIUM
	WHISPER_AHEADS_LARGE_V1
	WHISPER_AHEADS_LARGE_V2
	WHISPER_AHEADS_LARGE_V3
	WHISPER_AHEADS_LARGE_V3_TURBO
)

// Go struct for whisper_ahead
type WhisperAhead struct {
	NTextLayer int
	NHead      int
}

// Go struct for whisper_aheads
type WhisperAheads struct {
	NHeads int
	Heads  []WhisperAhead
}

// Go struct that mirrors the C whisper_context_params struct
type InitParams struct {
	UseGpu             bool
	FlashAttn          bool
	GpuDevice          int
	DtwTokenTimestamps bool
	DtwAheadsPreset    WhisperAlignmentHeadsPreset
	DtwNTop            int
	DtwAheads          WhisperAheads
	DtwMemSize         int // TODO: remove when C struct removes it
}

// /////////////////////////////////////////////////////////////////////////////
// PUBLIC METHODS

// Allocates all memory needed for the model and loads the model from the given file.
// Returns NULL on failure.
func Whisper_init(path string) *Context {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	if ctx := C.whisper_init_from_file_with_params(cPath, C.whisper_context_default_params()); ctx != nil {
		return (*Context)(ctx)
	} else {
		return nil
	}
}

// Helper function to create default parameters
func DefaultInitParams() InitParams {
	return InitParams{
		UseGpu:             false,
		FlashAttn:          false,
		GpuDevice:          0,
		DtwTokenTimestamps: false,
		DtwAheadsPreset:    WHISPER_AHEADS_NONE,
		DtwNTop:            0,
		DtwAheads:          WhisperAheads{NHeads: 0, Heads: nil},
		DtwMemSize:         0,
	}
}

// Helper function to convert Go InitParams to C whisper_context_params
func (p *InitParams) toCParams() C.struct_whisper_context_params {
	var cParams C.struct_whisper_context_params

	// Convert basic fields
	cParams.use_gpu = C.bool(p.UseGpu)
	cParams.flash_attn = C.bool(p.FlashAttn)
	cParams.gpu_device = C.int(p.GpuDevice)
	cParams.dtw_token_timestamps = C.bool(p.DtwTokenTimestamps)
	cParams.dtw_aheads_preset = C.enum_whisper_alignment_heads_preset(p.DtwAheadsPreset)
	cParams.dtw_n_top = C.int(p.DtwNTop)
	cParams.dtw_mem_size = C.size_t(p.DtwMemSize)

	// Convert whisper_aheads if there are heads
	if len(p.DtwAheads.Heads) > 0 {
		// Allocate C array for heads
		cHeads := C.malloc(C.size_t(len(p.DtwAheads.Heads)) * C.size_t(unsafe.Sizeof(C.struct_whisper_ahead{})))
		cHeadsSlice := (*[1 << 30]C.struct_whisper_ahead)(cHeads)[:len(p.DtwAheads.Heads):len(p.DtwAheads.Heads)]

		// Copy heads data
		for i, head := range p.DtwAheads.Heads {
			cHeadsSlice[i].n_text_layer = C.int(head.NTextLayer)
			cHeadsSlice[i].n_head = C.int(head.NHead)
		}

		cParams.dtw_aheads.n_heads = C.size_t(p.DtwAheads.NHeads)
		cParams.dtw_aheads.heads = (*C.struct_whisper_ahead)(cHeads)
	} else {
		cParams.dtw_aheads.n_heads = 0
		cParams.dtw_aheads.heads = nil
	}

	return cParams
}

// Helper function to free allocated memory in C params
func freeCParams(cParams *C.struct_whisper_context_params) {
	if cParams.dtw_aheads.heads != nil {
		C.free(unsafe.Pointer(cParams.dtw_aheads.heads))
	}
}

// Florin: Allow FlashAttention to be used.
// Allocates all memory needed for the model and loads the model from the given file.
// Returns NULL on failure.
func Whisper_init_with_params(path string, params InitParams) *Context {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	// Convert Go params to C params
	cParams := params.toCParams()
	defer freeCParams(&cParams)

	if ctx := C.whisper_init_from_file_with_params(cPath, cParams); ctx != nil {
		return (*Context)(ctx)
	} else {
		return nil
	}
}

// Bill: Added this to try and init the open vino encoder.
func (ctx *Context) InitOpenVINOEncoder(modelPath, device, cacheDir string) int {
	var cModelPath *C.char
	var cDevice *C.char
	var cCacheDir *C.char

	if modelPath != "" {
		cModelPath = C.CString(modelPath)
		defer C.free(unsafe.Pointer(cModelPath))
	}

	if device != "" {
		cDevice = C.CString(device)
		defer C.free(unsafe.Pointer(cDevice))
	}

	if cacheDir != "" {
		cCacheDir = C.CString(cacheDir)
		defer C.free(unsafe.Pointer(cCacheDir))
	}

	return int(C.whisper_ctx_init_openvino_encoder((*C.struct_whisper_context)(ctx), cModelPath, cDevice, cCacheDir))
}

// Frees all memory allocated by the model.
func (ctx *Context) Whisper_free() {
	C.whisper_free((*C.struct_whisper_context)(ctx))
}

// Convert RAW PCM audio to log mel spectrogram.
// The resulting spectrogram is stored inside the provided whisper context.
func (ctx *Context) Whisper_pcm_to_mel(data []float32, threads int) error {
	if C.whisper_pcm_to_mel((*C.struct_whisper_context)(ctx), (*C.float)(&data[0]), C.int(len(data)), C.int(threads)) == 0 {
		return nil
	} else {
		return ErrConversionFailed
	}
}

// This can be used to set a custom log mel spectrogram inside the provided whisper context.
// Use this instead of whisper_pcm_to_mel() if you want to provide your own log mel spectrogram.
// n_mel must be 80
func (ctx *Context) Whisper_set_mel(data []float32, n_mel int) error {
	if C.whisper_set_mel((*C.struct_whisper_context)(ctx), (*C.float)(&data[0]), C.int(len(data)), C.int(n_mel)) == 0 {
		return nil
	} else {
		return ErrConversionFailed
	}
}

// Run the Whisper encoder on the log mel spectrogram stored inside the provided whisper context.
// Make sure to call whisper_pcm_to_mel() or whisper_set_mel() first.
// offset can be used to specify the offset of the first frame in the spectrogram.
func (ctx *Context) Whisper_encode(offset, threads int) error {
	if C.whisper_encode((*C.struct_whisper_context)(ctx), C.int(offset), C.int(threads)) == 0 {
		return nil
	} else {
		return ErrConversionFailed
	}
}

// Run the Whisper decoder to obtain the logits and probabilities for the next token.
// Make sure to call whisper_encode() first.
// tokens + n_tokens is the provided context for the decoder.
// n_past is the number of tokens to use from previous decoder calls.
func (ctx *Context) Whisper_decode(tokens []Token, past, threads int) error {
	if C.whisper_decode((*C.struct_whisper_context)(ctx), (*C.whisper_token)(&tokens[0]), C.int(len(tokens)), C.int(past), C.int(threads)) == 0 {
		return nil
	} else {
		return ErrConversionFailed
	}
}

// Convert the provided text into tokens. The tokens pointer must be large enough to hold the resulting tokens.
// Returns the number of tokens on success
func (ctx *Context) Whisper_tokenize(text string, tokens []Token) (int, error) {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))
	if n := C.whisper_tokenize((*C.struct_whisper_context)(ctx), cText, (*C.whisper_token)(&tokens[0]), C.int(len(tokens))); n >= 0 {
		return int(n), nil
	} else {
		return 0, ErrTokenizerFailed
	}
}

// Return the id of the specified language, returns -1 if not found
// Examples:
//
//	"de" -> 2
//	"german" -> 2
func (ctx *Context) Whisper_lang_id(lang string) int {
	return int(C.whisper_lang_id(C.CString(lang)))
}

// Largest language id (i.e. number of available languages - 1)
func Whisper_lang_max_id() int {
	return int(C.whisper_lang_max_id())
}

// Return the short string of the specified language id (e.g. 2 -> "de"),
// returns empty string if not found
func Whisper_lang_str(id int) string {
	return C.GoString(C.whisper_lang_str(C.int(id)))
}

// Return the full string of the specified language id (e.g. 2 -> "german"),
// returns empty string if not found
func Whisper_lang_str_full(id int) string {
	return C.GoString(C.whisper_lang_str_full(C.int(id)))
}

// Use mel data at offset_ms to try and auto-detect the spoken language
// Make sure to call whisper_pcm_to_mel() or whisper_set_mel() first.
// Returns the probabilities of all languages.
// ref: https://github.com/openai/whisper/blob/main/whisper/decoding.py#L18-L69
func (ctx *Context) Whisper_lang_auto_detect(offset_ms, n_threads int) ([]float32, error) {
	probs := make([]float32, Whisper_lang_max_id()+1)
	if n := int(C.whisper_lang_auto_detect((*C.struct_whisper_context)(ctx), C.int(offset_ms), C.int(n_threads), (*C.float)(&probs[0]))); n < 0 {
		return nil, ErrAutoDetectFailed
	} else {
		return probs, nil
	}
}

func (ctx *Context) Whisper_n_len() int {
	return int(C.whisper_n_len((*C.struct_whisper_context)(ctx)))
}

func (ctx *Context) Whisper_n_vocab() int {
	return int(C.whisper_n_vocab((*C.struct_whisper_context)(ctx)))
}

func (ctx *Context) Whisper_n_text_ctx() int {
	return int(C.whisper_n_text_ctx((*C.struct_whisper_context)(ctx)))
}

func (ctx *Context) Whisper_n_audio_ctx() int {
	return int(C.whisper_n_audio_ctx((*C.struct_whisper_context)(ctx)))
}

func (ctx *Context) Whisper_is_multilingual() int {
	return int(C.whisper_is_multilingual((*C.struct_whisper_context)(ctx)))
}

// The probabilities for the next token
// func (ctx *Whisper_context) Whisper_get_probs() []float32 {
//	return (*[1 << 30]float32)(unsafe.Pointer(C.whisper_get_probs((*C.struct_whisper_context)(ctx))))[:ctx.Whisper_n_vocab()]
// }

// Token Id -> String. Uses the vocabulary in the provided context
func (ctx *Context) Whisper_token_to_str(token Token) string {
	return C.GoString(C.whisper_token_to_str((*C.struct_whisper_context)(ctx), C.whisper_token(token)))
}

// Special tokens
func (ctx *Context) Whisper_token_eot() Token {
	return Token(C.whisper_token_eot((*C.struct_whisper_context)(ctx)))
}

// Special tokens
func (ctx *Context) Whisper_token_sot() Token {
	return Token(C.whisper_token_sot((*C.struct_whisper_context)(ctx)))
}

// Special tokens
func (ctx *Context) Whisper_token_prev() Token {
	return Token(C.whisper_token_prev((*C.struct_whisper_context)(ctx)))
}

// Special tokens
func (ctx *Context) Whisper_token_solm() Token {
	return Token(C.whisper_token_solm((*C.struct_whisper_context)(ctx)))
}

// Special tokens
func (ctx *Context) Whisper_token_not() Token {
	return Token(C.whisper_token_not((*C.struct_whisper_context)(ctx)))
}

// Special tokens
func (ctx *Context) Whisper_token_beg() Token {
	return Token(C.whisper_token_beg((*C.struct_whisper_context)(ctx)))
}

// Special tokens
func (ctx *Context) Whisper_token_lang(lang_id int) Token {
	return Token(C.whisper_token_lang((*C.struct_whisper_context)(ctx), C.int(lang_id)))
}

// Task tokens
func (ctx *Context) Whisper_token_translate() Token {
	return Token(C.whisper_token_translate((*C.struct_whisper_context)(ctx)))
}

// Task tokens
func (ctx *Context) Whisper_token_transcribe() Token {
	return Token(C.whisper_token_transcribe((*C.struct_whisper_context)(ctx)))
}

// Performance information
func (ctx *Context) Whisper_print_timings() {
	C.whisper_print_timings((*C.struct_whisper_context)(ctx))
}

// Performance information
func (ctx *Context) Whisper_reset_timings() {
	C.whisper_reset_timings((*C.struct_whisper_context)(ctx))
}

// Print system information
func Whisper_print_system_info() string {
	return C.GoString(C.whisper_print_system_info())
}

// Return default parameters for a strategy
func (ctx *Context) Whisper_full_default_params(strategy SamplingStrategy) Params {
	// Get default parameters
	return Params(C.whisper_full_default_params_cb((*C.struct_whisper_context)(ctx), C.enum_whisper_sampling_strategy(strategy)))
}

// Run the entire model: PCM -> log mel spectrogram -> encoder -> decoder -> text
// Uses the specified decoding strategy to obtain the text.
func (ctx *Context) Whisper_full(
	params Params,
	samples []float32,
	encoderBeginCallback func() bool,
	newSegmentCallback func(int),
	progressCallback func(int),
) error {
	registerEncoderBeginCallback(ctx, encoderBeginCallback)
	registerNewSegmentCallback(ctx, newSegmentCallback)
	registerProgressCallback(ctx, progressCallback)
	defer registerEncoderBeginCallback(ctx, nil)
	defer registerNewSegmentCallback(ctx, nil)
	defer registerProgressCallback(ctx, nil)
	if C.whisper_full((*C.struct_whisper_context)(ctx), (C.struct_whisper_full_params)(params), (*C.float)(&samples[0]), C.int(len(samples))) == 0 {
		return nil
	} else {
		return ErrConversionFailed
	}
}

// Split the input audio in chunks and process each chunk separately using whisper_full()
// It seems this approach can offer some speedup in some cases.
// However, the transcription accuracy can be worse at the beginning and end of each chunk.
func (ctx *Context) Whisper_full_parallel(params Params, samples []float32, processors int, encoderBeginCallback func() bool, newSegmentCallback func(int)) error {
	registerEncoderBeginCallback(ctx, encoderBeginCallback)
	registerNewSegmentCallback(ctx, newSegmentCallback)
	defer registerEncoderBeginCallback(ctx, nil)
	defer registerNewSegmentCallback(ctx, nil)

	if C.whisper_full_parallel((*C.struct_whisper_context)(ctx), (C.struct_whisper_full_params)(params), (*C.float)(&samples[0]), C.int(len(samples)), C.int(processors)) == 0 {
		return nil
	} else {
		return ErrConversionFailed
	}
}

// Return the id of the autodetected language, returns -1 if not found
// Added to whisper.cpp in
// https://github.com/ggerganov/whisper.cpp/commit/a1c1583cc7cd8b75222857afc936f0638c5683d6
//
// Examples:
//
//	"de" -> 2
//	"german" -> 2
func (ctx *Context) Whisper_full_lang_id() int {
	return int(C.whisper_full_lang_id((*C.struct_whisper_context)(ctx)))
}

// Number of generated text segments.
// A segment can be a few words, a sentence, or even a paragraph.
func (ctx *Context) Whisper_full_n_segments() int {
	return int(C.whisper_full_n_segments((*C.struct_whisper_context)(ctx)))
}

// Get the start and end time of the specified segment.
func (ctx *Context) Whisper_full_get_segment_t0(segment int) int64 {
	return int64(C.whisper_full_get_segment_t0((*C.struct_whisper_context)(ctx), C.int(segment)))
}

// Get the start and end time of the specified segment.
func (ctx *Context) Whisper_full_get_segment_t1(segment int) int64 {
	return int64(C.whisper_full_get_segment_t1((*C.struct_whisper_context)(ctx), C.int(segment)))
}

// Get the text of the specified segment.
func (ctx *Context) Whisper_full_get_segment_text(segment int) string {
	return C.GoString(C.whisper_full_get_segment_text((*C.struct_whisper_context)(ctx), C.int(segment)))
}

// Get number of tokens in the specified segment.
func (ctx *Context) Whisper_full_n_tokens(segment int) int {
	return int(C.whisper_full_n_tokens((*C.struct_whisper_context)(ctx), C.int(segment)))
}

// Get the token text of the specified token index in the specified segment.
func (ctx *Context) Whisper_full_get_token_text(segment int, token int) string {
	return C.GoString(C.whisper_full_get_token_text((*C.struct_whisper_context)(ctx), C.int(segment), C.int(token)))
}

// Get the token of the specified token index in the specified segment.
func (ctx *Context) Whisper_full_get_token_id(segment int, token int) Token {
	return Token(C.whisper_full_get_token_id((*C.struct_whisper_context)(ctx), C.int(segment), C.int(token)))
}

// Get token data for the specified token in the specified segment.
// This contains probabilities, timestamps, etc.
func (ctx *Context) Whisper_full_get_token_data(segment int, token int) TokenData {
	return TokenData(C.whisper_full_get_token_data((*C.struct_whisper_context)(ctx), C.int(segment), C.int(token)))
}

// Get the probability of the specified token in the specified segment.
func (ctx *Context) Whisper_full_get_token_p(segment int, token int) float32 {
	return float32(C.whisper_full_get_token_p((*C.struct_whisper_context)(ctx), C.int(segment), C.int(token)))
}

// /////////////////////////////////////////////////////////////////////////////
// CALLBACKS

var (
	cbNewSegment   = make(map[unsafe.Pointer]func(int))
	cbProgress     = make(map[unsafe.Pointer]func(int))
	cbEncoderBegin = make(map[unsafe.Pointer]func() bool)
)

func registerNewSegmentCallback(ctx *Context, fn func(int)) {
	if fn == nil {
		delete(cbNewSegment, unsafe.Pointer(ctx))
	} else {
		cbNewSegment[unsafe.Pointer(ctx)] = fn
	}
}

func registerProgressCallback(ctx *Context, fn func(int)) {
	if fn == nil {
		delete(cbProgress, unsafe.Pointer(ctx))
	} else {
		cbProgress[unsafe.Pointer(ctx)] = fn
	}
}

func registerEncoderBeginCallback(ctx *Context, fn func() bool) {
	if fn == nil {
		delete(cbEncoderBegin, unsafe.Pointer(ctx))
	} else {
		cbEncoderBegin[unsafe.Pointer(ctx)] = fn
	}
}

//export callNewSegment
func callNewSegment(user_data unsafe.Pointer, new C.int) {
	if fn, ok := cbNewSegment[user_data]; ok {
		fn(int(new))
	}
}

//export callProgress
func callProgress(user_data unsafe.Pointer, progress C.int) {
	if fn, ok := cbProgress[user_data]; ok {
		fn(int(progress))
	}
}

//export callEncoderBegin
func callEncoderBegin(user_data unsafe.Pointer) C.bool {
	if fn, ok := cbEncoderBegin[user_data]; ok {
		if fn() {
			return C.bool(true)
		} else {
			return C.bool(false)
		}
	}
	return true
}

func (t TokenData) T0() int64 {
	return int64(t.t0)
}

func (t TokenData) T1() int64 {
	return int64(t.t1)
}

func (t TokenData) Id() Token {
	return Token(t.id)
}
