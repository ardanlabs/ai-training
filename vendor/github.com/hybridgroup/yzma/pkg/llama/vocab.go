package llama

import (
	"unsafe"

	"github.com/hybridgroup/yzma/pkg/utils"
	"github.com/jupiterrider/ffi"
)

var (
	// LLAMA_API const struct llama_vocab * llama_model_get_vocab(const struct llama_model * model);
	modelGetVocabFunc ffi.Fun

	// LLAMA_API llama_token llama_vocab_bos(const struct llama_vocab * vocab); // beginning-of-sentence
	vocabBOSFunc ffi.Fun

	// LLAMA_API llama_token llama_vocab_eos(const struct llama_vocab * vocab); // end-of-sentence
	vocabEOSFunc ffi.Fun

	// LLAMA_API llama_token llama_vocab_eot(const struct llama_vocab * vocab); // end-of-turn
	vocabEOTFunc ffi.Fun

	// LLAMA_API llama_token llama_vocab_sep(const struct llama_vocab * vocab); // sentence separator
	vocabSEPFunc ffi.Fun

	// LLAMA_API llama_token llama_vocab_nl(const struct llama_vocab * vocab); // next-line
	vocabNLFunc ffi.Fun

	// LLAMA_API llama_token llama_vocab_pad(const struct llama_vocab * vocab); // padding
	vocabPADFunc ffi.Fun

	// LLAMA_API llama_token llama_vocab_mask(const struct llama_vocab * vocab); // mask
	vocabMASKFunc ffi.Fun

	// LLAMA_API bool llama_vocab_get_add_bos(const struct llama_vocab * vocab);
	vocabGetAddBOSFunc ffi.Fun

	// LLAMA_API bool llama_vocab_get_add_eos(const struct llama_vocab * vocab);
	vocabGetAddEOSFunc ffi.Fun

	// LLAMA_API bool llama_vocab_get_add_sep(const struct llama_vocab * vocab);
	vocabGetAddSEPFunc ffi.Fun

	// LLAMA_API llama_token llama_vocab_fim_pre(const struct llama_vocab * vocab);
	vocabFIMPreFunc ffi.Fun

	// LLAMA_API llama_token llama_vocab_fim_suf(const struct llama_vocab * vocab);
	vocabFIMSufFunc ffi.Fun

	// LLAMA_API llama_token llama_vocab_fim_mid(const struct llama_vocab * vocab);
	vocabFIMMidFunc ffi.Fun

	// LLAMA_API llama_token llama_vocab_fim_pad(const struct llama_vocab * vocab);
	vocabFIMPadFunc ffi.Fun

	// LLAMA_API llama_token llama_vocab_fim_rep(const struct llama_vocab * vocab);
	vocabFIMRepFunc ffi.Fun

	// LLAMA_API llama_token llama_vocab_fim_sep(const struct llama_vocab * vocab);
	vocabFIMSepFunc ffi.Fun

	// LLAMA_API bool llama_vocab_is_eog(const struct llama_vocab * vocab, llama_token token);
	vocabIsEOGFunc ffi.Fun

	// LLAMA_API bool llama_vocab_is_control(const struct llama_vocab * vocab, llama_token token);
	vocabIsControlFunc ffi.Fun

	// LLAMA_API int32_t llama_vocab_n_tokens(const struct llama_vocab * vocab);
	vocabNTokensFunc ffi.Fun

	// LLAMA_API int32_t llama_token_to_piece(
	//              const struct llama_vocab * vocab,
	//                           llama_token   token,
	//                                  char * buf,
	//                               int32_t   length,
	//                               int32_t   lstrip,
	//                               bool   special);
	tokenToPieceFunc ffi.Fun

	// LLAMA_API int32_t llama_tokenize(
	//     const struct llama_vocab * vocab,
	//                   const char * text,
	//                      int32_t   text_len,
	//                  llama_token * tokens,
	//                      int32_t   n_tokens_max,
	//                         bool   add_special,
	//                         bool   parse_special);
	tokenizeFunc ffi.Fun

	// LLAMA_API enum llama_token_attr llama_vocab_get_attr(const struct llama_vocab * vocab, llama_token token);
	vocabGetAttrFunc ffi.Fun

	// LLAMA_API float llama_vocab_get_score(const struct llama_vocab * vocab, llama_token token);
	vocabGetScoreFunc ffi.Fun

	// LLAMA_API const char * llama_vocab_get_text(const struct llama_vocab * vocab, llama_token token);
	vocabGetTextFunc ffi.Fun

	// LLAMA_API enum llama_vocab_type llama_vocab_type(const struct llama_vocab * vocab);
	vocabTypeFunc ffi.Fun
)

func loadVocabFuncs(lib ffi.Lib) error {
	var err error

	if modelGetVocabFunc, err = lib.Prep("llama_model_get_vocab", &ffi.TypePointer, &ffi.TypePointer); err != nil {
		return loadError("llama_model_get_vocab", err)
	}

	if vocabBOSFunc, err = lib.Prep("llama_vocab_bos", &ffi.TypeSint32, &ffi.TypePointer); err != nil {
		return loadError("llama_vocab_bos", err)
	}

	if vocabEOSFunc, err = lib.Prep("llama_vocab_eos", &ffi.TypeSint32, &ffi.TypePointer); err != nil {
		return loadError("llama_vocab_eos", err)
	}

	if vocabEOTFunc, err = lib.Prep("llama_vocab_eot", &ffi.TypeSint32, &ffi.TypePointer); err != nil {
		return loadError("llama_vocab_eot", err)
	}

	if vocabSEPFunc, err = lib.Prep("llama_vocab_sep", &ffi.TypeSint32, &ffi.TypePointer); err != nil {
		return loadError("llama_vocab_sep", err)
	}

	if vocabNLFunc, err = lib.Prep("llama_vocab_nl", &ffi.TypeSint32, &ffi.TypePointer); err != nil {
		return loadError("llama_vocab_nl", err)
	}

	if vocabPADFunc, err = lib.Prep("llama_vocab_pad", &ffi.TypeSint32, &ffi.TypePointer); err != nil {
		return loadError("llama_vocab_pad", err)
	}

	if vocabMASKFunc, err = lib.Prep("llama_vocab_mask", &ffi.TypeSint32, &ffi.TypePointer); err != nil {
		return loadError("llama_vocab_mask", err)
	}

	if vocabGetAddBOSFunc, err = lib.Prep("llama_vocab_get_add_bos", &ffi.TypeUint8, &ffi.TypePointer); err != nil {
		return loadError("llama_vocab_get_add_bos", err)
	}

	if vocabGetAddEOSFunc, err = lib.Prep("llama_vocab_get_add_eos", &ffi.TypeUint8, &ffi.TypePointer); err != nil {
		return loadError("llama_vocab_get_add_eos", err)
	}

	if vocabGetAddSEPFunc, err = lib.Prep("llama_vocab_get_add_sep", &ffi.TypeUint8, &ffi.TypePointer); err != nil {
		return loadError("llama_vocab_get_add_sep", err)
	}

	if vocabFIMPreFunc, err = lib.Prep("llama_vocab_fim_pre", &ffi.TypeSint32, &ffi.TypePointer); err != nil {
		return loadError("llama_vocab_fim_pre", err)
	}

	if vocabFIMSufFunc, err = lib.Prep("llama_vocab_fim_suf", &ffi.TypeSint32, &ffi.TypePointer); err != nil {
		return loadError("llama_vocab_fim_suf", err)
	}

	if vocabFIMMidFunc, err = lib.Prep("llama_vocab_fim_mid", &ffi.TypeSint32, &ffi.TypePointer); err != nil {
		return loadError("llama_vocab_fim_mid", err)
	}

	if vocabFIMPadFunc, err = lib.Prep("llama_vocab_fim_pad", &ffi.TypeSint32, &ffi.TypePointer); err != nil {
		return loadError("llama_vocab_fim_pad", err)
	}

	if vocabFIMRepFunc, err = lib.Prep("llama_vocab_fim_rep", &ffi.TypeSint32, &ffi.TypePointer); err != nil {
		return loadError("llama_vocab_fim_rep", err)
	}

	if vocabFIMSepFunc, err = lib.Prep("llama_vocab_fim_sep", &ffi.TypeSint32, &ffi.TypePointer); err != nil {
		return loadError("llama_vocab_fim_sep", err)
	}

	if vocabIsEOGFunc, err = lib.Prep("llama_vocab_is_eog", &ffi.TypeUint8, &ffi.TypePointer, &ffi.TypeSint32); err != nil {
		return loadError("llama_vocab_is_eog", err)
	}

	if vocabIsControlFunc, err = lib.Prep("llama_vocab_is_control", &ffi.TypeUint8, &ffi.TypePointer, &ffi.TypeSint32); err != nil {
		return loadError("llama_vocab_is_control", err)
	}

	if vocabNTokensFunc, err = lib.Prep("llama_vocab_n_tokens", &ffi.TypeSint32, &ffi.TypePointer); err != nil {
		return loadError("llama_vocab_n_tokens", err)
	}

	if tokenToPieceFunc, err = lib.Prep("llama_token_to_piece", &ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeSint32,
		&ffi.TypePointer, &ffi.TypeSint32, &ffi.TypeSint32, &ffi.TypeUint8); err != nil {
		return loadError("llama_token_to_piece", err)
	}

	if tokenizeFunc, err = lib.Prep("llama_tokenize", &ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer, &ffi.TypeSint32,
		&ffi.TypePointer, &ffi.TypeSint32, &ffi.TypeUint8, &ffi.TypeUint8); err != nil {
		return loadError("llama_tokenize", err)
	}

	if vocabGetAttrFunc, err = lib.Prep("llama_vocab_get_attr", &ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeSint32); err != nil {
		return loadError("llama_vocab_get_attr", err)
	}

	if vocabGetScoreFunc, err = lib.Prep("llama_vocab_get_score", &ffi.TypeFloat, &ffi.TypePointer, &ffi.TypeSint32); err != nil {
		return loadError("llama_vocab_get_score", err)
	}

	if vocabGetTextFunc, err = lib.Prep("llama_vocab_get_text", &ffi.TypePointer, &ffi.TypePointer, &ffi.TypeSint32); err != nil {
		return loadError("llama_vocab_get_text", err)
	}

	if vocabTypeFunc, err = lib.Prep("llama_vocab_type", &ffi.TypeSint32, &ffi.TypePointer); err != nil {
		return loadError("llama_vocab_type", err)
	}

	return nil
}

func ModelGetVocab(model Model) Vocab {
	var vocab Vocab
	modelGetVocabFunc.Call(unsafe.Pointer(&vocab), unsafe.Pointer(&model))

	return vocab
}

func VocabBOS(vocab Vocab) Token {
	var token ffi.Arg
	vocabBOSFunc.Call(unsafe.Pointer(&token), unsafe.Pointer(&vocab))

	return Token(token)
}

func VocabEOS(vocab Vocab) Token {
	var token ffi.Arg
	vocabEOSFunc.Call(unsafe.Pointer(&token), unsafe.Pointer(&vocab))

	return Token(token)
}

func VocabEOT(vocab Vocab) Token {
	var token ffi.Arg
	vocabEOTFunc.Call(unsafe.Pointer(&token), unsafe.Pointer(&vocab))
	return Token(token)
}

func VocabSEP(vocab Vocab) Token {
	var token ffi.Arg
	vocabSEPFunc.Call(unsafe.Pointer(&token), unsafe.Pointer(&vocab))
	return Token(token)
}

func VocabNL(vocab Vocab) Token {
	var token ffi.Arg
	vocabNLFunc.Call(unsafe.Pointer(&token), unsafe.Pointer(&vocab))
	return Token(token)
}

func VocabPAD(vocab Vocab) Token {
	var token ffi.Arg
	vocabPADFunc.Call(unsafe.Pointer(&token), unsafe.Pointer(&vocab))
	return Token(token)
}

func VocabMASK(vocab Vocab) Token {
	var token ffi.Arg
	vocabMASKFunc.Call(unsafe.Pointer(&token), unsafe.Pointer(&vocab))
	return Token(token)
}

func VocabGetAddBOS(vocab Vocab) bool {
	var result ffi.Arg
	vocabGetAddBOSFunc.Call(unsafe.Pointer(&result), unsafe.Pointer(&vocab))
	return result.Bool()
}

func VocabGetAddEOS(vocab Vocab) bool {
	var result ffi.Arg
	vocabGetAddEOSFunc.Call(unsafe.Pointer(&result), unsafe.Pointer(&vocab))
	return result.Bool()
}

func VocabGetAddSEP(vocab Vocab) bool {
	var result ffi.Arg
	vocabGetAddSEPFunc.Call(unsafe.Pointer(&result), unsafe.Pointer(&vocab))
	return result.Bool()
}

func VocabFIMPre(vocab Vocab) Token {
	var token ffi.Arg
	vocabFIMPreFunc.Call(unsafe.Pointer(&token), unsafe.Pointer(&vocab))
	return Token(token)
}

func VocabFIMSuf(vocab Vocab) Token {
	var token ffi.Arg
	vocabFIMSufFunc.Call(unsafe.Pointer(&token), unsafe.Pointer(&vocab))
	return Token(token)
}

func VocabFIMMid(vocab Vocab) Token {
	var token ffi.Arg
	vocabFIMMidFunc.Call(unsafe.Pointer(&token), unsafe.Pointer(&vocab))
	return Token(token)
}

func VocabFIMPad(vocab Vocab) Token {
	var token ffi.Arg
	vocabFIMPadFunc.Call(unsafe.Pointer(&token), unsafe.Pointer(&vocab))
	return Token(token)
}

func VocabFIMRep(vocab Vocab) Token {
	var token ffi.Arg
	vocabFIMRepFunc.Call(unsafe.Pointer(&token), unsafe.Pointer(&vocab))
	return Token(token)
}

func VocabFIMSep(vocab Vocab) Token {
	var token ffi.Arg
	vocabFIMSepFunc.Call(unsafe.Pointer(&token), unsafe.Pointer(&vocab))
	return Token(token)
}

func VocabIsEOG(vocab Vocab, token Token) bool {
	var result ffi.Arg
	vocabIsEOGFunc.Call(unsafe.Pointer(&result), unsafe.Pointer(&vocab), unsafe.Pointer(&token))

	return result.Bool()
}

func VocabIsControl(vocab Vocab, token Token) bool {
	var result ffi.Arg
	vocabIsControlFunc.Call(unsafe.Pointer(&result), unsafe.Pointer(&vocab), unsafe.Pointer(&token))

	return result.Bool()
}

func VocabNTokens(vocab Vocab) int32 {
	var result ffi.Arg
	vocabNTokensFunc.Call(unsafe.Pointer(&result), unsafe.Pointer(&vocab))

	return int32(result)
}

func TokenToPiece(vocab Vocab, token Token, buf []byte, lstrip int32, special bool) int32 {
	piece := make([]byte, len(buf))
	b := unsafe.SliceData(piece)
	bLen := int32(len(piece))

	var result ffi.Arg
	tokenToPieceFunc.Call(unsafe.Pointer(&result), unsafe.Pointer(&vocab), &token, unsafe.Pointer(&b),
		&bLen, &lstrip, &special)

	copy(buf, piece)

	return int32(result)
}

func Tokenize(vocab Vocab, text string, tokens []Token, addSpecial bool, parseSpecial bool) int32 {
	txt, _ := utils.BytePtrFromString(text)
	txtLen := int32(len(text))

	var toks *Token
	if len(tokens) > 0 {
		toks = unsafe.SliceData(tokens)
	}
	nTokensMax := int32(len(tokens))

	var result ffi.Arg
	tokenizeFunc.Call(unsafe.Pointer(&result), unsafe.Pointer(&vocab), unsafe.Pointer(&txt), &txtLen,
		unsafe.Pointer(&toks), &nTokensMax, &addSpecial, &parseSpecial)

	return -int32(result)
}

// VocabGetAttr retrieves the attribute of a given token in the vocabulary.
func VocabGetAttr(vocab Vocab, token Token) TokenAttr {
	var attr ffi.Arg
	vocabGetAttrFunc.Call(unsafe.Pointer(&attr), unsafe.Pointer(&vocab), unsafe.Pointer(&token))
	return TokenAttr(int32(attr))
}

// VocabGetScore retrieves the score of a given token in the vocabulary.
func VocabGetScore(vocab Vocab, token Token) float32 {
	var score ffi.Arg
	vocabGetScoreFunc.Call(unsafe.Pointer(&score), unsafe.Pointer(&vocab), unsafe.Pointer(&token))
	return float32(score)
}

// VocabGetText retrieves the text representation of a given token in the vocabulary.
func VocabGetText(vocab Vocab, token Token) string {
	var textPtr *byte
	vocabGetTextFunc.Call(unsafe.Pointer(&textPtr), unsafe.Pointer(&vocab), unsafe.Pointer(&token))

	if textPtr == nil {
		return ""
	}

	return utils.BytePtrToString(textPtr)
}

// VocabType retrieves the type of the vocabulary.
func GetVocabType(vocab Vocab) VocabType {
	var vocabType ffi.Arg
	vocabTypeFunc.Call(unsafe.Pointer(&vocabType), unsafe.Pointer(&vocab))

	return VocabType(int32(vocabType))
}
