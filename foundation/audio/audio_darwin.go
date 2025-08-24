//go:build darwin

package audio

/*
#cgo CFLAGS: -I${SRCDIR}/../../zarf/whisper
#cgo LDFLAGS: -L${SRCDIR}/../../zarf/whisper/darwin -Wl,-rpath,${SRCDIR}/../../zarf/whisper/darwin
*/
import "C"
