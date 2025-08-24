//go:build linux

package audio

/*
#cgo CFLAGS: -I${SRCDIR}/../../zarf/whisper
#cgo LDFLAGS: -L${SRCDIR}/../../zarf/whisper/linux -Wl,-rpath,${SRCDIR}/../../zarf/whisper/linux
*/
import "C"
