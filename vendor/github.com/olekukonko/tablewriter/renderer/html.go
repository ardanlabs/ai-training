package renderer

import (
	"errors"
	"fmt"
	"github.com/olekukonko/ll"
	"html"
	"io"
	"strings"

	"github.com/olekukonko/tablewriter/tw"
)

// HTMLConfig defines settings for the HTML table renderer.
type HTMLConfig struct {
	EscapeContent  bool   // Whether to escape cell content
	AddLinesTag    bool   // Whether to wrap multiline content in <lines> tags
	TableClass     string // CSS class for <table>
	HeaderClass    string // CSS class for <thead>
	BodyClass      string // CSS class for <tbody>
	FooterClass    string // CSS class for <tfoot>
	RowClass       string // CSS class for <tr> in body
	HeaderRowClass string // CSS class for <tr> in header
	FooterRowClass string // CSS class for <tr> in footer
}

// HTML renders tables in HTML format with customizable classes and content handling.
type HTML struct {
	config       HTMLConfig  // Renderer configuration
	writer       io.Writer   // Output writer
	trace        []string    // Debug trace messages
	debug        bool        // Enables debug logging
	tableStarted bool        // Tracks if <table> tag is open
	tbodyStarted bool        // Tracks if <tbody> tag is open
	tfootStarted bool        // Tracks if <tfoot> tag is open
	vMergeTrack  map[int]int // Tracks vertical merge spans by column index
	logger       *ll.Logger
}

// NewHTML initializes an HTML renderer with the given writer, debug setting, and optional configuration.
// It panics if the writer is nil and applies defaults for unset config fields.
func NewHTML(w io.Writer, debug bool, configs ...HTMLConfig) *HTML {
	if w == nil {
		panic("NewHTML requires a non-nil writer")
	}
	cfg := HTMLConfig{
		EscapeContent: true,
		AddLinesTag:   false,
	}
	if len(configs) > 0 {
		userCfg := configs[0]
		cfg.EscapeContent = userCfg.EscapeContent
		cfg.AddLinesTag = userCfg.AddLinesTag
		cfg.TableClass = userCfg.TableClass
		cfg.HeaderClass = userCfg.HeaderClass
		cfg.BodyClass = userCfg.BodyClass
		cfg.FooterClass = userCfg.FooterClass
		cfg.RowClass = userCfg.RowClass
		cfg.HeaderRowClass = userCfg.HeaderRowClass
		cfg.FooterRowClass = userCfg.FooterRowClass
	}
	return &HTML{
		config:       cfg,
		writer:       w,
		debug:        debug,
		vMergeTrack:  make(map[int]int),
		tableStarted: false,
		tbodyStarted: false,
		tfootStarted: false,
	}
}

func (h *HTML) Logger(logger *ll.Logger) {
	h.logger = logger
}

// Config returns a Rendition representation of the current configuration.
func (h *HTML) Config() tw.Rendition {
	return tw.Rendition{
		Borders:   tw.BorderNone,
		Symbols:   tw.NewSymbols(tw.StyleNone),
		Settings:  tw.Settings{},
		Streaming: false,
	}
}

// debugLog appends a formatted message to the debug trace if debugging is enabled.
func (h *HTML) debugLog(format string, a ...interface{}) {
	if h.debug {
		msg := fmt.Sprintf(format, a...)
		h.trace = append(h.trace, fmt.Sprintf("[HTML] %s", msg))
	}
}

// Debug returns the accumulated debug trace messages.
func (h *HTML) Debug() []string {
	return h.trace
}

// Start begins the HTML table rendering by opening the <table> tag.
func (h *HTML) Start(w io.Writer) error {
	if h.writer == nil {
		return errors.New("HTML renderer writer not initialized")
	}
	h.Reset()
	h.debugLog("HTML.Start() called.")

	classAttr := ""
	if h.config.TableClass != "" {
		classAttr = fmt.Sprintf(` class="%s"`, h.config.TableClass)
	}
	h.debugLog("Writing opening <table%s> tag", classAttr)
	_, err := fmt.Fprintf(h.writer, "<table%s>\n", classAttr)
	if err != nil {
		return err
	}
	h.tableStarted = true
	return nil
}

// closePreviousSection closes any open <tbody> or <tfoot> sections.
func (h *HTML) closePreviousSection() {
	if h.tbodyStarted {
		h.debugLog("Closing <tbody> tag")
		fmt.Fprintln(h.writer, "</tbody>")
		h.tbodyStarted = false
	}
	if h.tfootStarted {
		h.debugLog("Closing <tfoot> tag")
		fmt.Fprintln(h.writer, "</tfoot>")
		h.tfootStarted = false
	}
}

// Header renders the <thead> section with header rows, supporting horizontal merges.
func (h *HTML) Header(w io.Writer, headers [][]string, ctx tw.Formatting) {
	if !h.tableStarted {
		h.debugLog("WARN: Header called before Start")
		return
	}
	if len(headers) == 0 || len(headers[0]) == 0 {
		h.debugLog("Header: No headers")
		return
	}

	h.closePreviousSection()
	classAttr := ""
	if h.config.HeaderClass != "" {
		classAttr = fmt.Sprintf(` class="%s"`, h.config.HeaderClass)
	}
	fmt.Fprintf(h.writer, "<thead%s>\n", classAttr)

	headerRow := headers[0]
	numCols := 0
	if len(ctx.Row.Current) > 0 {
		maxKey := -1
		for k := range ctx.Row.Current {
			if k > maxKey {
				maxKey = k
			}
		}
		numCols = maxKey + 1
	} else if len(headerRow) > 0 {
		numCols = len(headerRow)
	}

	indent := "  "
	rowClassAttr := ""
	if h.config.HeaderRowClass != "" {
		rowClassAttr = fmt.Sprintf(` class="%s"`, h.config.HeaderRowClass)
	}
	fmt.Fprintf(h.writer, "%s<tr%s>", indent, rowClassAttr)

	renderedCols := 0
	for colIdx := 0; renderedCols < numCols && colIdx < numCols; {
		// Skip columns consumed by vertical merges
		if remainingSpan, merging := h.vMergeTrack[colIdx]; merging && remainingSpan > 1 {
			h.debugLog("Header: Skipping col %d due to vmerge", colIdx)
			h.vMergeTrack[colIdx]--
			if h.vMergeTrack[colIdx] <= 1 {
				delete(h.vMergeTrack, colIdx)
			}
			colIdx++
			continue
		}

		// Render cell
		cellCtx, ok := ctx.Row.Current[colIdx]
		if !ok {
			cellCtx = tw.CellContext{Align: tw.AlignCenter}
		}
		originalContent := ""
		if colIdx < len(headerRow) {
			originalContent = headerRow[colIdx]
		}

		tag, attributes, processedContent := h.renderRowCell(originalContent, cellCtx, true, colIdx)
		fmt.Fprintf(h.writer, "<%s%s>%s</%s>", tag, attributes, processedContent, tag)
		renderedCols++

		// Handle horizontal merges
		hSpan := 1
		if cellCtx.Merge.Horizontal.Present && cellCtx.Merge.Horizontal.Start {
			hSpan = cellCtx.Merge.Horizontal.Span
			renderedCols += (hSpan - 1)
		}
		colIdx += hSpan
	}
	fmt.Fprintf(h.writer, "</tr>\n")
	fmt.Fprintln(h.writer, "</thead>")
}

// Row renders a <tr> element within <tbody>, supporting horizontal and vertical merges.
func (h *HTML) Row(w io.Writer, row []string, ctx tw.Formatting) {
	if !h.tableStarted {
		h.debugLog("WARN: Row called before Start")
		return
	}

	if !h.tbodyStarted {
		h.closePreviousSection()
		classAttr := ""
		if h.config.BodyClass != "" {
			classAttr = fmt.Sprintf(` class="%s"`, h.config.BodyClass)
		}
		h.debugLog("Writing opening <tbody%s> tag", classAttr)
		fmt.Fprintf(h.writer, "<tbody%s>\n", classAttr)
		h.tbodyStarted = true
	}

	h.debugLog("Rendering row data: %v", row)
	numCols := 0
	if len(ctx.Row.Current) > 0 {
		maxKey := -1
		for k := range ctx.Row.Current {
			if k > maxKey {
				maxKey = k
			}
		}
		numCols = maxKey + 1
	} else if len(row) > 0 {
		numCols = len(row)
	}

	indent := "  "
	rowClassAttr := ""
	if h.config.RowClass != "" {
		rowClassAttr = fmt.Sprintf(` class="%s"`, h.config.RowClass)
	}
	fmt.Fprintf(h.writer, "%s<tr%s>", indent, rowClassAttr)

	renderedCols := 0
	for colIdx := 0; renderedCols < numCols && colIdx < numCols; {
		// Skip columns consumed by vertical merges
		if remainingSpan, merging := h.vMergeTrack[colIdx]; merging && remainingSpan > 1 {
			h.debugLog("Row: Skipping render for col %d due to vertical merge (remaining %d)", colIdx, remainingSpan-1)
			h.vMergeTrack[colIdx]--
			if h.vMergeTrack[colIdx] <= 1 {
				delete(h.vMergeTrack, colIdx)
			}
			colIdx++
			continue
		}

		// Render cell
		cellCtx, ok := ctx.Row.Current[colIdx]
		if !ok {
			cellCtx = tw.CellContext{Align: tw.AlignLeft}
		}
		originalContent := ""
		if colIdx < len(row) {
			originalContent = row[colIdx]
		}

		tag, attributes, processedContent := h.renderRowCell(originalContent, cellCtx, false, colIdx)
		fmt.Fprintf(h.writer, "<%s%s>%s</%s>", tag, attributes, processedContent, tag)
		renderedCols++

		// Handle horizontal merges
		hSpan := 1
		if cellCtx.Merge.Horizontal.Present && cellCtx.Merge.Horizontal.Start {
			hSpan = cellCtx.Merge.Horizontal.Span
			renderedCols += (hSpan - 1)
		}
		colIdx += hSpan
	}
	fmt.Fprintf(h.writer, "</tr>\n")
}

// Footer renders the <tfoot> section with footer rows, supporting horizontal merges.
func (h *HTML) Footer(w io.Writer, footers [][]string, ctx tw.Formatting) {
	if !h.tableStarted {
		h.debugLog("WARN: Footer called before Start")
		return
	}
	if len(footers) == 0 || len(footers[0]) == 0 {
		h.debugLog("Footer: No footers")
		return
	}

	h.closePreviousSection()
	classAttr := ""
	if h.config.FooterClass != "" {
		classAttr = fmt.Sprintf(` class="%s"`, h.config.FooterClass)
	}
	fmt.Fprintf(h.writer, "<tfoot%s>\n", classAttr)
	h.tfootStarted = true

	footerRow := footers[0]
	numCols := 0
	if len(ctx.Row.Current) > 0 {
		maxKey := -1
		for k := range ctx.Row.Current {
			if k > maxKey {
				maxKey = k
			}
		}
		numCols = maxKey + 1
	} else if len(footerRow) > 0 {
		numCols = len(footerRow)
	}

	indent := "  "
	rowClassAttr := ""
	if h.config.FooterRowClass != "" {
		rowClassAttr = fmt.Sprintf(` class="%s"`, h.config.FooterRowClass)
	}
	fmt.Fprintf(h.writer, "%s<tr%s>", indent, rowClassAttr)

	renderedCols := 0
	for colIdx := 0; renderedCols < numCols && colIdx < numCols; {
		cellCtx, ok := ctx.Row.Current[colIdx]
		if !ok {
			cellCtx = tw.CellContext{Align: tw.AlignRight}
		}
		originalContent := ""
		if colIdx < len(footerRow) {
			originalContent = footerRow[colIdx]
		}

		tag, attributes, processedContent := h.renderRowCell(originalContent, cellCtx, false, colIdx)
		fmt.Fprintf(h.writer, "<%s%s>%s</%s>", tag, attributes, processedContent, tag)
		renderedCols++

		hSpan := 1
		if cellCtx.Merge.Horizontal.Present && cellCtx.Merge.Horizontal.Start {
			hSpan = cellCtx.Merge.Horizontal.Span
			renderedCols += (hSpan - 1)
		}
		colIdx += hSpan
	}
	fmt.Fprintf(h.writer, "</tr>\n")
	fmt.Fprintln(h.writer, "</tfoot>")
	h.tfootStarted = false
}

// renderRowCell generates HTML for a single cell, handling content escaping, merges, and alignment.
func (h *HTML) renderRowCell(originalContent string, cellCtx tw.CellContext, isHeader bool, colIdx int) (tag, attributes, processedContent string) {
	tag = "td"
	if isHeader {
		tag = "th"
	}

	// Process content
	processedContent = originalContent
	containsNewline := strings.Contains(originalContent, "\n")

	if h.config.EscapeContent {
		if containsNewline {
			const newlinePlaceholder = "[[--HTML_RENDERER_BR_PLACEHOLDER--]]"
			tempContent := strings.ReplaceAll(originalContent, "\n", newlinePlaceholder)
			escapedContent := html.EscapeString(tempContent)
			processedContent = strings.ReplaceAll(escapedContent, newlinePlaceholder, "<br>")
		} else {
			processedContent = html.EscapeString(originalContent)
		}
	} else if containsNewline {
		processedContent = strings.ReplaceAll(originalContent, "\n", "<br>")
	}

	if containsNewline && h.config.AddLinesTag {
		processedContent = "<lines>" + processedContent + "</lines>"
	}

	// Build attributes
	var attrBuilder strings.Builder
	merge := cellCtx.Merge

	if merge.Horizontal.Present && merge.Horizontal.Start && merge.Horizontal.Span > 1 {
		fmt.Fprintf(&attrBuilder, ` colspan="%d"`, merge.Horizontal.Span)
	}

	vSpan := 0
	if !isHeader {
		if merge.Vertical.Present && merge.Vertical.Start {
			vSpan = merge.Vertical.Span
		} else if merge.Hierarchical.Present && merge.Hierarchical.Start {
			vSpan = merge.Hierarchical.Span
		}
		if vSpan > 1 {
			fmt.Fprintf(&attrBuilder, ` rowspan="%d"`, vSpan)
			h.vMergeTrack[colIdx] = vSpan
			h.debugLog("renderRowCell: Tracking rowspan=%d for col %d", vSpan, colIdx)
		}
	}

	if style := getHTMLStyle(cellCtx.Align); style != "" {
		attrBuilder.WriteString(style)
	}
	attributes = attrBuilder.String()
	return
}

// Line is a no-op for HTML rendering, as structural lines are handled by tags.
func (h *HTML) Line(w io.Writer, ctx tw.Formatting) {}

// Reset clears the renderer's internal state, including debug traces and merge tracking.
func (h *HTML) Reset() {
	h.debugLog("HTML.Reset() called.")
	h.tableStarted = false
	h.tbodyStarted = false
	h.tfootStarted = false
	h.vMergeTrack = make(map[int]int)
	h.trace = nil
}

// Close ensures all open HTML tags (<table>, <tbody>, <tfoot>) are properly closed.
func (h *HTML) Close(w io.Writer) error {
	if h.writer == nil {
		return errors.New("HTML Renderer Close called on nil internal writer")
	}

	if h.tableStarted {
		h.debugLog("HTML.Close() called.")
		h.closePreviousSection()
		h.debugLog("Closing <table> tag.")
		_, err := fmt.Fprintln(h.writer, "</table>")
		h.tableStarted = false
		h.tbodyStarted = false
		h.tfootStarted = false
		h.vMergeTrack = make(map[int]int)
		return err
	}
	h.debugLog("HTML.Close() called, but table was not started (no-op).")
	return nil
}
