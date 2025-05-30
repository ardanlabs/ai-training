package renderer

import (
	"fmt"
	"github.com/fatih/color"
	"github.com/olekukonko/tablewriter/tw"
)

// defaultBlueprint returns a default Rendition for ASCII table rendering with borders and light symbols.
func defaultBlueprint() tw.Rendition {
	return tw.Rendition{
		Borders: tw.Border{
			Left:   tw.On,
			Right:  tw.On,
			Top:    tw.On,
			Bottom: tw.On,
		},
		Settings: tw.Settings{
			Separators: tw.Separators{
				ShowHeader:     tw.On,
				ShowFooter:     tw.On,
				BetweenRows:    tw.Off,
				BetweenColumns: tw.On,
			},
			Lines: tw.Lines{
				ShowTop:        tw.On,
				ShowBottom:     tw.On,
				ShowHeaderLine: tw.On,
				ShowFooterLine: tw.On,
			},

			CompactMode: tw.Off,
		},
		Symbols:   tw.NewSymbols(tw.StyleLight),
		Streaming: true,
	}
}

// defaultColorized returns a default ColorizedConfig optimized for dark terminal backgrounds with colored headers, rows, and borders.
func defaultColorized() ColorizedConfig {
	return ColorizedConfig{
		Borders: tw.Border{Left: tw.On, Right: tw.On, Top: tw.On, Bottom: tw.On},
		Settings: tw.Settings{
			Separators: tw.Separators{
				ShowHeader:     tw.On,
				ShowFooter:     tw.On,
				BetweenRows:    tw.Off,
				BetweenColumns: tw.On,
			},
			Lines: tw.Lines{
				ShowTop:        tw.On,
				ShowBottom:     tw.On,
				ShowHeaderLine: tw.On,
				ShowFooterLine: tw.On,
			},

			CompactMode: tw.Off,
		},
		Header: Tint{
			FG: Colors{color.FgWhite, color.Bold},
			BG: Colors{color.BgBlack},
		},
		Column: Tint{
			FG: Colors{color.FgCyan},
			BG: Colors{color.BgBlack},
		},
		Footer: Tint{
			FG: Colors{color.FgYellow},
			BG: Colors{color.BgBlack},
		},
		Border: Tint{
			FG: Colors{color.FgWhite},
			BG: Colors{color.BgBlack},
		},
		Separator: Tint{
			FG: Colors{color.FgWhite},
			BG: Colors{color.BgBlack},
		},
		Symbols: tw.NewSymbols(tw.StyleLight),
	}
}

// defaultOceanRendererConfig returns a base tw.Rendition for the Ocean renderer.
func defaultOceanRendererConfig() tw.Rendition {

	return tw.Rendition{
		Borders: tw.Border{
			Left: tw.On, Right: tw.On, Top: tw.On, Bottom: tw.On,
		},
		Settings: tw.Settings{
			Separators: tw.Separators{
				ShowHeader:     tw.On,
				ShowFooter:     tw.Off,
				BetweenRows:    tw.Off,
				BetweenColumns: tw.On,
			},
			Lines: tw.Lines{
				ShowTop:        tw.On,
				ShowBottom:     tw.On,
				ShowHeaderLine: tw.On,
				ShowFooterLine: tw.Off,
			},

			CompactMode: tw.Off,
		},
		Symbols:   tw.NewSymbols(tw.StyleDefault),
		Streaming: true,
	}
}

// getHTMLStyle remains the same
func getHTMLStyle(align tw.Align) string {
	styleContent := ""
	switch align {
	case tw.AlignRight:
		styleContent = "text-align: right;"
	case tw.AlignCenter:
		styleContent = "text-align: center;"
	case tw.AlignLeft:
		styleContent = "text-align: left;"
	}
	if styleContent != "" {
		return fmt.Sprintf(` style="%s"`, styleContent)
	}
	return ""
}

// mergeLines combines default and override line settings, preserving defaults for unset (zero) overrides.
func mergeLines(defaults, overrides tw.Lines) tw.Lines {
	if overrides.ShowTop != 0 {
		defaults.ShowTop = overrides.ShowTop
	}
	if overrides.ShowBottom != 0 {
		defaults.ShowBottom = overrides.ShowBottom
	}
	if overrides.ShowHeaderLine != 0 {
		defaults.ShowHeaderLine = overrides.ShowHeaderLine
	}
	if overrides.ShowFooterLine != 0 {
		defaults.ShowFooterLine = overrides.ShowFooterLine
	}
	return defaults
}

// mergeSeparators combines default and override separator settings, preserving defaults for unset (zero) overrides.
func mergeSeparators(defaults, overrides tw.Separators) tw.Separators {
	if overrides.ShowHeader != 0 {
		defaults.ShowHeader = overrides.ShowHeader
	}
	if overrides.ShowFooter != 0 {
		defaults.ShowFooter = overrides.ShowFooter
	}
	if overrides.BetweenRows != 0 {
		defaults.BetweenRows = overrides.BetweenRows
	}
	if overrides.BetweenColumns != 0 {
		defaults.BetweenColumns = overrides.BetweenColumns
	}
	return defaults
}

// mergeSettings combines default and override settings, preserving defaults for unset (zero) overrides.
func mergeSettings(defaults, overrides tw.Settings) tw.Settings {
	if overrides.Separators.ShowHeader != 0 {
		defaults.Separators.ShowHeader = overrides.Separators.ShowHeader
	}
	if overrides.Separators.ShowFooter != 0 {
		defaults.Separators.ShowFooter = overrides.Separators.ShowFooter
	}
	if overrides.Separators.BetweenRows != 0 {
		defaults.Separators.BetweenRows = overrides.Separators.BetweenRows
	}
	if overrides.Separators.BetweenColumns != 0 {
		defaults.Separators.BetweenColumns = overrides.Separators.BetweenColumns
	}
	if overrides.Lines.ShowTop != 0 {
		defaults.Lines.ShowTop = overrides.Lines.ShowTop
	}
	if overrides.Lines.ShowBottom != 0 {
		defaults.Lines.ShowBottom = overrides.Lines.ShowBottom
	}
	if overrides.Lines.ShowHeaderLine != 0 {
		defaults.Lines.ShowHeaderLine = overrides.Lines.ShowHeaderLine
	}
	if overrides.Lines.ShowFooterLine != 0 {
		defaults.Lines.ShowFooterLine = overrides.Lines.ShowFooterLine
	}

	if overrides.CompactMode != 0 {
		defaults.CompactMode = overrides.CompactMode
	}
	return defaults
}
