/**
 * drawio-to-svg.js
 * Converts draw.io mxGraphModel XML to standalone SVG
 * Uses shared XML utilities from ../shared/xml-utils.js
 */

import { attr, decodeEntities, escapeXml, extractCells, extractGraphAttrs, parseStyle } from '../shared/xml-utils.js'

/**
 * Parse mxGraphModel XML into a structured object
 * @param {string} xml
 * @returns {{ graph: object, cells: object[] }}
 */
function parseDrawioXml(xml) {
  const graph = extractGraphAttrs(xml)
  const cells = extractCells(xml)
  return { graph, cells }
}

// ============================================================================
// Shape Classification
// ============================================================================

/**
 * Determine the shape type from a parsed style map
 * @param {Map<string, string>} style
 * @returns {string}
 */
function classifyShape(style) {
  const shape = style.get('shape')
  if (shape === 'text' || style.has('text') || style.has('edgeLabel')) return 'text'
  if (shape === 'cylinder3' || shape === 'cylinder') return 'cylinder'
  if (shape === 'parallelogram') return 'parallelogram'
  if (shape === 'document') return 'document'
  if (shape === 'cloud') return 'cloud'
  if (shape === 'switch') return 'switch'
  if (shape === 'hexagon') return 'hexagon'
  if (shape === 'mxgraph.cisco.firewalls.firewall') return 'firewall'
  if (shape === 'mxgraph.cisco.wireless.access_point') return 'wirelessAp'
  if (style.has('rhombus')) return 'rhombus'
  if (style.has('ellipse')) return 'ellipse'
  const rounded = style.get('rounded')
  const arcSize = Number(style.get('arcSize')) || 0
  if (rounded === '1' && arcSize >= 50) return 'stadium'
  if (rounded === '1') return 'roundedRect'
  return 'rect'
}

// ============================================================================
// Arrow Marker Definitions
// ============================================================================

const ARROW_TYPES = ['block', 'open', 'classic', 'diamond']

/**
 * Build SVG <defs> with arrow markers
 * @returns {string}
 */
function buildMarkerDefs() {
  const markers = []

  // block arrow (filled triangle)
  markers.push(
    '<marker id="arrow-block" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">',
    '  <path d="M 0 0 L 10 5 L 0 10 Z" fill="currentColor"/>',
    '</marker>'
  )

  // open arrow (chevron)
  markers.push(
    '<marker id="arrow-open" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">',
    '  <path d="M 0 0 L 10 5 L 0 10" fill="none" stroke="currentColor" stroke-width="1.5"/>',
    '</marker>'
  )

  // classic arrow (filled arrow)
  markers.push(
    '<marker id="arrow-classic" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">',
    '  <path d="M 0 0 L 10 5 L 0 10 L 3 5 Z" fill="currentColor"/>',
    '</marker>'
  )

  // diamond
  markers.push(
    '<marker id="arrow-diamond" viewBox="0 0 12 12" refX="12" refY="6" markerWidth="10" markerHeight="10" orient="auto-start-reverse">',
    '  <path d="M 0 6 L 6 0 L 12 6 L 6 12 Z" fill="currentColor"/>',
    '</marker>'
  )

  return `<defs>\n${markers.join('\n')}\n</defs>`
}

/**
 * Resolve an arrow type name to a marker URL reference
 * @param {string} arrowType
 * @param {'start'|'end'} position
 * @returns {string} marker-start or marker-end attribute, or empty string
 */
function markerRef(arrowType, position) {
  if (!arrowType || arrowType === 'none') return ''
  const id = ARROW_TYPES.includes(arrowType) ? `arrow-${arrowType}` : 'arrow-block'
  const attrName = position === 'start' ? 'marker-start' : 'marker-end'
  return ` ${attrName}="url(#${id})"`
}

// ============================================================================
// Shape SVG Renderers
// ============================================================================

/**
 * Render a vertex cell to SVG elements
 * @param {object} cell - parsed cell
 * @param {Map<string, string>} style - parsed style
 * @returns {string} SVG markup
 */
function renderVertex(cell, style) {
  const geo = cell.geometry || { x: 0, y: 0, width: 120, height: 60 }
  const { x, y, width, height } = geo

  const fillColor = style.get('fillColor') || '#FFFFFF'
  const strokeColor = style.get('strokeColor') || '#000000'
  const strokeWidth = Number(style.get('strokeWidth')) || 1
  const fontColor = style.get('fontColor') || '#000000'
  const fontSize = Number(style.get('fontSize')) || 12
  const fontFamily = style.get('fontFamily') || 'sans-serif'
  const fontStyleBits = Number(style.get('fontStyle')) || 0

  let dashAttr = ''
  if (style.get('dashed') === '1') {
    const pattern = style.get('dashPattern') || '3 3'
    dashAttr = ` stroke-dasharray="${pattern}"`
  }

  const shapeType = classifyShape(style)
  const parts = []
  const baseAttrs = `fill="${fillColor}" stroke="${strokeColor}" stroke-width="${strokeWidth}"${dashAttr}`

  switch (shapeType) {
    case 'text': {
      break
    }

    case 'roundedRect': {
      const rx = Number(style.get('arcSize')) || 8
      parts.push(`<rect x="${x}" y="${y}" width="${width}" height="${height}" rx="${rx}" ${baseAttrs}/>`)
      break
    }

    case 'stadium': {
      const rx = height / 2
      parts.push(`<rect x="${x}" y="${y}" width="${width}" height="${height}" rx="${rx}" ${baseAttrs}/>`)
      break
    }

    case 'cylinder': {
      const ellipseRY = Math.min(12, height * 0.15)
      // Body rectangle
      parts.push(
        `<rect x="${x}" y="${y + ellipseRY}" width="${width}" height="${height - ellipseRY * 2}" ${baseAttrs}/>`
      )
      // Bottom ellipse
      parts.push(
        `<ellipse cx="${x + width / 2}" cy="${y + height - ellipseRY}" rx="${width / 2}" ry="${ellipseRY}" ${baseAttrs}/>`
      )
      // Top ellipse (drawn last so it's on top)
      parts.push(
        `<ellipse cx="${x + width / 2}" cy="${y + ellipseRY}" rx="${width / 2}" ry="${ellipseRY}" ${baseAttrs}/>`
      )
      // Side lines connecting top and bottom ellipses
      parts.push(
        `<line x1="${x}" y1="${y + ellipseRY}" x2="${x}" y2="${y + height - ellipseRY}" stroke="${strokeColor}" stroke-width="${strokeWidth}"/>`
      )
      parts.push(
        `<line x1="${x + width}" y1="${y + ellipseRY}" x2="${x + width}" y2="${y + height - ellipseRY}" stroke="${strokeColor}" stroke-width="${strokeWidth}"/>`
      )
      break
    }

    case 'rhombus': {
      const cx = x + width / 2
      const cy = y + height / 2
      const points = `${cx},${y} ${x + width},${cy} ${cx},${y + height} ${x},${cy}`
      parts.push(`<polygon points="${points}" ${baseAttrs}/>`)
      break
    }

    case 'ellipse': {
      const cx = x + width / 2
      const cy = y + height / 2
      parts.push(`<ellipse cx="${cx}" cy="${cy}" rx="${width / 2}" ry="${height / 2}" ${baseAttrs}/>`)
      break
    }

    case 'parallelogram': {
      const skew = width * 0.2
      const points = `${x + skew},${y} ${x + width},${y} ${x + width - skew},${y + height} ${x},${y + height}`
      parts.push(`<polygon points="${points}" ${baseAttrs}/>`)
      break
    }

    case 'hexagon': {
      const inset = Math.min(width * 0.22, 24)
      const points = [
        `${x + inset},${y}`,
        `${x + width - inset},${y}`,
        `${x + width},${y + height / 2}`,
        `${x + width - inset},${y + height}`,
        `${x + inset},${y + height}`,
        `${x},${y + height / 2}`
      ].join(' ')
      parts.push(`<polygon points="${points}" ${baseAttrs}/>`)
      break
    }

    case 'switch': {
      const inset = Math.min(width * 0.18, 18)
      const d = [
        `M ${x + inset} ${y}`,
        `L ${x + width - inset} ${y}`,
        `L ${x + width} ${y + height / 2}`,
        `L ${x + width - inset} ${y + height}`,
        `L ${x + inset} ${y + height}`,
        `L ${x} ${y + height / 2}`,
        'Z'
      ].join(' ')
      const portY1 = y + height * 0.35
      const portY2 = y + height * 0.65
      parts.push(`<path d="${d}" ${baseAttrs}/>`)
      parts.push(
        `<line x1="${x + inset}" y1="${portY1}" x2="${x + width - inset}" y2="${portY1}" stroke="${strokeColor}" stroke-width="${strokeWidth}"/>`
      )
      parts.push(
        `<line x1="${x + inset}" y1="${portY2}" x2="${x + width - inset}" y2="${portY2}" stroke="${strokeColor}" stroke-width="${strokeWidth}"/>`
      )
      break
    }

    case 'document': {
      const waveH = height * 0.1
      const d = [
        `M ${x} ${y}`,
        `L ${x + width} ${y}`,
        `L ${x + width} ${y + height - waveH}`,
        `Q ${x + width * 0.75} ${y + height + waveH} ${x + width / 2} ${y + height - waveH}`,
        `Q ${x + width * 0.25} ${y + height - waveH * 3} ${x} ${y + height - waveH}`,
        'Z'
      ].join(' ')
      parts.push(`<path d="${d}" ${baseAttrs}/>`)
      break
    }

    case 'cloud': {
      // Simplified cloud: overlapping circles
      const cx = x + width / 2
      const cy = y + height / 2
      const rx = width * 0.45
      const ry = height * 0.35
      const d = [
        `M ${x + width * 0.25} ${cy + ry * 0.5}`,
        `A ${rx * 0.5} ${ry * 0.6} 0 0 1 ${x + width * 0.15} ${cy - ry * 0.2}`,
        `A ${rx * 0.5} ${ry * 0.6} 0 0 1 ${x + width * 0.35} ${cy - ry * 0.8}`,
        `A ${rx * 0.5} ${ry * 0.5} 0 0 1 ${cx} ${y + height * 0.15}`,
        `A ${rx * 0.5} ${ry * 0.5} 0 0 1 ${x + width * 0.7} ${cy - ry * 0.7}`,
        `A ${rx * 0.6} ${ry * 0.7} 0 0 1 ${x + width * 0.85} ${cy}`,
        `A ${rx * 0.5} ${ry * 0.6} 0 0 1 ${x + width * 0.75} ${cy + ry * 0.7}`,
        `A ${rx * 0.6} ${ry * 0.4} 0 0 1 ${x + width * 0.5} ${cy + ry * 0.8}`,
        `A ${rx * 0.5} ${ry * 0.4} 0 0 1 ${x + width * 0.25} ${cy + ry * 0.5}`,
        'Z'
      ].join(' ')
      parts.push(`<path d="${d}" ${baseAttrs}/>`)
      break
    }

    case 'firewall': {
      const archHeight = height * 0.18
      const bodyTop = y + archHeight
      const brickWidth = width / 4
      const brickHeight = (height - archHeight) / 3
      const outer = [
        `M ${x} ${bodyTop}`,
        `Q ${x + width / 2} ${y - archHeight * 0.2} ${x + width} ${bodyTop}`,
        `L ${x + width} ${y + height}`,
        `L ${x} ${y + height}`,
        'Z'
      ].join(' ')
      const mortar = [
        `M ${x + brickWidth} ${bodyTop} L ${x + brickWidth} ${y + height}`,
        `M ${x + brickWidth * 2} ${bodyTop} L ${x + brickWidth * 2} ${y + height}`,
        `M ${x + brickWidth * 3} ${bodyTop} L ${x + brickWidth * 3} ${y + height}`,
        `M ${x} ${bodyTop + brickHeight} L ${x + width} ${bodyTop + brickHeight}`,
        `M ${x} ${bodyTop + brickHeight * 2} L ${x + width} ${bodyTop + brickHeight * 2}`
      ].join(' ')
      parts.push(`<path d="${outer}" ${baseAttrs}/>`)
      parts.push(
        `<path d="${mortar}" fill="none" stroke="${strokeColor}" stroke-width="${Math.max(strokeWidth * 0.8, 1)}"/>`
      )
      break
    }

    case 'wirelessAp': {
      const cx = x + width / 2
      const cy = y + height / 2
      const baseRy = height * 0.12
      const baseY = y + height * 0.78
      const arc1 = [
        `M ${cx - width * 0.16} ${cy + height * 0.02}`,
        `Q ${cx} ${cy - height * 0.18} ${cx + width * 0.16} ${cy + height * 0.02}`
      ].join(' ')
      const arc2 = [
        `M ${cx - width * 0.28} ${cy + height * 0.1}`,
        `Q ${cx} ${cy - height * 0.32} ${cx + width * 0.28} ${cy + height * 0.1}`
      ].join(' ')
      parts.push(`<ellipse cx="${cx}" cy="${baseY}" rx="${width * 0.16}" ry="${baseRy}" ${baseAttrs}/>`)
      parts.push(
        `<line x1="${cx}" y1="${baseY - baseRy}" x2="${cx}" y2="${cy + height * 0.12}" stroke="${strokeColor}" stroke-width="${strokeWidth}"/>`
      )
      parts.push(`<path d="${arc1}" fill="none" stroke="${strokeColor}" stroke-width="${strokeWidth}"/>`)
      parts.push(`<path d="${arc2}" fill="none" stroke="${strokeColor}" stroke-width="${strokeWidth}"/>`)
      break
    }

    default: {
      // Plain rectangle
      parts.push(`<rect x="${x}" y="${y}" width="${width}" height="${height}" ${baseAttrs}/>`)
      break
    }
  }

  // Text label
  const label = decodeEntities(cell.value)
  if (label) {
    const align = style.get('align') || 'center'
    const verticalAlign = style.get('verticalAlign') || 'middle'
    const spacingLeft = Number(style.get('spacingLeft')) || 0
    const spacingRight = Number(style.get('spacingRight')) || 0
    const spacingTop = Number(style.get('spacingTop')) || 0
    const spacingBottom = Number(style.get('spacingBottom')) || 0
    const textX = align === 'left' ? x + spacingLeft : align === 'right' ? x + width - spacingRight : x + width / 2
    // For cylinders the top ellipse pushes the usable body down; bias the
    // vertical center so multi-line labels sit inside the body, not on the rim.
    const cylinderBias = shapeType === 'cylinder' ? Math.min(22, height * 0.18) : 0
    const textY =
      verticalAlign === 'top'
        ? y + spacingTop
        : verticalAlign === 'bottom'
          ? y + height - spacingBottom
          : y + height / 2 + cylinderBias
    const textAnchor = align === 'left' ? 'start' : align === 'right' ? 'end' : 'middle'
    const dominantBaseline = verticalAlign === 'top' ? 'hanging' : verticalAlign === 'bottom' ? 'auto' : 'central'
    const fontWeightAttr = fontStyleBits & 1 ? ' font-weight="700"' : ''
    const fontStyleAttr = fontStyleBits & 2 ? ' font-style="italic"' : ''
    const lines = String(label).split(/\\n|\n/)
    const lineHeight = fontSize * 1.35
    // Vertically center the multi-line block around textY
    const firstDy = -((lines.length - 1) / 2) * lineHeight
    const tspans = lines
      .map((ln, i) => {
        const dy = i === 0 ? firstDy : lineHeight
        return `<tspan x="${textX}" dy="${dy}">${escapeXml(ln)}</tspan>`
      })
      .join('')
    parts.push(
      `<text x="${textX}" y="${textY}" text-anchor="${textAnchor}" dominant-baseline="${dominantBaseline}" ` +
        `font-family="${escapeXml(fontFamily)}" font-size="${fontSize}" fill="${fontColor}"${fontWeightAttr}${fontStyleAttr}>` +
        `${tspans}</text>`
    )
  }

  return parts.join('\n')
}

// ============================================================================
// Edge Rendering
// ============================================================================

/**
 * Compute center point of a cell's geometry
 * @param {object} cell
 * @returns {{ x: number, y: number }}
 */
function cellCenter(cell) {
  const geo = cell.geometry || { x: 0, y: 0, width: 120, height: 60 }
  return {
    x: geo.x + geo.width / 2,
    y: geo.y + geo.height / 2
  }
}

/**
 * Render an edge cell to SVG elements
 * @param {object} cell - parsed edge cell
 * @param {Map<string, string>} style - parsed style
 * @param {Map<string, object>} cellMap - id → cell lookup
 * @returns {string} SVG markup
 */
function renderEdge(cell, style, cellMap, suppressLabel = false) {
  const strokeColor = style.get('strokeColor') || '#000000'
  const strokeWidth = Number(style.get('strokeWidth')) || 1
  const fontColor = style.get('fontColor') || '#000000'
  const fontSize = Number(style.get('fontSize')) || 11

  let dashAttr = ''
  if (style.get('dashed') === '1') {
    const pattern = style.get('dashPattern') || '3 3'
    dashAttr = ` stroke-dasharray="${pattern}"`
  }

  const sourceCell = cell.source ? cellMap.get(cell.source) : null
  const targetCell = cell.target ? cellMap.get(cell.target) : null

  const parts = []

  // Arrow markers
  const endArrow = style.get('endArrow') || 'classic'
  const startArrow = style.get('startArrow') || ''
  const drawEndArrow = endArrow && endArrow !== 'none'
  const drawStartArrow = startArrow && startArrow !== 'none'

  // Build an orthogonal path that exits/enters node boundaries instead of
  // cutting a diagonal line through the box centers.
  let pathD = ''
  let labelX = 0
  let labelY = 0
  if (sourceCell && targetCell) {
    const sg = sourceCell.geometry || { x: 0, y: 0, width: 120, height: 60 }
    const tg = targetCell.geometry || { x: 0, y: 0, width: 120, height: 60 }
    const sc = { x: sg.x + sg.width / 2, y: sg.y + sg.height / 2 }
    const tc = { x: tg.x + tg.width / 2, y: tg.y + tg.height / 2 }

    const dx = tc.x - sc.x
    const dy = tc.y - sc.y
    let p1, p2
    if (Math.abs(dy) >= Math.abs(dx)) {
      // Predominantly vertical: exit top/bottom, enter opposite side
      p1 = { x: sc.x, y: dy >= 0 ? sg.y + sg.height : sg.y }
      p2 = { x: tc.x, y: dy >= 0 ? tg.y : tg.y + tg.height }
    } else {
      // Predominantly horizontal: exit left/right
      p1 = { x: dx >= 0 ? sg.x + sg.width : sg.x, y: sc.y }
      p2 = { x: dx >= 0 ? tg.x : tg.x + tg.width, y: tc.y }
    }

    // Orthogonal mid-routing when both axes differ
    const pts = [p1]
    if (p1.x !== p2.x && p1.y !== p2.y) {
      if (Math.abs(dy) >= Math.abs(dx)) {
        const midY = (p1.y + p2.y) / 2
        pts.push({ x: p1.x, y: midY }, { x: p2.x, y: midY })
      } else {
        const midX = (p1.x + p2.x) / 2
        pts.push({ x: midX, y: p1.y }, { x: midX, y: p2.y })
      }
    }
    pts.push(p2)
    pathD = pts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ')
    const midPt = pts[Math.floor(pts.length / 2)]
    labelX = midPt.x
    labelY = midPt.y

    // Arrowhead at p2, oriented from the previous point
    var arrowPrev = pts[pts.length - 2]
    var arrowEnd = p2
    var arrowStartPrev = pts[1] || p2
    var arrowStart = p1
  } else {
    // Fallback: straight line center-to-center
    const sc = sourceCell ? cellCenter(sourceCell) : { x: 0, y: 0 }
    const tc = targetCell ? cellCenter(targetCell) : { x: 100, y: 100 }
    pathD = `M ${sc.x} ${sc.y} L ${tc.x} ${tc.y}`
    labelX = (sc.x + tc.x) / 2
    labelY = (sc.y + tc.y) / 2
    var arrowPrev = sc
    var arrowEnd = tc
    var arrowStartPrev = tc
    var arrowStart = sc
  }

  parts.push(
    `<path d="${pathD}" ` +
      `stroke="${strokeColor}" stroke-width="${strokeWidth}"${dashAttr} fill="none"/>`
  )

  // Draw triangular arrowheads sized to the stroke (avoids oversized markers)
  const ah = 5 + strokeWidth * 1.8
  function arrowHead(tip, from, color) {
    const ang = Math.atan2(tip.y - from.y, tip.x - from.x)
    const a1 = ang + Math.PI - 0.42
    const a2 = ang + Math.PI + 0.42
    const x1 = tip.x + ah * Math.cos(a1)
    const y1 = tip.y + ah * Math.sin(a1)
    const x2 = tip.x + ah * Math.cos(a2)
    const y2 = tip.y + ah * Math.sin(a2)
    return `<path d="M ${tip.x} ${tip.y} L ${x1} ${y1} L ${x2} ${y2} Z" fill="${color}"/>`
  }
  if (drawEndArrow) parts.push(arrowHead(arrowEnd, arrowPrev, strokeColor))
  if (drawStartArrow) parts.push(arrowHead(arrowStart, arrowStartPrev, strokeColor))

  // Edge label — lift slightly off the line and use an opaque rounded pill so
  // it never appears to merge with a crossing edge underneath.
  const label = decodeEntities(cell.value)
  if (label && !suppressLabel) {
    const padX = 6
    const halfW = label.length * fontSize * 0.32 + padX
    const pillH = fontSize + 8
    const ly = labelY - 4
    parts.push(
      `<rect x="${labelX - halfW}" y="${ly - pillH + 4}" ` +
        `width="${halfW * 2}" height="${pillH}" rx="3" fill="#FAF8F4" stroke="#FAF8F4" stroke-width="1"/>`
    )
    parts.push(
      `<text x="${labelX}" y="${ly - 1}" text-anchor="middle" dominant-baseline="auto" ` +
        `font-size="${fontSize}" fill="${fontColor}">${escapeXml(label)}</text>`
    )
  }

  return parts.join('\n')
}

function renderEdgeLabel(cell, style, cellMap) {
  const parentEdge = cell.parent ? cellMap.get(cell.parent) : null
  if (!parentEdge) return ''
  const sourceCell = parentEdge.source ? cellMap.get(parentEdge.source) : null
  const targetCell = parentEdge.target ? cellMap.get(parentEdge.target) : null
  if (!sourceCell || !targetCell) return ''

  const source = cellCenter(sourceCell)
  const target = cellCenter(targetCell)
  const rawLabelX = Number(cell.geometry?.labelX)
  const labelX = Number.isFinite(rawLabelX) ? rawLabelX : 0.5
  const offset = cell.geometry?.offset || { x: 0, y: -6 }
  const x = source.x + (target.x - source.x) * labelX + offset.x
  const y = source.y + (target.y - source.y) * labelX + offset.y

  const label = decodeEntities(cell.value)
  if (!label) return ''
  const fontColor = style.get('fontColor') || '#000000'
  const fontSize = Number(style.get('fontSize')) || 11
  const fontFamily = style.get('fontFamily') || 'sans-serif'

  return (
    `<text x="${x}" y="${y}" text-anchor="middle" dominant-baseline="central" ` +
    `font-family="${escapeXml(fontFamily)}" font-size="${fontSize}" fill="${fontColor}">` +
    `${escapeXml(label)}</text>`
  )
}

// ============================================================================
// Main Converter
// ============================================================================

/**
 * Convert draw.io mxGraphModel XML to standalone SVG
 * @param {string} xmlString - draw.io XML content
 * @returns {string} SVG markup
 * @throws {Error} if input is empty or not a string
 */
export function drawioToSvg(xmlString) {
  if (!xmlString || typeof xmlString !== 'string' || xmlString.trim().length === 0) {
    throw new Error('Input XML string must be non-empty')
  }

  const { graph, cells } = parseDrawioXml(xmlString)

  // Build cell lookup map
  const cellMap = new Map()
  for (const cell of cells) {
    if (cell.id) cellMap.set(cell.id, cell)
  }

  // Separate vertices and edges
  const vertices = cells.filter((c) => c.vertex && c.parent !== '0')
  const edges = cells.filter((c) => c.edge)
  const edgeLabels = vertices.filter((v) => parseStyle(v.style).has('edgeLabel'))
  const edgeLabelParents = new Set(edgeLabels.map((v) => v.parent).filter(Boolean))
  const shapeVertices = vertices.filter((v) => !edgeLabelParents.has(v.id) && !parseStyle(v.style).has('edgeLabel'))

  // Calculate viewBox dimensions from content if default
  let svgWidth = graph.pageWidth
  let svgHeight = graph.pageHeight

  // Expand viewBox if any shape extends beyond page bounds
  for (const v of shapeVertices) {
    if (v.geometry) {
      svgWidth = Math.max(svgWidth, v.geometry.x + v.geometry.width + 20)
      svgHeight = Math.max(svgHeight, v.geometry.y + v.geometry.height + 20)
    }
  }

  // Encode original XML as base64 for round-trip editing
  const base64Xml = Buffer.from(xmlString, 'utf-8').toString('base64')

  // Build SVG
  const svgParts = []
  svgParts.push(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${svgWidth}" height="${svgHeight}" ` +
      `viewBox="0 0 ${svgWidth} ${svgHeight}" data-drawio="${base64Xml}">`
  )

  // Defs (arrow markers)
  svgParts.push(buildMarkerDefs())

  // Background
  if (graph.background && graph.background !== 'none') {
    svgParts.push(`<rect width="100%" height="100%" fill="${graph.background}"/>`)
  }

  // Render vertices first, then edges on top
  for (const v of shapeVertices) {
    const style = parseStyle(v.style)
    svgParts.push(renderVertex(v, style))
  }

  for (const e of edges) {
    const style = parseStyle(e.style)
    svgParts.push(renderEdge(e, style, cellMap, edgeLabelParents.has(e.id)))
  }

  for (const labelCell of edgeLabels) {
    const style = parseStyle(labelCell.style)
    const rendered = renderEdgeLabel(labelCell, style, cellMap)
    if (rendered) svgParts.push(rendered)
  }

  svgParts.push('</svg>')
  return svgParts.join('\n')
}
