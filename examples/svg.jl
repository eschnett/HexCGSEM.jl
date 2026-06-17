# Self-contained SVG plotting helpers shared by the HexCGSEM example scripts —
# a viridis-like colormap with colorbar, a grid `save_heatmap`, and a point
# `save_scatter`. No plotting-package dependency.

using Printf

const _VIRIDIS = ((0.267, 0.005, 0.329), (0.231, 0.318, 0.545),
                  (0.128, 0.567, 0.551), (0.369, 0.789, 0.383), (0.993, 0.906, 0.144))

# Viridis-ish color for `t ∈ [0, 1]` (NaN → light gray) as an SVG hex string.
function _color(t)
    isfinite(t) || return "#eeeeee"
    t = clamp(t, 0.0, 1.0) * (length(_VIRIDIS) - 1)
    k = clamp(floor(Int, t), 0, length(_VIRIDIS) - 2)
    f = t - k
    a = _VIRIDIS[k + 1]
    b = _VIRIDIS[k + 2]
    rgb = ntuple(c -> round(Int, 255 * (a[c] + f * (b[c] - a[c]))), 3)
    return @sprintf("#%02x%02x%02x", rgb[1], rgb[2], rgb[3])
end

const _PAD = 55
const _SZ = 380
const _CBW = 22
_svg_W() = _PAD + _SZ + 70 + _CBW
_svg_H() = _PAD + _SZ + 20

# Vertical colorbar [lo, hi] at x = `cbx`, height `sz` from y = `pad`.
function _colorbar!(io, cbx, pad, sz, cbw, lo, hi)
    nseg = 64
    for s in 0:(nseg - 1)
        y = pad + sz - (s + 1) * sz / nseg
        println(io, @sprintf("""<rect x="%.1f" y="%.2f" width="%d" height="%.2f" fill="%s"/>""",
                             cbx, y, cbw, sz / nseg + 0.6, _color(s / (nseg - 1))))
    end
    println(io, """<rect x="$cbx" y="$pad" width="$cbw" height="$sz" fill="none" stroke="#333"/>""")
    println(io, @sprintf("""<text x="%.1f" y="%.1f" font-size="11">%.3g</text>""", cbx + cbw + 4, pad + 6, hi))
    println(io, @sprintf("""<text x="%.1f" y="%.1f" font-size="11">%.3g</text>""", cbx + cbw + 4, pad + sz, lo))
    return nothing
end

"""
    save_heatmap(path, gx, Z, title; vmin, vmax)

SVG heatmap of `Z[i, j]` over the square grid `(gx[i], gx[j])`; NaN cells are
drawn light gray. `+y` points up.
"""
function save_heatmap(path, gx, Z, title; vmin = nothing, vmax = nothing)
    n = length(gx)
    vals = filter(isfinite, vec(Z))
    lo = vmin === nothing ? (isempty(vals) ? 0.0 : minimum(vals)) : vmin
    hi = vmax === nothing ? (isempty(vals) ? 1.0 : maximum(vals)) : vmax
    hi <= lo && (hi = lo + 1)
    pad = _PAD
    sz = _SZ
    cbw = _CBW
    cw = sz / n
    io = IOBuffer()
    println(io, """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 $(_svg_W()) $(_svg_H())" font-family="sans-serif">""")
    println(io, """<text x="$pad" y="28" font-size="16" font-weight="bold">$title</text>""")
    @inbounds for j in 1:n, i in 1:n
        v = Z[i, j]
        x = pad + (i - 1) * cw
        y = pad + (n - j) * cw          # flip so +y points up
        t = isfinite(v) ? (v - lo) / (hi - lo) : NaN
        println(io, @sprintf("""<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" fill="%s"/>""",
                             x, y, cw + 0.6, cw + 0.6, _color(t)))
    end
    println(io, """<rect x="$pad" y="$pad" width="$sz" height="$sz" fill="none" stroke="#333"/>""")
    _colorbar!(io, pad + sz + 30, pad, sz, cbw, lo, hi)
    println(io, "</svg>")
    write(path, take!(io))
    return path
end

"""
    save_scatter(path, xs, ys, vals, title; vmin, vmax, xlim, ylim, r)

SVG scatter plot: a dot at each physical point `(xs[i], ys[i])` coloured by
`vals[i]`, over the box `xlim × ylim` (defaults to the data extent). Faithful
for per-element spectral-element nodal data — no point-location/interpolation.
"""
function save_scatter(path, xs, ys, vals, title; vmin = nothing, vmax = nothing,
                      xlim = nothing, ylim = nothing, r = 1.8)
    finite = filter(isfinite, vals)
    lo = vmin === nothing ? (isempty(finite) ? 0.0 : minimum(finite)) : vmin
    hi = vmax === nothing ? (isempty(finite) ? 1.0 : maximum(finite)) : vmax
    hi <= lo && (hi = lo + 1)
    xlo, xhi = xlim === nothing ? extrema(xs) : xlim
    ylo, yhi = ylim === nothing ? extrema(ys) : ylim
    pad = _PAD
    sz = _SZ
    cbw = _CBW
    sx = sz / (xhi - xlo)
    sy = sz / (yhi - ylo)
    io = IOBuffer()
    println(io, """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 $(_svg_W()) $(_svg_H())" font-family="sans-serif">""")
    println(io, """<text x="$pad" y="28" font-size="16" font-weight="bold">$title</text>""")
    println(io, """<rect x="$pad" y="$pad" width="$sz" height="$sz" fill="#ffffff" stroke="#333"/>""")
    @inbounds for i in eachindex(vals)
        v = vals[i]
        isfinite(v) || continue
        (xlo ≤ xs[i] ≤ xhi && ylo ≤ ys[i] ≤ yhi) || continue
        px = pad + (xs[i] - xlo) * sx
        py = pad + (yhi - ys[i]) * sy          # flip so +y points up
        println(io, @sprintf("""<circle cx="%.2f" cy="%.2f" r="%.2f" fill="%s"/>""",
                             px, py, r, _color((v - lo) / (hi - lo))))
    end
    _colorbar!(io, pad + sz + 30, pad, sz, cbw, lo, hi)
    println(io, "</svg>")
    write(path, take!(io))
    return path
end
