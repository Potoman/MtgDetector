"""
Set symbols (expansion symbols) in the colors of the rarities, from the black SVG files of scryfall.

download.py (choice 3: set) downloads them into <scryfall dir>/set/: set_<code>.svg as downloaded, and
sets.json (the scryfall sets: release date, type... and "symbol_file"). This file adds, in the same
folder, one SVG per rarity: set_<code>_<rarity>.svg
- common, uncommon, rare: the sets released since Exodus (1998, the first set with rarity colors),
- mythic: the sets released since Shards of Alara (2008, the first set with mythic rares).
The older sets keep only their black symbol.

Look of each rarity: PAINT_FILE (resources/set_symbol_paint.json, see its "_help"), the same file as
the generated symbols of set_symbol_data_set.py. The symbol has 3 areas: 'symbol' (the black parts of
the downloaded SVG), 'inner' (its holes: the regions it encloses, always filled, as printed) and
'outline' (around it, OUTLINE wide; it also fills the thin gaps between the parts of the symbol). Each
area gets a paint: the rarity color (a flat color, or a gradient dark -> light -> dark from left to
right, slightly tilted, as on the cards), the line color of the rarity, white or black. The style of
a set: the one of its "sets" entry, else the default one since Exodus ("styles"), the one with the
biggest weight (a choice {paint: weight}: the paint with the biggest weight), without the random color
changes of the generated symbols. With the default paint file, as printed:
- a common: black, white outline and inner area; the other rarities: the rarity color, black outline
  and inner area,
- Dominaria (dom): the common is white, black outline and inner area,
- Aetherdrift (dft): the symbol stays black, its outline and inner area in the rarity color (white for
  a common).

How: the downloaded SVG is drawn TRACE_SIZE pixels wide, its 3 areas are traced back into SVG paths. A
gradient is GRADIENT_BANDS pieces of the area, each with one flat color, over the whole area in the
average color. So the colored SVG has only plain filled paths: it looks the same in every program,
PyMuPDF included (it draws neither SVG gradients nor clip paths), whatever the downloaded SVG uses
(styles, transforms...).

Each colored SVG says in a comment what it was drawn with (drawing_stamp: the paint hash and
DRAW_VERSION): a file drawn with another paint file or an older drawing is drawn again, also by
download.py (choice 3).

Run from anywhere (WSL: writes the Linux disk and the Windows disk, see download.symbol_dirs):
    python3 src/colorize_set_symbol.py            (draw again every colored symbol)
    python3 src/colorize_set_symbol.py dsk m15    (only these sets)
"""
import functools
import hashlib
import json
import sys
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import cv2
import numpy as np
import tqdm

import download
import local_paths

RGB = Tuple[int, int, int]


RARITIES = ["common", "uncommon", "rare", "mythic"]
# How the symbols are painted (see its "_help"), shared with set_symbol_data_set.py.
PAINT_FILE = local_paths.ROOT / "resources" / "set_symbol_paint.json"
AREAS = ["outline", "inner", "symbol"]  # The areas of a symbol, drawn in this order (from the bottom)
DRAW_VERSION = 2        # Change it when the drawing changes: every colored symbol is drawn again
# The first set of each rarity color: the sets released that day or later get it.
FIRST_SET = {"common": "exo", "uncommon": "exo", "rare": "exo", "mythic": "ala"}
FIRST_SET_DATE = {"exo": "1998-06-15", "ala": "2008-10-03"}  # Used when sets.json doesn't know them

GRADIENT_ANGLE = 110    # Degrees, as in CSS: 90 = from left to right, 110 = to the right and a bit down
GRADIENT_BANDS = 48     # The gradient is drawn as this many pieces of the symbol, each of one flat color
INNER_OVERLAP = 2      # The inner area goes this far (pixels) under the symbol: no gap between them
BAND_OVERLAP = 3.0      # Each piece goes this far (pixels) under the next one: no gap between them (the
                        # traced edges move up to ~1 px: half a pixel + TRACE_TOLERANCE)
TRACE_SIZE = 1024       # Longest side of the symbol when it is traced (pixels = units of the colored SVG)
OUTLINE = 0.03          # Outline width, share of the longest side (measured: ~1.2 px for a ~44 px symbol on a card)
TRACE_TOLERANCE = 0.6   # Largest gap between the traced path and the drawn shape (pixels of TRACE_SIZE)


def rarity_file_name(code: str, rarity: str) -> str:
    return f"set_{code}_{rarity}.svg"


def rarities_of(released_at: Optional[str], first_dates: Dict[str, str]) -> List[str]:
    """The rarities that have a colored symbol in a set released on released_at ('YYYY-MM-DD').
    An unknown date (a set not released yet) gets every rarity."""
    return [rarity for rarity in RARITIES
            if not released_at or released_at >= first_dates[FIRST_SET[rarity]]]


def first_dates(sets: List[dict]) -> Dict[str, str]:
    """Release date of each FIRST_SET, from the scryfall sets (FIRST_SET_DATE if missing)."""
    dates = dict(FIRST_SET_DATE)
    for set_data in sets:
        if set_data.get("code") in dates and set_data.get("released_at"):
            dates[set_data["code"]] = set_data["released_at"]
    return dates


@functools.lru_cache(maxsize=None)
def load_paint() -> dict:
    """PAINT_FILE, without its "_help" (checked in detail by set_symbol_data_set.load_paint)."""
    if not PAINT_FILE.exists():
        raise FileNotFoundError(f"No {PAINT_FILE}: the paint of the set symbols.")
    with open(PAINT_FILE, "r", encoding="utf-8") as fp:
        return {name: value for name, value in json.load(fp).items() if not name.startswith("_")}


def paint_hash() -> str:
    """Short hash of the paint (the same as set_symbol_data_set.paint_hash)."""
    return hashlib.sha256(json.dumps(load_paint(), sort_keys=True).encode("utf-8")).hexdigest()[:10]


def drawing_stamp() -> str:
    """Written in each colored SVG: a file with another stamp is drawn again."""
    return f"paint {paint_hash()}, version {DRAW_VERSION}"


def _heaviest(value) -> str:
    """A paint: the value itself, or the one of {paint: weight} with the biggest weight."""
    return max(value, key=value.get) if isinstance(value, dict) else value


def rarity_paints(code: str, rarity: str, inner_share: float = 1.0) -> Dict[str, Optional[List[Tuple[float, RGB]]]]:
    """
    The look of the symbol of the set `code` in a rarity, from PAINT_FILE: for each area (AREAS), its
    gradient stops [(position 0-1 along GRADIENT_ANGLE, RGB)] (one stop: a flat color), None: no outline.
    The style: of sets.<code>, else of "styles", the one with the biggest weight among those whose
    min_inner is at most inner_share (the inner area as a share of the symbol).
    """
    paint = load_paint()
    own = (paint.get("sets") or {}).get(code) or {}
    try:
        styles = own.get("styles") or paint["styles"]
        usable = [style for style in styles if inner_share >= style.get("min_inner", 0)] or styles
        style = max(usable, key=lambda style: style.get("weight", 1))
        stops: Dict[str, Optional[List[Tuple[float, RGB]]]] = {}
        for area in AREAS:
            name = _heaviest(style[area])
            if name == "none":
                stops[area] = None
            elif name == "rarity":
                colors = (own.get("rarity_colors") or {}).get(rarity) or paint["rarity_colors"][rarity]
                dark = tuple(colors["dark"])
                stops[area] = [(0.0, dark)] if not colors.get("light") else \
                    [(0.0, dark), (0.5, tuple(colors["light"])), (1.0, dark)]
            else:
                if name == "line":
                    name = (own.get("line_colors") or {}).get(rarity) or paint["line_colors"][rarity]
                stops[area] = [(0.0, tuple(paint["colors"][name]))]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{PAINT_FILE}: no paint for {code} {rarity} ({error!r}), see its \"_help\"") from error
    return stops


def render_shape(svg: bytes, size: int = TRACE_SIZE) -> np.ndarray:
    """
    The shape of a symbol: float32 0-1, its longest side `size` pixels (aspect ratio kept).
    The dark parts of the SVG are the symbol, the transparent and the white parts are not.
    """
    try:
        import pymupdf
    except ImportError:
        try:
            import fitz as pymupdf  # PyMuPDF before 1.24
        except ImportError as error:
            raise ImportError("The set symbols need PyMuPDF: pip install pymupdf") from error
    with pymupdf.open(stream=svg, filetype="svg") as document:
        page = document[0]
        scale = size / max(page.rect.width, page.rect.height)
        png = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=True).tobytes("png")
    image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] != 4:
        raise ValueError("Can't draw the SVG with transparency")
    image = image.astype(np.float32) / 255
    darkness = 1 - cv2.cvtColor(np.ascontiguousarray(image[:, :, :3]), cv2.COLOR_BGR2GRAY)
    return image[:, :, 3] * np.clip(darkness * 2, 0, 1)


def _path_data(mask: np.ndarray) -> str:
    """SVG path data of a 0/1 mask (pixel units), every contour (holes included): use fill-rule evenodd."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    parts = []
    for contour in contours:
        points = cv2.approxPolyDP(contour, TRACE_TOLERANCE, True).reshape(-1, 2)
        if len(points) >= 3:
            # +0.5: the contour goes through the centers of the edge pixels, the shape ends at their edge.
            parts.append("M" + " ".join(f"{x + 0.5:g},{y + 0.5:g}" for x, y in points) + "Z")
    return "".join(parts)


def _hex(color: RGB) -> str:
    return "#{:02x}{:02x}{:02x}".format(*color)


def _gradient_line(mask: np.ndarray) -> Tuple[float, float, float, float]:
    """(x1, y1, x2, y2) of a GRADIENT_ANGLE gradient that covers the bounding box of the mask (CSS rule)."""
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    angle = np.radians(GRADIENT_ANGLE)
    dx, dy = np.sin(angle), -np.cos(angle)  # CSS: 0 deg = up, 90 deg = right (y goes down)
    half = (abs((x1 - x0) * dx) + abs((y1 - y0) * dy)) / 2
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return cx - half * dx, cy - half * dy, cx + half * dx, cy + half * dy


class TracedSymbol(NamedTuple):
    width: int
    height: int
    masks: Dict[str, np.ndarray]                # Each area (AREAS): 0/1 mask, in pixels of TRACE_SIZE
    line: Tuple[float, float, float, float]     # The gradient line, over the symbol (_gradient_line)
    inner_share: float                          # The inner area as a share of the filled symbol
    paths: Dict[Tuple[str, bool], object]       # The paths traced so far (area_paths)


def _band_masks(mask: np.ndarray, line: Tuple[float, float, float, float]) -> List[np.ndarray]:
    """The mask cut in GRADIENT_BANDS pieces across the gradient line (x1, y1, x2, y2), each one going
    BAND_OVERLAP pixels under the next one (the first and last pieces: up to the edges of the mask)."""
    x1, y1, x2, y2 = line
    length = max(1e-6, float(np.hypot(x2 - x1, y2 - y1)))
    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    grid_y, grid_x = np.mgrid[0:mask.shape[0], 0:mask.shape[1]].astype(np.float32) + 0.5  # Pixel centers
    along = (grid_x - x1) * dx + (grid_y - y1) * dy  # Pixels along the gradient line
    edges = [band * length / GRADIENT_BANDS for band in range(GRADIENT_BANDS + 1)]
    pieces = []
    for band in range(GRADIENT_BANDS):
        piece = mask.astype(bool)
        if band > 0:
            piece &= along >= edges[band]
        if band < GRADIENT_BANDS - 1:
            piece &= along < edges[band + 1] + BAND_OVERLAP
        pieces.append(piece.astype(np.uint8))
    return pieces


def trace(svg: bytes) -> TracedSymbol:
    """The 3 areas of a downloaded SVG (AREAS), in pixels of TRACE_SIZE. Their paths: area_paths."""
    shape = render_shape(svg)
    outline_px = max(1, round(OUTLINE * max(shape.shape)))
    symbol = np.pad(shape > 0.5, outline_px + 2).astype(np.uint8)
    if not symbol.any():
        raise ValueError("empty symbol")
    # No hole, as printed: the regions enclosed by the symbol (inside its outer contours) are its inner area.
    contours, _ = cv2.findContours(symbol, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(symbol)
    cv2.drawContours(filled, contours, -1, 1, thickness=-1)
    holes = filled - symbol
    overlap = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * INNER_OVERLAP + 1, 2 * INNER_OVERLAP + 1))
    inner = cv2.dilate(holes, overlap) & filled  # Under the edge of the symbol too
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * outline_px + 1, 2 * outline_px + 1))
    outline = cv2.dilate(filled, kernel)  # The whole filled symbol grown: drawn first, under the others
    height, width = symbol.shape
    return TracedSymbol(width, height, {"outline": outline, "inner": inner, "symbol": symbol},
                        _gradient_line(symbol), float(holes.sum()) / float(filled.sum()), {})


def area_paths(symbol: TracedSymbol, area: str, gradient: bool):
    """Path data of an area: one path, or (gradient) its GRADIENT_BANDS pieces ('' = empty piece)."""
    key = (area, gradient)
    if key not in symbol.paths:
        mask = symbol.masks[area]
        symbol.paths[key] = ([_path_data(piece) if piece.any() else "" for piece in _band_masks(mask, symbol.line)]
                             if gradient else _path_data(mask))
    return symbol.paths[key]


def _color_at(stops: Sequence[Tuple[float, RGB]], t: float) -> RGB:
    """Color of the gradient `stops` at position t (0-1)."""
    positions = [position for position, _ in stops]
    return tuple(int(round(np.interp(t, positions, [color[channel] for _, color in stops]))) for channel in range(3))


def _area_svg(symbol: TracedSymbol, area: str, stops: Sequence[Tuple[float, RGB]]) -> str:
    """The paths of an area in a flat color (one stop) or a gradient."""
    if len(stops) == 1:
        return f'<path fill="{_hex(stops[0][1])}" fill-rule="evenodd" d="{area_paths(symbol, area, False)}"/>\n'
    # The gradient: each piece of the area in the color of the middle of its band, over the whole area
    # in the average color (drawn small, the partly covered edge pixels of two pieces would let the area
    # below show through: thin dark stripes).
    average = tuple(int(round(np.mean([_color_at(stops, t)[c] for t in np.linspace(0, 1, 101)]))) for c in range(3))
    return f'<path fill="{_hex(average)}" fill-rule="evenodd" d="{area_paths(symbol, area, False)}"/>\n' + "".join(
        f'<path fill="{_hex(_color_at(stops, (band + 0.5) / GRADIENT_BANDS))}" fill-rule="evenodd" d="{data}"/>\n'
        for band, data in enumerate(area_paths(symbol, area, True)) if data)


def rarity_svg(symbol: TracedSymbol, code: str, rarity: str, source: str = "") -> bytes:
    """SVG file content of a traced symbol in the colors of `rarity` in the set `code` (rarity_paints):
    the outline, the inner area, then the symbol over them."""
    paints = rarity_paints(code, rarity, symbol.inner_share)
    comment = f"<!-- {rarity} symbol drawn by colorize_set_symbol.py from {source or code}, {drawing_stamp()} -->\n"
    body = "".join(_area_svg(symbol, area, paints[area]) for area in AREAS
                   if paints[area] and symbol.masks[area].any())
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {symbol.width} {symbol.height}" '
            f'width="{symbol.width}" height="{symbol.height}">\n{comment}{body}</svg>\n').encode("utf-8")


def _up_to_date(path, stamp: str) -> bool:
    """True if the colored SVG is there, drawn with this drawing_stamp."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fp:
            return stamp in fp.read(1000)
    except OSError:
        return False


def colorize_set_symbols(codes: Optional[List[str]] = None, force: bool = False) -> Dict[str, int]:
    """
    Write set_<code>_<rarity>.svg in each folder of download.symbol_dirs(SET_DIR_NAME), for every
    downloaded set symbol (or only `codes`). The files already drawn with the same drawing_stamp are kept
    (force: draw them again).
    The downloaded SVG and the release dates are read in the first of these folders.
    :return: number of files 'drawn', 'kept' (up to date), sets 'too_old' (before Exodus), sets 'failed'.
    """
    dirs = download.symbol_dirs(download.SET_DIR_NAME)
    sets_file = dirs[0] / download.SETS_FILE_NAME
    sets = json.loads(sets_file.read_text(encoding="utf-8")) if sets_file.exists() else []
    dates = first_dates(sets)

    wanted = None if codes is None else {code.lower() for code in codes}
    todo = [set_data for set_data in sets if set_data.get(download.SET_SYMBOL_FILE_KEY)
            and (wanted is None or set_data["code"] in wanted)]
    counts = {"drawn": 0, "kept": 0, "too_old": 0, "failed": 0}
    stamp = drawing_stamp()
    for set_data in tqdm.tqdm(todo, desc="Rarity symbols", unit="set"):
        code, file_name = set_data["code"], set_data[download.SET_SYMBOL_FILE_KEY]
        rarities = rarities_of(set_data.get("released_at"), dates)
        if not rarities:
            counts["too_old"] += 1
            continue
        symbol = None
        for rarity in rarities:
            paths = [directory / rarity_file_name(code, rarity) for directory in dirs]
            missing = paths if force else [path for path in paths if not _up_to_date(path, stamp)]
            if not missing:
                counts["kept"] += 1
                continue
            try:
                symbol = symbol or trace((dirs[0] / file_name).read_bytes())
                content = rarity_svg(symbol, code, rarity, file_name)
            except Exception as error:  # A broken SVG must not stop the other sets
                tqdm.tqdm.write(f"Can't draw {file_name} ({rarity}): {error}")
                counts["failed"] += 1
                break
            for path in missing:
                download._write_file(path, content)
            counts["drawn"] += 1
    print(f"Rarity symbols: {counts['drawn']} drawn, {counts['kept']} up to date, "
          f"{counts['too_old']} sets before Exodus (black symbol only)"
          + (f", {counts['failed']} sets failed" if counts["failed"] else "") + ".")
    return counts


if __name__ == "__main__":
    colorize_set_symbols(sys.argv[1:] or None, force=True)
