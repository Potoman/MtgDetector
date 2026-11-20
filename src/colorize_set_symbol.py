"""
Set symbols (expansion symbols) in the colors of the rarities, from the black SVG files of scryfall.

download.py (choice 3: set) downloads them into <scryfall dir>/set/: set_<code>.svg as downloaded, and
sets.json (the scryfall sets: release date, type... and "symbol_file"). This file adds, in the same
folder, one SVG per rarity: set_<code>_<rarity>.svg
- common, uncommon, rare: every set. Before Exodus (1998, the first set with rarity colors), from the
  first sets to Stronghold, a symbol had one look: the 3 files are the same, in the colors of a common,
- mythic: the sets released since Shards of Alara (2008, the first set with mythic rares).

Their PNG (below) are the symbols of the dataset of the set symbol detector (set_symbol_data_set.py),
pasted as they are.

Each colored SVG also gets its PNG, one per set and rarity: <scryfall dir>/set_png/set_<code>_<rarity>.png
(PNG_DIR_NAME), PNG_SIZE x PNG_SIZE (75 x 75) pixels, transparent, the
symbol (its outline included) PNG_SYMBOL_HEIGHT (27) pixels high, the largest height on a card, its width
following its shape, at the center (symbol_png). A PNG says in a text chunk ("drawing") what it was made
with (png_stamp): it is made again with its SVG, or when it is missing or outdated.

Look of each rarity: PAINT_FILE (resources/set_symbol_paint.json, see its "_help"). The symbol has 3 areas: 'symbol' (the black parts of
the downloaded SVG), 'inner' (its holes: the regions it encloses, filled as printed, or left transparent
with the paint 'none'; with the style option close_gaps also the regions enclosed once its gaps up to
twice the outline are closed, see closed_keys) and 'outline' (around it, OUTLINE of the symbol height: 1.1
px when the whole symbol is 27 px high, as in the dataset, times the outline_width of the style; it also
fills the thin gaps between the parts of the symbol). A style can also draw a thin line inside the edge of
the symbol area (SYMBOL_LINE 'inline', its width 'inline_width', see core_key). Each
area gets a paint: the rarity color (a flat color, or a gradient dark -> light -> dark from left to
right, slightly tilted, as on the cards; 'rarity_white': the same, but white for a common), the line
color of the rarity, white or black (a choice
{paint: weight}: the paint with the biggest weight). The styles are named ("styles"); the style of a
set: the style whose "set" lists it (before anything else), else the one of its "sets" entry, else
the default one ("style"). With the default paint file, as
printed:
- black_hold_rarity, every set: a common black, white outline and inner area; the other rarities: the
  rarity color, black outline and inner area,
- Dominaria (dom): the common is white, black outline and inner area,
- Aetherdrift (dft), black_symbol: the symbol stays black, its outline and inner area in the rarity
  color (white for a common),
- Reality Fracture (fra), inner_hold_rarity: the symbol stays black (white for a common), its
  inner area in the rarity color (black for a common), no outline.

How: the downloaded SVG is drawn TRACE_SIZE pixels wide, its 3 areas are traced back into SVG paths. A
gradient is GRADIENT_BANDS pieces of the area, each with one flat color, over the whole area in the
average color. So the colored SVG has only plain filled paths: it looks the same in every program,
PyMuPDF included (it draws neither SVG gradients nor clip paths), whatever the downloaded SVG uses
(styles, transforms...).

Each colored SVG says in a comment what it was drawn with (drawing_stamp: the paint hash and
DRAW_VERSION): a file drawn with another paint file or an older drawing is drawn again, also by
download.py (choice 3).

Run from anywhere (WSL with paths.json: writes the Linux disk only, see download.symbol_dir):
    python3 src/colorize_set_symbol.py            (draw again every colored symbol and its PNG)
    python3 src/colorize_set_symbol.py dsk m15    (only these sets)
"""
import functools
import hashlib
import io
import json
import sys
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import cv2
import numpy as np
import tqdm
from PIL import Image, PngImagePlugin

import download
import local_paths

RGB = Tuple[int, int, int]


RARITIES = ["common", "uncommon", "rare", "mythic"]
# How the symbols are painted (see its "_help").
PAINT_FILE = local_paths.ROOT / "resources" / "set_symbol_paint.json"
PAINT_KEYS = ["colors", "rarity_colors", "line_colors", "styles", "style", "sets"]
# What an area can be painted with, besides the colors of the paint file ("colors": white, black, gray...):
# 'rarity_white': 'rarity', but white for a common; 'none': outline only (see paints_of)
PAINTS = ["rarity", "rarity_white", "line", "none"]
REQUIRED_COLORS = ["white", "black"]  # Colors every paint file has (line_colors, rarity_white)
AREAS = ["outline", "inner", "symbol"]  # The areas of a symbol, drawn in this order (from the bottom)
SYMBOL_LINE = "inline"          # Optional style key: the paint of a thin line inside the edge of the symbol area
SYMBOL_LINE_WIDTH_KEY = "inline_width"  # Optional style key: its width, times the usual outline (OUTLINE)
SYMBOL_LINE_WIDTH = 0.5         # Its default width: 0.56 px when the symbol is 27 px high
# The keys a style can have (see the "_help" of PAINT_FILE), besides AREAS
STYLE_KEYS = ["outline_width", "close_gaps", "min_inner", "set", SYMBOL_LINE, SYMBOL_LINE_WIDTH_KEY]
DRAW_VERSION = 6        # Change it when the drawing changes: every colored symbol is drawn again
# The first set of each rarity color: the sets released that day or later get it.
FIRST_SET = {"common": "exo", "uncommon": "exo", "rare": "exo", "mythic": "ala"}
OLD_RARITIES = ["common", "uncommon", "rare"]  # The files of a set before Exodus, all in the colors of a common
FIRST_SET_DATE = {"exo": "1998-06-15", "ala": "2008-10-03"}  # Used when sets.json doesn't know them

GRADIENT_ANGLE = 110    # Degrees, as in CSS: 90 = from left to right, 110 = to the right and a bit down
GRADIENT_BANDS = 48     # The gradient is drawn as this many pieces of the symbol, each of one flat color
INNER_OVERLAP = 2      # The inner area goes this far (pixels) under the symbol: no gap between them
BAND_OVERLAP = 3.0      # Each piece goes this far (pixels) under the next one: no gap between them (the
                        # traced edges move up to ~1 px: half a pixel + TRACE_TOLERANCE)
TRACE_SIZE = 1024       # Longest side of the symbol when it is traced (pixels = units of the colored SVG)
OUTLINE = 0.04515625    # Outline width, share of the height of the symbol: 1.1 px when the whole symbol (outline
                        # included) is 27 px high, the size of the dataset (24.8 + 2 x 1.1), whatever its width
MAX_OUTLINE_WIDTH = 3.0 # Largest outline_width of a style (times OUTLINE): the room left around the traced symbol
# The PNG of each colored SVG (symbol_png)
PNG_DIR_NAME = "set_png"    # <scryfall dir>/set_png/set_<code>_<rarity>.png
PNG_SIZE = 75               # Pixels, square, transparent
PNG_SYMBOL_HEIGHT = 27      # Height of the symbol in the PNG, its outline included: the largest on a card
PNG_RENDER_SIZE = 512       # The SVG is drawn this big (longest side), then reduced: smooth edges
PNG_TEXT_KEY = "drawing"    # Text chunk of the PNG: its png_stamp
TRACE_TOLERANCE = 0.6   # Largest gap between the traced path and the drawn shape (pixels of TRACE_SIZE)


def rarity_file_name(code: str, rarity: str) -> str:
    return f"set_{code}_{rarity}.svg"


def png_file_name(code: str, rarity: str) -> str:
    return f"set_{code}_{rarity}.png"


def before_exodus(released_at: Optional[str], first_dates: Dict[str, str]) -> bool:
    """True for a set released before Exodus (the first set with rarity colors): one look for every rarity."""
    return bool(released_at) and released_at < first_dates[FIRST_SET["common"]]


def rarities_of(released_at: Optional[str], first_dates: Dict[str, str]) -> List[str]:
    """The rarities that have a colored symbol in a set released on released_at ('YYYY-MM-DD'): before
    Exodus OLD_RARITIES (all in the colors of a common). An unknown date (a set not released yet) gets
    every rarity."""
    if before_exodus(released_at, first_dates):
        return list(OLD_RARITIES)
    return [rarity for rarity in RARITIES
            if not released_at or released_at >= first_dates[FIRST_SET[rarity]]]


def first_dates(sets: List[dict]) -> Dict[str, str]:
    """Release date of each FIRST_SET, from the scryfall sets (FIRST_SET_DATE if missing)."""
    dates = dict(FIRST_SET_DATE)
    for set_data in sets:
        if set_data.get("code") in dates and set_data.get("released_at"):
            dates[set_data["code"]] = set_data["released_at"]
    return dates


def paints_of(colors: dict, area: str = "") -> List[str]:
    """The paints of an area: PAINTS and the colors of the paint file, 'none' (not drawn) only for the
    outline and the inner area (its holes stay transparent)."""
    paints = PAINTS[:-1] + [name for name in colors if name not in PAINTS]
    return paints + ["none"] if area in ("outline", "inner") else paints


@functools.lru_cache(maxsize=None)
def load_paint() -> dict:
    """PAINT_FILE without its "_help", checked: a ValueError lists its problems."""
    if not PAINT_FILE.exists():
        raise FileNotFoundError(f"No {PAINT_FILE}: the paint of the set symbols.")
    with open(PAINT_FILE, "r", encoding="utf-8") as fp:
        paint = {name: value for name, value in json.load(fp).items() if not name.startswith("_")}
    problems = [f"{key}: unknown, only {PAINT_KEYS}" for key in paint if key not in PAINT_KEYS]
    color_table = paint.get("colors") if isinstance(paint.get("colors"), dict) else {}
    for name in REQUIRED_COLORS + [name for name in color_table if name not in REQUIRED_COLORS]:
        value = color_table.get(name)
        if not isinstance(value, list) or len(value) != 3 or not all(isinstance(v, (int, float)) for v in value):
            problems.append(f"colors.{name}: 3 numbers (RGB)" + (" (required)" if name in REQUIRED_COLORS else ""))
        if name in PAINTS:
            problems.append(f"colors.{name}: a paint name ({PAINTS}), choose another name")
    sets = paint.get("sets") or {}
    # The default colors: every rarity. Those of a set (sets.<code>): only the rarities it changes.
    for where, group, every in [("", paint, True)] + [(f"sets.{code}.", value, False) for code, value in sets.items()]:
        for key in ("rarity_colors", "line_colors"):
            unknown = [name for name in group.get(key) or {} if name not in RARITIES]
            if unknown:
                problems.append(f"{where}{key}: {unknown} is not a rarity ({RARITIES})")
        for rarity in RARITIES:
            colors = (group.get("rarity_colors") or {}).get(rarity)
            if (every or colors is not None) and len((colors or {}).get("dark") or []) != 3:
                problems.append(f"{where}rarity_colors.{rarity}.dark: 3 numbers (RGB)")
            line = (group.get("line_colors") or {}).get(rarity)
            if (every or line is not None) and line not in color_table:
                problems.append(f"{where}line_colors.{rarity}: one of the colors {list(color_table)}")
    for code, value in sets.items():
        unknown = [key for key in value if key not in ("rarity_colors", "line_colors", "style")]
        if unknown:
            problems.append(f"sets.{code}: {unknown}, only rarity_colors, line_colors, style")
    styles = paint.get("styles")
    if not isinstance(styles, dict) or not styles:
        problems.append("styles: at least one style, {name: {symbol, inner, outline}}")
        styles = {}
    for name, style in styles.items():
        unknown = [key for key in style if key not in AREAS + STYLE_KEYS]
        if unknown:
            problems.append(f"styles.{name}: {unknown} unknown, only {AREAS + STYLE_KEYS}")
        for area in AREAS:
            value = style.get(area)
            choices = value if isinstance(value, dict) else {value: 1}
            allowed = paints_of(color_table, area)
            if not choices or any(paint_name not in allowed or not weight > 0 for paint_name, weight in choices.items()):
                problems.append(f"styles.{name}.{area}: one of {allowed}, or {{paint: weight}}")
        for key, default in (("outline_width", 1.0), (SYMBOL_LINE_WIDTH_KEY, SYMBOL_LINE_WIDTH)):
            width = style.get(key, default)
            if isinstance(width, bool) or not isinstance(width, (int, float)) or not 0 < width <= MAX_OUTLINE_WIDTH:
                problems.append(f"styles.{name}.{key}: a number above 0, at most {MAX_OUTLINE_WIDTH:g} "
                                f"(times the usual outline)")
        if SYMBOL_LINE_WIDTH_KEY in style and SYMBOL_LINE not in style:
            problems.append(f"styles.{name}.{SYMBOL_LINE_WIDTH_KEY}: the width of the {SYMBOL_LINE}, no {SYMBOL_LINE} "
                            f"in the style")
        if not isinstance(style.get("close_gaps", False), bool):
            problems.append(f"styles.{name}.close_gaps: true or false")
        if SYMBOL_LINE in style:
            value = style[SYMBOL_LINE]
            choices = value if isinstance(value, dict) else {value: 1}
            allowed = paints_of(color_table)
            if not choices or any(paint_name not in allowed or not weight > 0 for paint_name, weight in choices.items()):
                problems.append(f"styles.{name}.{SYMBOL_LINE}: one of {allowed}, or {{paint: weight}} (or no key: no line)")
    # The sets of a style (styles.<name>.set): its style before any other, so in one style only
    owners: Dict[str, str] = {}
    for name, style in styles.items():
        codes = style.get("set", [])
        if not isinstance(codes, list) or not all(isinstance(code, str) and code for code in codes):
            problems.append(f"styles.{name}.set: a list of set codes, ex: [\"csp\"]")
            continue
        for code in codes:
            if code.lower() in owners:
                problems.append(f"styles.{name}.set: {code} is also in styles.{owners[code.lower()]}.set (one style per set)")
            owners[code.lower()] = name
    # The style of every set ("style", required), and of some sets (sets.<code>.style)
    for where, value in [("style", paint.get("style"))] + \
                        [(f"sets.{code}.style", value["style"]) for code, value in sets.items() if "style" in value]:
        choices = value if isinstance(value, dict) else {value: 1}
        if not choices or any(name not in styles or not weight > 0 for name, weight in choices.items()):
            problems.append(f"{where}: one of the styles {list(styles)}, or {{style: weight}}")
    if problems:
        raise ValueError(f"{PAINT_FILE}: " + "; ".join(problems))
    return paint


def paint_hash() -> str:
    """Short hash of the paint: in the drawing_stamp of every colored SVG."""
    return hashlib.sha256(json.dumps(load_paint(), sort_keys=True).encode("utf-8")).hexdigest()[:10]


def drawing_stamp() -> str:
    """Written in each colored SVG: a file with another stamp is drawn again."""
    return f"paint {paint_hash()}, version {DRAW_VERSION}"


def png_stamp() -> str:
    """Written in each PNG: a PNG with another stamp is made again."""
    return f"{drawing_stamp()}, png {PNG_SIZE} px, symbol {PNG_SYMBOL_HEIGHT} px high"


def _heaviest(value) -> str:
    """A paint: the value itself, or the one of {paint: weight} with the biggest weight."""
    return max(value, key=value.get) if isinstance(value, dict) else value


def style_of(code: str, inner_share: float = 1.0) -> dict:
    """
    The style of the set `code` (PAINT_FILE): the one whose "set" lists it (before anything else), else
    the one named by sets.<code>.style, else by "style" (a choice {name: weight}: the one with the biggest
    weight among those whose min_inner is at most inner_share, the inner area as a share of the symbol).
    """
    paint = load_paint()
    for style in paint["styles"].values():
        if code.lower() in (listed.lower() for listed in style.get("set", [])):
            return style
    choice = ((paint.get("sets") or {}).get(code) or {}).get("style") or paint["style"]
    choices = choice if isinstance(choice, dict) else {choice: 1}
    usable = {name: weight for name, weight in choices.items()
              if inner_share >= paint["styles"][name].get("min_inner", 0)} or choices
    return paint["styles"][_heaviest(usable)]


def rarity_paints(code: str, rarity: str, inner_share: float = 1.0) -> Dict[str, Optional[List[Tuple[float, RGB]]]]:
    """
    The look of the symbol of the set `code` in a rarity, from PAINT_FILE: for each area (AREAS), its
    gradient stops [(position 0-1 along GRADIENT_ANGLE, RGB)] (one stop: a flat color), None: no outline.
    Also SYMBOL_LINE when the style has one (the thin line inside the edge of the symbol area).
    The style: style_of.
    """
    paint = load_paint()
    own = (paint.get("sets") or {}).get(code) or {}
    try:
        style = style_of(code, inner_share)
        stops: Dict[str, Optional[List[Tuple[float, RGB]]]] = {}
        for area in AREAS + [SYMBOL_LINE] * (SYMBOL_LINE in style):
            name = _heaviest(style[area])
            if name == "none":
                stops[area] = None
            elif name == "rarity_white" and rarity == "common":  # 'rarity', but white for a common
                stops[area] = [(0.0, tuple(paint["colors"]["white"]))]
            elif name in ("rarity", "rarity_white"):
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


def _pymupdf():
    """The PyMuPDF module (imported here: only the drawing needs it)."""
    try:
        import pymupdf
    except ImportError:
        try:
            import fitz as pymupdf  # PyMuPDF before 1.24
        except ImportError as error:
            raise ImportError("The set symbols need PyMuPDF: pip install pymupdf") from error
    return pymupdf


def render_shape(svg: bytes, size: int = TRACE_SIZE) -> np.ndarray:
    """
    The shape of a symbol: float32 0-1, its longest side `size` pixels (aspect ratio kept).
    The dark parts of the SVG are the symbol, the transparent and the white parts are not.
    """
    pymupdf = _pymupdf()
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
    masks: Dict[str, np.ndarray]                # Each area (AREAS, outline_key): 0/1 mask, in pixels of
                                                # TRACE_SIZE, and 'filled' (the symbol and its inner area)
    outline_px: float                           # The usual outline width (OUTLINE of the height), pixels
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
    rows = np.nonzero((shape > 0.5).any(axis=1))[0]  # The height of the symbol (not of the page of the SVG)
    if not len(rows):
        raise ValueError("empty symbol")
    outline_px = OUTLINE * (rows[-1] - rows[0] + 1)
    # Room for the widest outline (MAX_OUTLINE_WIDTH): the outline of the style is grown later (outline_key).
    symbol = np.pad(shape > 0.5, int(np.ceil(outline_px * MAX_OUTLINE_WIDTH)) + 2).astype(np.uint8)
    if not symbol.any():
        raise ValueError("empty symbol")
    # No hole, as printed: the regions enclosed by the symbol (inside its outer contours) are its inner area.
    contours, _ = cv2.findContours(symbol, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(symbol)
    cv2.drawContours(filled, contours, -1, 1, thickness=-1)
    holes = filled - symbol
    overlap = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * INNER_OVERLAP + 1, 2 * INNER_OVERLAP + 1))
    inner = cv2.dilate(holes, overlap) & filled  # Under the edge of the symbol too
    height, width = symbol.shape
    return TracedSymbol(width, height, {"filled": filled, "inner": inner, "symbol": symbol}, outline_px,
                        _gradient_line(symbol), float(holes.sum()) / float(filled.sum()), {})


def _outline_pixels(symbol: TracedSymbol, outline_width: float) -> int:
    """The outline width in pixels of TRACE_SIZE: outline_width times the usual one, at least 1 pixel."""
    return max(1, round(symbol.outline_px * outline_width))


def _disk(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def outline_key(symbol: TracedSymbol, outline_width: float = 1.0, filled: str = "filled") -> str:
    """
    The key in symbol.masks of its outline, outline_width times the usual one (OUTLINE of the height, at
    least 1 pixel), made the first time: the filled area `filled` (a key of symbol.masks) grown, drawn
    first, under the others.
    """
    pixels = _outline_pixels(symbol, outline_width)
    key = f"outline_{pixels}_{filled}"
    if key not in symbol.masks:
        symbol.masks[key] = cv2.dilate(symbol.masks[filled], _disk(pixels))
    return key


def core_key(symbol: TracedSymbol, line_width: float = SYMBOL_LINE_WIDTH) -> str:
    """
    The style option SYMBOL_LINE: the key in symbol.masks of the symbol area shrunk by line_width times the
    usual outline (the style option SYMBOL_LINE_WIDTH_KEY), made the first time. Drawn in the paint of the
    symbol over the whole symbol area in the paint of the line: a thin line inside its edge (the outer edge
    and the edge of its holes), the parts thinner than twice the line all in the paint of the line.
    """
    pixels = _outline_pixels(symbol, line_width)
    key = f"symbol_core_{pixels}"
    if key not in symbol.masks:
        symbol.masks[key] = cv2.erode(symbol.masks["symbol"], _disk(pixels))
    return key


def closed_keys(symbol: TracedSymbol, outline_width: float = 1.0) -> Tuple[str, str]:
    """
    The style option close_gaps: (filled key, inner key) in symbol.masks of the symbol whose gaps up to
    twice the outline (outline_width times the usual one) are closed, made the first time. The regions
    that the closed gaps enclose are inner area too, as printed when the symbol has a background: ex the
    face of Vault Boy (pip), open beside the ears in the scryfall SVG, white on the cards. The filled
    symbol grown by the outline, its holes filled, shrunk back: also fills the narrow bays of its edge.
    """
    pixels = _outline_pixels(symbol, outline_width)
    filled_key, inner_key = f"filled_closed_{pixels}", f"inner_closed_{pixels}"
    if filled_key not in symbol.masks:
        grown = cv2.dilate(symbol.masks["filled"], _disk(pixels))
        contours, _ = cv2.findContours(grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(grown, contours, -1, 1, thickness=-1)
        filled = cv2.erode(grown, _disk(pixels)) | symbol.masks["filled"]
        holes = filled & (1 - symbol.masks["symbol"])
        symbol.masks[filled_key] = filled
        symbol.masks[inner_key] = cv2.dilate(holes, _disk(INNER_OVERLAP)) & filled  # Under the edge of the symbol too
    return filled_key, inner_key


def area_paths(symbol: TracedSymbol, area: str, gradient: bool):
    """Path data of an area (a key of symbol.masks): one path, or (gradient) its GRADIENT_BANDS pieces
    ('' = empty piece)."""
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


def rarity_svg(symbol: TracedSymbol, code: str, rarity: str, source: str = "",
               painted_as: Optional[str] = None) -> bytes:
    """SVG file content of a traced symbol in the colors of `rarity` in the set `code` (rarity_paints), or
    in those of `painted_as` (before Exodus: 'common'): the outline, the inner area, then the symbol (with a
    SYMBOL_LINE: the symbol in the paint of the line, then the symbol shrunk in its own paint, core_key)."""
    paints = rarity_paints(code, painted_as or rarity, symbol.inner_share)
    look = f" (in the colors of a {painted_as}: before Exodus)" if painted_as and painted_as != rarity else ""
    comment = (f"<!-- {rarity} symbol{look} drawn by colorize_set_symbol.py from {source or code}, "
               f"{drawing_stamp()} -->\n")
    style = style_of(code, symbol.inner_share)
    width = style.get("outline_width", 1.0)
    masks, filled = {area: area for area in AREAS}, "filled"
    if style.get("close_gaps"):
        filled, masks["inner"] = closed_keys(symbol, width)
    if paints["outline"]:
        masks["outline"] = outline_key(symbol, width, filled)
    layers = [(masks[area], paints[area]) for area in AREAS]  # From the bottom
    if paints.get(SYMBOL_LINE):
        layers[-1:] = [("symbol", paints[SYMBOL_LINE]),
                       (core_key(symbol, style.get(SYMBOL_LINE_WIDTH_KEY, SYMBOL_LINE_WIDTH)), paints["symbol"])]
    body = "".join(_area_svg(symbol, key, stops) for key, stops in layers if stops and symbol.masks[key].any())
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {symbol.width} {symbol.height}" '
            f'width="{symbol.width}" height="{symbol.height}">\n{comment}{body}</svg>\n').encode("utf-8")


def symbol_png(svg: bytes) -> bytes:
    """
    PNG file content of a colored SVG: PNG_SIZE x PNG_SIZE, transparent, the symbol (its outline included)
    PNG_SYMBOL_HEIGHT pixels high (narrower if it is too wide for the PNG), its width following its shape,
    at the center. Its png_stamp is in the text chunk PNG_TEXT_KEY.
    """
    pymupdf = _pymupdf()
    with pymupdf.open(stream=svg, filetype="svg") as document:
        page = document[0]
        scale = PNG_RENDER_SIZE / max(page.rect.width, page.rect.height)
        png = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=True).tobytes("png")
    image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)  # BGRA, not premultiplied
    if image is None or image.ndim != 3 or image.shape[2] != 4:
        raise ValueError("Can't draw the SVG with transparency")
    image = image.astype(np.float32)
    alpha = image[:, :, 3:] / 255
    ys, xs = np.nonzero(alpha[:, :, 0] > 0.02)
    if len(xs) == 0:
        raise ValueError("empty symbol")
    # Reduced premultiplied by the alpha: no dark fringe around the symbol.
    symbol = np.concatenate([image[:, :, :3] * alpha, image[:, :, 3:]], axis=2)[ys.min():ys.max() + 1,
                                                                               xs.min():xs.max() + 1]
    height, width = symbol.shape[:2]
    factor = min(PNG_SYMBOL_HEIGHT / height, (PNG_SIZE - 2) / width)
    size = (max(1, round(width * factor)), max(1, round(height * factor)))
    small = cv2.resize(symbol, size, interpolation=cv2.INTER_AREA)
    canvas = np.zeros((PNG_SIZE, PNG_SIZE, 4), np.float32)
    x, y = (PNG_SIZE - size[0]) // 2, (PNG_SIZE - size[1]) // 2
    canvas[y:y + size[1], x:x + size[0]] = small
    alpha = canvas[:, :, 3:] / 255
    bgr = np.where(alpha > 0, canvas[:, :, :3] / np.maximum(alpha, 1e-6), 0)
    rgba = np.clip(np.concatenate([bgr[:, :, ::-1], canvas[:, :, 3:]], axis=2) + 0.5, 0, 255).astype(np.uint8)
    info = PngImagePlugin.PngInfo()
    info.add_text(PNG_TEXT_KEY, png_stamp())
    buffer = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buffer, "PNG", pnginfo=info)
    return buffer.getvalue()


def _png_up_to_date(path, stamp: str) -> bool:
    """True if the PNG is there, made with this png_stamp."""
    try:
        with Image.open(path) as image:
            return image.info.get(PNG_TEXT_KEY) == stamp
    except (OSError, ValueError):
        return False


def _up_to_date(path, stamp: str) -> bool:
    """True if the colored SVG is there, drawn with this drawing_stamp."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fp:
            return stamp in fp.read(1000)
    except OSError:
        return False


def colorize_set_symbols(codes: Optional[List[str]] = None, force: bool = False) -> Dict[str, int]:
    """
    Write set_<code>_<rarity>.svg in download.symbol_dir(SET_DIR_NAME), for every downloaded set symbol
    (or only `codes`), next to the downloaded SVG and sets.json (release dates), and its PNG
    set_<code>_<rarity>.png in download.symbol_dir(PNG_DIR_NAME) (symbol_png). The files already made
    with the same drawing_stamp (png_stamp) are kept (force: make them again).
    :return: number of SVG 'drawn', 'kept' (up to date), of PNG 'png_made', 'png_kept', sets
             'before_exodus' (3 files in the colors of a common), sets 'failed'.
    """
    directory = download.symbol_dir(download.SET_DIR_NAME)
    sets_file = directory / download.SETS_FILE_NAME
    sets = json.loads(sets_file.read_text(encoding="utf-8")) if sets_file.exists() else []
    dates = first_dates(sets)

    wanted = None if codes is None else {code.lower() for code in codes}
    todo = [set_data for set_data in sets if set_data.get(download.SET_SYMBOL_FILE_KEY)
            and (wanted is None or set_data["code"] in wanted)]
    png_dir = download.symbol_dir(PNG_DIR_NAME)
    png_dir.mkdir(parents=True, exist_ok=True)
    counts = {"drawn": 0, "kept": 0, "png_made": 0, "png_kept": 0, "before_exodus": 0, "failed": 0}
    stamp, stamp_png = drawing_stamp(), png_stamp()
    for set_data in tqdm.tqdm(todo, desc="Rarity symbols", unit="set"):
        code, file_name = set_data["code"], set_data[download.SET_SYMBOL_FILE_KEY]
        rarities = rarities_of(set_data.get("released_at"), dates)
        old = before_exodus(set_data.get("released_at"), dates)
        counts["before_exodus"] += old
        symbol = None
        for rarity in rarities:
            path = directory / rarity_file_name(code, rarity)
            png_path = png_dir / png_file_name(code, rarity)
            missing = force or not _up_to_date(path, stamp)
            png_missing = force or not _png_up_to_date(png_path, stamp_png)
            counts["kept"] += not missing
            counts["png_kept"] += not png_missing
            if not missing and not png_missing:
                continue
            try:
                content = None
                if missing:
                    symbol = symbol or trace((directory / file_name).read_bytes())
                    content = rarity_svg(symbol, code, rarity, file_name, painted_as="common" if old else None)
                png = symbol_png(content or path.read_bytes()) if png_missing else None
            except Exception as error:  # A broken SVG must not stop the other sets
                tqdm.tqdm.write(f"Can't draw {file_name} ({rarity}): {error}")
                counts["failed"] += 1
                break
            if missing:
                download._write_file(path, content)
            if png_missing:
                download._write_file(png_path, png)
            counts["drawn"] += missing
            counts["png_made"] += png_missing
    print(f"Rarity symbols: {counts['drawn']} drawn, {counts['kept']} up to date; PNG: {counts['png_made']} made, "
          f"{counts['png_kept']} up to date; "
          f"{counts['before_exodus']} sets before Exodus (common, uncommon, rare in the colors of a common)"
          + (f", {counts['failed']} sets failed" if counts["failed"] else "") + ".")
    return counts


if __name__ == "__main__":
    colorize_set_symbols(sys.argv[1:] or None, force=True)
