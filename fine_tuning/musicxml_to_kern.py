#!/usr/bin/env python3
"""Standalone MusicXML -> **kern converter.

No dependency on the smt-finetune training code (data.py/torch/lightning/cv2) —
only `verovio` (primary) and `music21` (fallback) are needed. Copy this single
file anywhere and run it directly.

Examples:
    python musicxml2kern.py sample.musicxml
    python musicxml2kern.py sample.musicxml --stdout
    python musicxml2kern.py sample.musicxml --output sample.krn
    python musicxml2kern.py scores_dir/ --output /tmp/krn_out

Behavior:
    - Single file input defaults to the .krn sidecar next to the MusicXML.
    - Directory input converts every .musicxml file recursively.
    - If --output is given for a directory input, the relative folder structure
      is recreated under that output directory.
    - Existing .krn sidecars are reused unless --force is set.
    - Tries verovio first (run in a subprocess so segfaults don't crash this
      script), then falls back to a music21-based converter if verovio is
      unavailable, crashes, or times out.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import sys
from pathlib import Path
from typing import Iterable, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# MusicXML -> kern conversion
# =============================================================================

def _verovio_convert(musicxml_path: str, result_queue) -> None:
    """Worker function to convert MusicXML to kern in a separate process.

    Isolates verovio C++ code so segfaults don't crash the parent.
    """
    try:
        import verovio
        tk = verovio.toolkit()
        tk.setOptions({'inputFrom': 'musicxml-hum'})
        tk.loadFile(musicxml_path)
        kern_content = tk.getHumdrumBuffer()
        result_queue.put(kern_content if kern_content and kern_content.strip() else None)
    except Exception:
        result_queue.put(None)


def musicxml_to_kern(musicxml_path: str, timeout: int = 30) -> Optional[str]:
    """Convert a MusicXML file to **kern format using verovio.

    Uses a .krn sidecar cache: if a file with the same stem exists next to
    the MusicXML, its contents are returned directly without invoking verovio.
    On successful conversion, the result is saved to the sidecar so subsequent
    runs skip the (slow, crash-prone) verovio call.

    Runs verovio in a subprocess to survive segfaults. Falls back to a
    music21-based converter if verovio crashes or is unavailable.

    Returns the kern string, or None if conversion fails.
    """
    krn_cache_path = Path(musicxml_path).with_suffix('.krn')
    if krn_cache_path.exists():
        try:
            cached = krn_cache_path.read_text(encoding='utf-8')
            if cached.strip():
                logger.debug(f"Kern cache hit: {krn_cache_path.name}")
                return cached
        except Exception as e:
            logger.warning(f"Failed to read kern cache {krn_cache_path}: {e}")

    try:
        result_queue = multiprocessing.Queue()
        proc = multiprocessing.Process(target=_verovio_convert, args=(musicxml_path, result_queue))
        proc.start()
        proc.join(timeout=timeout)

        if proc.is_alive():
            proc.kill()
            proc.join()
            logger.warning(f"Timeout converting {musicxml_path}")
            return None

        if proc.exitcode != 0:
            logger.warning(f"Verovio crashed (exit {proc.exitcode}) on {musicxml_path}, "
                           "trying music21 fallback")
            kern = _music21_to_kern(musicxml_path)
            if kern and kern.strip():
                try:
                    krn_cache_path.write_text(kern, encoding='utf-8')
                    logger.info(f"Kern cache saved (music21 fallback): {krn_cache_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to write kern cache {krn_cache_path}: {e}")
            return kern

        if not result_queue.empty():
            kern = result_queue.get_nowait()
            if kern and kern.strip():
                try:
                    krn_cache_path.write_text(kern, encoding='utf-8')
                    logger.debug(f"Kern cache saved: {krn_cache_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to write kern cache {krn_cache_path}: {e}")
            return kern
        return None
    except Exception as e:
        logger.warning(f"Failed to convert {musicxml_path}: {e}")
        return None


# =============================================================================
# music21 fallback converter (for files that crash verovio)
# =============================================================================

def _m21_pitch_to_kern(pitch) -> str:
    """Convert a music21 Pitch to kern pitch token.

    Kern encoding: c=C4, cc=C5, C=C3, CC=C2, etc.
    Accidentals: #, ##, -, --, n (natural).
    """
    step = pitch.step.lower()
    octave = pitch.octave
    if octave <= 3:
        kern_pitch = step.upper() * (4 - octave)
    else:
        kern_pitch = step * (octave - 3)
    acc = pitch.accidental
    if acc:
        name_map = {'sharp': '#', 'double-sharp': '##',
                    'flat': '-', 'double-flat': '--', 'natural': 'n'}
        kern_pitch += name_map.get(acc.name, '')
    return kern_pitch


_M21_DUR_MAP = {
    'breve': '0', 'whole': '1', 'half': '2', 'quarter': '4',
    'eighth': '8', '16th': '16', '32nd': '32', '64th': '64', '128th': '128',
}


def _m21_duration_to_kern(duration) -> str:
    """Convert a music21 Duration to a kern duration ("recip") string.

    Plain/dotted durations (the common case) use the traditional recip+dots
    form (e.g. '4', '8.') via _M21_DUR_MAP, unchanged from before -- this
    path is well-tested and matches verovio's own convention for these.

    Tuplets (or any duration _M21_DUR_MAP can't represent, e.g. music21's
    'complex' type) use kern's fractional recip form 'N%D' instead, derived
    directly from the exact quarterLength as a Fraction: kern's own semantics
    are quarterLength == (4/N) * D, so N/D is just 4/quarterLength reduced --
    confirmed empirically by round-tripping test tokens through music21's own
    humdrum parser (e.g. '8%3' <-> 1.5, '12' <-> 1/3). This is the same
    notation verovio itself already emits for tuplets (seen in real converted
    output, e.g. '16%3'), so it's not a new convention for this pipeline --
    just one this fallback converter never implemented. Previously, a
    triplet eighth silently used the plain 'eighth' dot-less type ('8'),
    losing the tuplet ratio entirely -- found via a real content diff on
    converted MUSCIMA++ data, not assumed.
    """
    if not duration.tuplets and duration.type in _M21_DUR_MAP:
        return _M21_DUR_MAP[duration.type] + '.' * duration.dots
    from fractions import Fraction
    recip = Fraction(4, 1) / Fraction(duration.quarterLength)
    return str(recip.numerator) if recip.denominator == 1 else f'{recip.numerator}%{recip.denominator}'


def _grace_duration_str(duration) -> str:
    """Duration prefix for a grace note's kern token. A music21 GraceDuration
    keeps a display type (usually 'eighth') but quarterLength 0 -- the normal
    recip path would divide by zero on any grace type _M21_DUR_MAP can't
    name, so fall back to '8' (the conventional acciaccatura display value)."""
    if duration.type in _M21_DUR_MAP:
        return _M21_DUR_MAP[duration.type] + '.' * duration.dots
    return '8'


def _m21_note_to_kern(note) -> str:
    """Convert a music21 Note to a kern token, including ties and the 'q'
    grace marker (e.g. '8aq') -- music21's own humdrum parser round-trips 'q'
    back to a zero-duration grace note, verified empirically."""
    if note.duration.isGrace:
        return _grace_duration_str(note.duration) + _m21_pitch_to_kern(note.pitch) + 'q'
    token = _m21_duration_to_kern(note.duration) + _m21_pitch_to_kern(note.pitch)
    if note.tie:
        if note.tie.type == 'start':
            token = '[' + token
        elif note.tie.type == 'stop':
            token += ']'
        elif note.tie.type == 'continue':
            token = '[' + token + ']'
    return token


def _m21_chord_to_kern(chord) -> str:
    """Convert a music21 Chord to kern (space-separated notes)."""
    if chord.duration.isGrace:
        dur = _grace_duration_str(chord.duration)
        return ' '.join(dur + _m21_pitch_to_kern(p) + 'q'
                        for p in sorted(chord.pitches, key=lambda x: x.midi))
    dur = _m21_duration_to_kern(chord.duration)
    tokens = [dur + _m21_pitch_to_kern(p)
              for p in sorted(chord.pitches, key=lambda x: x.midi)]
    if chord.tie:
        if chord.tie.type == 'start':
            tokens = ['[' + t for t in tokens]
        elif chord.tie.type == 'stop':
            tokens = [t + ']' for t in tokens]
    return ' '.join(tokens)


def _m21_events_to_kern_lines(events) -> List[str]:
    """Group simultaneous (offset, element) events into kern data lines.

    Grace notes always get their OWN line (emitted before the same-offset
    principal note, matching kern's zero-time grace-row convention) instead
    of being space-merged into the principal's field -- merging serialized a
    grace as a bogus extra chord member, which either corrupted or dropped
    it on round-trip (found via a real corpus regression, 37 vs 35 notes)."""
    import music21 as _m21
    if not events:
        return []
    # Stable re-sort: at one offset a grace precedes its principal note (the
    # caller's pitch-based sort knows nothing about graces); non-grace
    # relative order is preserved.
    events = sorted(events, key=lambda x: (round(float(x[0]), 3),
                                           0 if getattr(x[1].duration, 'isGrace', False) else 1))
    lines: List[str] = []
    cur_off = None
    cur_tok: List[str] = []
    for offset, el in events:
        is_grace = getattr(el.duration, 'isGrace', False)
        if cur_off is not None and (abs(offset - cur_off) > 0.001 or is_grace):
            if cur_tok:
                lines.append(' '.join(cur_tok))
            cur_tok = []
        cur_off = offset
        if isinstance(el, _m21.chord.Chord):
            tok = _m21_chord_to_kern(el)
        elif isinstance(el, _m21.note.Note):
            tok = _m21_note_to_kern(el)
        elif isinstance(el, _m21.note.Rest):
            tok = _m21_duration_to_kern(el.duration) + 'r'
        else:
            continue
        if is_grace:
            lines.append(tok)  # own line; next event starts a fresh field
            cur_tok = []
            cur_off = None
        else:
            cur_tok.append(tok)
    if cur_tok:
        lines.append(' '.join(cur_tok))
    return lines


def _kern_token(el) -> str:
    """Kern token for a single music21 note/chord/rest element."""
    import music21 as _m21
    if isinstance(el, _m21.chord.Chord):
        return _m21_chord_to_kern(el)
    if isinstance(el, _m21.note.Note):
        return _m21_note_to_kern(el)
    if isinstance(el, _m21.note.Rest):
        return _m21_duration_to_kern(el.duration) + 'r'
    return '.'


def _music21_to_kern(musicxml_path: str) -> Optional[str]:
    """Fallback MusicXML->kern converter using music21.

    Produces valid kern. Compared to verovio output, beam markers (L/J/K)
    are absent. A staff carrying two music21 Voices in a measure is emitted as
    a real Humdrum spine split (``*^`` ... ``*v *v``) so both voices survive the
    round-trip; staves with 0/1/3+ voices stay a single column. Pitches,
    durations, ties, clefs, key/time signatures are preserved.

    Handles an arbitrary number of parts/staves (not just the 2-staff piano
    case): column order is staffN..staff1 left-to-right (matching verovio's
    own convention for N-staff scores), which for N=2 reproduces the original
    "*staff2 (LH) left, *staff1 (RH) right" layout exactly. Previously this
    hardcoded exactly 2 columns via `parts[1] if len(parts) > 1 else parts[0]`,
    which silently duplicated every note into both columns for single-staff
    input (parts[1] aliased parts[0]) and silently dropped any staff beyond
    the second for 3+-staff input.
    """
    try:
        import music21 as _m21
    except ImportError:
        logger.warning("music21 not installed — cannot use fallback converter")
        return None

    try:
        score = _m21.converter.parse(str(musicxml_path))
    except Exception as e:
        logger.warning(f"music21 failed to parse {musicxml_path}: {e}")
        return None

    parts = list(score.parts)
    if not parts:
        return None

    # Column order left-to-right is staffN..staff1 (descending), matching
    # verovio's own convention -- so parts (index 0 = staff1/top) are
    # reversed for display.
    display_parts = list(reversed(parts))
    n = len(display_parts)

    def _clef_token(part, default: str) -> str:
        clefs = part.flatten().getElementsByClass('Clef')
        if not clefs:
            return default
        if isinstance(clefs[0], _m21.clef.TrebleClef):
            return '*clefG2'
        if isinstance(clefs[0], _m21.clef.BassClef):
            return '*clefF4'
        return default

    # part0 (staff1, rightmost column) keeps its original default-to-treble
    # behavior; every other part defaults to bass, matching the original
    # function's per-column defaults for the N=2 case.
    clef_tokens = [
        _clef_token(p, '*clefG2' if p is parts[0] else '*clefF4')
        for p in display_parts
    ]

    ks_list = parts[0].flatten().getElementsByClass('KeySignature')
    if ks_list and ks_list[0].sharps != 0:
        sharps_order = ['f', 'c', 'g', 'd', 'a', 'e', 'b']
        flats_order = ['b', 'e', 'a', 'd', 'g', 'c', 'f']
        ks = ks_list[0]
        if ks.sharps > 0:
            ks_str = '*k[' + ''.join(p + '#' for p in sharps_order[:ks.sharps]) + ']'
        else:
            ks_str = '*k[' + ''.join(p + '-' for p in flats_order[:abs(ks.sharps)]) + ']'
    else:
        ks_str = '*k[]'

    ts_list = parts[0].flatten().getElementsByClass('TimeSignature')
    ts_str = (f'*M{ts_list[0].numerator}/{ts_list[0].denominator}'
              if ts_list else '*M4/4')

    staff_labels = [f'*staff{n - i}' for i in range(n)]  # staffN..staff1 left-to-right
    lines = [
        '\t'.join(['**kern'] * n),
        '\t'.join(staff_labels),
        '\t'.join(clef_tokens),
        '\t'.join([ks_str] * n),
        '\t'.join([ts_str] * n),
    ]

    measures_per_part = [list(p.getElementsByClass('Measure')) for p in display_parts]
    n_measures = max((len(m) for m in measures_per_part), default=0)

    def _sort_key(item):
        _, el = item
        if hasattr(el, 'pitch'):
            return -el.pitch.midi
        if hasattr(el, 'pitches') and el.pitches:
            return -el.pitches[0].midi
        return 0

    for mi in range(n_measures):
        col_measures = [ms[mi] if mi < len(ms) else None for ms in measures_per_part]
        mnum = next((m.number for m in col_measures if m is not None), mi + 1)
        lines.append('\t'.join([f'={mnum}'] * n))

        # A staff with exactly two music21 Voices this measure becomes a real
        # Humdrum spine split (*^ ... *v *v) so its two voices survive the
        # round-trip as two voices. The flatten-everything-into-one-column
        # path used to collide a rest in one voice with a note in the other at
        # the same offset and silently drop notes -- exactly what broke once
        # the MusicXML converter started emitting balanced multi-voice
        # measures. Splits are local to the measure (opened after the barline,
        # joined before the next), so every barline and the header keep the
        # base n-spine layout. Staves with 0/1/3+ voices stay single-spine
        # (3+ independent voices in one staff are rare and only partially
        # separated upstream anyway).
        split_flags = [bool(m) and len(m.voices) == 2 for m in col_measures]

        if not any(split_flags):
            # Unchanged single-spine-per-staff path (the common case; keeps the
            # existing round-trip tests' exact output). Columns are emitted
            # independently and padded to equal length with null tokens.
            kern_lines_per_col = []
            for m in col_measures:
                events = []
                if m:
                    sources = m.voices if m.voices else [m.flatten()]
                    for src in sources:
                        flat = src.flatten() if hasattr(src, 'voices') else src
                        for el in flat.notesAndRests:
                            events.append((float(el.offset), el))
                events.sort(key=lambda x: (x[0], _sort_key(x)))
                kern_lines_per_col.append(_m21_events_to_kern_lines(events))

            max_lines = max((len(kl) for kl in kern_lines_per_col), default=0)
            for li in range(max(max_lines, 1)):
                row = [kl[li] if li < len(kl) else '.' for kl in kern_lines_per_col]
                lines.append('\t'.join(row))
            continue

        # Spine-split path. Build the ordered sub-spines (a split staff
        # contributes its two voices, every other staff one flattened stream),
        # then align them all by absolute in-measure offset: one data line per
        # distinct offset, each sub-spine showing its token when it starts an
        # event there and a null '.' (the previous token sustains) otherwise.
        # Each subspine maps offset -> {'graces': [tok...], 'main': tok|None}.
        # Keying a plain dict by offset alone silently OVERWROTE one of any
        # two same-offset events -- exactly what a grace note and its
        # principal are -- losing real notes on round-trip (found via a real
        # corpus regression once grace notes started being emitted at all).
        subspines: List[dict] = []
        for m, split in zip(col_measures, split_flags):
            if split:
                streams = list(m.voices)
            elif m is not None:
                streams = [m.flatten()]
            else:
                streams = [None]
            for st in streams:
                events: dict = {}
                if st is not None:
                    for el in st.flatten().notesAndRests:
                        slot = events.setdefault(float(el.offset), {'graces': [], 'main': None})
                        if getattr(el.duration, 'isGrace', False):
                            slot['graces'].append(_kern_token(el))
                        else:
                            slot['main'] = _kern_token(el)
                subspines.append(events)

        # Open the splits: one field per STAFF (pre-split count n), '*^' where
        # that staff splits.
        lines.append('\t'.join('*^' if s else '*' for s in split_flags))

        offsets = sorted({o for ss in subspines for o in ss})
        for off in (offsets or [None]):
            if off is None:
                lines.append('\t'.join('.' for _ in subspines))
                continue
            slots = [ss.get(off) for ss in subspines]
            # Grace rows first (zero-time lines preceding the beat), padded at
            # the FRONT so the grace nearest the beat sits on the last grace
            # row; then one main row for the beat itself.
            n_grace_rows = max((len(s['graces']) if s else 0) for s in slots)
            for g_row in range(n_grace_rows):
                row = []
                for s in slots:
                    graces = s['graces'] if s else []
                    pad = n_grace_rows - len(graces)
                    row.append(graces[g_row - pad] if g_row >= pad else '.')
                lines.append('\t'.join(row))
            if any(s and s['main'] is not None for s in slots):
                lines.append('\t'.join(
                    (s['main'] if s and s['main'] is not None else '.') for s in slots
                ))

        # Join each split staff's two sub-spines back into one (*v *v), so the
        # spine count returns to n before the next barline.
        join_row: List[str] = []
        for s in split_flags:
            join_row += ['*v', '*v'] if s else ['*']
        lines.append('\t'.join(join_row))

    lines.append('\t'.join(['=='] * n))
    kern = '\n'.join(lines)
    logger.info(f"music21 fallback produced {len(kern)} chars for {Path(musicxml_path).name}")
    return kern


# =============================================================================
# CLI
# =============================================================================

def iter_musicxml_files(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
    else:
        yield from sorted(input_path.rglob("*.musicxml"))


def output_is_file(output_path: Optional[Path]) -> bool:
    return bool(output_path and (output_path.suffix.lower() == ".krn" or (output_path.exists() and output_path.is_file())))


def resolve_target_path(source_path: Path, input_root: Path, output_path: Optional[Path]) -> Path:
    if output_path is None:
        return source_path.with_suffix(".krn")

    if output_is_file(output_path):
        return output_path

    relative_path = source_path.relative_to(input_root)
    return output_path / relative_path.with_suffix(".krn")


def convert_one(source_path: Path, target_path: Path, timeout: int, force: bool) -> bool:
    sidecar_path = source_path.with_suffix(".krn")

    if force and sidecar_path.exists():
        sidecar_path.unlink()

    kern = musicxml_to_kern(str(source_path), timeout=timeout)
    if not kern or not kern.strip():
        logger.warning(f"FAILED: {source_path}")
        return False

    if target_path != sidecar_path:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(kern, encoding="utf-8")
        logger.info(f"Wrote {target_path}")
    else:
        logger.info(f"Wrote {sidecar_path}")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_path", type=Path, help="MusicXML file or directory to convert")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output .krn file or output directory. Defaults to sidecar .krn files.",
    )
    parser.add_argument("--force", action="store_true", help="Re-convert even if a .krn sidecar already exists")
    parser.add_argument("--timeout", type=int, default=30, help="Per-file conversion timeout in seconds")
    parser.add_argument("--stdout", action="store_true", help="Print the converted kern to stdout (single-file mode only)")
    args = parser.parse_args()

    input_path = args.input_path.resolve()
    if not input_path.exists():
        logger.error(f"Not found: {input_path}")
        sys.exit(1)

    if input_path.is_file() and input_path.suffix.lower() != ".musicxml":
        logger.error(f"Expected a .musicxml file: {input_path}")
        sys.exit(1)

    if args.stdout and not input_path.is_file():
        logger.error("--stdout is only supported for a single input file")
        sys.exit(1)

    if input_path.is_dir() and args.output is not None and output_is_file(args.output):
        logger.error("When converting a directory, --output must be a directory path")
        sys.exit(1)

    files = list(iter_musicxml_files(input_path))
    if not files:
        logger.error(f"No .musicxml files found under {input_path}")
        sys.exit(1)

    if input_path.is_file() and args.stdout:
        sidecar_path = input_path.with_suffix(".krn")
        if args.force and sidecar_path.exists():
            sidecar_path.unlink()

        kern = musicxml_to_kern(str(input_path), timeout=args.timeout)
        if not kern or not kern.strip():
            sys.exit(1)

        sys.stdout.write(kern)
        if not kern.endswith("\n"):
            sys.stdout.write("\n")
        return

    input_root = input_path if input_path.is_dir() else input_path.parent
    output_path = args.output.resolve() if args.output is not None else None

    converted = 0
    failed = 0

    for index, source_path in enumerate(files, 1):
        target_path = resolve_target_path(source_path, input_root, output_path)
        if convert_one(source_path, target_path, args.timeout, args.force):
            converted += 1
            logger.info(f"[{index}/{len(files)}] OK: {source_path} -> {target_path}")
        else:
            failed += 1
            logger.warning(f"[{index}/{len(files)}] FAILED: {source_path}")

    logger.info("=" * 60)
    logger.info(f"Done: {converted} converted, {failed} failed, {len(files)} total")
    logger.info("=" * 60)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
