"""TEDn (Tree Edit Distance, normalized) evaluation for both LMX and Kern formats.

Provides a format-independent evaluation metric by converting predictions
to MusicXML and comparing against gold MusicXML using the TEDn metric
from olimpic-icdar24.

TEDn reference:
  Hajič, Novotný, Pecina, Pokorný.
  "Further Steps Towards a Standard Testbed for Optical Music Recognition"
  ISMIR 2016.
"""

import os
import sys
import logging
import xml.etree.ElementTree as ET
import traceback
import io

# Import TEDn from olimpic-icdar24 (https://github.com/ufal/olimpic-icdar24).
# Clone that repo and point OLIMPIC_ICDAR24_DIR at it (env var, defaults to a sibling
# ./olimpic-icdar24 directory next to this repo) -- see fine_tuning/README.md. Set
# `compute_tedn: false` in a fine-tuning config to skip this dependency entirely.
OLIMPIC_ICDAR24_DIR = os.environ.get(
    'OLIMPIC_ICDAR24_DIR', os.path.join(os.path.dirname(__file__), '..', 'olimpic-icdar24')
)
sys.path.insert(0, OLIMPIC_ICDAR24_DIR)
from app.evaluation.TEDn import TEDn, TEDnResult
from app.evaluation.TEDn_lmx_xml import TEDn_lmx_xml
from app.symbolic.actual_durations_to_fractional import actual_durations_to_fractional
from app.symbolic.Pruner import Pruner

logger = logging.getLogger(__name__)


def _kern_to_musicxml_part(kern_string: str) -> ET.Element:
    """Convert a predicted Kern string to a MusicXML <part> element via music21.

    Returns an empty <part> element on conversion failure.
    """
    try:
        import music21
        score = music21.converter.parse(kern_string, format='humdrum')

        # Export to MusicXML string
        exporter = music21.musicxml.m21ToXml.GeneralObjectExporter(score)
        musicxml_bytes = exporter.parse()
        musicxml_string = musicxml_bytes.decode('utf-8')

        # Parse and extract the first <part>
        root = ET.fromstring(musicxml_string)
        parts = root.findall('.//part')
        if parts:
            return parts[0]
        else:
            logger.warning("music21 produced MusicXML with no <part> element")
            return ET.Element("part")
    except Exception as e:
        logger.warning(f"Kern→MusicXML conversion failed: {e}")
        return ET.Element("part")


def _load_gold_part(gold_musicxml_string: str) -> ET.Element:
    """Parse gold MusicXML string and extract the first <part> element.

    Also converts durations to fractional representation for proper
    TEDn comparison (matching TEDn_lmx_xml behavior).
    """
    gold_musicxml_string = ET.canonicalize(gold_musicxml_string, strip_text=True)
    root = ET.fromstring(gold_musicxml_string)
    parts = root.findall('.//part')
    if not parts:
        raise ValueError("Gold MusicXML has no <part> element")
    part = parts[0]
    actual_durations_to_fractional(part)
    return part


def compute_tedn_lmx(pred_lmx: str, gold_musicxml_string: str) -> TEDnResult:
    """Compute TEDn between a predicted LMX string and gold MusicXML.

    Uses the olimpic-icdar24 TEDn_lmx_xml function directly, with
    flavor="lmx" to prune gold down to LMX-representable concepts.
    """
    errout = io.StringIO()
    result = TEDn_lmx_xml(
        predicted_lmx=pred_lmx,
        gold_musicxml=gold_musicxml_string,
        flavor="lmx",
        errout=errout,
    )
    errors = errout.getvalue()
    if errors:
        logger.debug(f"Delinearizer messages: {errors[:200]}")
    return result


def compute_tedn_kern(pred_kern: str, gold_musicxml_string: str) -> TEDnResult:
    """Compute TEDn between a predicted Kern string and gold MusicXML.

    Converts the Kern prediction to a MusicXML <part> via music21,
    then runs TEDn against the gold <part>. Uses the same pruning
    as the LMX flavor to ensure fair comparison.
    """
    # Convert prediction
    predicted_part = _kern_to_musicxml_part(pred_kern)

    # Load and prepare gold
    gold_part = _load_gold_part(gold_musicxml_string)

    # Prune both sides to the same subset (matching TEDn_lmx_xml flavor="lmx")
    pruner = Pruner(
        prune_durations=False,
        prune_measure_attributes=False,
        prune_prints=True,
        prune_slur_numbering=True,
        prune_directions=True,
        prune_barlines=True,
        prune_harmony=True,
    )
    pruner.process_part(gold_part)
    pruner.process_part(predicted_part)

    return TEDn(predicted_part, gold_part)


def compute_tedn_metrics(
    predictions: list,
    gold_musicxml_strings: list,
    format_type: str = 'lmx',
) -> float:
    """Compute aggregate TEDn error (%) over a list of predictions.

    Returns the corpus-level normalized edit cost as a percentage,
    computed as: 100 * sum(edit_costs) / sum(gold_costs).
    This is the standard TEDn aggregation (micro-average).

    Args:
        predictions: List of predicted strings (LMX or Kern format).
        gold_musicxml_strings: List of gold MusicXML file contents.
        format_type: 'lmx' or 'kern'.

    Returns:
        TEDn error rate as a percentage (0-100+).
    """
    compute_fn = compute_tedn_lmx if format_type == 'lmx' else compute_tedn_kern

    total_edit_cost = 0
    total_gold_cost = 0
    n_errors = 0

    for pred, gold_xml in zip(predictions, gold_musicxml_strings):
        try:
            result = compute_fn(pred, gold_xml)
            total_edit_cost += result.edit_cost
            total_gold_cost += result.gold_cost
        except Exception as e:
            logger.warning(f"TEDn computation failed for one sample: {e}")
            n_errors += 1

    if n_errors > 0:
        logger.warning(f"TEDn: {n_errors}/{len(predictions)} samples failed")

    if total_gold_cost == 0:
        return 100.0

    return 100.0 * total_edit_cost / total_gold_cost
