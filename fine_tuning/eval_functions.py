"""Evaluation metrics for polyphonic music output (SER, CER, LER).

Supports both Kern and LMX (Linearized MusicXML) formats.
- Kern LER: lines split by newline (each line = one time step across all staves)
- LMX LER: segments split by voice boundaries (each voice:N block = one voice segment)
"""

import re


def levenshtein(a, b):
    """Computes the Levenshtein distance between a and b."""
    n, m = len(a), len(b)
    if n > m:
        a, b = b, a
        n, m = m, n
    current = list(range(n + 1))
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change += 1
            current[j] = min(add, delete, change)
    return current[n]


def parse_krn_content(krn, ler_parsing=False, cer_parsing=False):
    if cer_parsing:
        return list(krn)
    elif ler_parsing:
        return krn.split("\n")
    else:
        krn = krn.replace("\n", " <b> ")
        krn = krn.replace("\t", " <t> ")
        return krn.split(" ")


def parse_lmx_content(lmx, ler_parsing=False, cer_parsing=False):
    """Parse LMX string for metric computation.

    For LER: split into voice segments (delimited by pitch/rest + voice:N tokens).
      Each segment contains one voice's content at one time position,
      analogous to one line in Kern (one time step).
    For SER: split on spaces (each LMX token = one symbol).
    For CER: split into individual characters.
    """
    if cer_parsing:
        return list(lmx)
    elif ler_parsing:
        # Split before "pitch voice:N" or "rest voice:N" boundaries.
        # In LMX, the pitch token immediately precedes voice:N and belongs
        # to that voice segment (e.g., "E5 voice:1 half stem:up ...").
        segments = re.split(r'(?= (?:[A-G][#b]*\d|rest(?::measure)?) voice:\d)', lmx)
        # Also split on "measure" to separate measure headers from voice content
        result = []
        for seg in segments:
            parts = re.split(r'(?= measure )|(?= measure$)', seg)
            result.extend(parts)
        # Strip whitespace, remove empty segments and backup-only segments
        result = [s.strip() for s in result if s.strip()]
        result = [s for s in result if not re.match(r'^(backup\s+\S+\s*)+$', s)]
        return result if result else [lmx]
    else:
        return lmx.split(" ")


def compute_metric(a1, a2):
    acc_ed_dist = 0
    acc_len = 0
    for h, g in zip(a1, a2):
        acc_ed_dist += levenshtein(h, g)
        acc_len += len(g)
    if acc_len == 0:
        return 0.0
    return 100.0 * acc_ed_dist / acc_len


def compute_poliphony_metrics(hyp_array, gt_array, format_type='kern'):
    """Compute CER, SER, LER between hypothesis and ground truth strings.

    Args:
        format_type: 'kern' or 'lmx' — determines how LER splits are computed.
    """
    parse_fn = parse_lmx_content if format_type == 'lmx' else parse_krn_content

    hyp_cer, gt_cer = [], []
    hyp_ser, gt_ser = [], []
    hyp_ler, gt_ler = [], []

    for h_string, gt_string in zip(hyp_array, gt_array):
        hyp_ler.append(parse_fn(h_string, ler_parsing=True))
        gt_ler.append(parse_fn(gt_string, ler_parsing=True))

        hyp_ser.append(parse_fn(h_string))
        gt_ser.append(parse_fn(gt_string))

        hyp_cer.append(parse_fn(h_string, cer_parsing=True))
        gt_cer.append(parse_fn(gt_string, cer_parsing=True))

    cer = compute_metric(hyp_cer, gt_cer)
    ser = compute_metric(hyp_ser, gt_ser)
    ler = compute_metric(hyp_ler, gt_ler)

    return cer, ser, ler
