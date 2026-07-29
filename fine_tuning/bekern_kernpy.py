"""Official-bekern (kernpy) tokenization for the from-scratch system bekern experiment.

Produces the SMT `<t>`/`<b>`/`<s>` token stream from kernpy's basicExtendedKern (bEkern) output,
used IDENTICALLY for GrandStaff pretraining and Debussy fine-tuning so pretrain/finetune share one
official tokenizer (no home-grown `decompose_kern_token`, no `@`-vs-`·` confound — self-consistent).

Pipeline:
  GrandStaff:  .krn                       --kernpy--> bEkern text --this module--> token stream
  Debussy:     .musicxml --verovio(kern)--> .krn(str) --kernpy--> bEkern text --this module--> tokens

bEkern format (from kernpy): tab-separated spines, newline rows, within-cell chord notes
space-separated, each note's atoms joined by '@' (e.g. `4@.@C@#`). Non-kern spines (**bedynam,
**betext, ...) and comments are dropped. Spine splits/merges (*^/*v) fold into the parent spine
(origin tracking, same fix validated for the compound tokenizer).
"""
import re
import kernpy as kp

_MANIP = {'*^', '*v', '*-', '*+'}


# Characters bEkern actually KEEPS in a note atom: duration digits, augmentation dot,
# pitch letters, accidentals (#, -, n), the editorial/cautionary marker X, and rest 'r'.
# Everything else (ties [ ] _, slurs ( ), phrases { }, beams L J K k, stems / \,
# articulations ' ~ ^, fermata ;, breath ,, ornaments T M S W R O, grace q, gliss h ...)
# is DISCARDED by bEkern anyway -- verified atom-by-atom against kernpy.
_BEKERN_KEEP = set("0123456789.#-nrXaAbBcCdDeEfFgG")


def clean_kern_for_kernpy(krn):
    """Strip verovio artifacts kernpy's strict parser rejects, on data rows only:
    tuplet ratios (%N), invisible markers (yy), and null tokens carrying junk
    (`.ZZZ...`, `.<`, `.]` -> `.`). Leaves interpretation/barline/comment lines untouched.

    ALSO strips note-level signifiers (ties/slurs/beams/stems/articulations/ornaments)
    BEFORE handing the token to kernpy. This is required for correctness, not cosmetics:
    kernpy's bEkern exporter silently DROPS every chord note after the first whenever any
    note in the chord carries a signifier (`4c 4e 4g` -> `4@c 4@e 4@g`, but
    `[4c [4e [4g` -> `4@c`, losing two real pitches with no error raised). Since bEkern
    discards these signifiers anyway, removing them up front loses nothing bEkern would
    have kept, while raising note retention on the Debussy corpus from ~83% to ~99.5%.
    """
    out = []
    for line in krn.split('\n'):
        if line.startswith(('!', '*', '=')) or not line.strip():
            out.append(line)
            continue
        cells = []
        for cell in line.split('\t'):
            toks = []
            for t in cell.split(' '):
                if not t:
                    continue
                t = re.sub(r'%-?\d+', '', t)      # tuplet ratios
                t = t.replace('yy', '')           # invisibility markers
                if t and t[0] == '.' and t != '.':  # null carrying a spanner/junk suffix
                    t = '.'
                if t != '.':
                    t = ''.join(c for c in t if c in _BEKERN_KEEP)
                toks.append(t if t else '.')
            cells.append(' '.join(toks) if toks else '.')
        out.append('\t'.join(cells))
    return '\n'.join(out)


def kernpy_bekern_text(krn_text):
    """Humdrum **kern text -> kernpy bEkern text (official basic extended kern)."""
    import tempfile, os
    with tempfile.NamedTemporaryFile('w', suffix='.krn', delete=False) as f:
        f.write(clean_kern_for_kernpy(krn_text))
        tmp = f.name
    try:
        doc, _ = kp.load(tmp)
        return kp.dumps(doc, encoding=kp.Encoding.bEkern)
    finally:
        os.unlink(tmp)


def bekern_to_token_string(bekern_text):
    """kernpy bEkern text -> SMT token stream ('<t>'/'<b>'/'<s>' + '@'-split atoms).

    Fixed-width per ORIGINAL **bekern spine; *^ split voices fold back into their parent spine's
    cell (space-joined notes -> '<s>'-joined); atoms within a note come from splitting on '@'.
    """
    lines = bekern_text.strip().split('\n')

    origin = None      # origin[i] = original kern-spine index of current column i (or None)
    n_kern = 0
    for line in lines:
        if line.startswith('**'):
            cols = line.split('\t')
            origin = []
            for c in cols:
                if c.startswith('**bekern') or c.startswith('**kern'):
                    origin.append(n_kern); n_kern += 1
                else:
                    origin.append(None)   # **bedynam / **betext / etc. -> dropped
            break
    if origin is None:
        return ''

    def atoms(note):
        # a single bEkern note like '4@.@C@#' -> '4 . C #'
        return ' '.join(a for a in note.split('@') if a)

    out_rows = []
    for line in lines:
        line = line.rstrip()
        if not line or line.startswith('!'):
            continue
        cols = line.split('\t')
        is_header = line.startswith('**')
        is_interp = line.startswith('*')
        is_manip = is_interp and not is_header and any(c in _MANIP for c in cols)
        is_barline = all(c.startswith('=') for c in cols if c)

        # Filter interpretation lines to the same essential set as the compound tokenizer
        # (clef, key sig, meter, met, staff, spine manipulations, header). Drops *I"instrument,
        # *part, *8va, etc. -> keeps GrandStaff-pretrain and Debussy-finetune vocab consistent.
        if is_interp and not is_header and not is_manip:
            rep = next((cols[i] for i, o in enumerate(origin) if o is not None and i < len(cols)), cols[0])
            if not (rep.startswith('*clef') or rep.startswith('*k[') or rep.startswith('*M')
                    or rep.startswith('*met(') or rep.startswith('*staff')):
                continue  # drop this interpretation line

        # gather cells into their original spine slot
        per_slot = [[] for _ in range(n_kern)]
        for i, val in enumerate(cols):
            o = origin[i] if i < len(origin) else None
            if o is not None:
                per_slot[o].append(val)

        if is_header:
            # emit a single '**bekern' header per original spine
            out_cols = ['**bekern' for _ in range(n_kern)]
        elif is_interp or is_barline:
            # structural rows: one representative token per spine, atomized
            out_cols = []
            for vals in per_slot:
                if not vals:
                    out_cols.append('.')
                else:
                    v = vals[0]
                    # interpretations (*clef, *k[], *M, =, *^ ...) stay whole (no '@' inside)
                    out_cols.append(atoms(v) if '@' in v else v)
        else:
            # real note/rest data: chords/split-voices -> '<s>'-joined atomized notes
            out_cols = []
            for vals in per_slot:
                notes = [n for cell in vals for n in cell.split(' ') if n and n != '.']
                if not notes:
                    out_cols.append('.')
                else:
                    out_cols.append(' <s> '.join(atoms(n) for n in notes))
        out_rows.append(' <t> '.join(out_cols))

        # advance origin past this row's spine manipulations
        if is_manip:
            new_origin = []
            i = 0
            while i < len(cols):
                tok = cols[i]
                o = origin[i] if i < len(origin) else None
                if tok == '*^':
                    new_origin += [o, o]; i += 1
                elif tok == '*v':
                    while i < len(cols) and cols[i] == '*v':
                        i += 1
                    new_origin.append(o)
                elif tok == '*-':
                    i += 1
                elif tok == '*+':
                    new_origin.append(None); i += 1
                else:
                    new_origin.append(o); i += 1
            origin = new_origin

    return ' <b> '.join(out_rows)


def krn_to_bekern_tokens(krn_text):
    """Convenience: Humdrum **kern text -> SMT bekern token stream (kernpy end-to-end)."""
    return bekern_to_token_string(kernpy_bekern_text(krn_text))
