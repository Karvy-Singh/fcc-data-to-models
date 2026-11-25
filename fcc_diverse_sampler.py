#!/usr/bin/env python3
"""Reproducible FCC sample selection (Python standard library only).

Two modes:
  exact      : Recreate the *previously supplied* 1,000 observations using
               their saved 1-based Excel row numbers. Exact ID reproducibility.
  regenerate : Select a new 1,000-row sample algorithmically with a fixed seed:
               extrema anchors -> greedy observed pair coverage -> balanced
               seeded selection -> quota-improving swaps.

Blue columns AJ, AL and AM are NEVER used to choose observations. Input columns
are A:G. Every other column is a target. Source sheet: first sheet ("Set-2").

Example:
  python fcc_diverse_sampler.py FCC_NN_Data_Set2.xlsx --mode exact \
      --manifest fcc_selected_rows_1000.txt --outdir exact_result
  python fcc_diverse_sampler.py FCC_NN_Data_Set2.xlsx --mode regenerate \
      --seed 2026 --outdir new_result

Outputs in --outdir:
  NN_1000.csv              : 7 inputs and 32 non-blue targets; header+units
  All_Columns_1000.csv     : all original columns + Source Excel Row
  selected_excel_rows.txt  : exact original Excel row indices (1-based)
  audit_input_levels.csv   : marginal input-level distribution checks
  audit_output_ranges.csv  : target minima/maxima and quantiles
  audit_summary.json       : counts, coverage and seed, hash of input workbook
"""

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

INPUT_COLUMNS = tuple(range(7))
BLUE_COLUMNS = (35, 37, 38)
OUTPUT_COLUMNS = tuple(i for i in range(7, 42) if i not in BLUE_COLUMNS)
REQUIRED_COLUMNS = INPUT_COLUMNS + OUTPUT_COLUMNS
ROW_TARGET_DEFAULT = 1000
NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
CELL_REF = re.compile(r'^([A-Z]+)')


def colnum(text):
    n = 0
    for c in text:
        n = n * 26 + (ord(c) - ord('A') + 1)
    return n - 1


def colname(i):
    name = ''
    i += 1
    while i:
        i, rem = divmod(i - 1, 26)
        name = chr(65 + rem) + name
    return name


def read_strings(z):
    if 'xl/sharedStrings.xml' not in z.namelist():
        return []
    strings = []
    with z.open('xl/sharedStrings.xml') as f:
        for _, elem in ET.iterparse(f, events=('end',)):
            if elem.tag == NS + 'si':
                strings.append(''.join(t.text or '' for t in elem.iter(NS + 't')))
                elem.clear()
    return strings


def read_source(path, keep_full_ids=None):
    """Streaming .xlsx reader, retaining only A:G plus non-blue targets in RAM.

    keep_full_ids (exact mode) additionally retains entire selected rows, so
    the original 49,935-row file needs only one pass in that mode.
    """
    data = []      # dict: row, inputs(7 numbers), targets(16 numbers)
    full_rows = {}
    header = None
    units = None
    with zipfile.ZipFile(path) as z:
        strings = read_strings(z)
        sheet = 'xl/worksheets/sheet1.xml'
        if sheet not in z.namelist():
            raise ValueError('Expected the Set-2 worksheet in xl/worksheets/sheet1.xml')
        with z.open(sheet) as f:
            for _, elem in ET.iterparse(f, events=('end',)):
                if elem.tag != NS + 'row':
                    continue
                rownum = int(elem.get('r'))
                record = {}
                for c in elem.findall(NS + 'c'):
                    cref = c.get('r')
                    if not cref:
                        continue
                    col = colnum(CELL_REF.match(cref).group(1))
                    v = c.find(NS + 'v')
                    if v is not None and v.text is not None:
                        raw = strings[int(v.text)] if c.get('t') == 's' else v.text
                    else:
                        inline = c.find(NS + 'is')
                        raw = ''.join(t.text or '' for t in inline.iter(NS + 't')) if inline is not None else ''
                    if raw != '':
                        record[col] = raw
                if rownum == 1:
                    header = record
                elif rownum == 2:
                    units = record
                elif rownum >= 3 and all(k in record for k in REQUIRED_COLUMNS):
                    try:
                        inputs = tuple(float(record[j]) for j in INPUT_COLUMNS)
                        targets = tuple(float(record[j]) for j in OUTPUT_COLUMNS)
                    except (ValueError, TypeError) as exc:
                        raise ValueError(f'Non-numeric input/target at Excel row {rownum}') from exc
                    if not all(math.isfinite(x) for x in inputs + targets):
                        raise ValueError(f'Nonfinite input/target at Excel row {rownum}')
                    data.append((rownum, inputs, targets))
                    if keep_full_ids is not None and rownum in keep_full_ids:
                        full_rows[rownum] = record
                elem.clear()
    if header is None or units is None or not data:
        raise ValueError('Missing expected headers, units, or valid numeric data')
    for i in REQUIRED_COLUMNS:
        if i not in header:
            raise ValueError(f'Missing expected header for column {colname(i)}')
    return data, header, units, full_rows


def read_selected_full_rows(path, desired_ids):
    """Read 42 original columns for selected IDs (reusing streaming reader)."""
    _, _, _, full = read_source(path, keep_full_ids=set(desired_ids))
    return full


def exact_ids(manifest):
    ids = []
    with open(manifest, encoding='utf-8') as f:
        for line in f:
            val = line.strip()
            if val and not val.startswith('#'):
                ids.append(int(val))
    if len(ids) != len(set(ids)):
        raise ValueError('Manifest contains duplicate Excel row numbers')
    return ids


def input_quotas(data, n):
    """Largest-remainder allocation: exact quotas per input, summing to n."""
    counts = [Counter(row[1][j] for row in data) for j in INPUT_COLUMNS]
    quotas = []
    for c in counts:
        expected = {v: count * n / len(data) for v, count in c.items()}
        q = {v: math.floor(x) for v, x in expected.items()}
        remaining = n - sum(q.values())
        order = sorted(expected, key=lambda v: (-(expected[v] - q[v]), v))
        for v in order[:remaining]:
            q[v] += 1
        quotas.append(q)
    return counts, quotas


def select_extrema(data):
    """Anchor the first row reaching global min/max for each target."""
    mins = [(float('inf'), None) for _ in OUTPUT_COLUMNS]
    maxs = [(float('-inf'), None) for _ in OUTPUT_COLUMNS]
    for idx, (_, _, y) in enumerate(data):
        for k, v in enumerate(y):
            if v < mins[k][0]:
                mins[k] = (v, idx)
            if v > maxs[k][0]:
                maxs[k] = (v, idx)
    return {i for _, i in mins + maxs}


def pair_masks(data):
    """One big-integer bit mask of observed two-factor levels per data row."""
    all_pairs = set()
    for _, x, _ in data:
        all_pairs.update((i, x[i], j, x[j]) for i in range(7) for j in range(i + 1, 7))
    ordered = sorted(all_pairs)
    lookup = {key: p for p, key in enumerate(ordered)}
    masks = []
    for _, x, _ in data:
        mask = 0
        for i in range(7):
            for j in range(i + 1, 7):
                mask |= 1 << lookup[(i, x[i], j, x[j])]
        masks.append(mask)
    return masks, len(ordered)


def cover_all_pairs(masks, pair_count, selected, rng, limit):
    """Greedy set cover; seeded random priorities resolve exact ties."""
    missing = (1 << pair_count) - 1
    for idx in selected:
        missing &= ~masks[idx]
    priority = [rng.random() for _ in masks]
    while missing:
        if len(selected) >= limit:
            raise ValueError('Cannot cover all input pairs within the requested sample size')
        best = -1
        best_gain = 0
        best_priority = 1.1
        for idx, m in enumerate(masks):
            if idx in selected:
                continue
            gain = (m & missing).bit_count()
            if gain > best_gain or (gain == best_gain and gain > 0 and priority[idx] < best_priority):
                best, best_gain, best_priority = idx, gain, priority[idx]
        if best == -1 or best_gain == 0:
            raise RuntimeError('Pair coverage is inconsistent')
        selected.add(best)
        missing &= ~masks[best]
    return selected


def balance_score(x, counts, quotas):
    """Highest score for under-represented levels; penalize overshoot."""
    score = 0.0
    for j, v in enumerate(x):
        target = quotas[j][v]
        deficit = target - counts[j][v]
        if deficit > 0:
            score += deficit / max(target, 1)
        else:
            score += 2.0 * deficit / max(target, 1)
    return score


def delta_swap(x_out, x_in, counts, quotas):
    """Decrease in sum of squared quota errors if out/in are exchanged."""
    improvement = 0
    for j in range(7):
        a, b = x_out[j], x_in[j]
        if a == b:
            continue
        ca, cb = counts[j][a], counts[j][b]
        ta, tb = quotas[j][a], quotas[j][b]
        improvement += (ca-ta)**2 + (cb-tb)**2 - ((ca-1-ta)**2 + (cb+1-tb)**2)
    return improvement


def generate_ids(data, requested, seed):
    rng = random.Random(seed)
    _, quotas = input_quotas(data, requested)
    selected = select_extrema(data)
    masks, pair_count = pair_masks(data)
    selected = cover_all_pairs(masks, pair_count, selected, rng, requested)
    protected = set(selected)  # Never swap out extrema or pairwise coverage anchors.
    counts = [Counter(data[idx][1][j] for idx in selected) for j in range(7)]
    N = len(data)

    # Seeded candidate-search filling (candidate pool size is part of method).
    while len(selected) < requested:
        candidates = rng.sample(range(N), min(384, N))
        best, best_score, best_tie = None, -1e300, -1.0
        for idx in candidates:
            if idx in selected:
                continue
            score = balance_score(data[idx][1], counts, quotas)
            tie = rng.random()
            if score > best_score or (score == best_score and tie > best_tie):
                best, best_score, best_tie = idx, score, tie
        if best is None:
            raise RuntimeError('No available rows to fill sample')
        selected.add(best)
        for j, val in enumerate(data[best][1]):
            counts[j][val] += 1

    # Quota repair: swap *only* unprotected rows. Every pair and output extreme
    # already protected remains present by construction.
    unprotected = set(selected) - protected
    for iteration in range(3000):
        excesses = [(counts[j][v]-q, j, v) for j, qs in enumerate(quotas)
                    for v, q in qs.items() if counts[j][v] > q]
        if not excesses:
            break
        excesses.sort(key=lambda t: (-t[0], t[1], t[2]))
        improvement_made = False
        # Focus moves on an over-quota input level, randomized among its rows.
        for _, j, val in excesses[:min(12, len(excesses))]:
            outs = [i for i in unprotected if data[i][1][j] == val]
            if not outs:
                continue
            rng.shuffle(outs)
            outs = outs[:16]
            insiders = rng.sample(range(N), min(384, N))
            candidate_in = [i for i in insiders if i not in selected and
                            counts[j][data[i][1][j]] < quotas[j][data[i][1][j]]]
            best_gain = 0
            best_pair = None
            for iout in outs:
                out = data[iout][1]
                for iin in candidate_in:
                    gain = delta_swap(out, data[iin][1], counts, quotas)
                    if gain > best_gain:
                        best_gain, best_pair = gain, (iout, iin)
            if best_pair is not None:
                old, new = best_pair
                for k in range(7):
                    counts[k][data[old][1][k]] -= 1
                    counts[k][data[new][1][k]] += 1
                selected.remove(old)
                selected.add(new)
                unprotected.remove(old)
                unprotected.add(new)
                improvement_made = True
                break
        if not improvement_made:
            break

    return sorted(data[idx][0] for idx in selected), len(protected), pair_count


def percentile(sorted_values, pct):
    if not sorted_values:
        return None
    at = (len(sorted_values) - 1) * pct
    low = int(math.floor(at))
    frac = at - low
    if low + 1 >= len(sorted_values):
        return sorted_values[low]
    return sorted_values[low] * (1 - frac) + sorted_values[low + 1] * frac


def write_csv(path, rows):
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def audit(data, ids, header, outdir, requested, mode, seed, protected_count, workbook_path):
    sample_ids = set(ids)
    sample = [r for r in data if r[0] in sample_ids]
    if len(sample) != requested:
        raise ValueError('Number of found source rows does not equal requested n')
    full_counts, _ = input_quotas(data, requested)
    sample_counts = [Counter(row[1][j] for row in sample) for j in range(7)]
    level_audit = [['Input', 'Excel column', 'Level', 'Source count', 'Source %',
                    'Sample count', 'Sample %', 'Difference (percentage points)']]
    max_diff = 0
    levels_retained = 0
    for j, c in enumerate(full_counts):
        for val in sorted(c):
            p_full = 100*c[val]/len(data)
            p_sample = 100*sample_counts[j][val]/len(sample)
            diff = p_sample-p_full
            max_diff = max(max_diff, abs(diff))
            levels_retained += sample_counts[j][val] > 0
            level_audit.append([header[j], colname(j), val, c[val],
                                round(p_full, 6), sample_counts[j][val],
                                round(p_sample, 6), round(diff, 6)])
    write_csv(outdir/'audit_input_levels.csv', level_audit)

    output_audit = [['Target', 'Excel column', 'Source min', 'Sample min',
                     'Source max', 'Sample max', 'Source P5', 'Sample P5',
                     'Source P50', 'Sample P50', 'Source P95', 'Sample P95']]
    preserved_ranges = 0
    for pos, col in enumerate(OUTPUT_COLUMNS):
        all_vals = sorted(r[2][pos] for r in data)
        sampled = sorted(r[2][pos] for r in sample)
        if all_vals[0] == sampled[0] and all_vals[-1] == sampled[-1]:
            preserved_ranges += 1
        output_audit.append([header[col], colname(col), all_vals[0], sampled[0],
                             all_vals[-1], sampled[-1],
                             percentile(all_vals, .05), percentile(sampled, .05),
                             percentile(all_vals, .50), percentile(sampled, .50),
                             percentile(all_vals, .95), percentile(sampled, .95)])
    write_csv(outdir/'audit_output_ranges.csv', output_audit)

    def pairs(rows):
        return {(j, row[1][j], k, row[1][k]) for row in rows for j in range(7)
                for k in range(j + 1, 7)}
    available_pairs = pairs(data)
    covered_pairs = pairs(sample)
    hash_digest = hashlib.sha256()
    with open(workbook_path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            hash_digest.update(chunk)
    result = {
        'mode': mode, 'seed': seed if mode == 'regenerate' else None,
        'source_file': workbook_path.name, 'source_sha256': hash_digest.hexdigest(),
        'source_valid_data_rows': len(data), 'selected_rows': len(sample),
        'input_columns': [colname(i) for i in INPUT_COLUMNS],
        'target_columns': [colname(i) for i in OUTPUT_COLUMNS],
        'individual_input_levels_source': sum(map(len, full_counts)),
        'individual_input_levels_retained': levels_retained,
        'pairwise_combinations_source': len(available_pairs),
        'pairwise_combinations_retained': len(covered_pairs),
        'output_ranges_preserved': preserved_ranges,
        'output_count': len(OUTPUT_COLUMNS),
        'max_input_level_proportion_deviation_pp': round(max_diff, 6),
        'protected_extrema_and_pairwise_anchors': protected_count,
        'selected_ids_sha256': hashlib.sha256(('\n'.join(map(str,ids))+'\n').encode()).hexdigest(),
        'interpretation': 'Blue columns were excluded from all selection criteria.'
    }
    (outdir/'audit_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('workbook', type=Path, help='Original FCC_NN_Data_Set2.xlsx')
    parser.add_argument('--mode', choices=['exact', 'regenerate'], default='exact')
    parser.add_argument('--manifest', type=Path, default=Path(__file__).with_name('fcc_selected_rows_1000.txt'))
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--outdir', type=Path, default=Path('fcc_reproduced'))
    args = parser.parse_args()
    if args.n <= 0:
        parser.error('--n must be positive')
    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest_ids = exact_ids(args.manifest) if args.mode == 'exact' else None
    if args.mode == 'exact' and len(manifest_ids) != args.n:
        parser.error(f'Exact manifest has {len(manifest_ids)} rows but --n is {args.n}')

    data, header, units, full_records = read_source(
        args.workbook, set(manifest_ids) if manifest_ids is not None else None)
    if len(data) < args.n:
        raise ValueError(f'Only {len(data)} valid rows; requested {args.n}')
    known_rows = {row[0] for row in data}
    if args.mode == 'exact':
        selected_ids = sorted(manifest_ids)
        protected_count = None
    else:
        selected_ids, protected_count, _ = generate_ids(data, args.n, args.seed)
        # Second streaming read for full 42-column rows of newly selected IDs.
        full_records = read_selected_full_rows(args.workbook, selected_ids)
    if any(i not in known_rows for i in selected_ids):
        raise ValueError('Selection contains an Excel row that is missing or invalid')
    if any(i not in full_records for i in selected_ids):
        raise ValueError('Could not read all selected original rows')

    all_columns = sorted(header)
    nn_columns = list(REQUIRED_COLUMNS)
    write_csv(args.outdir/'NN_1000.csv',
              [[header[i] for i in nn_columns], [units.get(i,'') for i in nn_columns]] +
              [[full_records[r].get(i,'') for i in nn_columns] for r in selected_ids])
    write_csv(args.outdir/'All_Columns_1000.csv',
              [[header[i] for i in all_columns] + ['Source Excel Row'],
               [units.get(i,'') for i in all_columns] + ['Original Set-2']] +
              [[full_records[r].get(i,'') for i in all_columns] + [r] for r in selected_ids])
    (args.outdir/'selected_excel_rows.txt').write_text('\n'.join(map(str, selected_ids))+'\n', encoding='utf-8')
    report = audit(data, selected_ids, header, args.outdir, args.n, args.mode,
                   args.seed, protected_count, args.workbook)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
