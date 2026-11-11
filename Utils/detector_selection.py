"""Choose detector operating points without rewarding raw box count."""


def selection_key(row):
    return (bool(row.get('eligible', True)), row['f1'] if row['f1'] is not None else -1.)


def select_operating_point(rows, *, score=.4, sweep=False, min_precision=0., min_recall=0.):
    candidates = []
    for source in rows:
        if source['sequence'] != 'COMBINED' or source['iou'] != .5 or source['f1'] is None:
            continue
        if not sweep and source['score'] != score:
            continue
        row = dict(source)
        row['eligible'] = (row['precision'] is not None and row['recall'] is not None
                           and row['precision'] >= min_precision and row['recall'] >= min_recall)
        candidates.append(row)
    if not candidates:
        raise ValueError('No matching validation operating point')
    # An eligible candidate always outranks an ineligible one. If none qualify,
    # retain the highest F1 as an explicitly ineligible fallback for inspection.
    return max(candidates, key=lambda r: (selection_key(r), r['recall'], r['precision'] or 0.))
