"""Report fold variation and held-out-image bootstrap uncertainty for a grid selection."""
import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path


def pq(items):
    totals = {key: sum(item['score']['official'][key] for item in items)
              for key in ('tp', 'fp', 'fn', 'sum_iou')}
    denominator = totals['tp'] + 0.5 * (totals['fp'] + totals['fn'])
    return totals['sum_iou'] / denominator if denominator else 0.0


def percentile(sorted_values, fraction):
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (position - lower) * (sorted_values[upper] - sorted_values[lower])


def summarize(docs, config, seed=20261110, samples=1000):
    if samples <= 0 or len(docs) != 5 or sorted(doc['fold'] for doc in docs) != list(range(5)):
        raise ValueError('Expected five distinct folds and a positive sample count')
    folds = []
    all_items = []
    for doc in sorted(docs, key=lambda d: d['fold']):
        indexes = [i for i, candidate in enumerate(doc['configs']) if candidate == config]
        if len(indexes) != 1:
            raise ValueError('Selected config is missing or duplicated')
        items = doc['observations'][str(indexes[0])]
        if not items:
            raise ValueError('Empty fold')
        folds.append({'fold': doc['fold'], 'observations': len(items), 'official_pq': pq(items)})
        all_items.extend(items)
    if len(all_items) != 707 or len({item['filename'] for item in all_items}) != 707:
        raise ValueError('Expected 707 unique held-out observations')
    generator = random.Random(seed)
    n = len(all_items)
    resamples = sorted(pq([all_items[generator.randrange(n)] for _ in range(n)]) for _ in range(samples))
    values = [fold['official_pq'] for fold in folds]
    return {'config': config, 'folds': folds,
            'fold_mean': statistics.mean(values), 'fold_sample_std': statistics.stdev(values),
            'pooled_official_pq': pq(all_items),
            'bootstrap': {'unit': 'held-out observation', 'observations': n, 'samples': samples,
                          'seed': seed, 'method': 'percentile, linear interpolation',
                          'official_pq_ci95': [percentile(resamples, 0.025), percentile(resamples, 0.975)]},
            'interpretation': 'Held-out medoid OOF only; selection on these folds is optimistic. Not a public test-score interval.'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--train-json', type=Path, required=True)
    parser.add_argument('--manifests', type=Path, nargs=5, required=True)
    parser.add_argument('--results', type=Path, nargs=5, required=True)
    parser.add_argument('--seed', type=int, default=20261110)
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from baseline.grid_pooled import pool
    selected_doc = json.loads(args.selection.read_text(encoding='utf8'))
    config = selected_doc.get('config', selected_doc)
    validated = pool(args.results, args.train_json, args.manifests)
    if sum(row['config'] == config for row in validated) != 1:
        raise ValueError('Selection not found uniquely in validated pool')
    docs = [json.loads(path.read_text(encoding='utf8')) for path in args.results]
    report = summarize(docs, config, seed=args.seed, samples=args.samples)
    pooled = next(row for row in validated if row['config'] == config)
    if abs(report['pooled_official_pq'] - pooled['official']['pq']) > 1e-12:
        raise ValueError('Pooled score disagrees with grid validator')
    report['validation'] = {'train_json_sha256': sha(args.train_json),
                            'selection_sha256': sha(args.selection),
                            'result_sha256': {str(path): sha(path) for path in args.results},
                            'manifest_sha256': {str(path): sha(path) for path in args.manifests}}
    args.output.write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
