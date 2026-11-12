"""Split the original model state into small, lossless CPU checkpoint files."""
import argparse
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    state = torch.load(args.source, map_location='cpu', weights_only=True)['model']
    parts = [{}, {}]
    sizes = [0, 0]
    for name, tensor in sorted(state.items(), key=lambda x: x[1].numel() * x[1].element_size(), reverse=True):
        index = 0 if sizes[0] <= sizes[1] else 1
        parts[index][name] = tensor
        sizes[index] += tensor.numel() * tensor.element_size()
    args.destination.mkdir(parents=True, exist_ok=True)
    for index, part in enumerate(parts):
        path = args.destination / f'detector-part{index}.pt'
        if path.exists():
            raise FileExistsError(path)
        torch.save(part, path)
        if path.stat().st_size > 100_000_000:
            raise ValueError(f'{path} exceeds 100 MB')
    restored = {}
    for index in range(2):
        restored.update(torch.load(args.destination / f'detector-part{index}.pt', map_location='cpu', weights_only=True))
    if set(restored) != set(state) or any(not torch.equal(state[k], restored[k]) for k in state):
        raise ValueError('Lossless split verification failed')
    print('Verified exact tensors:', len(state), 'bytes per part:', sizes)


if __name__ == '__main__':
    main()
