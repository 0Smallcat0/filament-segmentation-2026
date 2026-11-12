import csv, json, os, subprocess, sys, tempfile, unittest
from pathlib import Path
from pycocotools import mask as mask_utils
from src.data import prepare_train

class DataPipelineTests(unittest.TestCase):
    def test_grouping_medoid_and_cli(self):
        temp_root = Path(os.environ.get('FILAMENT_TMP', Path(__file__).resolve().parents[2] / 'tmp'))
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as td:
            root = Path(td)
            # Two agreeing annotators and one outlier for image a.
            images = [{'id': i, 'file_name': name+'.jpeg', 'height': 2048, 'width': 2048}
                      for i, name in [('a1','a'),('a2','a'),('a3','a'),('b1','b')]]
            def annotation(i, x):
                return {'image_id': i, 'segmentation': [[x,10,x+10,10,x+10,20,x,20]]}
            annotations = [annotation('a1',10), annotation('a2',10), annotation('a3',100), annotation('b1',40)]
            path = root / 'train.json'; path.write_text(json.dumps({'images':images,'annotations':annotations}))
            records = prepare_train(path, folds=2, seed=1)
            self.assertEqual(len(records), 2)
            a = next(r for r in records if r['filename'] == 'a.jpeg')
            self.assertEqual(a['image_id'], 'a1')
            self.assertEqual(len({r['fold'] for r in records}), 2)
            csv_path = root / 'oof.csv'
            with csv_path.open('w', newline='') as f:
                writer=csv.writer(f); writer.writerow(['filament_id','segmentation_rle'])
                writer.writerow(['a_1', a['rles'][0]['counts'].decode('ascii')])
            result = subprocess.run([sys.executable, 'score_oof.py', '--train-json', str(path),
                '--predictions', str(csv_path), '--fold', str(a['fold']), '--folds', '2', '--seed', '1'],
                cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, check=True)
            score=json.loads(result.stdout)
            self.assertEqual(score['observations'], 1)
            self.assertEqual(score['one_to_one']['pq'], 1.0)
            self.assertEqual(score['official']['pq'], 1.0)

if __name__ == '__main__': unittest.main()
