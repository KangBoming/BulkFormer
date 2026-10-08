"""Regression checks for release input mapping, pooling, and download handling."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from utils.latest import (extract_feature, main_gene_selection, normalize_data,
                          prepare_latest_bundle)
from utils.v1plus import prepare_v1plus_bundle


class PoolingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(num_genes=3, mask_value=-10.0)

    def forward(self, expression, return_gene_embeddings=False):
        hidden = torch.tensor([[1., 10.], [2., 20.], [9., 90.]])
        return {'gene_embeddings': hidden.expand(len(expression), -1, -1),
                'expression_prediction': expression - 1.0}


class InputTests(unittest.TestCase):
    def setUp(self):
        self.genes = ['ENSG1', 'ENSG2', 'ENSG3']
        self.vocab = pd.DataFrame({'canonical_ensg': self.genes,
                                   'gene_length_bp': [1000., 2000., 1000.]})

    def test_counts_mapping_and_missing_values(self):
        counts = pd.DataFrame([[20., 10., np.nan, 1e8], [0., 20., 10., 1e8]],
                              columns=['ENSG2.4', 'ENSG1', 'ENSG3', 'OTHER'])
        normalized = normalize_data(counts, self.vocab)
        np.testing.assert_allclose(normalized.iloc[0, :2], np.log1p([500000., 500000.]), rtol=1e-6)
        self.assertTrue(np.isnan(normalized.iloc[0, 2]))
        self.assertEqual(normalized.iloc[1, 1], 0.)
        aligned, missing, _ = main_gene_selection(normalized, self.genes)
        self.assertEqual(missing, [])
        self.assertEqual(aligned.iloc[0, 2], -10.)
        self.assertEqual(aligned.iloc[1, 1], 0.)

    def test_absent_gene_keeps_complete_axis(self):
        frame = pd.DataFrame([[0., 2.]], columns=['ENSG2', 'ENSG1'])
        aligned, missing, metadata = main_gene_selection(frame, self.genes)
        np.testing.assert_equal(aligned.values, [[2., 0., -10.]])
        self.assertEqual(missing, ['ENSG3'])
        self.assertEqual(metadata['mask'].tolist(), [0, 0, 1])

    def test_invalid_inputs_fail_before_inference(self):
        for frame in [pd.DataFrame([[1., 2.]], columns=['ENSG1.1', 'ENSG1.2']),
                      pd.DataFrame([[-1.]], columns=['ENSG1']),
                      pd.DataFrame([[0.]], columns=['ENSG1'])]:
            with self.assertRaises(ValueError):
                normalize_data(frame, self.vocab)
        for value in [-1., np.inf, 20., np.nan]:
            with self.assertRaises(ValueError):
                main_gene_selection(pd.DataFrame([[value]], columns=['ENSG1']), self.genes)

    def test_pooling_includes_zero_excludes_missing_per_sample(self):
        model = PoolingModel()
        expression = np.array([[0., 3., -10.], [-10., 0., 4.]], dtype=np.float32)
        pooled = extract_feature(model, expression, batch_size=2)
        np.testing.assert_equal(pooled, [[1.5, 15.], [5.5, 55.]])
        genes = extract_feature(model, expression, 'gene_level', interested_gene_idx=[2, 0])
        self.assertEqual(genes.shape, (2, 2, 2))
        np.testing.assert_equal(genes[0], [[9., 90.], [1., 10.]])
        predictions = extract_feature(model, expression, 'expr_level')
        self.assertEqual(predictions[0, 0], -1.)
        with self.assertRaises(ValueError):
            extract_feature(model, expression, interested_gene_idx=[0])
        with self.assertRaises(ValueError):
            extract_feature(model, expression, interested_gene_idx=[3])


class DownloadTests(unittest.TestCase):
    def test_complete_files_corruption_missing_and_existing_directory(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            downloads = root / 'downloads'
            downloads.mkdir()
            manifest = {'format_version': 2, 'files': {}}
            for name in ['model.pt', 'model_config.json', 'gene_vocab.csv',
                         'edge_index.pt', 'edge_weight.pt', 'manifest.json']:
                (downloads / name).write_bytes(b'test')
                manifest['files'][name] = {'bytes': 4,
                                          'sha256': hashlib.sha256(b'test').hexdigest()}
            metadata = root / 'downloads.json'
            metadata.write_text(json.dumps(manifest))
            destination = prepare_latest_bundle(downloads, root / 'output', metadata)
            self.assertEqual(destination.name, 'bulkformer-latest')
            self.assertEqual((destination / 'model.pt').read_bytes(), b'test')
            self.assertEqual(prepare_latest_bundle(destination, root / 'output', metadata), destination)
            with self.assertRaises(FileExistsError):
                prepare_latest_bundle(downloads, root / 'output', metadata)
            (downloads / 'model.pt').write_bytes(b'bad!')
            with self.assertRaises(ValueError):
                prepare_latest_bundle(downloads, root / 'bad', metadata)
            self.assertFalse((root / 'bad' / 'bulkformer-latest').exists())
            (downloads / 'model.pt').write_bytes(b'test')
            manifest['model_directory'] = 'bulkformer_v1plus'
            metadata.write_text(json.dumps(manifest))
            compatibility = prepare_v1plus_bundle(downloads, root / 'legacy', metadata)
            self.assertEqual(compatibility.name, 'bulkformer_v1plus')
            self.assertEqual((compatibility / 'model.pt').read_bytes(), b'test')
            manifest['model_directory'] = '../../outside'
            metadata.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                prepare_latest_bundle(downloads, root / 'invalid', metadata)
            manifest.pop('model_directory')
            metadata.write_text(json.dumps(manifest))
            (downloads / 'model_config.json').unlink()
            with self.assertRaises(FileNotFoundError):
                prepare_latest_bundle(downloads, root / 'missing', metadata)


if __name__ == '__main__':
    unittest.main()
