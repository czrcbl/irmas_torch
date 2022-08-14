import torch
import unittest
import os
from os.path import join as pjoin
from irmas_torch import transforms, datasets
from irmas_torch import config as cfg


project_path = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))

class Tests(unittest.TestCase):
    
    def test_transforms(self):
        
        fn = pjoin(project_path, 'data/IRMAS/IRMAS-TrainingData/cel/[cel][cla]0001__1.wav')
        
        trans = transforms.MelSpecTransformRosa(mono=False)
        spec = trans(fn)
        self.assertEqual(spec.shape[0], 2)
        self.assertEqual(spec.shape[1], 128)
        
        trans = transforms.MelSpecTransformTorchAudio(mono=False)
        spec = trans(fn)
        self.assertEqual(spec.shape[0], 2)
        self.assertEqual(spec.shape[1], 128)
        
        trans = transforms.MelSpecTransformRosa(mono=True)
        spec = trans(fn)
        self.assertEqual(spec.shape[0], 1)
        self.assertEqual(spec.shape[1], 128)
        
        trans = transforms.MelSpecTransformTorchAudio(mono=True)
        spec = trans(fn)
        self.assertEqual(spec.shape[0], 1)
        self.assertEqual(spec.shape[1], 128)


    def test_dataset(self):
        
        train_fraq = 0.7
        trn_ds = datasets.IRMAS(mode='train', train_fraq=train_fraq)
        val_ds = datasets.IRMAS(mode='val', train_fraq=train_fraq)
        
        total = len(trn_ds) + len(val_ds)
        
        self.assertEqual(int(train_fraq * total), len(trn_ds))
        self.assertGreater(len(trn_ds), len(val_ds))
#        print(ds.fns)
        
        test_ds = datasets.IRMAS(mode='test')
        
        trn_ds = datasets.IRMAS(seed=200)
        spec0 = trn_ds[0][0]
        trn_ds = datasets.IRMAS(seed=200)
        spec1 = trn_ds[0][0]
        self.assertTrue(torch.all(torch.eq(spec0, spec1)))
        
        trn_ds = datasets.IRMAS(seed=100)
        spec0 = trn_ds[0][0]
        trn_ds = datasets.IRMAS(seed=200)
        spec1 = trn_ds[0][0]
        self.assertFalse(torch.all(torch.eq(spec0, spec1)))
        
        
        
if __name__ == '__main__':
    unittest.main()


