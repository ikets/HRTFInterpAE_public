import torch as th
import torch.nn.functional as F
import torchaudio as ta

class LSD(th.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, data, target):
        return self._loss(data, target)
    def _loss(self, data, target):
        '''
        args
        -----
        data:   (B,2,L,S) complex (or float) tensor
        target: (B,2,L,S) complex (or float) tensor

        return
        -----
        a scalar loss value
        '''
        mag2db = ta.transforms.AmplitudeToDB(stype = 'magnitude')
        data_db   = mag2db(th.abs(data))
        target_db = mag2db(th.abs(target))
        return th.mean(th.sqrt(th.mean((data_db - target_db).pow(2), dim=2)))

class LSD_before_mean(th.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, data, target):
        return self._loss(data, target)
    def _loss(self, data, target):
        '''
        args
        -----
        data:   (B,2,L,S) complex (or float) tensor
        target: (B,2,L,S) complex (or float) tensor

        return
        -----
        LSD: (B,2,S) float tensor
        '''
        mag2db = ta.transforms.AmplitudeToDB(stype = 'magnitude')
        data_db   = mag2db(th.abs(data))
        target_db = mag2db(th.abs(target))
        return th.sqrt(th.mean((data_db - target_db).pow(2), dim=2))

class CosDistIntra(th.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, z):
        return self._loss(z)
    def _loss(self, z):
        '''
        args
        -----
        z: (B,dimz,S) tensor

        return
        -----
        a scalar loss value
        '''
        c = th.mean(z,dim=0) # centroid; prototype
        cs = F.cosine_similarity(z, c[None,:,:], dim=1)
        return th.mean((1-cs).pow(2))**0.5