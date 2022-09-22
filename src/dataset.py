import torch as th
import torch.nn.functional as F
import torchaudio as ta
import numpy as np
import sofa

class HRTFDataset:
    def __init__(self, config, sofa_path = "../HUTUBS/HRIRs"):
        '''
        args
        -----
            config: (dict)
            sofa_path: (str) path to dir contains *.sofa
        '''
        super().__init__()
        self.sofa_path = sofa_path
        self.config = config
        filter_length = round(self.config["fft_length"]/2) # 128
        max_f = self.config["max_frequency"] # 16000 (Hz)
        

        #==== load HUTBUS database [Brinkmann+19] =====
        subject_list = np.arange(1,96+1) # 1-indexed
        for sub_id in subject_list:
            path = self.sofa_path + "/pp" + str(sub_id) + "_HRIRs_measured.sofa"
            # path = self.sofa_path + "/pp" + str(sub_id) + "_HRIRs_simulated.sofa"
            HRTF, srcpos_sph = self.sofa2hrtf_pos(path,max_f,filter_length,self.config["green"])
            
            if sub_id == 1:
                self.srcpos = th.zeros(srcpos_sph.shape[0],srcpos_sph.shape[1],len(subject_list))
                self.HRTF = th.zeros(HRTF.shape[0],HRTF.shape[1],HRTF.shape[2],len(subject_list),dtype=th.complex64)
            self.srcpos[:,:,sub_id-1] = srcpos_sph
            self.HRTF[:,:,:,sub_id-1] = HRTF 
        #---------
        # self.srcpos: B x 3 x S = 440 x 3 x 96 tensor
        # self.HRTF:   B x 2 x filter_length x S = 440 x 2 x 128 x 96 tensor
        # B = # srcpos, S = # sub
        #==============================================
        
        #================ devide data =================
        # train (77): sub  0--76
        # valid (10): sub 77--86
        # test   (7): sub 88--94
        # [Note] sub95==sub0, sub87==sub21 
        
        self.HRTF_train = self.HRTF[:,:,:,:77]
        self.srcpos_train = self.srcpos[:,:,:77]

        self.HRTF_valid = self.HRTF[:,:,:,77:87] # sub 77--86
        self.srcpos_valid = self.srcpos[:,:,77:87]

        self.HRTF_test = self.HRTF[:,:,:,88:95] # sub 88--94
        self.srcpos_test = self.srcpos[:,:,88:95]
        #==============================================


        #=== calculate mean & std of train HRTF =======
        mag_l = th.abs(self.HRTF_train[:,0,:,:])
        mag_r = th.abs(self.HRTF_train[:,1,:,:])
        eps = 1e-10*th.ones(mag_l.shape).to(mag_l.device)
        magdb_l = th.log10(th.max(mag_l, eps))
        magdb_r = th.log10(th.max(mag_r, eps))
        magdb_lr = th.cat((magdb_l,magdb_r), dim=1)
        print(f"train data's mean:{th.mean(magdb_lr)}, std:{th.mean(th.std(magdb_lr,dim=1))}")
        #==============================================

    def __len__(self):
        '''
        :return: 
            number of training chunks in dataset
        '''
        return self.srcpos_train.shape[-1]

    def __getitem__(self, index): # for train (1 sub)
        '''
        returns: 
            srcpos as B x 3 = 440 x 3 tensor
            hrtf as B x 2 x filter_length = 440 x 2 x 128 tensor
        '''
        return self.srcpos_train[:,:,index], self.HRTF_train[:,:,:,index]
    
    def trainitem(self): # for train (all)
        '''
        returns: 
            srcpos as B x 3 x S_train = 440 x 3 x 77 tensor
            hrtf as B x 2 x filter_length x S_train = 440 x 2 x 128 x 77 tensor
        '''
        return self.srcpos_train, self.HRTF_train
    
    def validitem(self): # for validation
        '''
        returns: 
            srcpos as B x 3 x S_valid = 440 x 3 x 10 tensor
            hrtf as B x 2 x filter_length x S_valid = 440 x 2 x 128 x 10 tensor
        '''
        return self.srcpos_valid, self.HRTF_valid
    
    def testitem(self): # for test
        '''
        returns: 
            srcpos as B x 3 x S_test = 440 x 3 x 7 tensor
            hrtf as B x 2 x filter_length x S_test = 440 x 2 x 128 x 7 tensor
        '''
        return self.srcpos_test, self.HRTF_test
    
    def sofa2hrtf_pos(self, path, max_f, filter_length, multiple_green_func=False):
        '''
        args:
            path: (str) path to *.sofa
            max_f: (int) max frequency (= desired sampling frequency / 2)
            filter_length: (int)
            multiple_green_func: (bool)
        returns:
            HRTF: B x 2 x filter_length = 440 x 2 x 128 tensor
            srcpos_sph: B x 3 = 440 x 3 tensor
                        srcpos_sph[*,:]=[radius (m), azimuth in [0,2*pi) (rad), zenith in [0,pi] (rad)]
        '''
        SOFA = sofa.Database.open(path) # load *.sofa

        #==== srcpos =====
        srcpos_ori = th.tensor(SOFA.Source.Position.get_values()) # azimuth in [0,360),elevation in [-90,90], radius in {1.47}
        srcpos_sph = srcpos_ori
        srcpos_sph[:,0] = srcpos_sph[:, 0] % 360 # azimuth in [0,360)
        srcpos_sph[:,1] = 90 - srcpos_sph[:, 1] # elevation in [-90,90] -> zenith in [180,0]
        srcpos_sph[:,:2] = srcpos_sph[:,:2] / 180 * np.pi # azimuth in [0,2*pi), zenith in [0,pi]
        srcpos_sph = th.cat((srcpos_sph[:,2].unsqueeze(1), srcpos_sph[:,:2]), dim=1) # radius, azimuth in [0,2*pi), zenith in [0,pi]
        #=================

        self.sr_ori = SOFA.Data.SamplingRate.get_values()[0] # original sampling frequancy, 44100 (Hz)
        self.HRIR = th.tensor(SOFA.Data.IR.get_values()) # 440 x 2 x 256 tensor
        self.fft_length_ori = self.HRIR.shape[-1] # 256

        # Downsampling
        downsampler = ta.transforms.Resample(self.sr_ori,2*max_f)
        HRIR_us = downsampler(self.HRIR.to(th.float32))

        if 2*filter_length > HRIR_us.shape[-1]:
            HRIR_us = F.pad(input=HRIR_us,pad=(0,2*filter_length-HRIR_us.shape[-1]))
        else:
            HRIR_us = HRIR_us[:,:,HRIR_us.shape[-1]-2*filter_length:]


        HRTF_pm = th.conj(th.fft.fft(HRIR_us, dim=-1)) # FFT & conj
        HRTF = HRTF_pm[:,:,1:filter_length+1]          # Extract positive frequency
        
        if multiple_green_func:
            r = srcpos_sph[0,0]
            self.freq = th.arange(1,filter_length+1) * max_f/filter_length
            self.k = self.freq * 2 * np.pi / 343.18
            self.green = th.exp(1j* self.k * r) / (4*np.pi*r)
            # * green function
            HRTF = HRTF * self.green[None,None,:]
        
        return HRTF, srcpos_sph