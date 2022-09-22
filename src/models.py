import torch as th
import torchaudio as ta
import torch.nn as nn
import torch.nn.functional as F
from src.utils import Net, sph2cart, aprox_t_des

class HyperLinear(nn.Module):
    def __init__(self, ch_in, ch_out, aux_size=3, ch_hidden=32, num_hidden=1, droprate=0.0):
        '''
        args:
            ch_in:  (int) input feature size
            ch_out: (int) output feature size
            aux_size: (int) feature size of auxiliary information
            ch_hidden:  (int) feature size of hypernet
            num_hidden: (int) # layers of hypernet
            droprate: (float) rate of dropout
        '''
        super().__init__()
        self.ch_in  = ch_in
        self.ch_out = ch_out

        #===== weight generator ======
        modules = [
            nn.Linear(aux_size, ch_hidden),
            nn.LayerNorm(ch_hidden),
            nn.ReLU(),
            nn.Dropout(droprate)]
        for _ in range(num_hidden):
            modules.extend([
                nn.Linear(ch_hidden, ch_hidden),
                nn.LayerNorm(ch_hidden),
                nn.ReLU(),
                nn.Dropout(droprate)
            ])
        modules.extend([
             nn.Linear(ch_hidden, ch_out * ch_in)
        ])
        self.weight_layers = nn.Sequential(*modules)

        #====== bias generator =======
        modules = [
            nn.Linear(aux_size, ch_hidden),
            nn.LayerNorm(ch_hidden),
            nn.ReLU(),
            nn.Dropout(droprate)]
        for _ in range(num_hidden):
            modules.extend([
                nn.Linear(ch_hidden, ch_hidden),
                nn.LayerNorm(ch_hidden),
                nn.ReLU(),
                nn.Dropout(droprate)
            ])
        modules.extend([
             nn.Linear(ch_hidden, ch_out)
        ])
        self.bias_layers = nn.Sequential(*modules)

    def forward(self, input):
        '''
        args:
            input: (dict)
                x: S x B x ch_in tensor
                z: S x B x aux_size tensor
        return:
            output: (dict)
                x: S x B x ch_out tensor
                z: S x B x aux_size tensor (=input["z"])
        '''
        x = input["x"] 
        z = input["z"]
        S = x.shape[0] 
        B = x.shape[1] 
        weight = self.weight_layers(z) # S x B x ch_out * ch_in
        weight = weight.view(S, B, self.ch_out, self.ch_in)
        bias = self.bias_layers(z) # S x B x ch_out

        output = {}
        wx = th.matmul(weight, x.view(S,B,-1,1)).view(S,B,-1)
        output["x"] = wx + bias
        output["z"] = z
        
        return output

class HyperLinearBlock(nn.Module):
    def __init__(self, ch_in, ch_out, ch_hidden=32, num_hidden=1, droprate=0.0, aux_size=3, post_prcs=True):
        '''
        args:
            ch_in:  (int) input feature size
            ch_out: (int) output feature size
            ch_hidden:  (int) feature size of hypernet
            num_hidden: (int) # layers of hypernet
            droprate: (float) rate of dropout
            aux_size: (int) feature size of auxiliary information
            post_prcs: (bool) if true, [LayerNorm, ReLU, Dropout] is added after HyperLinear
        '''
        super().__init__()
        self.hyperlinear = HyperLinear(ch_in, ch_out, ch_hidden=ch_hidden, num_hidden=num_hidden, aux_size=aux_size, droprate=droprate)
        if post_prcs:
            self.layers_post = nn.Sequential(
                nn.LayerNorm(ch_out),
                nn.ReLU(),
                nn.Dropout(droprate)
            )
        else:
            self.layers_post = nn.Sequential(
                nn.Identity()
            )

    def forward(self, input):
        '''
        args:
            input: (dict)
                x: S x B x ch_in tensor
                z: S x B x aux_size tensor
        return:
            output: (dict)
                x: S x B x ch_out tensor
                z: S x B x aux_size tensor (=input["z"])
        '''
        z = input["z"] 
        xz = self.hyperlinear(input)

        output = {}
        output["x"] = self.layers_post(xz["x"])
        output["z"] = z
        
        return output

class HRTF_Interp_AE(Net):
    def __init__(self, config, model_name='hrtf_interp_ae', use_cuda=True):
        super().__init__(model_name, use_cuda)
        self.config = config
        self.filter_length = round(config["fft_length"]/2)

        #====== Encoder =======
        # HyperLinear(256, 128), LayerNorm(128), ReLU
        # HyperLinear(128,  64), LayerNorm(64), ReLU
        # HyperLinear( 64,  64)
        #----------------------
        modules = []
        modules.extend([
            HyperLinearBlock(ch_in=2*1*self.filter_length, ch_out=config["channel_En_z"], ch_hidden=config["channel_hyper"], num_hidden=config["hlayers_hyper"], droprate=config["droprate"]), 
            HyperLinearBlock(ch_in=config["channel_En_z"], ch_out=config["dim_z"], ch_hidden=config["channel_hyper"], num_hidden=config["hlayers_hyper"], droprate=config["droprate"]),
            HyperLinearBlock(ch_in=config["dim_z"], ch_out=config["dim_z"], ch_hidden=config["channel_hyper"], num_hidden=config["hlayers_hyper"], droprate=config["droprate"], post_prcs=False),
            ])
        self.Encoder_z = nn.Sequential(*modules)
        #======================
        

        #====== Decoder =======
        # HyperLinear( 64, 128), LayerNorm(128), ReLU
        # HyperLinear(128, 256)
        #----------------------
        modules = []
        modules.extend([
            HyperLinearBlock(ch_in=config["dim_z"], ch_out=config["channel_De_z"], ch_hidden=config["channel_hyper"], num_hidden=config["hlayers_hyper"], droprate=config["droprate"]),
            HyperLinearBlock(ch_in=config["channel_De_z"], ch_out=2*1*self.filter_length, ch_hidden=config["channel_hyper"], num_hidden=config["hlayers_hyper"], droprate=config["droprate"], post_prcs=False)
            ])
        self.Decoder_z = nn.Sequential(*modules)
        #======================
    
            
    def forward(self, input, srcpos):
        '''
        args:
            input: B x 4L x S tensor
                   input[*,:,*]=[real_left, imag_left, real_right, imag_right]
            srcpos: B x 3 x S tensor
                    srcpos[*,:,*]=[radius (m), azimuth in [0,2*pi) (rad), zenith in [0,pi] (rad)]
        return:
            returns: (dict) 
                output: B x 2 x L x S complex tensor, output HRTFs
                z: B x dimz=64 x S tensor, latent variables
                idx_mes_pos: (list) indexes of B' measurement positions
        '''
 
        returns = {}
        L = self.filter_length
        B = input.shape[0]
        S = input.shape[-1]
        
        #======== source position =======
        srcpos = srcpos.cuda()
        r,phi,theta = srcpos[:,0,:], srcpos[:,1,:], srcpos[:,2,:]
        vec_cart_gt = sph2cart(phi,theta,r) # x,y,z

        #============ HRTF ==============
        #---- re/im -> mag (dB/20) ------
        mag_l = th.sqrt(th.abs(input[:,0*L:1*L])**2 + 
                        th.abs(input[:,1*L:2*L])**2)
        mag_r = th.sqrt(th.abs(input[:,2*L:3*L])**2 + 
                        th.abs(input[:,3*L:4*L])**2)
        mag2db = ta.transforms.AmplitudeToDB(stype = 'magnitude', top_db = 80) 
        magdb_l = mag2db(th.abs(mag_l))/20 # th.log10(th.max(mag_l, eps))
        magdb_r = mag2db(th.abs(mag_r))/20 # th.log10(th.max(mag_r, eps))
        magdb_lr = th.cat((magdb_l,magdb_r), dim=1)

        #------- standardization --------
        magdb_mean = -1.5852 if self.config["green"] else -0.3187
        magdb_std = 0.5688
        magdb_lr = (magdb_lr - magdb_mean) / magdb_std 

        #===== concat HRTF & srcpos =====
        input = th.cat((magdb_lr, vec_cart_gt), dim=1) # B x (2L+3) x S 
        
        #======== sample B' measurement positions from B=440 positions ========
        if self.config["num_pts"] < min(input.shape[0], (13+1)**2+1):
            idx_mes = aprox_t_des(pts=vec_cart_gt[:,:,0]/1.47, t=round(self.config["num_pts"]**0.5-1), plot=False)
            input_rs = input[idx_mes] # B' x (2L+3) x S 
        else:
            idx_mes = range(0,B)
            input_rs = input # B x (2L+3) x S tensor (B'=B)
        returns["idx_mes_pos"] = idx_mes
        #===============================


        input_rs = input_rs.permute(2,0,1) # S x B x (2L+3) tensor

        #=========== Encoder ===========
        input_rs_x = input_rs[:,:,:-3] # S x B x 3 
        input_rs_z = input_rs[:,:,-3:] # S x B x 2L
        input_rs_dict = {
            "x": input_rs_x,
            "z": input_rs_z,
        }
        z = self.Encoder_z(input_rs_dict)["x"] # S x B x dimz tensor

        #===== Aggregation module ======
        z = F.normalize(z,dim=-1)
        returns['z'] = z.permute(1,2,0) # B x dimz x S
        z = z.permute(1,0,2)            # B x S x dimz
        z = th.mean(z, dim=0)           # S x dimz
        z = F.normalize(z,dim=-1)

        #=========== Decoder ===========
        z = th.tile(z.unsqueeze(1),dims=(1,B,1)) # S x dimz -> S x B x dimz
        input_dec_dict = {
            "x": z,                             # S x B x dimz
            "z": vec_cart_gt.permute(2,0,1)     # S x B x 3
        }
        out_f = self.Decoder_z(input_dec_dict)["x"] # S x B x 2L

        #========= Transform ===========
        # out_f: S x B x 2L float -> out: S x B x 2 x L complex
        out = th.zeros(S,B,2,L, dtype=th.complex64, device=out_f.device)
        out[:,:,0,:] = 10**(out_f[:,:,0*L:1*L] * magdb_std + magdb_mean)
        out[:,:,1,:] = 10**(out_f[:,:,1*L:2*L] * magdb_std + magdb_mean)

        returns['output'] = out.permute(1,2,3,0) # B x 2 x L x S 
        
        return returns
