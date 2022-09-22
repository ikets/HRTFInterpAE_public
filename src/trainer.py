import time 
import numpy as np
import torch as th
from torch.utils.data import DataLoader
from src.utils import plotmaghrtf, plothrir, hrir2itd, minphase_recon, assign_itd, posTF2IR_dim4
from src.losses import LSD, CosDistIntra


class Trainer:
    def __init__(self, config, net, dataset):
        '''
        args
        -----
        config: a dict containing parameters
        net: the network to be trained, must be of type src.utils.Net
        dataset: the dataset to be trained on
        '''
        self.config = config
        self.dataset = dataset
        gpus = [i for i in range(config["num_gpus"])]
        self.dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, num_workers=1)
        self.net = th.nn.DataParallel(net, gpus)
        weights = filter(lambda x: x.requires_grad, net.parameters())
        self.optimizer = th.optim.Adam(weights, lr=config["learning_rate"], eps=1e-8)

        self.lsd_loss = LSD()
        self.cosdist = CosDistIntra()
        
        self.total_iters = 0
        self.net.train()


    def save(self, suffix=""):
        self.net.module.save(self.config["artifacts_dir"], suffix)

    def train_1ep(self, epoch):
        '''1 training epoch
        '''
        t_start = time.time()
        loss_stats = {}
        num_data = np.ceil(min(
            self.config["num_sub"]/self.config["batch_size"], len(self.dataloader)
            ))
        if epoch == 1:
            print(f'num_sub: {self.config["num_sub"]}, batch_size: {self.config["batch_size"]}, num_batch:{num_data}')
        
        for itr, data in enumerate(self.dataloader):
            if itr > num_data-1:
                break

            bs = self.config["batch_size"]
            slice = range(itr*bs, itr*bs + min(bs, data[0].shape[0]))
            label = th.tensor(slice).cuda()
            loss_new = self.train_iteration(data, sub=label, itr=itr)

            # logging
            for k, v in loss_new.items():
                if k == 'z':
                    if itr == 0:
                        z_dim_last = 77
                        z_train = th.zeros((loss_new["z"].shape[0], loss_new["z"].shape[1],  z_dim_last),dtype=th.float32).to(loss_new["z"].device)
                    z_train[:,:,slice] = loss_new["z"] 
                else:
                    loss_stats[k] = loss_stats[k]+v if k in loss_stats else v
            #===== progress bar ======
            prog_step = 20
            if round(num_data/prog_step) != 0:
                if itr == 0:
                    print('[',end='')
                elif itr == num_data-1:
                    print('#]')
                elif itr % round(num_data/prog_step) == 0:
                    print('#',end='')
            #=========================

        for k in loss_stats:
            loss_stats[k] /= num_data
        t_end = time.time()
        
        loss_str = "    ".join([f"{k}:{loss_stats[k]:.4}" for k in sorted(loss_stats.keys())])
        time_str = f"({time.strftime('%H:%M:%S', time.gmtime(t_end-t_start))})"
        print(f"epoch {epoch} (train) ")
        print(loss_str + "        " + time_str)

        return loss_stats, z_train

    def train_iteration(self, data, sub, itr=1):
        '''
        one optimization step
        
        args
        -----
        data: tuple of tensors, [source position, hrtf]
        sub: (list) indexes of subjects
        itr: (int) 

        returns
        -----

        '''
        returns = {}
        
        #=== forward ====
        self.optimizer.zero_grad()

        srcpos, hrtf_gt = data
        srcpos = srcpos.cuda().permute(1,2,0)
        hrtf_gt = hrtf_gt.cuda().permute(1,2,3,0)

        hrtf_gt_l = hrtf_gt[:,0,:,:]
        hrtf_gt_r = hrtf_gt[:,1,:,:]

        input = th.cat((th.real(hrtf_gt_l), th.imag(hrtf_gt_l), th.real(hrtf_gt_r), th.imag(hrtf_gt_r)), dim=1)
        prediction = self.net.forward(input=input, srcpos=srcpos) # dict key: "output", "z", "idx_mes_pos"

        hrtf_est = prediction["output"]
        z_est = prediction["z"]
        #==================

        loss_dict = {}
        loss_dict["cdintra_z"] = self.cosdist(z_est)
        loss_dict["lsd"] = self.lsd_loss(hrtf_est, hrtf_gt)
        
        #=== plot HRTFs / HRIRs ====
        if itr==0:
            for sub_id in range(min(3,len(sub))): # sub 1--3
                # plot HRTFs
                plotmaghrtf(srcpos=srcpos[:,:,sub_id],hrtf_gt=hrtf_gt[:,:,:,sub_id],hrtf_est=hrtf_est[:,:,:,sub_id],mode=f'train_sub-{sub_id+1}', idx_plot_list=np.array([202,211,220,229]),config=self.config)

                # plot HRIRs
                fs = self.config['max_frequency']*2
                f_us = self.config['fs_upsampling']
                
                hrtf_est_s = hrtf_est.permute(3,0,1,2) # (B,2,L,S) -> (S,B,2,L)
                hrtf_gt_s = hrtf_gt.permute(3,0,1,2) # (B,2,L,S) -> (S,B,2,L)

                hrir_gt_s = posTF2IR_dim4(hrtf_gt_s)
                if self.config["use_itd_gt"]: # use true ITD
                    itd_des = hrir2itd(hrir=hrir_gt_s, fs=fs, f_us=f_us)
                else:
                    raise NotImplementedError()
                _, hrir_min = minphase_recon(tf=hrtf_est_s)
                itd_ori = hrir2itd(hrir=hrir_min, fs=fs, f_us=f_us)
                hrir_min_itd = assign_itd(hrir_ori=hrir_min, itd_ori=itd_ori, itd_des=itd_des, fs=fs)
                hrir_est_s = hrir_min_itd.permute(1,2,3,0)
                hrir_gt_s = hrir_gt_s.permute(1,2,3,0)

                plothrir(srcpos=srcpos[:,:,sub_id],mode=f'train_sub-{sub_id+1}',idx_plot_list=np.array([202,211,220,229]), config=self.config, hrir_gt=hrir_gt_s[:,:,:,sub_id], hrir_est=hrir_est_s[:,:,:,sub_id])

        returns["z"] = prediction["z"].detach().clone()
        
        loss = 0
        for k,v in loss_dict.items():
            if k in self.config["loss_weights"]:
                if self.config["loss_weights"][k] > 0:
                    loss += self.config["loss_weights"][k] * v
            returns[k] = v.detach().clone()

        returns["loss"] = loss.detach().clone()

        # update model parameters
        loss.backward() 
        self.optimizer.step()
        self.total_iters += 1
        
        return returns