import os
import argparse
import datetime
import time
import pprint

import numpy as np
import torch as th
import matplotlib.pyplot as plt   

from src.dataset import HRTFDataset
from src.models import HRTF_Interp_AE
from src.trainer import Trainer
from src.losses import LSD, LSD_before_mean, CosDistIntra
from src.configs import *
from src.utils import plotmaghrtf, plothrir, plotcz, hrir2itd, minphase_recon, assign_itd, posTF2IR_dim4, plotazimzeni

from torch.utils.tensorboard import SummaryWriter # tensorboard

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-datadir","--dataset_directory",
                        type=str,
                        default='../HUTUBS/HRIRs',
                        help="path to the train data")
    parser.add_argument("-a","--artifacts_directory",
                        type=str,
                        default="",
                        help="directory to write model files to")
    parser.add_argument("-c","--num_config",
                        type=int,
                        default=0,
                        help="idx of config you use")
    parser.add_argument("-lf", "--load_from",
                    type=str,
                    default="",
                    help="model file containing the trained network weights")
    parser.add_argument("-test", "--test", action="store_true", help="")
    args = parser.parse_args()
    return args

def train(trainer):
    BEST_LOSS = 1e+16
    LAST_SAVED = -1
    for epoch in range(1, 1+config["epochs"]):
        trainer.net.cuda()
        trainer.net.train()

        #========== Train ==========
        loss_train, z_train = trainer.train_1ep(epoch) 

        #-- logging(TensorBoard) ---
        for k, v in loss_train.items():
            writer.add_scalar(k + " / train", v, epoch)

        #--- save latent vector ----
        ptpath_z = config["artifacts_dir"] + "/z_train.pt"
        th.save(z_train.cpu(), ptpath_z)
        #===========================

        #======= Validation ========
        print("----------")
        print(f"epoch {epoch} (valid)")
        t_start = time.time()
        use_cuda = True
        
        loss_valid = test(trainer.net, False, BEST_LOSS, validdataset, 'valid', use_cuda=use_cuda)
        t_end = time.time()
        time_str = f"({time.strftime('%H:%M:%S', time.gmtime(t_end-t_start))})"
        print(time_str)
        #-- Logging(TensorBoard) ---
        for k, v in loss_valid.items():
            writer.add_scalar(k + " / valid", v, epoch)
        #===========================

        if loss_valid["loss"] < BEST_LOSS:
            BEST_LOSS = loss_valid["loss"]
            LAST_SAVED = epoch
            print("Best Loss. Saving model!")
            trainer.save(suffix="best")
        elif config["save_frequency"] > 0 and epoch % config["save_frequency"] == 0:
            print("Saving model!")
            trainer.save(suffix="log_"+f'{epoch:03}'+"ep")
        else:
            print("Not saving model! Last saved: {}".format(LAST_SAVED))
        print("---------------------")
        #--- Plot latent variables ---
        plotcz(dir=config["artifacts_dir"])
        plt.clf()
        plt.close()

    #--- Save final model ----
    trainer.save(suffix="final_"+f'{epoch:03}'+"ep")

def test(net, flg_save, best_loss, data, mode, use_cuda=True):
    device = 'cuda' if use_cuda else 'cpu'
    net.to(device).eval()

    srcpos, hrtf_gt = data
    srcpos, hrtf_gt = srcpos.to(device), hrtf_gt.to(device)

    returns = {}
    lsd_loss = LSD()
    cosdist_loss = CosDistIntra()
    lsd_bm_loss = LSD_before_mean()

    keys = ["lsd", "cdintra_z"]
    for k in keys:
        returns.setdefault(k,0)
    
    subject_list = np.arange(0,srcpos.shape[-1])
    for sub_id in subject_list:
        hrtf_sub = hrtf_gt[:,:,:,sub_id:sub_id+1]
        input = th.cat((th.real(hrtf_sub[:,0,:]), th.imag(hrtf_sub[:,0,:]), th.real(hrtf_sub[:,1,:]), th.imag(hrtf_sub[:,1,:])), dim=1)
        srcpos_sub = srcpos[:,:,sub_id:sub_id+1]

        prediction = net.forward(input=input, srcpos=srcpos_sub)
        hrtf_est = prediction["output"].detach().clone()
        z_est = prediction["z"].detach().clone()

        returns["lsd"] += lsd_loss(hrtf_est, hrtf_sub)
        returns["cdintra_z"] += cosdist_loss(z_est)

        if flg_save:
            #===== Plot HRTF magnitude =====
            plotmaghrtf(srcpos=srcpos[:,:,sub_id],hrtf_gt=hrtf_gt[:,:,:,sub_id],hrtf_est=hrtf_est,mode=f'{mode}_sub-{sub_id+1}',idx_plot_list=np.array([202,211,220,229]), config=config)

            #======== Obtain HRIRs =========
            fs = config['max_frequency']*2 # sampling frequency
            f_us = config['fs_upsampling'] # frequency to calc ITD
            
            #------- Obtain true ITDs ------
            hrtf_sub = hrtf_sub.permute(3,0,1,2) # B x 2 x L x S -> S x B x 2 x L
            hrir_sub = posTF2IR_dim4(hrtf_sub)   # S x B x 2 x 2L
            if config["use_itd_gt"]: # Use true ITDs
                itd_des = hrir2itd(hrir=hrir_sub, fs=fs, f_us=f_us)
            else:
                raise NotImplementedError()

            #-- minimum phase reconstruction ---
            hrtf_est = hrtf_est.permute(3,0,1,2) # B x 2 x L x S -> S x B x 2 x L
            _, hrir_min = minphase_recon(tf=hrtf_est) # S x B x 2 x 2L
            #--------- Assign ITD ----------
            itd_ori = hrir2itd(hrir=hrir_min, fs=fs, f_us=f_us) # original ITD
            hrir_min_itd = assign_itd(hrir_ori=hrir_min, itd_ori=itd_ori, itd_des=itd_des, fs=fs) # HRIR with minimum phase & true ITD

            hrir_est = hrir_min_itd.permute(1,2,3,0) # (B,2,L,S)
            hrir_sub = hrir_sub.permute(1,2,3,0) # (B,2,L,S)
            hrtf_est = hrtf_est.permute(1,2,3,0) # (B,2,L,S)
            hrtf_sub = hrtf_sub.permute(1,2,3,0) # (B,2,L,S)
            
            #========= Plot HRIRs ==========
            plothrir(srcpos=srcpos[:,:,sub_id],mode=f'{mode}_sub-{sub_id+1}',idx_plot_list=np.array([202,211,220,229]), config=config, hrir_gt=hrir_sub.squeeze(), hrir_est=hrir_est)
            if sub_id == 0:
                hrir_gt_all = th.zeros(hrir_est.shape[0],hrir_est.shape[1],hrir_est.shape[2],len(subject_list))
                hrir_est_all = th.zeros(hrir_est.shape[0],hrir_est.shape[1],hrir_est.shape[2],len(subject_list))
                lsd_bm_all = th.zeros(hrir_est.shape[0],hrir_est.shape[1],len(subject_list)) # B,2,S
            hrir_gt_all[:,:,:,sub_id] = hrir_sub[:,:,:,0]
            hrir_est_all[:,:,:,sub_id] = hrir_est[:,:,:,0]

            #===== Plot LSDs in azimuth-zenith plane =====
            fig_dir_lsd_bm = f'{config["artifacts_dir"]}/figure/LSD/'
            os.makedirs(fig_dir_lsd_bm, exist_ok=True)
            lsd_bm = lsd_bm_loss(hrtf_est, hrtf_sub).squeeze().cpu() # (B,2)

            emphasize_mes_pos = True if "idx_mes_pos" in prediction else False
                
            for ch in range(2):
                ch_str_l = ['left', 'right']
                ch_str = ch_str_l[ch]
                plotazimzeni(pos=srcpos[:,:,sub_id].cpu(),c=lsd_bm[:,ch],fname=f'{fig_dir_lsd_bm}lsd_{mode}_sub-{sub_id+1}_{ch_str}',title=f'sub:{sub_id+1}, {ch_str}',cblabel=f'LSD (dB)',cmap='gist_heat',figsize=(10.5,5),dpi=300,emphasize_mes_pos=emphasize_mes_pos, idx_mes_pos=prediction["idx_mes_pos"],vmin=0, vmax=10)
            #===========================================

            #======== store LSD (before mean) ==========
            lsd_bm_all[:,:,sub_id] = lsd_bm

            #======== save HRIR, LSD (before mean) =====
            if sub_id == subject_list[-1]:
                hrir_dir = f'{config["artifacts_dir"]}/HRIR'
                lsd_bm_dir = f'{config["artifacts_dir"]}/LSD'
                os.makedirs(hrir_dir, exist_ok=True)
                os.makedirs(lsd_bm_dir, exist_ok=True)

                th.save(hrir_est_all, f'{hrir_dir}/HRIR_est_{mode}.pt') 
                th.save(hrir_gt_all, f'{hrir_dir}/HRIR_gt.pt') 
                th.save(lsd_bm_all, f'{lsd_bm_dir}/LSD_{mode}_before_mean.pt')

    loss = returns["lsd"]
    returns["loss"] = loss.detach().clone()
        
    for k in returns:
        returns[k] /= len(subject_list)
    
    loss_str = "    ".join([f"{k}:{returns[k]:.4}" for k in sorted(returns.keys())])
    print(loss_str)

    
    if returns["loss"] < best_loss:
        flg_save = True
        print("Best Loss.")
    if flg_save:
        print("Saving figures...")
        # Visualise
        plotmaghrtf(srcpos=srcpos[:,:,-1],hrtf_gt=hrtf_gt[:,:,:,-1],hrtf_est=hrtf_est,mode=mode,idx_plot_list=np.array([202,211,220,229]), config=config)

    return returns

if __name__ == "__main__":
    #========== set random seed ===========
    seed = 0
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)

    #========== time stamp =================
    dt_now = datetime.datetime.now() + datetime.timedelta(hours=9)
    datestamp = dt_now.strftime('_%Y%m%d')
    timestamp = dt_now.strftime('_%m%d_%H%M')

    #========== args =================
    args = arg_parse()
    print("=====================")
    print("args: ")
    print(args)

    #========== load config from src/configs.py =================
    config = {}
    config_name = f"config_{args.num_config}"
    config.update(eval(config_name))

    print("=====================")
    print("config: ")
    pprint.pprint(config)
    print("=====================")
    
    #========== init model ===========
    net = HRTF_Interp_AE(config=config)

    if args.load_from != '': # load model from file
        net.load_from_file(args.load_from)
    # print(net)
    # print("=====================")

    #========== make dir contains model ===========
    if args.artifacts_directory == "":
        config["artifacts_dir"] = "outputs/"+ datestamp + '_' + config["model"]
    else:
        config["artifacts_dir"] = args.artifacts_directory
    print("artifacts_dir: " + config["artifacts_dir"])

    os.makedirs(config["artifacts_dir"], exist_ok=True)
    print("---------------------")

    #========== load dataset ===========
    dataset = HRTFDataset(config=config, sofa_path=args.dataset_directory)
    # traindataset_all = dataset.trainitem()
    validdataset = dataset.validitem()
    testdataset = dataset.testitem()
    
    #========== train model ===========
    if not args.test:
        print("=====================")
        print(net)
        print("=====================")
        
        print("Train model.")
        print(f"number of trainable parameters: {net.num_trainable_parameters()}")
        print("---------------------")
        
        trainer = Trainer(config, net, dataset)
        #---- TensorBoard -----
        log_dir=config["artifacts_dir"]+"/logs/"+config["model"]+timestamp
        writer = SummaryWriter(log_dir)
        print("logdir: "+log_dir) 
        print("---------------------")

        train(trainer=trainer)

    #====== test model =======
    print("Test model.")
    print("=====================")
    if args.test and args.load_from != '':
        net.load_from_file(args.load_from)
    else:
        net.load_from_file(config["artifacts_dir"]+'/hrtf_interp_ae.best.net')
    print(net)
    print("=====================")

    #--------- Estimate from B' = 9, 16,..., 196 pts (spherical t-design) ---------
    t_list = range(2,13+1) # t = 2, ..., 13

    loss_test = np.zeros((len(t_list),1))
    for t in t_list:
        config["num_pts"] = round((t+1)**2)
        print("---------------")
        print(f'{config["num_pts"]} pts (spherical t-design, t={t})')
        print("- - - - - - - -")
        
        loss = test(net=net, flg_save=True, best_loss=-1, data=testdataset, mode=f'test_t-{round(config["num_pts"]**0.5-1)}') 
        loss_test[t-2,0] = loss["lsd"]
    th.save(loss_test,config["artifacts_dir"] + f"/loss_test_t-{t_list[0]}-{t_list[-1]}.pt")

    #--------- Estimate from 440 pts (all) ----------
    print("---------------")
    print(f'440 pts (all)')
    print("- - - - - - - -")
    loss_test_all = np.zeros((1,1))
    config["num_pts"] = 440
    loss = test(net=net, flg_save=True, best_loss=-1, data=testdataset, mode=f'test_440pts')
    loss_test_all[0,0] = loss["lsd"]
    th.save(loss_test_all,config["artifacts_dir"] + f"/loss_test_all.pt")

    print("Finished!")