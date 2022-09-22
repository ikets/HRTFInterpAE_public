import numpy as np
import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn
import torchaudio as ta   
import torch.nn.functional as F
import os
import scipy.io
from scipy.spatial import KDTree

class Net(nn.Module):

    def __init__(self, model_name="network", use_cuda=True):
        super().__init__()
        self.use_cuda = use_cuda
        if th.cuda.is_available(): 
            self.use_cuda = True
        self.model_name = model_name

    def save(self, model_dir, suffix=''):
        '''
        save the network to model_dir/model_name.suffix.net
        args
        -----
        model_dir: directory to save the model to

        returns
        -----
        suffix: suffix to append after model name
        '''
        if self.use_cuda:
            self.cpu()

        if suffix == "":
            fname = f"{model_dir}/{self.model_name}.net"
        else:
            fname = f"{model_dir}/{self.model_name}.{suffix}.net"

        th.save(self.state_dict(), fname)
        if self.use_cuda:
            self.cuda()

    def load_from_file(self, model_file):
        '''load network parameters from model_file

        args
        -----
        model_file: file containing the model parameters
        '''
        if self.use_cuda:
            self.cpu()

        states = th.load(model_file)
        self.load_state_dict(states)

        if self.use_cuda:
            self.cuda()
        print(f"Loaded: {model_file}")

    def load(self, model_dir, suffix=''):
        '''
        load network parameters from model_dir/model_name.suffix.net

        args
        -----
        model_dir: directory to load the model from
        suffix: suffix to append after model name
        '''
        if suffix == "":
            fname = f"{model_dir}/{self.model_name}.net"
        else:
            fname = f"{model_dir}/{self.model_name}.{suffix}.net"
        self.load_from_file(fname)

    def num_trainable_parameters(self):
        '''
        returns
        -----
        the number of trainable parameters in the model
        '''
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def plotmaghrtf(srcpos,hrtf_gt,hrtf_est,mode,config,idx_plot_list=np.arange(5)):
    '''plot magnitude frequency response

    args
    -----
    srcpos:   B x 3 tensor
    hrtf_gt:  B x 2 x L tensor, true HRTF
    hrtf_est: B x 2 x L tensor, estimated HRTF
    mode: (str)
    config: (dict)
    idx_plot_list: np.array, indexes of source position for which you want to plot HRTF mag
    '''
    mag2db = ta.transforms.AmplitudeToDB(stype = 'magnitude')     
    f_bin = np.linspace(0,config["max_frequency"],round(config["fft_length"]/2)+1)[1:]
    plt.figure(figsize=(12,round(6*len(idx_plot_list))))
    plt.subplots_adjust(wspace=0.4, hspace=0.6)

    for itr, idx_plot in enumerate(idx_plot_list):
        plt.subplot(len(idx_plot_list),1, itr+1)

        plt.plot(f_bin, mag2db(th.abs(hrtf_gt[idx_plot,0,:])).to('cpu').detach().numpy().copy(), label="Left (Ground Truth)", color='b',linestyle=':')
        plt.plot(f_bin, mag2db(th.abs(hrtf_gt[idx_plot,1,:])).to('cpu').detach().numpy().copy(), label="Right (Ground Truth)", color='r',linestyle=':')
        plt.plot(f_bin, mag2db(th.abs(hrtf_est[idx_plot,0,:])).to('cpu').detach().numpy().copy(), label="Left (Estimated)", color='b')
        plt.plot(f_bin, mag2db(th.abs(hrtf_est[idx_plot,1,:])).to('cpu').detach().numpy().copy(), label="Right (Estimated)", color='r')
        plt.grid()
        plt.legend()
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude [dB]")
        ylim = [-80,0] if config["green"] else [-50,30]
        plt.ylim(ylim)
        plt.xlim([0,config["max_frequency"]])
        srcpos_np = srcpos.to("cpu").detach().numpy().copy()
        # print(srcpos_np.shape)
        plt.title(f"HRTF (radius={srcpos_np[idx_plot,0]:.2f} m, azimuth={srcpos_np[idx_plot,1]/np.pi*180:.1f} deg, zenith={srcpos_np[idx_plot,2]/np.pi*180:.1f} deg)")

    figure_dir = config["artifacts_dir"] + "/figure/HRTF/"
    os.makedirs(figure_dir, exist_ok=True)
    plt.savefig(figure_dir + "HRTF_mag_"+mode+".png", dpi=300)
    plt.close()

def posTF2IR_dim4(tf):
    '''Transfer function (positive freq. bins) -> Impulse response
    args
    -----
    tf: S x B x 2 x L complex tensor

    returns
    -----
    ir: S x B x 2 x 2L float tensor
    '''
    zeros = th.zeros([tf.shape[0],tf.shape[1],tf.shape[2],1]).to(tf.device).to(tf.dtype)
    tf_fc = th.conj(th.flip(tf[:,:,:,:-1],dims=(-1,)))

    tf = th.cat((zeros,tf,tf_fc), dim=-1) # DC, positive freq. bins, negative freq. bins
    ir = th.fft.ifft(th.conj(tf), dim=-1)
    ir = th.real(ir)

    return ir

def plothrir(srcpos,mode,config,idx_plot_list=np.arange(5), hrir_gt=None, hrir_est=None):
    '''plot HRIRs

    args
    -----
    srcpos:   B x 3 tensor
    mode: (str)
    config: (dict)
    idx_plot_list: np.array, indexes of source position for which you want to plot HRIRs
    hrir_gt:  B x 2 x 2L tensor, true HRIR
    hrir_est: B x 2 x 2L tensor, estimated HRIR
    '''
    t_bin = np.linspace(0, config["fft_length"]/(2*config["max_frequency"]), config["fft_length"])
    plt.figure(figsize=(12,round(6*len(idx_plot_list))))
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    for itr, idx_plot in enumerate(idx_plot_list):
        plt.subplot(len(idx_plot_list),1, itr+1)

        plt.plot(t_bin, hrir_gt[idx_plot,0,:].to('cpu').detach().numpy().copy(), label="Left (Ground Truth)", color='b',linestyle=':')
        plt.plot(t_bin, hrir_gt[idx_plot,1,:].to('cpu').detach().numpy().copy(), label="Right (Ground Truth)", color='r',linestyle=':')
        plt.plot(t_bin, hrir_est[idx_plot,0,:].to('cpu').detach().numpy().copy(), label="Left (Estimated)", color='b')
        plt.plot(t_bin, hrir_est[idx_plot,1,:].to('cpu').detach().numpy().copy(), label="Right (Estimated)", color='r')
        plt.grid()
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Magnitude")
        plt.xlim([t_bin[0], t_bin[-1]])
        srcpos_np = srcpos.to("cpu").detach().numpy().copy()
        plt.title(f"HRIR (radius={srcpos_np[idx_plot,0]:.2f} m, azimuth={srcpos_np[idx_plot,1]/np.pi*180:.1f} deg, zenith={srcpos_np[idx_plot,2]/np.pi*180:.1f} deg)")

    figure_dir = config["artifacts_dir"] + "/figure/HRIR/"
    os.makedirs(figure_dir, exist_ok=True)
    plt.savefig(figure_dir + "HRIR_"+mode+".png", dpi=300)
    plt.close()

def sph2cart(phi, theta, r):
    """Conversion from spherical to Cartesian coordinates

    Parameters
    ------
    phi, theta, r: Azimuth angle, zenith angle, distance

    Returns
    ------
    x, y, z : Position in Cartesian coordinates
    """
    x = r * th.sin(theta) * th.cos(phi)
    y = r * th.sin(theta) * th.sin(phi)
    z = r * th.cos(theta)
    return th.hstack((x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1)))

def aprox_t_des(pts, t, plot=False):
    '''From a given set of points, sample points closest to each point in the spherical t-design
    
    args
    -----
    pts: B x 3 tensor, pts[*,:]=[x,y,z], x**2+y**2+z**2=1
    t: (int) t for spherical t-design
    plot: (bool)

    returns
    -----
    idx: (list) indexes for (t+1)**2 sampled points
    '''
    t_grid_dic = scipy.io.loadmat(f't_des/grid_t{t}d{(t+1)**2}.mat') 
    t_grid = th.tensor(t_grid_dic["Y"]).T

    kdt = KDTree(pts.cpu())
    _, idx = kdt.query(t_grid)
    idx = sorted(idx)
    idx_prev = idx.copy()
    idx = list(set(idx)) # remove duplicated indexes
    if len(idx_prev) > len(idx):
        print(f"[aprox_t_des] detected duplication. {len(idx_prev)}->{len(idx)} pts")
        
    t_grid_aprox = pts[idx]

    #=========================
    if plot:
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(t_grid[:,0], t_grid[:,1], t_grid[:,2],label='t-design',marker='+')
        ax.scatter(t_grid_aprox[:,0], t_grid_aprox[:,1], t_grid_aprox[:,2], label='nearest',marker='x')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f'Spherical t-design with d=(t+1)^2 points (t={t})') 
        ax.legend(loc=2, title='legend')

        # Make data
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))

        # Plot the surface of sphere
        ax.plot_surface(x, y, z,color="gray",rcount=100, ccount=100, antialiased=False, alpha=0.05)
        ax.set_box_aspect((1,1,1))
    #=========================
    return idx

def plotz(ptpath_est,sub,clim = [-10,10],flag_save=False,fname=None):
    '''
    args
    -----
    ptpath_est: (str) path to latent variables (*.pt)
    sub: (itr) idx of sub
    clim: (list) [min_value, max_value]
    flag_save: (bool)
    fname: (str) name of fig to be saved
    '''
    # print("Load z from " + ptpath_est)
    z_est = th.load(ptpath_est)
    # print(z_est.shape)

    plt.figure(figsize=(30,10))
    fig_num = 3
    fig_itr = 1
    cmap = 'bwr'#'RdBu'
    aspect = z_est.shape[1] / 100.0

    # sub_1, 
    plt.subplot(1,fig_num,fig_itr)
    fig_itr += 1
    plt.imshow(z_est[:,:,sub],cmap=cmap,interpolation='none',aspect=aspect) # B,dim_z,sub
    plt.colorbar()
    plt.clim(clim)
    plt.xlim()
    plt.title(f"z, Sub={sub+1}")

    # sub_2, 
    plt.subplot(1,fig_num,fig_itr)
    fig_itr += 1
    plt.imshow(z_est[:,:,sub+1],cmap=cmap,interpolation='none',aspect=aspect) # B,dim_z,sub
    plt.colorbar()
    plt.clim(clim)
    plt.xlim()
    plt.title(f"z, Sub={sub+2}")

    plt.subplot(1,fig_num,fig_itr)
    fig_itr += 1
     # print(coeff_est.shape)
    plt.imshow(th.var(z_est,dim=-1),cmap='Reds',interpolation='none',aspect=aspect) # B,dim_z,sub
    plt.colorbar()
    plt.clim([0,clim[1]/10])
    plt.title(f"Variance of z (dim=-1:sub)")

    if flag_save:
        plt.savefig(fname)

     # plt.show()
    plt.clf()
    plt.close()

def plotcz(dir):
    '''plot latent variables
    '''
    dirname = f'{dir}/'
    ptpath_z = dirname + 'z_train.pt'
    fname = dirname + 'figure/z_train.jpg'
    clim = [-0.5,0.5]
    plotz(ptpath_z,sub=0,clim=clim,flag_save=True,fname=fname)

def hrir2itd(hrir,fs,f_us=8*48000,thrsh_ms=1000,lpf=True,upsample_via_cpu=True):
    '''calculate ITD from HRIR

    args
    -----
        hrir: (S,B,2,L) tensor
        fs: (int) sampling freq. of original hrir
        f_us: (int) sampling freq. after upsampling
        thrsh_ms: (float) threshold [ms]. (computed ITD is forced to be in [-thrsh_ms, +thrsh_ms] )
        lpf: (bool) If True, Low-pass filter is filtered to hrir.
    returns
    -----
        ITD: (S,B) tensor, interaural time difference [s] (-:src@left, +:src@right)
    '''
    if lpf:
        hrir = ta.functional.lowpass_biquad(waveform=hrir, sample_rate=fs, cutoff_freq=1600)
    else:
        pass
    if upsample_via_cpu:
        hrir = hrir.cpu()
    upsampler = ta.transforms.Resample(fs,f_us)
    hrir_us = upsampler(hrir.contiguous())
    if upsample_via_cpu:
        hrir_us = hrir_us.cuda()
    S, B, _, L = hrir_us.shape
    thrsh_idx = round(f_us/thrsh_ms)
    #===============================
    HRIR_l = hrir_us[:,:,0,:]
    HRIR_r = hrir_us[:,:,1,:]
    HRIR_l_pad = F.pad(HRIR_l,(L,L)) # torch.Size([77, 440, 384])
    HRIR_l_pad_in = HRIR_l_pad.reshape(1, S*B, -1)
    HRIR_r_wt = HRIR_r.reshape(S*B, 1, -1)
    crs_cor = F.conv1d(HRIR_l_pad_in, HRIR_r_wt, groups=S*B)
    crs_cor = crs_cor.reshape(S, B, -1)
    idx_beg = L - thrsh_idx
    idx_end = L + thrsh_idx + 1
    idx_max = th.argmax(crs_cor[:,:,idx_beg:idx_end], dim=-1) - thrsh_idx
    ITD = idx_max/f_us

    return ITD

def HilbertTransform(data, detach=False):
    '''perform Hilbert transformation
    '''
    # https://stackoverflow.com/questions/50902981/hilbert-transform-using-cuda
    assert data.dim()==4
    N = data.shape[-1]
    # Allocates memory on GPU with size/dimensions of signal
    if detach:
        transforms = data.clone().detach()
    else:
        transforms = data.clone()
    transforms = th.fft.fft(transforms, axis=-1)
    transforms[:,:,:,1:N//2]      *= -1j      # positive frequency
    transforms[:,:,:,(N+2)//2 + 1: N] *= +1j  # negative frequency
    transforms[:,:,:,0] = 0; # DC signal
    if N % 2 == 0:
        transforms[:,:,:,N//2] = 0; # the (-1)**n term
    # Do IFFT on GPU: in place (same memory)
    return th.fft.ifft(transforms, axis=-1)

def minphase_recon(tf, contain_neg_fbin=False):
    '''minimum phase reconstruction

    args
    -----
        tf: (S,B,2,L) or (S,B,2,2L) tensor (L: # of freq. bins)
        contain_neg_fbin: bool. If True, mag.shape == (S,B,2,2L)
        conj: bool. 
    return
    -----
        phase_min: (S,B,2,2L) tensor
        ir_min:  (S,B,2,2L) tensor. Impulse response with minimum phase.
    '''
    if contain_neg_fbin:
        tf_pm = tf
    else:
        tf_nf = th.conj(th.flip(tf[:,:,:,:-1],dims=(-1,))) # negatibe freq.
        tf_pm = th.cat((th.ones_like(tf)[:,:,:,0:1], tf, tf_nf), dim=-1) # [1,pos,neg]
        # DC 成分が 0 でなく 1 なのは，対数を取ったときに-infに発散するのを防ぐため←これでOK?
    mag_pm_log = th.log(th.abs(tf_pm)) # magnitude の対数振幅（底: e）
    phase_min =  - HilbertTransform(mag_pm_log)
    ir_min = th.real(th.fft.ifft(th.abs(tf_pm)*th.exp(1j * phase_min), axis=-1))

    return phase_min, ir_min

def assign_itd(hrir_ori,itd_ori,itd_des,fs,shift_s=1e-3):
    '''assign ITD

    args
    -----
        hrir_ori: (S,B,2,L) tensor. (L: filter length)
        itd_ori: (S,B) tensor. ITD [s] of hrir_ori
        itd_des: (S,B) tensor. desired ITD [s]
        fs: (int) [Hz]. Sampling Frequency.
        shift_lr: (float) [s]. offset when ITD==0.
    return
    -----
        ir_itd_des:  (S,B,2,L) tensor. Impulse response with desired ITD.
    '''
    S, B = itd_ori.shape
    L = hrir_ori.shape[-1]
    shift_idx = shift_s * fs
    ITD_idx_fs_half = (itd_des-itd_ori) * fs / 2
    offset = th.ones(S,B,2).to(ITD_idx_fs_half.device) * shift_idx
    offset[:,:,0] += ITD_idx_fs_half # left
    offset[:,:,1] -= ITD_idx_fs_half # right
    offset = th.round(offset).to(int)

    arange = th.arange(L).reshape(1,1,1,L).tile(S,B,2,1).to(ITD_idx_fs_half.device)
    arange = (arange - offset[:,:,:,None]) % L

    # square window to remove pre-echo
    window_length = int(L - shift_idx)
    window_sq = th.cat((th.ones(window_length), th.zeros(L-window_length))).to(hrir_ori.device)
    hrir_ori_w = hrir_ori * window_sq[None,None,None,:]
    ir_itd_des = th.gather(hrir_ori_w, -1, arange)

    return ir_itd_des

def vhlines(ax, linestyle='-', color='gray', zorder=1, alpha=0.8, lw=0.75):
    ax.axhline(y=np.pi/2, linestyle=linestyle, color=color,zorder=zorder, alpha=alpha, lw=lw)
    ax.axvline(x=np.pi/2, linestyle=linestyle, color=color,zorder=zorder, alpha=alpha, lw=lw)
    ax.axvline(x=np.pi, linestyle=linestyle, color=color,zorder=zorder, alpha=alpha, lw=lw)
    ax.axvline(x=np.pi*3/2, linestyle=linestyle, color=color,zorder=zorder, alpha=alpha, lw=lw)
    ax.text(np.pi/2, np.pi+0.05, "Left", ha='center')
    ax.text(np.pi*3/2, np.pi+0.05, "Right", ha='center')
    ax.text(np.pi, np.pi+0.05, "Back", ha='center')

def plotazimzeni(pos,c,fname,title,cblabel,cmap='gist_heat',figsize=(10.5,5),dpi=300, emphasize_mes_pos=False, idx_mes_pos=None, vmin=None, vmax=None, save=True, clf=True):
    '''plot values on azimuth-zenith plane

    args
    -----
        pos: (B,*>3) tensor. (:,1):azimuth, (:,2):zenith
        c: (B) tensor.
        fname: str. filename
        title: str. title.
        cblabel: str. label of colorbar.
        cmap: colormap.
        figsie: (*,*) tuple.
        dpi: scalar.
    '''
    fig, ax = plt.subplots(figsize=figsize)
    vhlines(ax)
    if vmin==None:
        vmin=th.min(c)
    if vmax == None:
        vmax=th.max(c)
    mappable = ax.scatter(pos[:,1], pos[:,2], c=c, cmap=cmap, s=60, lw=0.3, ec="gray", zorder=2, vmin=vmin, vmax=vmax)
    fig.colorbar(mappable=mappable,label=cblabel)
    if emphasize_mes_pos:
        ax.scatter(pos[idx_mes_pos,1], pos[idx_mes_pos,2], s=120, lw=0.5, c="None", marker="o", ec="k", zorder=1)
    ds = 0.1
    xlim = [0-ds, 2*np.pi]
    ylim = [0-ds, ds+np.pi]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.invert_yaxis()
    ax.set_xlabel('Azimuth (rad)')
    ax.set_ylabel('Zenith (rad)')
    ax.set_title(title)
    if save:
        fig.savefig(f'{fname}.png', dpi=dpi)

    if clf:
        fig.clf()
        plt.close()