import torch

import os
from logging import getLogger

from MOTSPEnv import TSPEnv as Env
from MOTSPModel_HV import TSPModelHV as Model

from MOTSProblemDef import get_random_problems, augment_xy_data_by_64_fold_2obj


from einops import rearrange

from utils.utils import *


class TSPTesterHV:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()


        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)
        
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint_motsp-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # HV-calculation
        self.ref = torch.tensor(self.tester_params['reference'], device=device)
        self.HV_const =  torch.pi / 4 # \Phi/(m*2^m) = 2*\pi^(m/2)/(\Gamma(m/2)*m*2^m)

        # utility
        self.time_estimator = TimeEstimator()

    def run(self, n_prefs, shared_problem):
        self.time_estimator.reset()
    
        aug_score_AM = {}

        self.n_prefs = n_prefs

        prefs = torch.zeros(n_prefs)
        for p in range(n_prefs):
            prefs[p] = torch.rand([1]) * torch.pi / 2
        
        Y = torch.zeros(n_prefs, 2, 2) # Store average solutions for implicit and explicit inference
            
        test_num_episode = self.tester_params['test_episodes']
        episode = 0
        
        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            aug_score_imp, aug_score_exp = self._test_one_batch(shared_problem, prefs, batch_size, episode) 
            # (n_prefs, 2)
            
            # Update Y
            Y[:, 0, :] = Y[:, 0, :] + batch_size * aug_score_imp / test_num_episode
            Y[:, 1, :] = Y[:, 1, :] + batch_size * aug_score_exp / test_num_episode

            episode += batch_size

        Y = self._LSSA(Y)
            
        elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
        for p in range(n_prefs):
            self.logger.info("AUG_OBJ_1 SCORE: {:.4f}, AUG_OBJ_2 SCORE: {:.4f} ".format(Y[p, 0], Y[p, 1]))
                
        return Y
                
    def _test_one_batch(self, shared_problem, prefs, batch_size, episode):

        Y_batch_imp = torch.zeros(self.n_prefs, 2) # Store average solutions for implicit inference
        Y_batch_exp = torch.zeros(self.n_prefs, 2) # Store average solutions for explicit inference

        self.proj_dists = None

        for p in range(self.n_prefs):
            pref = prefs[p].unsqueeze(0)
            aug_score_imp, aug_score_exp = self._test_one_batch_and_pref(shared_problem, pref, batch_size, episode) 
            Y_batch_imp[p, :] = aug_score_imp
            Y_batch_exp[p, :] = aug_score_exp

        return Y_batch_imp, Y_batch_exp

    def _test_one_batch_and_pref(self, shared_probelm, pref, batch_size, episode):

        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1
            
        self.env.batch_size = batch_size 
        self.env.problems = shared_probelm[episode: episode + batch_size]
        
        if aug_factor == 64:
            self.env.batch_size = self.env.batch_size * 64
            self.env.problems = augment_xy_data_by_64_fold_2obj(self.env.problems)
            
        self.env.BATCH_IDX = torch.arange(self.env.batch_size)[:, None].expand(self.env.batch_size, self.env.pomo_size)
        self.env.POMO_IDX = torch.arange(self.env.pomo_size)[None, :].expand(self.env.batch_size, self.env.pomo_size)
        
        self.model.eval()
        with torch.no_grad():
            reset_state, _, _ = self.env.reset()
            
            self.model.decoder.assign(pref)
            self.model.pre_forward(reset_state)
            
        state, reward, done = self.env.pre_step()
        
        while not done:
            selected, _ = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)
        
        # reward was negative, here we set it to positive
        reward = - reward
        proj_dist, HV, HV_old = self._calculate_proj_distance_and_hv(pref, reward) 
        # (B, P)

        ehvi = torch.ceil(HV - HV_old) # Current HV - previous HV, (B, P)
                        
        omega = 1
        R_exp = omega * proj_dist
        R_imp = R_exp + ehvi * HV

        R_exp = R_exp.reshape(aug_factor, batch_size, self.env.pomo_size)
        R_imp = R_imp.reshape(aug_factor, batch_size, self.env.pomo_size)
        R_exp_aug = rearrange(R_exp, 'c b h -> b (c h)') 
        R_imp_aug = rearrange(R_imp, 'c b h -> b (c h)')

        _ , max_idx_exp = R_exp_aug.max(dim=1)
        _, max_idx_imp = R_imp_aug.max(dim=1)
        max_idx_exp = max_idx_exp.reshape(max_idx_exp.shape[0],1)
        max_idx_imp = max_idx_imp.reshape(max_idx_imp.shape[0],1)

        reward_obj1_exp = rearrange(reward[:,:,0].reshape(aug_factor, batch_size, self.env.pomo_size), 'c b h -> b (c h)').gather(1, max_idx_exp)
        reward_obj2_exp = rearrange(reward[:,:,1].reshape(aug_factor, batch_size, self.env.pomo_size), 'c b h -> b (c h)').gather(1, max_idx_exp)
        reward_obj1_imp = rearrange(reward[:,:,0].reshape(aug_factor, batch_size, self.env.pomo_size), 'c b h -> b (c h)').gather(1, max_idx_imp)
        reward_obj2_imp = rearrange(reward[:,:,1].reshape(aug_factor, batch_size, self.env.pomo_size), 'c b h -> b (c h)').gather(1, max_idx_imp)
     
        aug_score_imp = torch.zeros(2)
        aug_score_imp[0] = reward_obj1_imp.float().mean()
        aug_score_imp[1] = reward_obj2_imp.float().mean()

        aug_score_exp = torch.zeros(2)
        aug_score_exp[0] = reward_obj1_exp.float().mean()
        aug_score_exp[1] = reward_obj2_exp.float().mean()

        # Update proj_dists
        proj_dists_exp = rearrange(proj_dist.reshape(aug_factor, batch_size, self.env.pomo_size), 'c b h -> b (c h)').gather(1, max_idx_exp)
        proj_dists_imp = rearrange(proj_dist.reshape(aug_factor, batch_size, self.env.pomo_size), 'c b h -> b (c h)').gather(1, max_idx_imp)
        
        proj_dists_exp = proj_dists_exp.unsqueeze(1).expand(self.env.batch_size, self.env.pomo_size).repeat(aug_factor, 1).reshape(aug_factor * self.env.batch_size, self.env.pomo_size)
        proj_dists_imp = proj_dists_imp.unsqueeze(1).expand(self.env.batch_size, self.env.pomo_size).repeat(aug_factor, 1).reshape(aug_factor * self.env.batch_size, self.env.pomo_size)

        if self.proj_dists is None:
            self.proj_dists = torch.cat((proj_dists_exp, proj_dists_imp), dim=2)
        else: 
            self.proj_dists = torch.cat((self.proj_dists, proj_dists_exp, proj_dists_imp), dim=2)

        return aug_score_imp, aug_score_exp
     
      ### HV functionality ###
    def _calculate_proj_distance_and_hv(self, pref, obj_value):
        # obj_value: (B, P, Nobj)

        # Projection distance for current preference
        lambda1 = torch.sin(pref) # Here we assume 2D objective
        lambda2 = torch.cos(pref)
        lambda_tot = torch.cat((lambda1, lambda2)).unsqueeze(0).unsqueeze(1) # (1, 1, Nobj)
        ref_grouped = self.ref.unsqueeze(0).unsqueeze(1) # (1, 1, Nobj)
        G_current = torch.min((ref_grouped - obj_value) / lambda_tot, dim=2).values
        proj_dist = torch.maximum(G_current, torch.zeros_like(G_current)) # (B, P)

        if self.proj_dists is None:
            proj_dists = proj_dist.unsqueeze(2)
            HV_old = 0
        else: 
            proj_dists = torch.cat((self.proj_dists, proj_dist.unsqueeze(2)), dim=2) # Add current distance, (B, P, 2*p'-1)
            HV_old = self.HV_const * torch.mean(torch.pow(self.proj_dists, 2), dim=2)
            HV_old = HV_old / (self.ref[0]*self.ref[1])
        
        HV = self.HV_const * torch.mean(torch.pow(proj_dists, 2), dim=2) # (B, P, p') -> (B, P)
        HV = HV / (self.ref[0]*self.ref[1])

        norm_const = torch.minimum(self.ref[0]/lambda1, self.ref[1]/lambda2)
        proj_dist = proj_dist /  norm_const # Normalize between 0 and 1

        return proj_dist, HV, HV_old

    def _LSSA(Y):
        Y1 = Y[:, :, 0]
        Y2 = Y[:, :, 1]

        Y = torch.cat((Y1, Y2), dim=0)
        return Y