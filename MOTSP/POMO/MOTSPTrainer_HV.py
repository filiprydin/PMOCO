import torch
from logging import getLogger

from MOTSPEnv import TSPEnv as Env
from MOTSPModel_HV import TSPModelHV as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils.utils import *

class TSPTrainerHV:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # HV-calculation
        self.ref = torch.tensor(self.trainer_params['reference'], device=device)
        self.HV_const =  torch.pi / 4 # \Phi/(m*2^m) = 2*\pi^(m/2)/(\Gamma(m/2)*m*2^m)

        # Main Components
        self.model = Model(**self.model_params)
        
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint_motsp-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')

            # Train
            self.epoch = epoch # Set epoch, needed for HV-training
            HVs, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('hv', epoch, HVs)
            self.result_log.append('train_loss', epoch, train_loss)

            # LR Decay
            #self.scheduler.step()

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
       
            if epoch == self.start_epoch or all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint_motsp-{}.pt'.format(self.result_folder, epoch))

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):

        HVs_AM = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_hv, avg_loss = self._train_one_batch(batch_size)
            HVs_AM.update(avg_hv, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  HV: {:.4f},  Loss: {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                            HVs_AM.avg, loss_AM.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  HV: {:.4f}, Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 HVs_AM.avg, loss_AM.avg))

        return HVs_AM.avg, loss_AM.avg

    def _train_one_batch(self, batch_size):

        HVs_AM = AverageMeter()
        loss_AM = AverageMeter()

        # Prep
        ###############################################
        self.model.train()
        self.env.load_problems(batch_size)

        # Batch data for HV maximization
        self.HVs = None # Hypervolume in previous iteration, (B, P)
        self.proj_dists = None # Projection distances this far, (B, P, p'-1)
      
        for pprime in range(self.trainer_params["n_prefs"]):
            pref = torch.rand([1]) * torch.pi / 2

            HVs_mean, loss_mean = self._train_one_batch_and_pref(batch_size, pref) # For now do not take into account previous obj_values

            HVs_AM.update(HVs_mean, batch_size)
            loss_AM.update(loss_mean, batch_size)

        return HVs_AM.avg, loss_AM.avg
            
    
    def _train_one_batch_and_pref(self, batch_size, pref):
        reset_state, _, _ = self.env.reset()

        self.model.decoder.assign(pref)
        self.model.pre_forward(reset_state)
        
        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
      
        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        
        while not done:
            selected, prob = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
            
        # Loss
        ###############################################
        # reward is negative objective values, here we set it to positive
        reward = - reward
        proj_dist, HV = self._calculate_proj_distance_and_hv(pref, reward) 
        # proj_dist and HV: (B, P)
        if self.HVs is None: 
            ehvi = 1 # Expected hypervolume improvement
        else: 
            ehvi = torch.ceil(HV - self.HVs) # Current HV - previous HV, (B, P)
                
        self.HVs = HV # Set current to old

        omega = 1 - self.epoch / self.trainer_params['epochs']

        R = omega * proj_dist + ehvi * HV # Final reward, (B, P)
        baseline = torch.mean(R, dim=1, keepdim=True) # Mean along POMO dimension, (B, P)
       
        log_prob = prob_list.log().sum(dim=2) # (B, P)
    
        advantage = R - baseline # (B, P)
    
        loss = -advantage * log_prob # Minus Sign, (B, P)
        # shape = (batch, group)
        loss_mean = loss.mean()
        
        #Step & Return
        ################################################
        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()
        
        return HV.mean().item(), loss_mean.item()
    
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
        else: 
            proj_dists = torch.cat((proj_dists, proj_dist.unsqueeze(2)), dim=2) # Add current distance, (B, P, p')
        
        HV = self.HV_const * torch.mean(torch.pow(proj_dists, 2), dim=2) # (B, P, p') -> (B, P)
        HV = HV / (self.ref[0]*self.ref[1])

        norm_const = torch.minimum(self.ref[0]/lambda1, self.ref[1]/lambda2)
        proj_dist = proj_dist /  norm_const # Normalize between 0 and 1

        return proj_dist, HV
