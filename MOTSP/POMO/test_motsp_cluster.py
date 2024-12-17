##########################################################################################
# Machine Environment Config
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

##########################################################################################
# Path Config
import os
import sys
import torch
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

##########################################################################################
# import
import logging
from utils.utils import create_logger, copy_all_src, get_result_folder


from MOTSPTester import TSPTester
from MOTSPTester_HV import TSPTesterHV
from MOTSProblemDef import get_random_problems

##########################################################################################
import time

##########################################################################################
# parameters
env_params = {
    'problem_size': 20,
    'pomo_size': 20,
}

training_method = "HV" # HV or Obj

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': './Final_result/HV',  # directory path of pre-trained model and log files saved.
        'epoch': 100, 
    },
    'reference': [15, 15], # Only relevant if training method is "HV"
    'test_episodes': 40, 
    'test_batch_size': 40,
    'augmentation_enable': True,
    'aug_factor': 64, #64,
    'aug_batch_size': 100 
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']

logger_params = {
    'log_file': {
        'desc': 'test__tsp_n20',
        'filename': 'run_log'
    }
}

##########################################################################################
def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 100

def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('Training Method: {}'.format(training_method))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]

##########################################################################################
def main(n_sols = 101):

    timer_start = time.time()
    logger_start = time.time()
    
    if DEBUG_MODE:
        _set_debug_mode()
    
    create_logger(**logger_params)
    _print_config()
    
    if training_method == "Obj":
        tester = TSPTester(env_params=env_params,
                        model_params=model_params,
                        tester_params=tester_params)
    else: 
        tester = TSPTesterHV(env_params=env_params,
                        model_params=model_params,
                        tester_params=tester_params)

    
    copy_all_src(tester.result_folder)
    
    sols = np.zeros([n_sols, 2])
    
    shared_problem = get_random_problems(tester_params['test_episodes'], env_params['problem_size'])
    
    if training_method == "Obj":
        for i in range(n_sols):
            pref = torch.zeros(2).cuda()
            pref[0] = 1 - 0.01 * i
            pref[1] = 0.01 * i
            pref = pref / torch.sum(pref)
            aug_score = tester.run(shared_problem,pref)
            sols[i] = np.array(aug_score)
    else: 
        aug_score = tester.run(n_sols, shared_problem) # (n_sols, n_obj)
        sols = aug_score.cpu().numpy()
    
    timer_end = time.time()
    
    total_time = timer_end - timer_start
   
    result_folder = get_result_folder()
    np.savetxt(os.path.join(result_folder, "{}.txt".format("PMOCO")), sols, fmt='%f', delimiter=' ', newline='\n')

    print('Run Time(s): {:.4f}'.format(total_time))

##########################################################################################
if __name__ == "__main__":
    main()
