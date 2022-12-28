
from nni.experiment import Experiment
experiment = Experiment('local')

    
experiment.config.trial_command = 'python example_xvector_experiment.py'
experiment.config.trial_code_directory = '.'


search_space = {
    'lr': {'_type': 'loguniform', '_value': [0.0001, 0.1]},
    'batch_size': {'_type': 'choice', '_value': [32,64]},
    # 'xvector_tdnn_ch0': {'_type': 'choice', '_value': [256, 512, 1024]},
    # 'xvector_tdnn_ch1': {'_type': 'choice', '_value': [256, 512, 1024]},
    # 'xvector_tdnn_ch2': {'_type': 'choice', '_value': [256, 512, 1024]},
    # 'xvector_tdnn_ch3': {'_type': 'choice', '_value': [256, 512, 1024]},
    # 'xvector_tdnn_ch4': 1500, 
}
experiment.config.search_space = search_space

experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

experiment.config.max_trial_number = 10
experiment.config.trial_concurrency = 1

experiment.run(8080)


input('Press enter to quit')
experiment.stop()