
import ml_collections
from jraph_MPEU_configs.default_mp_test import get_config as get_config_super

def get_config() -> ml_collections.ConfigDict():
    config = get_config_super() # inherit from default mp config
    config.eval_every_steps = 100_000
    config.num_train_steps_max = 10000
    config.log_every_steps = 100_000
    config.checkpoint_every_steps = 100_000
    config.limit_data = None
    config.selection = None
    config.data_file = '/u/dansp/jraph_MPEU/qm9/qm9_graphs_fc.db'
    config.label_str = 'U0'
    config.num_edges_max = None
    config.dynamic_batch = True
    config.compute_device = 'cpu'
    config.batch_size = 128
    config.static_round_to_multiple = True
    # MPNN hyperparameters we use the defaults.

    return config
