from ..models.tacotron2 import DEFAULTS as TACOTRON2_DEFAULTS


def tacotron2_training_to_model_config(training_config):

    shared_keys = set(TACOTRON2_DEFAULTS.values().keys()).intersection(
        training_config.keys()
    )
    # NOTE (Sam): only need to save non-default parameters in config unless defaults change.
    minimal_model_config = {k: training_config[k] for k in shared_keys}
    return minimal_model_config
