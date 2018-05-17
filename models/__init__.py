from models import BasicModel
from models import SimpleModel

__all__ = [
    "BasicModel",
    "SimpleModel"
]


def make_model(config, env):
    if config['model_name'] in __all__:
        # print("===")
        # print(globals())
        # print("===")
        return globals()[config['model_name']](config, env)
    else:
        raise Exception('The model name %s does not exist' % config['model_name'])


def get_model_class(config):
    if config['model_name'] in __all__:
        return globals()[config['model_name']]
    else:
        raise Exception('The model name %s does not exist' % config['model_name'])