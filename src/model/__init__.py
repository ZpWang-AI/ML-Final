from model.LSTM import LSTM
from model.Transformer import Transformer
from model.GRU import GRU

from model.configs import *


def get_model(model_name:str, model_config:ModelConfig):
    model_dict = {
        'lstm':LSTM,
        'transformer':Transformer,
        'gru':GRU,
    }
    model_name = model_name.lower()
    if model_name in model_dict:
        return model_dict[model_name](**model_config)
    else:
        raise Exception('wrong model_name')