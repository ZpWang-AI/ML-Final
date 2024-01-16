from model.LSTM import LSTM, LSTMResidual, LSTMWithoutLinear
from model.Transformer import Transformer
from model.GRU import GRU
from model.MLP import MLP
from model.CNN import CNN
from model.ExtendedVector import ExtendedVector

from model.configs import *


def get_model(model_name:str, model_config:ModelConfig):
    model_dict = {
        'lstm':LSTM, 
        'lstmresidual':LSTMResidual, 
        'lstmwithoutlinear':LSTMWithoutLinear,
        'transformer':Transformer,
        'gru':GRU,
        'mlp':MLP,
        'cnn':CNN,
        'extended vector':ExtendedVector,
    }
    model_name = model_name.lower()
    if model_name in model_dict:
        return model_dict[model_name](**model_config)
    else:
        raise Exception('wrong model_name')