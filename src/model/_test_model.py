import sys, os
from pathlib import Path as path
sys.path.insert(0, str(path(__file__).parent.parent))


from model import *

def test_model(sample_net):
    print(sample_net)
    import torch
    sample_inputs = torch.rand((5, 96+336, 8))
    sample_net = Transformer(**TransformerConfig())
    print(sample_net)
    sample_output = sample_net(sample_inputs)
    print(sample_output['pred'].shape, sample_output['gt'].shape, sample_output['loss'])
    sample_net.eval()
    sample_output = sample_net(sample_inputs)
    print(sample_output['pred'].shape, sample_output['gt'].shape, sample_output['loss'])
    print('='*30)


test_model(LSTM(**LSTMConfig()))
# test_model(Transformer(**TransformerConfig()))
# test_model(GRU(**GRUConfig()))