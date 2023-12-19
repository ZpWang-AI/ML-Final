import sys, os
from pathlib import Path as path
sys.path.insert(0, str(path(__file__).parent.parent))

import time
from model import *

def test_model(sample_net):
    print(sample_net)
    import torch
    sample_inputs = torch.rand((5, 96+336, 8))
    start_time = time.time()
    sample_output = sample_net(sample_inputs)
    print(sample_output['pred'].shape, sample_output['gt'].shape, sample_output['loss'])
    print(f'train time: {time.time()-start_time}')
    sample_net.eval()
    with torch.no_grad():
        start_time = time.time()
        sample_output = sample_net(sample_inputs)
        print(sample_output['pred'].shape, sample_output['gt'].shape, sample_output['loss'])
        print(f'eval time: {time.time()-start_time}')
    print('='*30)


# test_model(LSTM(**LSTMConfig()))
# test_model(LSTMResidual(**LSTMConfig()))
# test_model(LSTMWithoutLinear(**LSTMConfig()))
# test_model(Transformer(**TransformerConfig()))
# test_model(GRU(**GRUConfig()))