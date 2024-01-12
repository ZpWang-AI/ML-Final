class ModelConfig(dict):
    def __init__(self):
        pass
    
    def __setattr__(self, __name: str, __value) -> None:
        self.__dict__[__name] = __value
        super().__init__(self.__dict__)
        
        
class LSTMConfig(ModelConfig):
    def __init__(
        self,
        data_dim=7,
        hidden_size=128,
        num_layers=3,
        dropout=0.,
    ) -> None:
        self.data_dim = data_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        
class TransformerConfig(ModelConfig):
    def __init__(
        self, 
        data_dim=7,
        channels=128,
        num_layers=3,
        nhead=8,
        dropout=0.1,
    ) -> None:
        self.data_dim = data_dim
        self.channels = channels
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout


class GRUConfig(ModelConfig):
    def __init__(
        self,
        data_dim=7,
        hidden_size=128,
        num_layers=3,
        dropout=0.,
    ) -> None:
        self.data_dim = data_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        

class MLPConfig(ModelConfig):
    def __init__(
        self,
        data_dim=7,
        hidden_size=1280,
        num_layers=3,
        dropout=0.,
    ) -> None:
        self.data_dim = data_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout


class CNNConfig(ModelConfig):
    def __init__(
        self,
        data_dim=7,
        num_layers=3,
        dropout=0.,
    ) -> None:
        self.data_dim = data_dim
        self.num_layers = num_layers
        self.dropout = dropout
        

if __name__ == '__main__':
    def generate_config(s):
        items = [i.strip().strip(',')for i in s.split() if i.strip()]
        for i in items:
            print(f'{i}=,')
        print()
        for i in items:
            print(f'self.{i} = {i}')
    
    s = '''
        data_dim,
        hidden_size,
        num_layers,
        dropout,
    '''
    generate_config(s)