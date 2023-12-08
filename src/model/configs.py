class ModelConfig(dict):
    def __init__(self):
        pass
    
    def __setattr__(self, __name: str, __value) -> None:
        self.__dict__[__name] = __value
        super().__init__(self.__dict__)
        
        
class LSTMConfig(ModelConfig):
    def __init__(self, data_dim=7, hidden_size=128, num_layers=3, dropout=0.) -> None:
        self.data_dim = data_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        
class TransformerConfig(ModelConfig):
    def __init__(self, data_dim=8, hidden_size=2048, num_layers=3, nhead=8, dropout=0.1,):
        self.data_dim = data_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout