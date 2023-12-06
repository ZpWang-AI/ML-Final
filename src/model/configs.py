class LSTMConfig(dict):
    def __init__(self, data_dim=7, hidden_size=128, num_layers=3, dropout=0.) -> None:
        self.data_dim = data_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
    def __setattr__(self, __name: str, __value) -> None:
        self.__dict__[__name] = __value
        super().__init__(self.__dict__)