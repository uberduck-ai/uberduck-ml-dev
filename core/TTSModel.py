import torch

class TTSModel(torch.nn.Module):
    
    def infer(self):
        raise NotImplemented  

    def forward(self):
        raise NotImplemented  

    def from_pretrained(self):
        raise NotImplemented  

    @classmethod 
    def create(cls, name, opts):

        model_cls = cls.get_class(name)
        
        return(model_cls(opts))