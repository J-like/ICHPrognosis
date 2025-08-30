import torch
import torch.nn as nn
import torch.nn.functional as F
from net.encoder     import *
from net.attention   import *

class CombinedModel(nn.Module):

    def __init__(self,):
        super().__init__()

        self.image3d_encoder  = Encoder3D(n_channels=1, n_filters=16,
                                          normalization='groupnorm', has_dropout=True)
        self.t2t       = nn.Sequential(
                                        nn.Linear(768, 512),
                                        nn.GELU(),
                                        nn.Dropout(0.1)
                                    )
        self.fusion = BCA()
        self.classifier = nn.Linear(512,2)

    def forward(self, img3d, text):
        
        img3d_feat = self.image3d_encoder(img3d)              
        img3d_feat = img3d_feat.reshape(img3d_feat.size(0), -1, 512) 
        
        text_feat  = self.t2t(text)                 
        x = self.fusion(text_feat,  img3d_feat)  

        x = x.mean(dim=1)
        logits = self.classifier(x)                      
        
        return logits

def main():
    B = 2
    x = torch.randn(B, 1, 128, 512, 512).to("cuda")
    t = torch.randn(B, 64,768).to("cuda")
        
    model = CombinedModel().to("cuda")
    
    output = model(x, t)
    
    print(f"Shape of model output: {output.shape}")

if __name__ == "__main__":
    main()




