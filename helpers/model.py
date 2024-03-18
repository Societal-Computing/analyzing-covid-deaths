import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Define the model we'll be using
class CovidClassifier(nn.Module):
    def __init__(self, base_model, base_model_out=1024, n_classes=2, dropout=0.1):
        super(CovidClassifier, self).__init__()

        self.base_model = base_model
        self.dropout = nn.Dropout(p=dropout)
        self.linear_out = nn.Linear(in_features=base_model_out, out_features=n_classes)

    def forward(self, b_input_ids, token_type_ids, attention_mask):
        # transfer all inputs to GPU
        b_input_ids = b_input_ids.to(DEVICE)

        if token_type_ids:
            token_type_ids = token_type_ids.to(DEVICE)

        attention_mask = attention_mask.to(DEVICE)

        out = self.base_model(b_input_ids, token_type_ids, attention_mask)

        # get the logits from the base model
        out = out.pooler_output

        out = self.dropout(out)
        out = self.linear_out(out)

        return out
