# coding=utf-8

import torch
import torch.nn as nn
from transformers import DistilBertModel


class DistilBERTEncoder(nn.Module):
    """DistilBERT language encoder for REC tasks.

    Replaces LSTM+GloVe with contextual embeddings from DistilBERT.
    Output format matches LSTM_SA: dict with flat_lang_feat, lang_feat, lang_feat_mask.

    DistilBERT output dim: 768 (vs LSTM's HIDDEN_SIZE=512).
    Downstream layers (linear_vs, linear_ts, linear_decoder) must use 768 accordingly.
    """

    def __init__(self, __C):
        super(DistilBERTEncoder, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.output_dim = 768

        # Optionally freeze lower layers for efficiency
        freeze_layers = getattr(__C, 'BERT_FREEZE_LAYERS', 0)
        if freeze_layers > 0:
            # Freeze embeddings
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            # Freeze first N transformer layers
            for layer in self.bert.transformer.layer[:freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, y):
        """
        Args:
            y: tuple (input_ids, attention_mask) or just input_ids
               input_ids: (B, seq_len) LongTensor
               attention_mask: (B, seq_len) LongTensor

        Returns:
            dict with:
                'flat_lang_feat': (B, 768) - [CLS] token representation
                'lang_feat': (B, seq_len, 768) - full sequence
                'lang_feat_mask': (B, 1, 1, seq_len) - padding mask (True = masked)
        """
        if isinstance(y, (tuple, list)):
            input_ids, attention_mask = y
        else:
            input_ids = y
            attention_mask = (input_ids != 0).long()

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lang_feat = outputs.last_hidden_state           # (B, seq_len, 768)
        flat_lang_feat = lang_feat[:, 0, :]             # [CLS] token: (B, 768)

        # Build mask compatible with existing SA / AttFlat code
        # True means masked (padding position)
        lang_feat_mask = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2).bool()

        return {
            'flat_lang_feat': flat_lang_feat,
            'lang_feat': lang_feat,
            'lang_feat_mask': lang_feat_mask
        }
