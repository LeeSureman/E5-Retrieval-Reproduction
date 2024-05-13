import torch.nn as nn
import torch
from grad_cache import GradCache
from grad_cache.loss import DistributedContrastiveLoss
from transformers import PreTrainedModel


class DenseEmbedder(PreTrainedModel):
    def __init__(self, model, representation_id, config):
        super().__init__(config)
        self.model = model
        self.representation_id = representation_id
        self.config = config

    def forward(self, *args, **kwargs):
        input_ids = kwargs['input_ids']
        batch_size = input_ids.size()[0]
        is_representation_id = (input_ids == self.representation_id)

        attention_mask = kwargs['attention_mask']
        seq_len = input_ids.size()[1]
        range_tensor = torch.arange(seq_len).unsqueeze(0).to(is_representation_id.device)
        seq_len_each = torch.sum(attention_mask, dim=1)
        first_representation_token_pos = seq_len_each - (self.representation_token_num)
        mask = range_tensor < (first_representation_token_pos.unsqueeze(1))
        # mask the representation_token in the original input
        is_representation_id[mask] = False

        encoded = self.model(*args, **kwargs)
        last_hidden_states = encoded.hidden_states[-1]
        hidden_size = last_hidden_states.size()[-1]

        sequence_representation_embeds = last_hidden_states[is_representation_id]
        sequence_representation_embeds = sequence_representation_embeds.view(batch_size, -1, hidden_size)
        sequence_representation = torch.mean(sequence_representation_embeds, dim=1)
        sequence_representation = nn.functional.normalize(sequence_representation, p=2, dim=-1)

        return sequence_representation

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def save_pretrained(self, *args, **kwargs):
        self.model.save_pretrained(*args, **kwargs)


def retrieval_forward(
    dense_embedder, 
    query_ids, 
    query_attention_mask, 
    positive_doc_ids, 
    positive_doc_attention_mask,
    hard_negative_docs_ids=None, 
    hard_negative_docs_attention_mask=None,
    do_grad_cache=1, 
    chunk_sizes=2,
    temperature=50, 
    n_hard_negative=0
):
    loss_fn = DistributedContrastiveLoss(temperature, n_hard_negative)
    gc = GradCache(
        models=[dense_embedder, dense_embedder],
        chunk_sizes=chunk_sizes,
        loss_fn=loss_fn,
        get_rep_fn=None,
        do_cache=do_grad_cache,
    )
    query_input = {'input_ids': query_ids, 'attention_mask': query_attention_mask}

    # prepare inputs of docs:
    batch_size = positive_doc_ids.size()[0]
    seq_len = positive_doc_ids.size()[1]

    if hard_negative_docs_ids != None:
        positive_doc_ids = positive_doc_ids.view(batch_size, 1, seq_len)
        positive_doc_and_hard_negative_docs_ids = torch.cat([positive_doc_ids, hard_negative_docs_ids], dim=1)
        positive_doc_and_hard_negative_docs_ids = positive_doc_and_hard_negative_docs_ids.view(-1, seq_len)
    else:
        positive_doc_and_hard_negative_docs_ids = positive_doc_ids

    if hard_negative_docs_ids != None:
        positive_doc_attention_mask = positive_doc_attention_mask.view(batch_size, 1, seq_len)
        positive_doc_and_hard_negative_docs_attention_mask = torch.cat([positive_doc_attention_mask, hard_negative_docs_attention_mask], dim=1)
        positive_doc_and_hard_negative_docs_attention_mask = positive_doc_and_hard_negative_docs_attention_mask.view(-1, seq_len)
    else:
        positive_doc_and_hard_negative_docs_attention_mask = positive_doc_attention_mask

    positive_doc_and_hard_negative_docs_input = {
        'input_ids': positive_doc_and_hard_negative_docs_ids,
        'attention_mask': positive_doc_and_hard_negative_docs_attention_mask
    }
    detached_loss = gc(query_input, positive_doc_and_hard_negative_docs_input).requires_grad_()
    return {'loss': detached_loss}
