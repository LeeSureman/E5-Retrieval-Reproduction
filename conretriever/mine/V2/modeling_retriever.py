import torch.nn as nn
import torch
from grad_cache.grad_cache import GradCache
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
        encoded = self.model(*args, **kwargs)

        last_hidden_states = encoded.hidden_states[-1]

        # print('last_hidden_states: {}'.format(last_hidden_states.size()))


        hidden_size = last_hidden_states.size()[-1]

        sequence_representation_embeds = last_hidden_states[is_representation_id]
        sequence_representation_embeds = sequence_representation_embeds.view(batch_size, -1, hidden_size)
        sequence_representation = torch.mean(sequence_representation_embeds, dim=1)
        sequence_representation = sequence_representation / torch.norm(sequence_representation, p=2, dim=1, keepdim=True)

        return sequence_representation

    def gradient_checkpointing_enable(self,gradient_checkpointing_kwargs):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def save_pretrained(self,*args,**kwargs):
        self.model.save_pretrained(*args,**kwargs)


def retrieval_forward(dense_embedder, query_ids, query_attention_mask, positive_doc_ids, positive_doc_attention_mask, chunk_sizes=2,
                      temperature=50, n_hard_negative=0):
    loss_fn = DistributedContrastiveLoss(temperature, n_hard_negative)
    gc = GradCache(
        models=[dense_embedder, dense_embedder],
        chunk_sizes=chunk_sizes,
        loss_fn=loss_fn,
        get_rep_fn=None
    )
    query_input = {'input_ids':query_ids,'attention_mask':query_attention_mask}
    positive_doc_input = {'input_ids':positive_doc_ids,'attention_mask':positive_doc_attention_mask}

    detached_loss = gc(query_input, positive_doc_input).requires_grad_()
    # print('loss: {}'.format(detached_loss))

    return {'loss':detached_loss}


# class Retriever(nn.Module):
#     def __init__(self, model, representation_id):
#         '''
#
#         :param model: ***ForCausalLM
#         :param representation_id:
#         '''
#         super().__init__()
#         self.model = model
#         self.representation_id = representation_id
#         self.dense_embedder = DenseEmbedder(model.model, representation_id)
#
#     def forward(self, *args, **kwargs):
#         query_ids = kwargs['query_ids']
#         positive_doc_ids = kwargs['positive_doc_ids']
