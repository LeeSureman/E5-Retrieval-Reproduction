from typing import Callable

import torch
from torch import Tensor
from torch.nn import functional as F
from torch import distributed as dist


class SimpleContrastiveLoss:
    def __init__(self, temperature, n_hard_negatives: int = 0):
        self.target_per_qry = n_hard_negatives + 1
        self.temperature = temperature

    def __call__(self, x: Tensor, y: Tensor, target: Tensor = None, reduction: str = 'mean'):
        if target is None:
            assert x.size(0) * self.target_per_qry == y.size(0)
            target = torch.arange(0, x.size(0) * self.target_per_qry, self.target_per_qry, device=x.device)

        logits = torch.matmul(x, y.transpose(0, 1))
        logits = logits*self.temperature
        return F.cross_entropy(logits, target, reduction=reduction)


class DistributedContrastiveLoss(SimpleContrastiveLoss):
    def __init__(self, temperature, n_hard_negatives: int = 0):
        assert dist.is_initialized(), "Distributed training has not been properly initialized."

        super().__init__(temperature=temperature,n_hard_negatives=n_hard_negatives)
        self.word_size = dist.get_world_size()
        self.rank = dist.get_rank()

    def __call__(self, x: Tensor, y: Tensor, **kwargs):
        dist_x = self.gather_tensor(x)
        dist_y = self.gather_tensor(y)


        return super().__call__(dist_x, dist_y, **kwargs)

    def gather_tensor(self, t):
        gathered = [torch.empty_like(t) for _ in range(self.word_size)]
        dist.all_gather(gathered, t)
        gathered[self.rank] = t

        return torch.cat(gathered, dim=0)

class SimpleContrastiveLoss_debug:
    def __init__(self, temperature, n_hard_negatives: int = 0):
        self.target_per_qry = n_hard_negatives + 1
        self.temperature = temperature

    def __call__(self, x: Tensor, y: Tensor, target: Tensor = None, reduction: str = 'mean'):
        if target is None:
            assert x.size(0) * self.target_per_qry == y.size(0)
            target = torch.arange(0, x.size(0) * self.target_per_qry, self.target_per_qry, device=x.device)

        logits = torch.matmul(x, y.transpose(0, 1))
        logits = logits*self.temperature

        loss = F.cross_entropy(logits, target, reduction=reduction)

        print('scores: {}'.format(logits.size()))
        print('scores: {}'.format(logits))
        # print('scores[4:7,7:10]: {}'.format(logits[4:7, 7:10]))
        # print('scores[-7:-4,-7:-4]:{}'.format(logits[-7:-4, -7:-4]))
        # print('scores[4:10,-7:-4]:{}'.format(logits[4:10, -7:-4]))
        print('loss: {}'.format(loss))
        print('self.temperature: {}'.format(self.temperature))




        return loss

class DistributedContrastiveLoss_debug(SimpleContrastiveLoss_debug):
    def __init__(self, temperature, n_hard_negatives: int = 0):
        assert dist.is_initialized(), "Distributed training has not been properly initialized."

        super().__init__(temperature=temperature,n_hard_negatives=n_hard_negatives)
        self.word_size = dist.get_world_size()
        self.rank = dist.get_rank()

    def __call__(self, x: Tensor, y: Tensor, **kwargs):
        dist_x = self.gather_tensor(x)
        dist_y = self.gather_tensor(y)

        if self.rank == 0:
            print('dist_x:{}, {}'.format(dist_x.size(), dist_x.dtype))
            print('dist_y:{}, {}'.format(dist_y.size(), dist_y.dtype))

            print('dist_x:{}'.format(dist_x[:2,:4]))
            print('dist_y:{}'.format(dist_y[:2, :4]))


        return super().__call__(dist_x, dist_y, **kwargs)

    def gather_tensor(self, t):
        gathered = [torch.empty_like(t) for _ in range(self.word_size)]
        dist.all_gather(gathered, t)
        gathered[self.rank] = t


        # for i, tmp in enumerate(gathered):
        #     if i != self.rank:
        #         gathered[i].detach_()
        return torch.cat(gathered, dim=0)


class ContrastiveLossWithQueryClosure(SimpleContrastiveLoss):
    def __call__(
            self,
            *reps: Tensor,
            query_closure: Callable[[], Tensor] = None,
            target: Tensor = None,
            reduction: str = 'mean'
    ):
        if len(reps) == 0 or len(reps) > 2:
            raise ValueError(f'Expecting 1 or 2 tensor input, got {len(reps)} tensors')

        # no closure evaluation
        if len(reps) == 2:
            assert query_closure is None, 'received 2 representation tensors while query_closure is also set'
            return super().__call__(*reps, target=target, reduction=reduction)

        # run the closure
        assert query_closure is not None
        x = query_closure()
        y = reps[0]
        return super().__call__(x, y, target=target, reduction=reduction)
