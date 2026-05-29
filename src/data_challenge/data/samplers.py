import math

import torch
from torch.utils.data import BatchSampler


class BalancedGenderBatchSampler(BatchSampler):
    """Build batches with an even gender mix using replacement for the minority group.

    This targets the challenge metric directly: female and male errors are scored
    separately, while the training set is substantially imbalanced.
    """

    def __init__(
        self,
        genders: list[float],
        batch_size: int,
        generator: torch.Generator | None = None,
        drop_last: bool = False,
    ):
        if batch_size < 2:
            raise ValueError("BalancedGenderBatchSampler requires batch_size >= 2")

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.generator = generator

        self.group0 = [idx for idx, gender in enumerate(genders) if float(gender) == 0.0]
        self.group1 = [idx for idx, gender in enumerate(genders) if float(gender) == 1.0]

        if not self.group0 or not self.group1:
            raise ValueError("BalancedGenderBatchSampler requires both gender groups to be present")

        self.num_samples = len(genders)
        self.majority_size = max(len(self.group0), len(self.group1))

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        return math.ceil(self.num_samples / self.batch_size)

    def __iter__(self):
        group0 = self._expand_and_shuffle(self.group0, self.majority_size)
        group1 = self._expand_and_shuffle(self.group1, self.majority_size)

        half = self.batch_size // 2
        remainder = self.batch_size - (2 * half)
        batches = []

        for start in range(0, self.majority_size, half):
            batch = group0[start:start + half] + group1[start:start + half]
            if len(batch) < self.batch_size:
                needed = self.batch_size - len(batch)
                if len(group0) >= len(group1):
                    batch.extend(group0[:needed])
                else:
                    batch.extend(group1[:needed])

            if remainder:
                source = group0 if len(group0) >= len(group1) else group1
                batch.extend(source[start:start + remainder])
                if len(batch) < self.batch_size:
                    batch.extend(source[: self.batch_size - len(batch)])

            batch = batch[: self.batch_size]
            batches.append(self._shuffle_batch(batch))

        if not self.drop_last and self.num_samples % self.batch_size:
            batches = batches[: math.ceil(self.num_samples / self.batch_size)]
        elif self.drop_last:
            batches = batches[: self.num_samples // self.batch_size]

        for batch in batches:
            yield batch

    def _expand_and_shuffle(self, indices: list[int], target_size: int) -> list[int]:
        if not indices:
            return []

        repeats = math.ceil(target_size / len(indices))
        expanded = (indices * repeats)[:target_size]
        return self._shuffle_list(expanded)

    def _shuffle_batch(self, batch: list[int]) -> list[int]:
        return self._shuffle_list(batch)

    def _shuffle_list(self, items: list[int]) -> list[int]:
        order = torch.randperm(len(items), generator=self.generator).tolist()
        return [items[i] for i in order]
