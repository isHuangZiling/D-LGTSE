import random
import torch as th
import numpy as np

from torch.utils.data.dataloader import default_collate
import torch.utils.data as dat
from torch.nn.utils.rnn import pad_sequence
from libs.audio import WaveReader

def make_dataloader(train=True,
                    data_kwargs=None,
                    num_workers=16,
                    chunk_size=32000,
                    batch_size=16):
    dataset = Dataset(**data_kwargs)
    return DataLoader(dataset,
                      train=train,
                      chunk_size=chunk_size,
                      batch_size=batch_size,
                      num_workers=num_workers)

def get_spk_ivec(key):
    '''
      409o030h_1.7445_029o0304_-1.7445_409c0211
    '''
    spk = key.split('_')[-1][0:3]
    print(spk)



class Dataset(object):
    """
    Per Utterance Loader
    """
    def __init__(self, mix_both_scp="", mix_clean_scp="",  ref_scp=None, aux_scp=None, sample_rate=8000):
        self.mix_both = WaveReader(mix_both_scp, sample_rate=sample_rate)
        self.mix_clean = WaveReader(mix_clean_scp, sample_rate=sample_rate)
        self.ref = WaveReader(ref_scp, sample_rate=sample_rate)
        self.aux = WaveReader(aux_scp, sample_rate=sample_rate)
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.mix_both)

    def __getitem__(self, index):
        key = self.mix_both.index_keys[index]
        mix_both = self.mix_both[key]
        mix_clean = self.mix_clean[key]
        ref = self.ref[key]
        aux = self.aux[key]

        # ---- 加这两行 ----
        min_len = min(len(mix_both), len(mix_clean), len(ref))
        mix_both, mix_clean, ref = mix_both[:min_len], mix_clean[:min_len], ref[:min_len]
        # ------------------

        return {
            "mix_both": mix_both.astype(np.float32),
            "mix_clean": mix_clean.astype(np.float32),
            "ref": ref.astype(np.float32),
            "aux": aux.astype(np.float32),
            "aux_len": len(aux),
            "key": key
        }


class ChunkSplitter(object):
    """
    Split utterance into small chunks
    """
    def __init__(self, chunk_size, train=True, least=8000):
        self.chunk_size = chunk_size
        self.least = least
        self.train = train

    def _make_chunk(self, eg, s):
        """
        Make a chunk instance, which contains:
            "mix": ndarray,
            "ref": [ndarray...]
        """
        chunk = dict()
        chunk["mix_both"] = eg["mix_both"][s:s + self.chunk_size]
        chunk["mix_clean"] = eg["mix_clean"][s:s + self.chunk_size]
        chunk["ref"] = eg["ref"][s:s + self.chunk_size]
        chunk["aux"] = eg["aux"]
        chunk["aux_len"] = eg["aux_len"]
        chunk["valid_len"] = int(self.chunk_size)
        return chunk

    def split(self, eg):
        N = eg["mix_both"].size
        # too short, throw away
        if N < self.least:
            return []
        chunks = []
        # padding zeros
        if N < self.chunk_size:
            P = self.chunk_size - N
            chunk = dict()
            chunk["mix_both"] = np.pad(eg["mix_both"], (0, P), "constant")
            chunk["mix_clean"] = np.pad(eg["mix_clean"], (0, P), "constant")
            chunk["ref"] = np.pad(eg["ref"], (0, P), "constant")
            chunk["aux"] = eg["aux"]
            chunk["aux_len"] = eg["aux_len"]
            chunk["valid_len"] = int(N)
            chunks.append(chunk)
        else:
            # random select start point for training
            s = random.randint(0, N % self.least) if self.train else 0
            while True:
                if s + self.chunk_size > N:
                    break
                chunk = self._make_chunk(eg, s)
                chunks.append(chunk)
                s += self.least
        return chunks


class DataLoader(object):
    """
    Online dataloader for chunk-level PIT
    """
    def __init__(self,
                 dataset,
                 num_workers=16,
                 chunk_size=32000,
                 batch_size=16,
                 train=True):
        self.batch_size = batch_size
        self.train = train
        self.splitter = ChunkSplitter(chunk_size,
                                      train=train,
                                      least=chunk_size // 2)
        # just return batch of egs, support multiple workers
        self.eg_loader = dat.DataLoader(dataset,
                                        batch_size=batch_size // 2,
                                        num_workers=num_workers,
                                        shuffle=train,
                                        collate_fn=self._collate)

    def _collate(self, batch):
        """
        Online split utterances
        """
        chunk = []
        for eg in batch:
            chunk += self.splitter.split(eg)
        return chunk

    def _pad_aux(self, chunk_list):
        # NOTE: `aux_len` may be stale/inconsistent if upstream modifies `aux`.
        # Always use the true current length of `aux` to decide padding so that
        # `default_collate` can `torch.stack` safely.
        lens_list = [len(chunk_item["aux"]) for chunk_item in chunk_list]
        max_len = int(np.max(lens_list)) if lens_list else 0

        for idx in range(len(chunk_list)):
            aux = chunk_list[idx]["aux"]
            cur_len = len(aux)
            P = max_len - cur_len
            if P > 0:
                chunk_list[idx]["aux"] = np.pad(aux, (0, P), "constant")

        return chunk_list

    def _merge(self, chunk_list):
        """
        Merge chunk list into mini-batch
        """
        N = len(chunk_list)
        if self.train:
            random.shuffle(chunk_list)
        blist = []
        for s in range(0, N - self.batch_size + 1, self.batch_size):
            batch = default_collate(self._pad_aux(chunk_list[s:s + self.batch_size]))
            blist.append(batch)
        rn = N % self.batch_size
        return blist, chunk_list[-rn:] if rn else []

    def __iter__(self):
        chunk_list = []
        for chunks in self.eg_loader:
            chunk_list += chunks
            batch, chunk_list = self._merge(chunk_list)
            for obj in batch:
                yield obj
                