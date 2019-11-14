from __future__ import absolute_import

from functools import reduce

from laia.losses.ctc_loss import transform_output


class CTCGreedyDecoder(object):
    def __init__(self):
        self._output = None
        self._segm = None

    def __call__(self, x, getSeg=False):
        x, xs = transform_output(x)
        _, idx = x.max(dim=2)
        idx = idx.t().tolist()
        x = [idx_n[: int(xs[n])] for n, idx_n in enumerate(idx)]
        if getSeg:
            self._segm = [
                [p for p, v in enumerate(x_n) if p==0 or (v!=x_n[p-1] and x_n[p-1]!=0)] + [len(x_n)-1]
                for x_n in x
            ]
        # Remove repeated symbols
        x = [
            reduce(lambda z, x: z if z[-1] == x else z + [x], x_n[1:], [x_n[0]])
            for x_n in x
        ]
        # Remove CTC blank symbol
        self._output = [[x for x in x_n if x != 0] for x_n in x]
        if self._segm:
            assert len(self._segm) == len(self._output)+1 or len(self._segm) == len(self._output)+2, ( 
                "Number of char segmentations is not consistent"
                "with the number of recognized chars"
            )
        return self._output

    @property
    def output(self):
        return self._output

    @property
    def seg(self):
        return self._segm
