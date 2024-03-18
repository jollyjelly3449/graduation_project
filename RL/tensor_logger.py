import numpy as np
import torch
import torch.nn as nn


class TensorLogger(object):
    def __init__(self, path: str, slots: list[str] = None):
        self.__path = path

        if slots is None:
            self.__memory = {"": []}
            self.__slots = [""]

        else:
            self.__memory = {}
            for slot in slots:
                self.__memory[slot] = []
            self.__slots = slots

    def append(self, **kwargs):
        for key in kwargs:
            assert key in self.__slots, "key is not in slots."
            val = kwargs[key]
            if isinstance(val, torch.Tensor):
                val = val.detach().cpu().numpy()
            self.__memory[key].append(val)

    def save(self, path=None):
        if path is None:
            path = self.__path

        for slot in self.__slots:
            np.save(path + slot, np.array(self.__memory[slot]))

    def reset(self):
        self.__memory = {}
        for slot in self.__slots:
            self.__memory[slot] = []
