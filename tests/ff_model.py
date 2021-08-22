import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Protocol, Dict, Callable

import jax.numpy as jnp
import numpy as np
import numpy.random as npr
from jax import jit, grad
from jax.scipy.special import logsumexp

from supervised_benchmarks.dataset_protocols import Input, Output, Port, Sampler, FullBatchSampler, DataContent
from supervised_benchmarks.dataset_utils import subset_all
from supervised_benchmarks.mnist import MnistDataConfig, Mnist, FixedTrain, FixedTest
from supervised_benchmarks.mnist_variations import MnistConfigIn, MnistConfigOut
from supervised_benchmarks.sampler import get_fixed_epoch_sampler, get_full_batch_sampler


class Model(Protocol[DataContent]):
    def predict(self) -> Dict[Tuple[List[Port], List[Port]],
                              Callable[[List[DataContent]], List[DataContent]]]: ...


def measure(measurement_fn, predict_fn, port_pair: Tuple[Port, Port], sampler: Sampler) -> float:
    src, tgt = port_pair
    if sampler.tag == 'FullBatchSampler':
        assert isinstance(sampler, FullBatchSampler)
        return measurement_fn(predict_fn(sampler.full_batch[src]), sampler.full_batch[tgt])
    else:
        raise NotImplementedError


@dataclass
class MlpMnistModel:
    params: List[Tuple[np.ndarray, np.ndarray]]
    step_size: float
    num_epochs: int
    batch_size: int

    @staticmethod
    def _forward(params, inputs):
        activations = inputs
        for w, b in params[:-1]:
            outputs = jnp.dot(activations, w) + b
            activations = jnp.tanh(outputs)

        final_w, final_b = params[-1]
        logits = jnp.dot(activations, final_w) + final_b
        return logits - logsumexp(logits, axis=1, keepdims=True)

    @staticmethod
    def _loss(params, batch):
        inputs, targets = batch
        preds = MlpMnistModel._forward(params, inputs)
        return -jnp.mean(jnp.sum(preds * targets, axis=1))

    @staticmethod
    @jit
    def _update(params, step_size, batch):
        grads = grad(MlpMnistModel._loss)(params, batch)
        return [(w - step_size * dw, b - step_size * db)
                for (w, b), (dw, db) in zip(params, grads)]

    def update_(self, x: np.ndarray, y: np.ndarray):
        self.params = self._update(self.params, self.step_size, (x, y))

    def predict(self, inputs: np.ndarray):
        return self._forward(self.params, inputs)

    def fit_(self):
        k = Mnist(MnistDataConfig(base_path=Path('/Data/torchvision/')))
        mnist_in_flattened = MnistConfigIn(is_float=True, is_flat=True).get_var()
        mnist_out_1hot = MnistConfigOut(is_1hot=True).get_var()
        pool_dict = k.retrieve({Input: mnist_in_flattened, Output: mnist_out_1hot})
        train_pool = subset_all(pool_dict, FixedTrain)
        test_pool = subset_all(pool_dict, FixedTest)
        train_sampler = get_fixed_epoch_sampler(self.batch_size, train_pool)
        train_all = get_full_batch_sampler(train_pool)
        test_all = get_full_batch_sampler(test_pool)
        flow = (Input, Output)
        for epoch in range(self.num_epochs):
            start_time = time.time()
            for _ in range(train_sampler.num_batches):
                model.update_(next(train_sampler.iter[Input]), next(train_sampler.iter[Output]))
            epoch_time = time.time() - start_time

            train_acc = measure(accuracy, model.predict, flow, train_all)
            test_acc = measure(accuracy, model.predict, flow, test_all)
            print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
            print("Training set accuracy {}".format(train_acc))
            print("Test set accuracy {}".format(test_acc))


def init_model_():
    rng = npr.RandomState(0)
    layer_sizes = [784, 784, 256, 10]
    param_scale = 0.01
    return MlpMnistModel(
        params=[(param_scale * rng.standard_normal((m, n)), param_scale * rng.standard_normal(n))
                for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])],
        step_size=0.01,
        num_epochs=50,
        batch_size=32
    )


def accuracy(output: np.ndarray, target: np.ndarray) -> float:
    target_class = jnp.argmax(target, axis=1)
    output_class = jnp.argmax(output, axis=1)
    return jnp.mean(output_class == target_class)


model = init_model_()
model.fit_()
