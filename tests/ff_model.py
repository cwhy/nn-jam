import time
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from jax import jit, grad
import numpy.random as npr
from typing import Tuple, List
from dataclasses import dataclass
from jax.scipy.special import logsumexp
from supervised_benchmarks.dataset_protocols import Input, Output
from supervised_benchmarks.mnist_variations import MnistConfigIn, MnistConfigOut
from supervised_benchmarks.mnist import MnistDataConfig, Mnist, FixedTrain, FixedTest


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

    def update_(self, batch: Tuple[np.ndarray, np.ndarray]):
        self.params = self._update(self.params, self.step_size, batch)

    def predict(self, inputs):
        return self._forward(self.params, inputs)

    def get_sampler(self):
        k = Mnist(MnistDataConfig(base_path=Path('/Data/torchvision/')))
        mnist_in_flattened = MnistConfigIn(is_float=True, is_flat=True).get_var()
        mnist_out_1hot = MnistConfigOut(is_1hot=True).get_var()
        pool_dict = k.retrieve({Input: mnist_in_flattened, Output: mnist_out_1hot})
        pool_input = pool_dict[Input]
        pool_output = pool_dict[Output]
        train_images = pool_input.subset(FixedTrain).content
        test_images = pool_input.subset(FixedTest).content
        train_labels = pool_output.subset(FixedTrain).content
        test_labels = pool_output.subset(FixedTest).content

        num_train = len(FixedTrain.indices)
        num_complete_batches, leftover = divmod(num_train, self.batch_size)
        num_batches = num_complete_batches + int(bool(leftover))

        def data_stream():
            rng = npr.RandomState(0)
            while True:
                perm = rng.permutation(num_train)
                for i in range(num_batches):
                    batch_idx = perm[i * self.batch_size:(i + 1) * self.batch_size]
                    yield train_images[batch_idx], train_labels[batch_idx]

        return (data_stream(), num_batches), (train_images, train_labels), (test_images, test_labels)

    def fit_(self):
        (batches, num_batches), (train_images, train_labels), (test_images, test_labels) = self.get_sampler()
        for epoch in range(self.num_epochs):
            start_time = time.time()
            for _ in range(num_batches):
                model.update_(next(batches))
            epoch_time = time.time() - start_time

            train_acc = accuracy(model.predict(train_images), train_labels)
            test_acc = accuracy(model.predict(test_images), test_labels)
            print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
            print("Training set accuracy {}".format(train_acc))
            print("Test set accuracy {}".format(test_acc))


def init_model(rng=npr.RandomState(0)):
    layer_sizes = [784, 784, 256, 10]
    param_scale = 0.01
    return MlpMnistModel(
        params=[(param_scale * rng.randn(m, n), param_scale * rng.randn(n))
                for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])],
        step_size=0.01,
        num_epochs=50,
        batch_size=32
    )


def accuracy(output, target):
    target_class = jnp.argmax(target, axis=1)
    output_class = jnp.argmax(output, axis=1)
    return jnp.mean(output_class == target_class)


model = init_model()
model.fit_()
