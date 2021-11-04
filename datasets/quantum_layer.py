import pennylane as qml
from pennylane.templates import RandomLayers
from pennylane import numpy as np


class QuantumLayer():
    def __init__(self, depth=1, kernel_size=3, stride=1, devname='default.qubit'):
        # infer parameters
        self.kernel_size = kernel_size
        self.wires = kernel_size * kernel_size
        self.dev = qml.device(devname, wires=self.wires)
        self.rand_params = np.random.uniform(high=2 * np.pi, size=(depth, self.wires))
        self.stride = stride

        @qml.qnode(self.dev)
        def random_circuit(x):
            n = len(x)
            assert n == self.wires, 'Number of wires (%d) must be same as input size (%d)' % (self.wires, n)
            # This is equivalent to a nxn = wires convolution
            for j in range(n):
                qml.RY(np.pi * x[j], wires=j)  # mapping, can be changed

            # Random quantum circuit
            RandomLayers(self.rand_params, wires=list(range(n)))

            # Measurement producing n classical output values
            # Thus, n kernels
            return [qml.expval(qml.PauliZ(j)) for j in range(n)]

        self.random_circuit = random_circuit


    def quanv(self, in_img):
        W1, H1 = in_img.shape
        P = round((self.kernel_size - 1) / 2)
        W = round((W1 - self.kernel_size + 2 * P) / self.stride + 1)
        H = round((H1 - self.kernel_size + 2 * P) / self.stride + 1)
        out = np.zeros((W, H, self.wires))
        # print(out.shape)

        # create an image with a padding
        pad_img = np.pad(in_img, (P, P), 'constant', constant_values=(0, 0))
        Wp, Hp = pad_img.shape
        # print(pad_img.shape)

        for j in range(0, Wp, self.stride):
            for k in range(0, Hp, self.stride):
                y = pad_img[j:j + self.kernel_size, k:k + self.kernel_size]
                if y.shape[0] != self.kernel_size or y.shape[1] != self.kernel_size:
                    continue
                # get a flatten patch
                y = y.flatten()

                # get output from random circuit
                q_results = self.random_circuit(y)

                # Assign expectation values to different channels of the output pixel (j/stride, k/stride)
                for c in range(self.wires):
                    out[j // self.stride, k // self.stride, c] = q_results[c]

        return out

    #return quanv

"""
image = np.random.rand(10,10)
x = image.copy()
z = QuantumLayer(depth=1, kernel_size=3, stride=1)(x)

print(x.shape)
print(z.shape)
"""