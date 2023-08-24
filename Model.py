import numpy


class Model:

    def __init__(self, layers):
        self.layers = layers

    def forward_prop(self, x_in):
        output = numpy.zeros(len(self.layers))
        a_i = x_in
        for i in range(len(self.layers)):
            a_i = self.layers[i].predict(a_i)
            output[i] = a_i
        return output

    # need to add a back prop function here.

