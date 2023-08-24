import numpy


class Layer:

    def __int__(self, units, activation_function):
        self.units = units
        self.activation_function = activation_function
        self.W = numpy.array([[]])
        self.B = numpy.array([[]])

    def predict(self, x_in):
        output = numpy.zeros(self.units)
        for i in range(self.units):
            a_i = numpy.dot(self.W, x_in) + self.B
            output[i] = self.activation_function(a_i)
        return output
