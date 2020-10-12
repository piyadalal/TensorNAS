from tensornas.layerargs import *
from tensornas.mutator import *
from tensorflow import keras
from tensornas.mutator import dimension_mag
from enum import Enum, auto


class SupportedLayers(Enum):
    Conv2D = auto()
    MaxPool2D = auto()
    MaxPool3D = auto()
    Reshape = auto()
    HiddenDense = auto()
    OutputDense = auto()
    Flatten = auto()
    Dropout = auto()


class LayerShape:
    def __init__(self, dimensions=None):
        self.dimensions = dimensions

    def __str__(self):
        if self.dimensions:
            return str(self.dimensions)
        else:
            return "?"

    def __eq__(self, other):
        if isinstance(other, LayerShape):
            return self.dimensions == other.dimensions
        return False

    def set(self, dimensions):
        self.dimensions = dimensions

    def get(self):
        return self.dimensions


class ModelLayer:
    "Common layer properties"

    def __init__(self, name, args=None, input_shape=(0, 0, 0)):
        self.name = name
        if args:
            self.args = args
        else:
            self.args = {}

        self.inputshape = LayerShape()
        self.outputshape = LayerShape()

        self.inputshape.set(input_shape)

    def getname(self):
        return self.name

    def getargs(self):
        return self.args

    def print(self):
        print("{} {}-> {}".format(self.name, self.inputshape, self.outputshape))
        for param_name, param_value in self.args.items():
            print("{}: {}".format(param_name, param_value))

    # The use of DEAP to instantiate individuals and in turn Models with ModelLayers means that the use
    # of abstract classes (using the abs package) cannot be done. As such the following methods are "abstract"
    # and should be implemented by all layer sub-classes.

    # repair is required for some layers when they possibly return an invalid configuration, a good example is the
    # reshape layer. If the input has changed from a previous mutation, eg. a previous Conv2D layer's filter count
    # changed which in turn changed the layers output dimensioning, then the reshape layer will need to adjust its
    # parameters, in the case of a reshape layer its output dimensioning, such that it is valid. A valid reshape
    # layer is one that has the same dimension magnitude in as out, eg. [2, 4, 6] in has a magnitude of 2*4*6=48 so
    # the output must also have a product of 48. A simple solution to repairing a reshape layer would be calling its
    # mutation function with the correct output magnitude.
    #
    # The repair function should solve all the conditions that can cause the validate function to fail.
    def repair(self):
        pass

    # The mutate function should randomly choose one parameter of the layer to modify. As mutation should perform
    # small changes it is best the mutation function performs small changes to one one parameter.
    def mutate(self):
        pass

    # The validate function is used to validate the layer at a layer level, this means that the layer only has access
    # to information on parameters that are local to it. Validating the arrangement of various layers should be
    # performed at a high level of the model tree.
    def validate(self, repair=True):
        pass

    # Returns the size of the layer's output dimension given the layer's current configuration.
    def output_shape(self):
        pass

    # Returns the keras layer object used for constructing the keras model for training.
    def get_keras_layer(self):
        pass


class Conv2DLayer(ModelLayer):
    MUTATABLE_PARAMETERS = 6
    MAX_FILTER_COUNT = 128
    MAX_KERNEL_DIMENSION = 7
    MAX_STRIDE = 7
    MAX_DILATION = 5

    def __init__(
        self,
        filters,
        kernel_size,
        strides,
        padding=PaddingArgs.SAME.value,
        dilation_rate=(0, 0),
        activation=Activations.RELU.value,
        input_shape=(0, 0, 0),
    ):
        super().__init__("Conv2D", input_shape=input_shape)

        self.args[Conv2DArgs.FILTERS.name] = filters
        self.args[Conv2DArgs.KERNEL_SIZE.name] = kernel_size
        self.args[Conv2DArgs.STRIDES.name] = strides
        self.args[Conv2DArgs.INPUT_SHAPE.name] = input_shape
        self.args[Conv2DArgs.PADDING.name] = padding
        self.args[Conv2DArgs.DILATION_RATE.name] = dilation_rate
        self.args[Conv2DArgs.ACTIVATION.name] = activation

        self.outputshape.set(self.output_shape())
        self.validate(repair=True)

    def _filters(self):
        return self.args[Conv2DArgs.FILTERS.name]

    def _kernel_size(self):
        return self.args[Conv2DArgs.KERNEL_SIZE.name]

    def _strides(self):
        return self.args[Conv2DArgs.STRIDES.name]

    def _input_size(self):
        return self.args[Conv2DArgs.INPUT_SHAPE.name]

    def _padding(self):
        return self.args[Conv2DArgs.PADDING.name]

    def _dilation_rate(self):
        return self.args[Conv2DArgs.DILATION_RATE.name]

    def _activation(self):
        return self.args[Conv2DArgs.ACTIVATION.name]

    def _single_stride(self):
        st = self._strides()
        if st[0] == 1 and st[1] == 1:
            return True
        return False

    def _single_dilation_rate(self):
        dr = self._dilation_rate()
        if dr[0] == 1 and dr[1]:
            return True
        return False

    def _mutate_filters(self, operator=MutationOperators.STEP):
        self.args[Conv2DArgs.FILTERS.name] = mutate_int(
            self._filters(), 1, Conv2DLayer.MAX_FILTER_COUNT, operator
        )

    def _mutate_kernel_size(self, operator=MutationOperators.SYNC_STEP):
        self.args[Conv2DArgs.KERNEL_SIZE.name] = mutate_tuple(
            self._kernel_size(), 1, Conv2DLayer.MAX_KERNEL_DIMENSION, operator
        )

    def _mutate_strides(self, operator=MutationOperators.SYNC_STEP):
        self.args[Conv2DArgs.STRIDES.name] = mutate_tuple(
            self._strides(), 1, Conv2DLayer.MAX_STRIDE, operator
        )

    def _mutate_padding(self):
        self.args[Conv2DArgs.PADDING.name] = mutate_enum(self._padding(), PaddingArgs)

    def _mutate_dilation_rate(self, operator=MutationOperators.SYNC_STEP):
        self.args[Conv2DArgs.DILATION_RATE.name] = mutate_tuple(
            self._dilation_rate(), 1, Conv2DLayer.MAX_DILATION, operator
        )

    def _mutate_activation(self):
        self.args[Conv2DArgs.ACTIVATION.name] = mutate_enum(
            self._activation(), Activations
        )

    def repair(self):
        # TODO
        pass

    def mutate(self):
        random.choice(
            [
                self._mutate_activation,
                self._mutate_filters,
                self._mutate_kernel_size,
                self._mutate_padding,
                self._mutate_strides,
                self._mutate_dilation_rate,
            ]
        )()

    def validate(self, repair=True):
        if not self._filters() > 0:
            return False

        if not self._single_stride() and not self._single_dilation_rate():
            return False

        if not self._strides()[0] > 0 or not self._strides()[1] > 0:
            return False

        return True

    def output_shape(self):
        return Conv2DLayer.conv2Doutputshape(
            input_size=self._input_size(),
            stride=self._strides(),
            kernel_size=self._kernel_size(),
            filter_count=self._filters(),
            padding=self._padding(),
        )

    @staticmethod
    def _valid_pad_output_shape(input, kernel, stride):
        return ((input - kernel) // stride) + 1

    @staticmethod
    def _same_pad_output_shape(input, stride):
        return ((input - 1) // stride) + 1

    @staticmethod
    def conv2Doutputshape(input_size, stride, kernel_size, filter_count, padding):
        if padding == PaddingArgs.SAME.value:
            X = Conv2DLayer._same_pad_output_shape(input_size[0], stride[0])
            Y = Conv2DLayer._same_pad_output_shape(input_size[1], stride[1])
            return X, Y, filter_count
        elif padding == PaddingArgs.VALID.value:
            X = Conv2DLayer._valid_pad_output_shape(
                input_size[0], kernel_size[0], stride[0]
            )
            Y = Conv2DLayer._valid_pad_output_shape(
                input_size[1], kernel_size[1], stride[1]
            )
            return X, Y, filter_count
        return 0, 0, 0

    def get_keras_layer(self):
        return keras.layers.Conv2D(
            self.args.get(Conv2DArgs.FILTERS.name),
            kernel_size=self.args.get(Conv2DArgs.KERNEL_SIZE.name),
            strides=self.args.get(Conv2DArgs.STRIDES.name),
            input_shape=self.args.get(Conv2DArgs.INPUT_SHAPE.name),
            activation=self.args.get(Conv2DArgs.ACTIVATION.name),
            padding=self.args.get(Conv2DArgs.PADDING.name),
            dilation_rate=self.args.get(Conv2DArgs.DILATION_RATE.name),
        )


class MaxPool2DLayer(ModelLayer):
    MUTATABLE_PARAMETERS = 3
    MAX_POOL_SIZE = 7
    MAX_STRIDE = 7

    def __init__(
        self,
        pool_size,
        strides=(1, 1),
        padding=PaddingArgs.SAME.value,
        name="MaxPool2D",
        input_shape=(0, 0, 0),
    ):
        super().__init__(name, input_shape=input_shape)
        self.args[MaxPool2DArgs.POOL_SIZE.name] = pool_size
        self.args[MaxPool2DArgs.STRIDES.name] = strides
        self.args[MaxPool2DArgs.PADDING.name] = padding

        self.outputshape.set(self.output_shape())
        self.validate(repair=True)

    def _pool_size(self):
        return self.args[MaxPool2DArgs.POOL_SIZE.name]

    def _strides(self):
        return self.args[MaxPool2DArgs.STRIDES.name]

    def _padding(self):
        return self.args[MaxPool2DArgs.PADDING.name]

    def _mutate_pool_size(self, operator=MutationOperators.SYNC_STEP):
        self.args[MaxPool2DArgs.POOL_SIZE.name] = mutate_tuple(
            self._pool_size(), 1, MaxPool2DLayer.MAX_POOL_SIZE, operator=operator
        )

    def _mutate_strides(self, operator=MutationOperators.SYNC_STEP):
        self.args[MaxPool2DArgs.STRIDES.name] = mutate_tuple(
            self._strides(), 1, MaxPool2DLayer.MAX_STRIDE, operator=operator
        )

    @staticmethod
    def _valid_pad_output_shape(input, pool, stride):
        return ((input - pool) // stride) + 1

    @staticmethod
    def _same_pad_output_shape(input, pool, stride):
        return MaxPool2DLayer._valid_pad_output_shape(input, pool, stride) + (
            1 if ((input - pool) % stride) else 0
        )

    def repair(self):
        for x, val in enumerate(self._strides()):
            if not val > 0:
                self.args[MaxPool2DArgs.STRIDES.name][x] = 1

        for x, val in enumerate(self._pool_size()):
            if not val > 0:
                self.args[MaxPool2DArgs.POOL_SIZE.name][x] = 1

    def mutate(self):
        random.choice([self._mutate_pool_size, self._mutate_strides])()

    def validate(self, repair=True):
        if not self._strides()[0] > 0 or not self._strides()[1] > 0:
            if repair:
                while not self.validate(repair):
                    self.repair()
            else:
                return False
        if not self._pool_size()[0] > 0 or not self._pool_size()[1] > 0:
            if repair:
                while not self.validate(repair):
                    self.repair()
            else:
                return False
        return True

    def output_shape(self):
        inp = self.inputshape.get()
        pool = self._pool_size()
        stri = self._strides()
        if self._padding() == PaddingArgs.SAME.value:
            x = self._same_pad_output_shape(inp[0], pool[0], stri[0])
            y = self._same_pad_output_shape(inp[1], pool[1], stri[1])
            return x, y, inp[2]
        elif self._padding() == PaddingArgs.VALID.value:
            x = self._valid_pad_output_shape(inp[0], pool[0], stri[0])
            y = self._valid_pad_output_shape(inp[1], pool[1], stri[1])
            return x, y, inp[2]
        return 0, 0, 0

    def get_keras_layer(self):
        return keras.layers.MaxPool2D(
            pool_size=self.args.get(MaxPool2DArgs.POOL_SIZE.name),
            strides=self.args.get(MaxPool2DArgs.STRIDES.name),
            padding=self.args.get(MaxPool2DArgs.PADDING.name),
        )


class MaxPool3DLayer(MaxPool2DLayer):
    def __init__(self, pool_size, strides, padding, input_shape=(0, 0, 0)):
        super().__init__(
            input_shape=input_shape,
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            name="MaxPool2D",
        )

    def repair(self):
        # TODO
        pass

    def mutate(self):
        # TODO
        pass

    def validate(self, repair=True):
        if (
            not self._strides()[0] > 0
            or not self._strides()[1] > 0
            or not self._strides()[2] > 0
        ):
            return False
        if (
            not self._pool_size()[0] > 0
            or not self._pool_size()[1] > 0
            or not self._pool_size()[2] > 0
        ):
            return False
        return True

    def output_shape(self):
        # TODO
        return self.inputshape

    def get_keras_layer(self):
        return keras.layers.MaxPool3D(
            pool_size=self.args.get(MaxPool2DArgs.POOL_SIZE.name),
            strides=self.args.get(MaxPool2DArgs.STRIDES.name),
            padding=self.args.get(MaxPool2DArgs.PADDING.name),
        )


class ReshapeLayer(ModelLayer):
    MUTATABLE_PARAMETERS = 0

    def __init__(self, target_shape, input_shape=(0, 0, 0)):
        super().__init__("Reshape", input_shape=input_shape)
        self.args[ReshapeArgs.TARGET_SHAPE.name] = target_shape

        self.outputshape.set(self.output_shape())
        self.validate(repair=True)

    def _target_shape(self):
        return self.args.get(ReshapeArgs.TARGET_SHAPE.name, self.inputshape.get())

    def _mutate_target_shape(self):
        self.args[ReshapeArgs.TARGET_SHAPE.name] = mutate_dimension(
            self._target_shape()
        )

    def repair(self):
        self.inputshape.set(self.outputshape.get())
        self._mutate_target_shape()

    def mutate(self):
        self._mutate_target_shape()

    def validate(self, repair=True):
        input_mag = dimension_mag(list(self.inputshape.get()))
        output_mag = dimension_mag(list(self.output_shape()))

        if not input_mag == output_mag:
            if repair:
                while not input_mag == output_mag:
                    self.repair()
                    input_mag = dimension_mag(list(self.inputshape.get()))
                    output_mag = dimension_mag(list(self.output_shape()))
            else:
                return False
        return True

    def output_shape(self):
        return self._target_shape()

    def get_keras_layer(self):
        target_shape = self.args.get(ReshapeArgs.TARGET_SHAPE.name)
        return keras.layers.Reshape(target_shape)


class DenseLayer(ModelLayer):
    def __init__(self, units, activation, input_shape=(0, 0, 0)):
        super().__init__("Dense", input_shape=input_shape)
        self.args[DenseArgs.UNITS.name] = units
        self.args[DenseArgs.ACTIVATION.name] = activation

        self.outputshape.set(self.output_shape())
        self.validate(repair=True)

    def _activation(self):
        return self.args[DenseArgs.ACTIVATION.name]

    def _units(self):
        return self.args[DenseArgs.UNITS.name]

    def _mutate_activation(self):
        self.args[DenseArgs.ACTIVATION.name] = mutate_enum(
            self._activation(), Activations
        )

    def repair(self):
        # TODO
        pass

    def mutate(self):
        random.choice([self._mutate_activation])()

    def validate(self, repair=True):
        # Dense layers take in a 1D tensor array, ie. previous layer should be a flatten layer
        if not len(self.inputshape.get()) == 1:
            return False

        return True

    def output_shape(self):
        return (self._units(),)

    def get_keras_layer(self):
        return keras.layers.Dense(
            self.args.get(DenseArgs.UNITS.name),
            activation=self.args.get(DenseArgs.ACTIVATION.name),
        )


class HiddenDenseLayer(DenseLayer):
    MAX_UNITS = 256
    MUTATABLE_PARAMETERS = 2

    def __init__(self, units, activation, input_shape=(0, 0, 0)):
        super().__init__(input_shape=input_shape, units=units, activation=activation)

    def _mutate_units(self):
        self.args[DenseArgs.UNITS.name] = mutate_int(
            self._unit(), 1, HiddenDenseLayer.MAX_UNITS
        )

    def repair(self):
        # TODO
        pass

    def mutate(self):
        random.choice([self._mutate_activation, self._mutate_units])()

    def validate(self, repair=True):
        super().validate()

    def output_shape(self):
        super().output_shape()

    def get_keras_layer(self):
        super().get_keras_layer()


class OutputDenseLayer(DenseLayer):
    MUTATABLE_PARAMETERS = 1

    def __init__(self, units, activation, input_shape=(0, 0, 0)):
        super().__init__(input_shape=input_shape, units=units, activation=activation)

    def repair(self):
        # TODO
        pass

    def mutate(self):
        super().mutate()

    def validate(self, repair=True):
        super().validate()

    def output_shape(self):
        super().output_shape()

    def get_keras_layer(self):
        super().get_keras_layer()


class FlattenLayer(ModelLayer):
    MUTATABLE_PARAMETERS = 0

    def __init__(self, input_shape=(0, 0, 0)):
        super().__init__("Flatten", input_shape=input_shape)

        self.outputshape.set(self.output_shape())
        self.validate(repair=True)

    def repair(self):
        # TODO
        pass

    def mutate(self):
        pass

    def validate(self, repair=True):
        return True

    def output_shape(self):
        return (dimension_mag(self.inputshape.get()),)

    def get_keras_layer(self):
        return keras.layers.Flatten()


class DropoutLayer(ModelLayer):
    MUTATABLE_PARAMETERS = 1
    MAX_RATE = 0.5

    def __init__(self, rate, input_shape=(0, 0, 0)):
        super().__init__("Dropout", input_shape=input_shape)
        self.args[DropoutArgs.RATE.name] = rate
        self.inputshape.set(input_shape)

        self.outputshape.set(self.output_shape())
        self.validate(repair=True)

    def _rate(self):
        return self.args[DropoutArgs.RATE.name]

    def _mutate_rate(self):
        self.args[DropoutArgs.RATE.name] = mutate_unit_interval(
            self._rate(), 0, DropoutLayer.MAX_RATE
        )

    def repair(self):
        # TODO
        pass

    def mutate(self):
        random.choice([self._mutate_rate])()

    def validate(self, repair=True):
        return True

    def output_shape(self):
        return self.inputshape.get()

    def get_keras_layer(self):
        rate = self.args.get(DropoutArgs.RATE.name)
        return keras.layers.Dropout(rate)