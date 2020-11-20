from tensornas.blocktemplates.blockarchitectures import SqueezeNetBlockArchitecture
from demos.DemoMNISTInput import *
from tensornas.core.util import list_available_blocks, save_model

print("##########################################")
print("Testing Squeeze Net block architecture")
print("##########################################")

list_available_blocks()

model = SqueezeNetBlockArchitecture.SqueezeNetBlockArchitecture(
    input_tensor_shape, mnist_class_count
)

save_model(model, "squeezenet.tflite")

model.print()

metrics = model.evaluate(
    train_data=images_train,
    train_labels=labels_train,
    test_data=images_test,
    test_labels=labels_test,
    epochs=2,
    batch_size=32,
    steps=5,
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print(metrics)

model.mutate(verbose=True)

model.print()

metrics = model.evaluate(
    train_data=images_train,
    train_labels=labels_train,
    test_data=images_test,
    test_labels=labels_test,
    epochs=2,
    batch_size=32,
    steps=5,
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print(metrics)

print("Done")