from tensornas.blocktemplates.blockarchitectures import GhostNetBlockArchitecture
from demos.DemoMNISTInput import *
from tensornas.core.util import list_available_blocks

print("##########################################")
print("Testing Ghost Net block architecture")
print("##########################################")

list_available_blocks()

model = GhostNetBlockArchitecture.GhostNetBlockArchitecture(
    input_tensor_shape, mnist_class_count
)

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
