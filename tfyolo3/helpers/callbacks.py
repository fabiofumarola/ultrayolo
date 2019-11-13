from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LambdaCallback, ReduceLROnPlateau, LearningRateScheduler
from pathlib import Path
from ..learningrates import cyclic_learning_rate


def save_model(model, checkpoints_path):
    save_checkpoints_path = str(Path(checkpoints_path).absolute())

    def save_model_callback(epoch, logs):
        loss = logs['loss']
        file_path = save_checkpoints_path + "/" + \
            f"weights_train.{epoch:03d}-{loss:.3f}.h5"
        if epoch % 10 == 0:
            model.save(file_path, save_format='h5')

    return LambdaCallback(on_epoch_end=save_model_callback)


def lr_scheduler(lrate_mode, lrate_value):
    if (lrate_mode == 'cyclic') or (lrate_mode == 'exp_range'):
        lrate_fn = cyclic_learning_rate(
            learning_rate=lrate_value,
            # one order greather that the selected
            max_lr=lrate_value * 1e+1,
            step_size=20,
            mode=lrate_mode)
        return LearningRateScheduler(lrate_fn, verbose=1)

    elif lrate_mode == 'reduce_on_plateau':
        return ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=lrate_value / 1e+2,
            verbose=1)


def default_callbacks(model, checkpoints_path, lrate_mode, lrate_value):
    save_checkpoints_path = str(Path(checkpoints_path).absolute())

    callbacks = [
        ModelCheckpoint(save_checkpoints_path + '/weights.{epoch:03d}-{loss:.3f}.h5',
                        verbose=1, save_best_only=True, monitor='val_loss'),
        TensorBoard(save_checkpoints_path),
        save_model(model, save_checkpoints_path),
        lr_scheduler(lrate_mode, lrate_value)
    ]
    return callbacks
