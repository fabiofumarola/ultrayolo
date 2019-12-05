from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LambdaCallback, ReduceLROnPlateau, LearningRateScheduler
from pathlib import Path
from ..learningrates import cyclic_learning_rate


def save_model(model, checkpoints_path):
    """method to save the model

    Arguments:
        model {tf.keras.Model} -- the model to save
        checkpoints_path {str} -- the path to save the chckpoints

    Returns:
        [tf.keras.callbacks.LambdaCallback]
    """
    save_checkpoints_path = str(Path(checkpoints_path).absolute())

    def save_model_callback(epoch, logs):
        loss = logs['loss']
        file_path = save_checkpoints_path + "/" + \
            f"weights_train.{epoch:03d}-{loss:.3f}.h5"
        print(file_path)
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
    """create the callbacks for the model

    Arguments:
        model {tf.keras.Model} -- a valid tensorflow model
        checkpoints_path {str} -- the path to save the checkpoints
        lrate_mode {str} -- the mode for the learning rate scheduler (values: cyclic, exp_range, reduce_on_plateau)
        lrate_value {float} -- the initial value for the learning rate

    Returns:
        list  -- a list of tf.keras.callbacks
    """
    save_checkpoints_path = str(Path(checkpoints_path).absolute())

    callbacks = [
        ModelCheckpoint(save_checkpoints_path + '/weights.{epoch:03d}-{loss:.3f}.h5',
                        verbose=1, save_best_only=True, monitor='val_loss'),
        TensorBoard(save_checkpoints_path),
        ModelCheckpoint(save_checkpoints_path + '/weights.{epoch:03d}-{loss:.3f}.h5',
                        verbose=1, save_best_only=True, monitor='loss'),
        lr_scheduler(lrate_mode, lrate_value)
    ]
    return callbacks
