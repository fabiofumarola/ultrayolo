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
        if epoch % 10 == 0:
            model.save(file_path, save_format='h5')

    return LambdaCallback(on_epoch_end=save_model_callback)


def lr_scheduler(lrate_mode, lrate_value, verbose=1):
    if (lrate_mode == 'cyclic') or (lrate_mode == 'exp_range'):
        lrate_fn = cyclic_learning_rate(
            learning_rate=lrate_value,
        # one order greather that the selected
            max_lr=lrate_value * 1e+1,
            step_size=20,
            mode=lrate_mode)
        return LearningRateScheduler(lrate_fn, verbose=verbose)

    elif lrate_mode == 'reduce_on_plateau':
        return ReduceLROnPlateau(monitor='val_loss',
                                 factor=0.5,
                                 patience=10,
                                 min_lr=lrate_value / 1e+2,
                                 verbose=verbose)


def default_callbacks(model, run_path, lrate_mode, lrate_value, verbose=1):
    """create the callbacks for the model

    Arguments:
        model {tf.keras.Model} -- a valid tensorflow model
        run_path {str} -- the path to save the checkpoints
        lrate_mode {str} -- the mode for the learning rate scheduler
            (values: cyclic, exp_range, reduce_on_plateau)
        lrate_value {float} -- the initial value for the learning rate
        verbose {int} -- 0 for no verbose, 1 for verbose (default: 1)
    Returns:
        list -- a list of tf.keras.callbacks
    """
    run_path = Path(run_path)
    run_path_str = str(run_path.absolute())

    callbacks = [
        ModelCheckpoint(run_path_str + '/weights_val.{epoch:03d}-{val_loss:.3f}.h5',
                        verbose=verbose,
                        save_best_only=True,
                        monitor='val_loss'),
        TensorBoard(run_path_str),
        ModelCheckpoint(run_path_str + '/weights_train_best.h5',
                        verbose=verbose,
                        save_best_only=True,
                        monitor='loss'),
        lr_scheduler(lrate_mode, lrate_value, verbose)
    ]
    return callbacks
