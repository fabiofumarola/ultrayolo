.. code:: ipython3

    import sys
    # add the code path
    sys.path.append('..')

.. code:: ipython3

    %load_ext autoreload
    %autoreload 2

Sample to train Yolo for object detection
-----------------------------------------

.. code:: ipython3

    from ultrayolo import YoloV3, callbacks
    from ultrayolo import datasets
    from pathlib import Path
    import tensorflow as tf
    from ultrayolo import losses
    import matplotlib.pyplot as plt

Define the parameters for the run

.. code:: ipython3

    image_shape = (256,256,3)
    batch_shape = 2
    max_objects = 100
    train_dataset_path = '../tests/data/manifest.txt'
    anchors = datasets.load_anchors('../tests/data/yolov3_anchors.txt')
    classes = datasets.load_classes('../tests/data/classes.txt')

Create the model

.. code:: ipython3

    model = YoloV3(image_shape, max_objects, 
                   anchors=anchors, num_classes=len(classes), 
                   training=True, backbone='MobileNetV2', base_grid_size=64)


.. parsed-literal::

    num pooling 1


.. code:: ipython3

    tf.keras.utils.plot_model(model.model, show_shapes=True)




.. image:: 4_train_example_files/4_train_example_8_0.png



Create the dataset

.. code:: ipython3

    train_dataset = datasets.YoloDatasetMultiFile(
        train_dataset_path, image_shape, max_objects, batch_shape, 
        model.anchors, model.masks, 64
    )

.. code:: ipython3

    print('num batches', len(train_dataset))


.. parsed-literal::

    num batches 2


Make optimizer and loss

.. code:: ipython3

    optimizer = model.get_optimizer('adam', 1e-4)
    model_loss = model.get_loss_function(num_batches = len(train_dataset))


.. parsed-literal::

      9596 MainThread using adam optimize


.. code:: ipython3

    model_loss




.. parsed-literal::

    [yolo_loss_large at 0x15bded350,
     yolo_loss_medium at 0x15bdeddd0,
     yolo_loss_small at 0x15be3a310]



compile the model

.. code:: ipython3

    model.compile(optimizer, model_loss, run_eagerly=True, summary=False)

Create the callbacks

.. code:: ipython3

    model_callbacks = callbacks.default_callbacks(model,
        run_path='./checkpoints', lrate_mode='exp_range',
        lrate_value=1e-5, verbose=0)

Set the model in transfer mode

.. code:: ipython3

    model.set_mode_transfer()


.. parsed-literal::

    164272 MainThread freeze backbone


.. code:: ipython3

    history = model.fit(train_dataset, train_dataset, 5, callbacks=model_callbacks)


.. parsed-literal::

    164428 MainThread training for 5 epochs on the dataset /Users/fumarolaf/git/ultrayolo/notebooks/../tests/data


.. parsed-literal::

    Train for 2 steps, validate for 2 steps
    Epoch 1/5
    2/2 [==============================] - 5s 2s/step - loss: 934.6033 - yolo_output_0_loss: 23.9461 - yolo_output_1_loss: 50.1620 - yolo_output_2_loss: 759.6201 - val_loss: 5264.7634 - val_yolo_output_0_loss: 1555.0167 - val_yolo_output_1_loss: 199.1157 - val_yolo_output_2_loss: 3409.6306
    Epoch 2/5
    2/2 [==============================] - 5s 3s/step - loss: 915.7275 - yolo_output_0_loss: 23.5616 - yolo_output_1_loss: 48.3204 - yolo_output_2_loss: 742.7931 - val_loss: 4735.4019 - val_yolo_output_0_loss: 1105.7004 - val_yolo_output_1_loss: 219.6431 - val_yolo_output_2_loss: 3308.8633
    Epoch 3/5
    2/2 [==============================] - 5s 2s/step - loss: 890.7028 - yolo_output_0_loss: 22.7319 - yolo_output_1_loss: 45.7045 - yolo_output_2_loss: 721.0179 - val_loss: 4345.3047 - val_yolo_output_0_loss: 814.5522 - val_yolo_output_1_loss: 223.1448 - val_yolo_output_2_loss: 3206.2117
    Epoch 4/5
    2/2 [==============================] - 5s 3s/step - loss: 859.4015 - yolo_output_0_loss: 21.0472 - yolo_output_1_loss: 42.8375 - yolo_output_2_loss: 694.0692 - val_loss: 4047.0514 - val_yolo_output_0_loss: 618.8140 - val_yolo_output_1_loss: 223.1597 - val_yolo_output_2_loss: 3103.4873
    Epoch 5/5
    1/2 [==============>...............] - ETA: 1s - loss: 815.9636 - yolo_output_0_loss: 17.7589 - yolo_output_1_loss: 40.5396 - yolo_output_2_loss: 656.0746

::


    ---------------------------------------------------------------------------

    NotFoundError                             Traceback (most recent call last)

    ~/miniconda3/envs/dl/lib/python3.7/site-packages/tensorflow_core/python/keras/engine/training_v2.py in on_epoch(self, epoch, mode)
        766     try:
    --> 767       yield epoch_logs
        768     finally:


    ~/miniconda3/envs/dl/lib/python3.7/site-packages/tensorflow_core/python/keras/engine/training_v2.py in fit(self, model, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_freq, max_queue_size, workers, use_multiprocessing, **kwargs)
        341                 training_context=training_context,
    --> 342                 total_epochs=epochs)
        343             cbks.make_logs(model, epoch_logs, training_result, ModeKeys.TRAIN)


    ~/miniconda3/envs/dl/lib/python3.7/site-packages/tensorflow_core/python/keras/engine/training_v2.py in run_one_epoch(model, iterator, execution_function, dataset_size, batch_size, strategy, steps_per_epoch, num_samples, mode, training_context, total_epochs)
        127       try:
    --> 128         batch_outs = execution_function(iterator)
        129       except (StopIteration, errors.OutOfRangeError):


    ~/miniconda3/envs/dl/lib/python3.7/site-packages/tensorflow_core/python/keras/engine/training_v2_utils.py in execution_function(input_fn)
         97     return nest.map_structure(_non_none_constant_value,
    ---> 98                               distributed_function(input_fn))
         99 


    ~/miniconda3/envs/dl/lib/python3.7/site-packages/tensorflow_core/python/keras/engine/training_v2_utils.py in distributed_function(input_iterator)
         84     outputs = strategy.experimental_run_v2(
    ---> 85         per_replica_function, args=args)
         86     # Out of PerReplica outputs reduce or pick values to return.


    ~/miniconda3/envs/dl/lib/python3.7/site-packages/tensorflow_core/python/distribute/distribute_lib.py in experimental_run_v2(self, fn, args, kwargs)
        762                                 convert_by_default=False)
    --> 763       return self._extended.call_for_each_replica(fn, args=args, kwargs=kwargs)
        764 


    ~/miniconda3/envs/dl/lib/python3.7/site-packages/tensorflow_core/python/distribute/distribute_lib.py in call_for_each_replica(self, fn, args, kwargs)
       1818     with self._container_strategy().scope():
    -> 1819       return self._call_for_each_replica(fn, args, kwargs)
       1820 


    ~/miniconda3/envs/dl/lib/python3.7/site-packages/tensorflow_core/python/distribute/distribute_lib.py in _call_for_each_replica(self, fn, args, kwargs)
       2163         replica_id_in_sync_group=constant_op.constant(0, dtypes.int32)):
    -> 2164       return fn(*args, **kwargs)
       2165 


    ~/miniconda3/envs/dl/lib/python3.7/site-packages/tensorflow_core/python/autograph/impl/api.py in wrapper(*args, **kwargs)
        257     with ag_ctx.ControlStatusCtx(status=ag_ctx.Status.UNSPECIFIED):
    --> 258       return func(*args, **kwargs)
        259 


    ~/miniconda3/envs/dl/lib/python3.7/site-packages/tensorflow_core/python/keras/engine/training_v2_utils.py in train_on_batch(model, x, y, sample_weight, class_weight, reset_metrics, standalone)
        432       sample_weights=sample_weights,
    --> 433       output_loss_metrics=model._output_loss_metrics)
        434 


    ~/miniconda3/envs/dl/lib/python3.7/site-packages/tensorflow_core/python/keras/engine/training_eager.py in train_on_batch(model, inputs, targets, sample_weights, output_loss_metrics)
        311           training=True,
    --> 312           output_loss_metrics=output_loss_metrics))
        313   if not isinstance(outs, list):


    ~/miniconda3/envs/dl/lib/python3.7/site-packages/tensorflow_core/python/keras/engine/training_eager.py in _process_single_batch(model, inputs, targets, output_loss_metrics, sample_weights, training)
        252               sample_weights=sample_weights,
    --> 253               training=training))
        254       if total_loss is None:


    ~/miniconda3/envs/dl/lib/python3.7/site-packages/tensorflow_core/python/keras/engine/training_eager.py in _model_loss(model, inputs, targets, output_loss_metrics, sample_weights, training)
        166         if hasattr(loss_fn, 'reduction'):
    --> 167           per_sample_losses = loss_fn.call(targets[i], outs[i])
        168           weighted_losses = losses_utils.compute_weighted_loss(


    ~/miniconda3/envs/dl/lib/python3.7/site-packages/tensorflow_core/python/keras/losses.py in call(self, y_true, y_pred)
        220           y_pred, y_true)
    --> 221     return self.fn(y_true, y_pred, **self._fn_kwargs)
        222 


    ~/git/ultrayolo/ultrayolo/losses.py in __call__(self, y_true, y_pred, **kvargs)
        329         self.count_batches.assign_add(1)
    --> 330         self.save_metrics()
        331 


    ~/miniconda3/envs/dl/lib/python3.7/site-packages/tensorflow_core/python/eager/def_function.py in __call__(self, *args, **kwds)
        567     else:
    --> 568       result = self._call(*args, **kwds)
        569 


    ~/miniconda3/envs/dl/lib/python3.7/site-packages/tensorflow_core/python/eager/def_function.py in _call(self, *args, **kwds)
        605       # run the first trace but we should fail if variables are created.
    --> 606       results = self._stateful_fn(*args, **kwds)
        607       if self._created_variables:


    ~/miniconda3/envs/dl/lib/python3.7/site-packages/tensorflow_core/python/eager/function.py in __call__(self, *args, **kwargs)
       2362       graph_function, args, kwargs = self._maybe_define_function(args, kwargs)
    -> 2363     return graph_function._filtered_call(args, kwargs)  # pylint: disable=protected-access
       2364 


    ~/miniconda3/envs/dl/lib/python3.7/site-packages/tensorflow_core/python/eager/function.py in _filtered_call(self, args, kwargs)
       1610                            resource_variable_ops.BaseResourceVariable))),
    -> 1611         self.captured_inputs)
       1612 


    ~/miniconda3/envs/dl/lib/python3.7/site-packages/tensorflow_core/python/eager/function.py in _call_flat(self, args, captured_inputs, cancellation_manager)
       1691       return self._build_call_outputs(self._inference_function.call(
    -> 1692           ctx, args, cancellation_manager=cancellation_manager))
       1693     forward_backward = self._select_forward_and_backward_functions(


    ~/miniconda3/envs/dl/lib/python3.7/site-packages/tensorflow_core/python/eager/function.py in call(self, ctx, args, cancellation_manager)
        544               attrs=("executor_type", executor_type, "config_proto", config),
    --> 545               ctx=ctx)
        546         else:


    ~/miniconda3/envs/dl/lib/python3.7/site-packages/tensorflow_core/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
         66       message = e.message
    ---> 67     six.raise_from(core._status_to_exception(e.code, message), None)
         68   except TypeError as e:


    ~/miniconda3/envs/dl/lib/python3.7/site-packages/six.py in raise_from(value, from_value)


    NotFoundError:  Resource localhost/_AnonymousVar408/N10tensorflow22SummaryWriterInterfaceE does not exist.
    	 [[{{node cond/then/_0/yolo_loss_large_xy_loss/write_summary}}]] [Op:__inference_save_metrics_15892]
    
    Function call stack:
    save_metrics


    
    During handling of the above exception, another exception occurred:


    KeyError                                  Traceback (most recent call last)

    <ipython-input-21-0f662bf2a06d> in <module>
    ----> 1 history = model.fit(train_dataset, train_dataset, 5, callbacks=model_callbacks)
    

    ~/git/ultrayolo/ultrayolo/ultrayolo.py in fit(self, train_dataset, val_dataset, epochs, initial_epoch, callbacks, workers, max_queue_size)
        233                               max_queue_size=64,
        234                               initial_epoch=initial_epoch,
    --> 235                               verbose=1)
        236 
        237     def save(self, path, save_format='h5'):


    ~/miniconda3/envs/dl/lib/python3.7/site-packages/tensorflow_core/python/keras/engine/training.py in fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_freq, max_queue_size, workers, use_multiprocessing, **kwargs)
        817         max_queue_size=max_queue_size,
        818         workers=workers,
    --> 819         use_multiprocessing=use_multiprocessing)
        820 
        821   def evaluate(self,


    ~/miniconda3/envs/dl/lib/python3.7/site-packages/tensorflow_core/python/keras/engine/training_v2.py in fit(self, model, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_freq, max_queue_size, workers, use_multiprocessing, **kwargs)
        395                       total_epochs=1)
        396                   cbks.make_logs(model, epoch_logs, eval_result, ModeKeys.TEST,
    --> 397                                  prefix='val_')
        398 
        399     return model.history


    ~/miniconda3/envs/dl/lib/python3.7/contextlib.py in __exit__(self, type, value, traceback)
        128                 value = type()
        129             try:
    --> 130                 self.gen.throw(type, value, traceback)
        131             except StopIteration as exc:
        132                 # Suppress StopIteration *unless* it's the same exception that


    ~/miniconda3/envs/dl/lib/python3.7/site-packages/tensorflow_core/python/keras/engine/training_v2.py in on_epoch(self, epoch, mode)
        769       if mode == ModeKeys.TRAIN:
        770         # Epochs only apply to `fit`.
    --> 771         self.callbacks.on_epoch_end(epoch, epoch_logs)
        772       self.progbar.on_epoch_end(epoch, epoch_logs)
        773 


    ~/miniconda3/envs/dl/lib/python3.7/site-packages/tensorflow_core/python/keras/callbacks.py in on_epoch_end(self, epoch, logs)
        300     logs = logs or {}
        301     for callback in self.callbacks:
    --> 302       callback.on_epoch_end(epoch, logs)
        303 
        304   def on_train_batch_begin(self, batch, logs=None):


    ~/miniconda3/envs/dl/lib/python3.7/site-packages/tensorflow_core/python/keras/callbacks.py in on_epoch_end(self, epoch, logs)
        990           self._save_model(epoch=epoch, logs=logs)
        991       else:
    --> 992         self._save_model(epoch=epoch, logs=logs)
        993     if self.model._in_multi_worker_mode():
        994       # For multi-worker training, back up the weights and current training


    ~/miniconda3/envs/dl/lib/python3.7/site-packages/tensorflow_core/python/keras/callbacks.py in _save_model(self, epoch, logs)
       1009                   int) or self.epochs_since_last_save >= self.period:
       1010       self.epochs_since_last_save = 0
    -> 1011       filepath = self._get_file_path(epoch, logs)
       1012 
       1013       try:


    ~/miniconda3/envs/dl/lib/python3.7/site-packages/tensorflow_core/python/keras/callbacks.py in _get_file_path(self, epoch, logs)
       1053     if not self.model._in_multi_worker_mode(
       1054     ) or multi_worker_util.should_save_checkpoint():
    -> 1055       return self.filepath.format(epoch=epoch + 1, **logs)
       1056     else:
       1057       # If this is multi-worker training, and this worker should not


    KeyError: 'val_loss'


Evaluate model Loss
-------------------

.. code:: ipython3

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(loss) + 1)
    
    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()

Yolo loss for large-sized objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    loss = history.history['yolo_output_0_loss']
    val_loss = history.history['val_yolo_output_0_loss']
    
    epochs = range(1, len(loss) + 1)
    
    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss large size object')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss large size object')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()

Yolo loss for medium-sized objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    loss = history.history['yolo_output_1_loss']
    val_loss = history.history['val_yolo_output_1_loss']
    
    epochs = range(1, len(loss) + 1)
    
    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss medium size object')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss medium size object')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()

Yolo loss for small-sized objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    loss = history.history['yolo_output_2_loss']
    val_loss = history.history['val_yolo_output_2_loss']
    
    epochs = range(1, len(loss) + 1)
    
    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss small size object')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss small size object')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()

model.save('./save_model/model.h5')
