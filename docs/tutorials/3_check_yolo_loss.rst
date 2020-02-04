.. code:: ipython3

    import sys
    if '..' not in sys.path:
        sys.path.append('..')
    
    from ultrayolo import YoloV3, losses
    from ultrayolo.datasets import YoloDatasetSingleFile,common
    from ultrayolo.helpers import draw
    from pathlib import Path
    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt

Yolo Loss Tutorial
==================

The first create some random anchors and use the default masks

.. code:: ipython3

    x = np.arange(5, 46, 5)
    anchors = np.array(list(zip(x,x)), dtype=np.float32)
    anchors[:,1] += np.random.randint(0, 10, 9)
    anchors




.. parsed-literal::

    array([[ 5., 10.],
           [10., 14.],
           [15., 19.],
           [20., 22.],
           [25., 32.],
           [30., 37.],
           [35., 44.],
           [40., 42.],
           [45., 45.]], dtype=float32)



.. code:: ipython3

    masks = YoloV3.default_masks
    masks




.. parsed-literal::

    array([[6, 7, 8],
           [3, 4, 5],
           [0, 1, 2]])



Download a toy dataset
----------------------

make the following cell executable to download the toy dataset

%%capture

!wget https://www.dropbox.com/s/drjtl8hjy42lq42/toy_dataset.tar.gz
!tar -xf toy_dataset.tar.gz

And load the dataset using the SequenceDataset
----------------------------------------------

.. code:: ipython3

    filepath = Path('./toy_dataset/data_annotations_train.txt')
    target_shape = (256, 256, 3)
    batch_size = 2
    gridlen = 8
    is_training = True
    max_objects = 10
    
    
    train_seq = YoloDatasetSingleFile(filepath,
                                      target_shape,
                                      max_objects,
                                      batch_size, 
                                      anchors,
                                      masks,
                                      is_training=True)

.. code:: ipython3

    train_seq.classes




.. parsed-literal::

    ['chat', 'rat', 'test', 'none']



Now we take a batch from the dataset
------------------------------------

.. code:: ipython3

    x_true, y_true_grids = train_seq[0]

The batch contains: - 2 images

.. code:: ipython3

    x_true.shape




.. parsed-literal::

    (2, 256, 256, 3)



-  3 grids

.. code:: ipython3

    for i in range(len(y_true_grids)): 
        print(i, '-->', y_true_grids[i].shape, target_shape[0] / y_true_grids[i].shape[1])


.. parsed-literal::

    0 --> (2, 8, 8, 3, 9) 32.0
    1 --> (2, 16, 16, 3, 9) 16.0
    2 --> (2, 32, 32, 3, 9) 8.0


The third value plotted represents the size in number of pixel of grid
cells

Check that the dataset transformed is correct
---------------------------------------------

.. code:: ipython3

    for img_idx in range(len(x_true)):
        print('Show annotations for image', img_idx)
        img = x_true[img_idx]
    
        for i in range(len(y_true_grids)):
            y_data_grid_img = y_true_grids[i][img_idx]
            
            ax = draw.show_img(img)
            grid_len = y_data_grid_img.shape[1]
            draw.grid(ax, img.shape[:2], grid_len)
    
            grid_cell_size = target_shape[1] / grid_len
            
            for grid_y, grid_x, box in np.argwhere(np.sum(y_data_grid_img[..., :4], axis=-1) > 0):
                box_xyxy = (y_data_grid_img[grid_y,grid_x,box, :4] * target_shape[0]).astype(int)
                class_id = np.argwhere(y_data_grid_img[grid_y,grid_x,box, 5:])[0][0]
                draw.rect(ax, box_xyxy, 'white', 1)
    
                rect_resp = np.array([grid_x, grid_y]) * grid_cell_size
                rect_resp = np.concatenate([rect_resp, rect_resp + grid_cell_size])
                draw.rect(ax, rect_resp, 'blue', 2)
    
                draw.point(ax, common.to_center_width_height(box_xyxy)[:2])
            plt.show()


.. parsed-literal::

    Show annotations for image 0



.. image:: 3_check_yolo_loss_files/3_check_yolo_loss_17_1.png



.. image:: 3_check_yolo_loss_files/3_check_yolo_loss_17_2.png



.. image:: 3_check_yolo_loss_files/3_check_yolo_loss_17_3.png


.. parsed-literal::

    Show annotations for image 1



.. image:: 3_check_yolo_loss_files/3_check_yolo_loss_17_5.png



.. image:: 3_check_yolo_loss_files/3_check_yolo_loss_17_6.png



.. image:: 3_check_yolo_loss_files/3_check_yolo_loss_17_7.png


Create the model

.. code:: ipython3

    model = YoloV3(target_shape, max_objects, anchors=anchors, num_classes=train_seq.num_classes, training=True)

.. code:: ipython3

    tf.keras.utils.plot_model(model.model, show_shapes=True)




.. image:: 3_check_yolo_loss_files/3_check_yolo_loss_20_0.png



Evaluate how the loss works
---------------------------

We consider two cases:

1. when we got an initialized network the predictions should be around
   0.5 (max entropy)
2. when we give as prediction the right labels the loss should be close
   to zero

.. code:: ipython3

    y_pred_grids = model(x_true)
    for y_pred in y_pred_grids:
        print(y_pred.shape)


.. parsed-literal::

    (2, 8, 8, 3, 9)
    (2, 16, 16, 3, 9)
    (2, 32, 32, 3, 9)


We take i=0 since all the images are in the first grid

.. code:: ipython3

    i = 0
    y_true = y_true_grids[i]
    y_pred = y_pred_grids[i]
    anchors_masks = anchors[masks[i]]
    img_size = target_shape[0]
    loss_fn = losses.make_loss(train_seq.num_classes, anchors, masks, img_size)
    ignore_threshold = 0.7

First Case
~~~~~~~~~~

.. code:: ipython3

    from ultrayolo.losses import Loss

.. code:: ipython3

    # 1. transform all pred outputs
    # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
    anchors_masks_scaled = anchors_masks / img_size
    pred_xyxy, pred_obj, pred_class, pred_xywh = losses.process_predictions(
                tf.cast(y_pred, tf.float32), train_seq.num_classes, anchors_masks_scaled
    )
    pred_xy = pred_xywh[..., 0:2]
    pred_wh = pred_xywh[..., 2:4]

We expect that considering the variable ``pred_xywh`` the predictions
should be: - for xy in in average 0.5 - for wh close to 0 - for xy1, xy2
close to 0.5

While considering ``pred_xyxy`` it should be around 0.5

.. code:: ipython3

    print('average xy', tf.reduce_mean(pred_xy))
    print('average hw', tf.reduce_mean(pred_wh))
    print('average xyxy', tf.reduce_mean(pred_xyxy))


.. parsed-literal::

    average xy tf.Tensor(0.5, shape=(), dtype=float32)
    average hw tf.Tensor(3.8821852e-11, shape=(), dtype=float32)
    average xyxy tf.Tensor(0.5, shape=(), dtype=float32)


This is valid for all the objecteness and classes

.. code:: ipython3

    print('average pred_obj', tf.reduce_mean(pred_obj))
    print('average pred_class', tf.reduce_mean(pred_class))


.. parsed-literal::

    average pred_obj tf.Tensor(0.5, shape=(), dtype=float32)
    average pred_class tf.Tensor(0.5, shape=(), dtype=float32)


.. code:: ipython3

    # 2. transform all true outputs
    # y_true: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
    true_box_xyxy, true_obj, true_class = tf.split(
        y_true, (4, 1, train_seq.num_classes), axis=-1)
    true_xy = (true_box_xyxy[..., 0:2] + true_box_xyxy[..., 2:4]) / 2
    true_wh = true_box_xyxy[..., 2:4] - true_box_xyxy[..., 0:2]

.. code:: ipython3

    box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

3. inverting the pred box equations, to make it comparable with the
   transformations done for the predictions

.. code:: ipython3

    grid_size = tf.shape(y_true)[1]
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
    true_xy = true_xy * tf.cast(grid_size, tf.float32) - \
        tf.cast(grid, tf.float32)
    
    true_wh = tf.math.log(true_wh / anchors_masks_scaled)
    true_wh = tf.where(tf.math.is_inf(true_wh),
                       tf.zeros_like(true_wh), true_wh)

The line 8 contains the opposite transformation made for the predictions

::

   box_wh = tf.exp(box_wh) * anchors_masks

The masks are used to: 1. separate the boxes that contain objects and
should be considered in the objects loss 2. from the boxes that not
contain objects and should be considered in the **no object loss**

.. code:: ipython3

    # 4. calculate all masks
    obj_mask = tf.squeeze(true_obj, -1)
    # ignore false positive when iou is over threshold
    true_box_mask = tf.boolean_mask(
        true_box_xyxy, tf.cast(obj_mask, tf.bool))
    best_iou = tf.reduce_max(Loss.broadcast_iou(
        pred_xyxy, true_box_mask), axis=-1)
    ignore_mask = tf.cast(best_iou < ignore_threshold, tf.float32)

Compute all the losses

-  xy, wh only with respect the objects that contains elements

.. code:: ipython3

    xy_loss = obj_mask * box_loss_scale * \
        tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
    wh_loss = obj_mask * box_loss_scale * \
        tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)

-  the object and no object loss

You can check the whenever the loss is different to zero in the
``obj_loss`` is zero in the ``no_obj_loss`` and vice-versa

.. code:: ipython3

    obj_cross_entropy = tf.keras.metrics.binary_crossentropy(
        true_obj, pred_obj, from_logits=False)
    obj_loss = obj_mask * obj_cross_entropy
    no_obj_loss = (1 - obj_mask) * ignore_mask * obj_cross_entropy

-  The class loss is computed only for the cells the contains objects

.. code:: ipython3

    class_loss = obj_mask * tf.keras.metrics.binary_crossentropy(
                true_class, pred_class, from_logits=False)

-  everything is reduced to one value per image

.. code:: ipython3

    xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
    wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
    obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
    no_obj_loss = tf.reduce_sum(no_obj_loss, axis=(1, 2, 3))
    class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))
    
    loss = xy_loss + wh_loss + obj_loss + no_obj_loss + class_loss
    
    loss




.. parsed-literal::

    <tf.Tensor: shape=(2,), dtype=float32, numpy=array([197.02797, 193.08067], dtype=float32)>



Second Case
~~~~~~~~~~~

-  y_true == y_pred

.. code:: ipython3

    i = 0
    y_true = y_true_grids[i]
    y_pred = y_true_grids[i]

-  Remember that y_pred is in format xy_min xy_max

.. code:: ipython3

    pred_xyxy, pred_obj, pred_class = tf.split(
            y_pred, (4, 1, train_seq.num_classes), axis=-1)
    
    pred_xy = (pred_xyxy[..., 0:2] + pred_xyxy[..., 2:4]) / 2
    pred_wh = pred_xyxy[..., 2:4] - pred_xyxy[..., 0:2]
    
    pred_xywh = tf.concat((pred_xy, pred_wh), axis=-1)

.. code:: ipython3

    true_box_xyxy, true_obj, true_class = tf.split(
                y_true, (4, 1, train_seq.num_classes), axis=-1)
    true_xy = (true_box_xyxy[..., 0:2] + true_box_xyxy[..., 2:4]) / 2
    true_wh = true_box_xyxy[..., 2:4] - true_box_xyxy[..., 0:2]

.. code:: ipython3

    box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]
    
    # 4. calculate all masks
    obj_mask = tf.squeeze(true_obj, -1)
    # ignore false positive when iou is over threshold
    true_box_mask = tf.boolean_mask(
        true_box_xyxy, tf.cast(obj_mask, tf.bool))
    best_iou = tf.reduce_max(Loss.broadcast_iou(
        pred_xyxy, true_box_mask), axis=-1)
    ignore_mask = tf.cast(best_iou < ignore_threshold, tf.float32)

.. code:: ipython3

    # 5. compute all the losses
    xy_loss = obj_mask * box_loss_scale * \
        tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
    wh_loss = obj_mask * box_loss_scale * \
        tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)

.. code:: ipython3

    obj_cross_entropy = tf.keras.metrics.binary_crossentropy(
        true_obj, pred_obj, from_logits=False)
    obj_loss = obj_mask * obj_cross_entropy
    no_obj_loss = (1 - obj_mask) * ignore_mask * obj_cross_entropy
    
    class_loss = obj_mask * tf.keras.metrics.binary_crossentropy(
        true_class, pred_class, from_logits=False)

.. code:: ipython3

    xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
    wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
    obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
    no_obj_loss = tf.reduce_sum(no_obj_loss, axis=(1, 2, 3))
    class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))
    
    loss = xy_loss + wh_loss + obj_loss + no_obj_loss + class_loss
    loss




.. parsed-literal::

    <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0., 0.], dtype=float32)>



The loss is 0 when the prediction is equal to the true values

Conclusion
----------

we have verified that the loss: - return max entropy value when the
network is initialized, and - return 0 when the y_pred is equal to
y_true
