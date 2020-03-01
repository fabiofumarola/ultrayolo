.. code:: ipython3

    import sys
    if '..' not in sys.path:
        sys.path.append('..')

%load_ext autoreload
%autoreload 2

.. code:: ipython3

    import tensorflow as tf
    from ultrayolo import YoloV3, datasets
    from ultrayolo.helpers import draw
    from pathlib import Path
    import numpy as np
    import logging
    
    from matplotlib import patches
    import matplotlib.pyplot as plt

Predict using a custom pretrained model
=======================================

The classes contained in the dataset

.. code:: ipython3

    classes_dict = datasets.load_classes('./mini_classes.txt', True)
    target_shape = (512, 512, 3)
    max_objects = 100
    num_classes = len(classes_dict)
    print(f'number of classes {num_classes}')
    classes_dict


.. parsed-literal::

    number of classes 3




.. parsed-literal::

    {0: 'bear', 1: 'toaster', 2: 'hair drier'}



.. code:: ipython3

    anchors = datasets.load_anchors('./mini_anchors.txt')
    anchors




.. parsed-literal::

    array([[ 79.2,  77. ],
           [100.1,  90.2],
           [104.5,  89.1],
           [112.2,  99. ],
           [178.2, 193.6],
           [207.9, 199.1],
           [227.7, 205.7],
           [235.4, 204.6],
           [237.6, 202.4]])



.. code:: ipython3

    model = YoloV3(target_shape, max_objects, anchors=anchors,
                   num_classes=num_classes, score_threshold=0.5, iou_threshold=0.5, 
                   training=False, backbone='DenseNet121')

tf.keras.utils.plot_model(model.model, show_shapes=True)

Load the weights
----------------

load a custom model from `here <add%20a%20valid%20link%20here>`__

.. code:: ipython3

    w_path = Path('weights_val.118-7.478.h5')
    model.load_weights(w_path)


.. parsed-literal::

     54385 MainThread loading checkpoint from /Users/fumarolaf/git/ultrayolo/notebooks/weights_val.118-7.478.h5


Predict
-------

we predict the objects using an image from the web. You can try with
your.

Bear
~~~~

.. code:: ipython3

    img = datasets.open_image('https://upload.wikimedia.org/wikipedia/commons/thumb/5/5d/Kamchatka_Brown_Bear_near_Dvuhyurtochnoe_on_2015-07-23.jpg/1200px-Kamchatka_Brown_Bear_near_Dvuhyurtochnoe_on_2015-07-23.jpg')
    img_pad = datasets.pad_to_fixed_size(img, target_shape)
    img_resized = datasets.resize(img, target_shape)
    #preprocess the image
    x = np.divide(img_pad, 255.)
    x = np.expand_dims(x, 0)
    plt.imshow(x[0])
    x.shape




.. parsed-literal::

    (1, 512, 512, 3)




.. image:: 2_predict_custom_dataset_files/2_predict_custom_dataset_11_1.png


Perform the prediction
~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    boxes, scores, classes, sel = model.predict(x)
    print(f'found {sel[0]} objects')


.. parsed-literal::

    found 1 objects


Show the image with the discovered objects

.. code:: ipython3

    ax = draw.show_img(img_resized, figsize=(8,8))
    for i,b in enumerate(boxes[0,:sel[0]]):
        draw.rect(ax, b, color='#9cff1d')
        name_score = f'{classes_dict[classes[0, i]]} {str(round(scores[0,i],2))}'
        draw.text(ax, b[:2], name_score, sz=12)
        print(classes_dict[classes[0, i]], scores[0,i])
        
    plt.show()


.. parsed-literal::

    bear 0.9999901



.. image:: 2_predict_custom_dataset_files/2_predict_custom_dataset_15_1.png


Perform a prediction for the class Hair Drier
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    img = datasets.open_image('https://cdn.mos.cms.futurecdn.net/UA2XKzB6zdzm486hC4TXPf.jpg')
    # img = datasets.open_image('https://reviewed-com-res.cloudinary.com/image/fetch/s--IAWIW5ff--/b_white,c_limit,cs_srgb,f_auto,fl_progressive.strip_profile,g_center,q_auto,w_1200/https://reviewed-production.s3.amazonaws.com/1521023219800/toaster-newhhero.jpg')
    img_pad = datasets.pad_to_fixed_size(img, target_shape)
    img_resized = datasets.resize(img, target_shape)
    #preprocess the image
    x = np.divide(img_pad, 255.)
    x = np.expand_dims(x, 0)
    plt.imshow(x[0])
    x.shape




.. parsed-literal::

    (1, 512, 512, 3)




.. image:: 2_predict_custom_dataset_files/2_predict_custom_dataset_17_1.png


.. code:: ipython3

    boxes, scores, classes, sel = model.predict(x)
    print(f'found {sel[0]} objects')


.. parsed-literal::

    found 1 objects


.. code:: ipython3

    ax = draw.show_img(img_resized, figsize=(8,8))
    for i,b in enumerate(boxes[0,:sel[0]]):
        draw.rect(ax, b, color='#9cff1d')
        name_score = f'{classes_dict[classes[0, i]]} {str(round(scores[0,i],2))}'
        draw.text(ax, b[:2], name_score, sz=12)
        print(classes_dict[classes[0, i]], scores[0,i])
        
    plt.show()


.. parsed-literal::

    hair drier 0.9821011



.. image:: 2_predict_custom_dataset_files/2_predict_custom_dataset_19_1.png


Perform a prediction for the class Toaster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    img = datasets.open_image('https://reviewed-com-res.cloudinary.com/image/fetch/s--IAWIW5ff--/b_white,c_limit,cs_srgb,f_auto,fl_progressive.strip_profile,g_center,q_auto,w_1200/https://reviewed-production.s3.amazonaws.com/1521023219800/toaster-newhhero.jpg')
    img_pad = datasets.pad_to_fixed_size(img, target_shape)
    img_resized = datasets.resize(img, target_shape)
    #preprocess the image
    x = np.divide(img_pad, 255.)
    x = np.expand_dims(x, 0)
    plt.imshow(x[0])
    x.shape




.. parsed-literal::

    (1, 512, 512, 3)




.. image:: 2_predict_custom_dataset_files/2_predict_custom_dataset_21_1.png


.. code:: ipython3

    boxes, scores, classes, sel = model.predict(x)
    print(f'found {sel[0]} objects')


.. parsed-literal::

    found 2 objects


.. code:: ipython3

    ax = draw.show_img(img_resized, figsize=(8,8))
    for i,b in enumerate(boxes[0,:sel[0]]):
        draw.rect(ax, b, color='#9cff1d')
        name_score = f'{classes_dict[classes[0, i]]} {str(round(scores[0,i],2))}'
        draw.text(ax, b[:2], name_score, sz=12)
        print(classes_dict[classes[0, i]], scores[0,i])
        
    plt.show()


.. parsed-literal::

    toaster 0.89197004
    toaster 0.5734626



.. image:: 2_predict_custom_dataset_files/2_predict_custom_dataset_23_1.png

