=====
Usage
=====

To use ultrayolo in a project::

    from ultrayolo import YoloV3
    from ultrayolo import datasets
    from ultrayolo.losses import Loss

    image_shape = (256,256,3)
    max_objects = 100
    anchors = datasets.load_anchors('the path of the anchors')
    classes = datasets.load_classes('the file of the classes')
    
    train_annotation_path = ''
    train_dataset = datasets.YoloDatasetMultiFile(
        train_annotation_path, image_shape, max_objects, 2, 
        anchors, YoloV3.default_masks, len(classes)
    )

    val_annotation_path = ''
    val_dataset = datasets.YoloDatasetMultiFile(
        val_annotation_path, image_shape, max_objects, 2, 
        anchors, YoloV3.default_masks, len(classes)
    )

    model = YoloV3(image_shape, max_objects, backbone='DarkNet',
        anchors=anchors, num_classes=len(classes), training=True)

    loss_fn = Loss(len(test_classes), test_anchors, test_masks, img_shape[0])
    optimizer = model.get_optimizer('sgd', 1e-4)
    model.compile(optimizer, loss_fn, run_eagerly=False)
    history = model.fit(train_dataset, val_dataset, 5)


