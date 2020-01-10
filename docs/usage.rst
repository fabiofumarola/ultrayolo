=====
Usage
=====

To use tfyolo3 in a project::

    from tfyolo3 import YoloV3
    from tfyolo3 import dataloaders
    from tfyolo3.losses import Loss

    image_shape = (256,256,3)
    max_objects = 100
    anchors = dataloaders.load_anchors('the path of the anchors')
    classes = dataloaders.load_classes('the file of the classes')
    
    train_annotation_path = ''
    train_dataset = dataloaders.YoloDatasetMultiFile(
        train_annotation_path, image_shape, max_objects, 2, 
        anchors, YoloV3.default_masks, len(classes)
    )

    val_annotation_path = ''
    val_dataset = dataloaders.YoloDatasetMultiFile(
        val_annotation_path, image_shape, max_objects, 2, 
        anchors, YoloV3.default_masks, len(classes)
    )

    model = YoloV3(image_shape, max_objects, backbone='DarkNet',
        anchors=anchors, num_classes=len(classes), training=True)

    loss_fn = Loss(len(test_classes), test_anchors, test_masks, img_shape[0])
    optimizer = model.get_optimizer('sgd', 1e-4)
    model.compile(optimizer, loss_fn, run_eagerly=False)
    history = model.fit(train_dataset, val_dataset, 5)


