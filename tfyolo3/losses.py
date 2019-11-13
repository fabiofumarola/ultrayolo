import tensorflow as tf


def non_max_suppression(outputs, anchors, masks, classes, iou_threshold, score_threshold, max_boxes_per_image, img_size):
    """an implementation of non max suppression

    Arguments:
        outputs {tf.tensor} -- the outputs of the yolo branches
        anchors {np.ndarray} -- the anchors scaled in [0,1]
        masks {np.ndarray} -- the list of the anchors to use
        classes {list} -- the list of classes
        iou_threshold {float} -- the minimum intersection over union threshold
        score_threshold {float} -- the minimum confidence score to use
        max_boxes_per_image {int} -- the number of maximum boxes to show
        img_size {int} -- the size of the image

    Returns:
        (boxes, scores, classes, valid_detections) -- a tuple of the results
    """
    # boxes, conf, type
    b, c, t = [], [], []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)

    scores = confidence * class_probs

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(
            scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=max_boxes_per_image,
        max_total_size=max_boxes_per_image,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold
    )

    return tf.math.ceil(boxes * img_size), scores, classes, valid_detections


def process_predictions(y_pred, num_classes, anchors_masks):
    """process the predictions

    Arguments:
        y_pred {tf.tensor} -- the predictions
        num_classes {int} -- the number of classes
        anchors_masks {tf.tensor} -- the anchors masks

    Returns:
        tuple -- box,xyxy, perd_obj, pred_class, pred_xywh
    """
    pred_xy, pred_wh, pred_obj, pred_class = tf.split(
        y_pred, (2, 2, 1, num_classes), axis=-1
    )

    pred_xy = tf.sigmoid(pred_xy)
    pred_obj = tf.sigmoid(pred_obj)
    pred_class = tf.sigmoid(pred_class)
    pred_xywh = tf.concat((pred_xy, pred_wh), axis=-1)

    grid_size = tf.shape(y_pred)[1]
    box_xyxy = to_box_xyxy(pred_xy, pred_wh, grid_size, anchors_masks)

    return box_xyxy, pred_obj, pred_class, pred_xywh


def to_box_xyxy(box_xy, box_wh, grid_size, anchors_masks):
    """conrvet the given boxes into the xy_min xy_max format

    Arguments:
        box_xy {tf.tensor} -- 
        box_wh {tf,tensor} -- 
        grid_size {float} -- the size of the grid used
        anchors_masks {tf.tensor} -- the anchor masks

    Returns:
        tf.tensor -- the boxes
    """
    # !!! grid[x][y] == (y, x)
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
        tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors_masks

    box_wh = tf.where(tf.math.is_inf(box_wh),
                      tf.zeros_like(box_wh), box_wh)

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    box_xyxy = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return box_xyxy


def __broadcast_iou(box_1, box_2):
    """brodcast intersection over union

    Arguments:
        box_1 {tf.tensor} --  (..., (x1, y1, x2, y2))
        box_2 {tf.tensor} --  (N, (x1, y1, x2, y2))

    Returns:
        tf.tensor  -- intersection over union
    """

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                       tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                       tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
        (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
        (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)


def Loss(num_classes, anchors_masks, img_size, ignore_iou_threshold=0.7):
    """the default Yolo Loss

    Arguments:
        num_classes {int} -- the number of classes
        anchors_masks {tf.tensor} -- the anchors
        img_size {int} -- the size of the image

    Keyword Arguments:
        ignore_iou_threshold {float} -- the value below of which do not consider the predictions (default: {0.7})

    Returns:
        function  -- a function that compute the loss 
    """

    def yolo_loss(y_true, y_pred):
        anchors_masks_scaled = anchors_masks / img_size

        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        pred_xyxy, pred_obj, pred_class, pred_xywh = process_predictions(
            tf.cast(y_pred, tf.float32), num_classes, anchors_masks_scaled
        )
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        true_box_xyxy, true_obj, true_class = tf.split(
            y_true, (4, 1, num_classes), axis=-1)
        true_xy = (true_box_xyxy[..., 0:2] + true_box_xyxy[..., 2:4]) / 2
        true_wh = true_box_xyxy[..., 2:4] - true_box_xyxy[..., 0:2]

        # give more weight to smaller boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # 3. inverting the pred box equations for true values
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - \
            tf.cast(grid, tf.float32)
        true_wh = tf.math.log(true_wh / anchors_masks_scaled)
        true_wh = tf.where(tf.math.is_inf(true_wh),
                           tf.zeros_like(true_wh), true_wh)

        # 4. calculate all masks
        obj_mask = tf.squeeze(true_obj, -1)
        # ignore false positive when iou is over threshold
        true_box_mask = tf.boolean_mask(
            true_box_xyxy, tf.cast(obj_mask, tf.bool))
        best_iou = tf.reduce_max(__broadcast_iou(
            pred_xyxy, true_box_mask), axis=-1)
        ignore_mask = tf.cast(best_iou < ignore_iou_threshold, tf.float32)

        # 5. compute all the losses
        xy_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)

        obj_cross_entropy = tf.keras.metrics.binary_crossentropy(
            true_obj, pred_obj, from_logits=False)
        obj_loss = obj_mask * obj_cross_entropy
        no_obj_loss = (1 - obj_mask) * ignore_mask * obj_cross_entropy

        class_loss = obj_mask * tf.keras.metrics.binary_crossentropy(
            true_class, pred_class, from_logits=False)

        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        no_obj_loss = tf.reduce_sum(no_obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        loss = 5. * xy_loss + 5. * wh_loss + obj_loss + 0.5 * no_obj_loss + class_loss
        # tf.print('xy_loss', xy_loss)
        # tf.print('wh_loss', wh_loss)
        # tf.print('obj_loss', obj_loss)
        # tf.print('no_obj_loss', 0.5 * no_obj_loss)
        # tf.print('class_loss', class_loss)

        return loss
    return yolo_loss
