import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


@tf.function
def to_box_xyxy(box_xy, box_wh, grid_size, anchors_masks):
    """convert the given boxes into the xy_min xy_max format
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
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)    # [gx, gy, 1, 2]
    grid = tf.cast(grid, tf.float32)

    box_xy = (box_xy + grid) / tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors_masks

    box_wh = tf.where(tf.math.is_inf(box_wh), tf.zeros_like(box_wh), box_wh)

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    box_xyxy = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return box_xyxy


@tf.function
def process_predictions(y_pred, num_classes, anchors_masks):
    """process the predictions to transform from:
    -  pred_xy, pred_wh, pred_obj, pred_class
    into
    - box_xyxy, pred_obj, pred_class, pred_xywh

    Arguments:
        y_pred {tf.tensor} -- the predictions in the format 
            (NBATCH, x_center, y_center, width, heigth, obj, one_hot_classes)
        num_classes {int} -- the number of classes
        anchors_masks {tf.tensor} -- the anchors masks

    Returns:
        tuple -- box_xyxy, pred_obj, pred_class, pred_xywh
    """
    # anchors_masks = tf.gather(anchors, masks)

    pred_xy, pred_wh, pred_obj, pred_class = tf.split(y_pred,
                                                      (2, 2, 1, num_classes),
                                                      axis=-1)

    pred_xy = tf.sigmoid(pred_xy)
    pred_obj = tf.sigmoid(pred_obj)
    pred_class = tf.sigmoid(pred_class)
    pred_xywh = tf.concat((pred_xy, pred_wh), axis=-1)

    grid_size = tf.shape(y_pred)[1]
    box_xyxy = to_box_xyxy(pred_xy, pred_wh, grid_size, anchors_masks)

    return box_xyxy, pred_obj, pred_class, pred_xywh


class FocalLoss():

    def __init__(self, anchor_masks, ignore_iou_threshold, img_size,
                 num_classes, name):
        self.anchor_masks = anchor_masks.astype(np.float32)
        self.ignore_iou_threshold = ignore_iou_threshold
        self.img_size = img_size
        self.num_classes = num_classes
        self.anchors_masks_scaled = self.anchor_masks / img_size
        self.__name__ = name

    @staticmethod
    def broadcast_iou(box_1, box_2):
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

        int_w = tf.maximum(
            tf.minimum(box_1[..., 2], box_2[..., 2]) -
            tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
        int_h = tf.maximum(
            tf.minimum(box_1[..., 3], box_2[..., 3]) -
            tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
        int_area = int_w * int_h
        box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
            (box_1[..., 3] - box_1[..., 1])
        box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
            (box_2[..., 3] - box_2[..., 1])
        return int_area / (box_1_area + box_2_area - int_area)

    def __call__(self, y_true, y_pred, **kvargs):

        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        pred_xyxy, pred_obj, pred_class, pred_xywh = process_predictions(
            tf.cast(y_pred, tf.float32), self.num_classes,
            self.anchors_masks_scaled)
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # 2. prepare the true outputs
        # y_true: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        true_box_xyxy, true_obj, true_class = tf.split(y_true,
                                                       (4, 1, self.num_classes),
                                                       axis=-1)
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
        true_wh = tf.math.log(true_wh / self.anchors_masks_scaled)
        true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh),
                           true_wh)

        # 4. calculate all masks
        # remove the last dimension
        obj_mask = tf.squeeze(true_obj, -1)
        # ignore false positive when iou is over threshold
        true_box_mask = tf.boolean_mask(true_box_xyxy,
                                        tf.cast(obj_mask, tf.bool))
        best_iou = tf.reduce_max(FocalLoss.broadcast_iou(
            pred_xyxy, true_box_mask),
                                 axis=-1)
        ignore_mask = tf.cast(best_iou < self.ignore_iou_threshold, tf.float32)

        # 5. compute all the losses
        xy_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.abs(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.abs(true_wh - pred_wh), axis=-1)

        obj_cross_entropy = tfa.losses.sigmoid_focal_crossentropy(
            true_obj, pred_obj, from_logits=False)
        obj_loss = obj_mask * obj_cross_entropy
        no_obj_loss = (1 - obj_mask) * ignore_mask * obj_cross_entropy

        class_loss = obj_mask * tfa.losses.sigmoid_focal_crossentropy(
            true_class, pred_class, from_logits=False)

        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        no_obj_loss = tf.reduce_sum(no_obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        loss = xy_loss + wh_loss + obj_loss + 0.5 * no_obj_loss + class_loss
        # tf.print('xy_loss', xy_loss)
        # tf.print('wh_loss', wh_loss)
        # tf.print('obj_loss', obj_loss)
        # tf.print('no_obj_loss', 0.5 * no_obj_loss)
        # tf.print('class_loss', class_loss)

        return loss

    def __str__(self):
        return self.__name__

    def __repr__(self):
        return f'{self.__name__} at {hex(id(self))}'


class YoloLoss():

    def __init__(self, anchor_masks, ignore_iou_threshold, img_size,
                 num_classes, name, num_batches):
        self.anchor_masks = anchor_masks.astype(np.float32)
        self.ignore_iou_threshold = ignore_iou_threshold
        self.img_size = img_size
        self.num_classes = num_classes
        self.anchors_masks_scaled = self.anchor_masks / img_size
        self.__name__ = name
        self.num_batches = num_batches

        self.xy_losses = []
        self.wh_losses = []
        self.obj_losses = []
        self.no_obj_losses = []
        self.class_losses = []
        self.epoch = 0
        self.count_batches = 0

    @staticmethod
    def broadcast_iou(box_1, box_2):
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

        int_w = tf.maximum(
            tf.minimum(box_1[..., 2], box_2[..., 2]) -
            tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
        int_h = tf.maximum(
            tf.minimum(box_1[..., 3], box_2[..., 3]) -
            tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
        int_area = int_w * int_h
        box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
            (box_1[..., 3] - box_1[..., 1])
        box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
            (box_2[..., 3] - box_2[..., 1])
        return int_area / (box_1_area + box_2_area - int_area)

    def __call__(self, y_true, y_pred, **kvargs):
        """[summary]
        
        Arguments:
            y_true {tf.Tensor} -- a tensor of shape (batch_size, grid, grid, anchors, (x, y, x, y, obj, ...cls))
            y_pred {tf.Tensor} -- a tensor of shape (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        
        Returns:
            [type] -- [description]
        """

        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        pred_xyxy, pred_obj, pred_class, pred_xywh = process_predictions(
            tf.cast(y_pred, tf.float32), self.num_classes,
            self.anchors_masks_scaled)
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # 2. prepare the true outputs
        # y_true: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        true_box_xyxy, true_obj, true_class = tf.split(y_true,
                                                       (4, 1, self.num_classes),
                                                       axis=-1)
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
        true_wh = tf.math.log(true_wh / self.anchors_masks_scaled)
        true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh),
                           true_wh)

        # 4. calculate all masks
        obj_mask = tf.squeeze(true_obj, -1)
        # ignore false positive when iou is over threshold
        true_box_mask = tf.boolean_mask(true_box_xyxy,
                                        tf.cast(obj_mask, tf.bool))
        best_iou = tf.reduce_max(YoloLoss.broadcast_iou(pred_xyxy,
                                                        true_box_mask),
                                 axis=-1)
        ignore_mask = tf.cast(best_iou < self.ignore_iou_threshold, tf.float32)

        # 5. compute all the losses
        xy_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)

        obj_cross_entropy = tf.keras.losses.binary_crossentropy(
            true_obj, pred_obj, from_logits=False)
        obj_loss = obj_mask * obj_cross_entropy
        no_obj_loss = (1 - obj_mask) * ignore_mask * obj_cross_entropy

        class_loss = obj_mask * tf.keras.losses.binary_crossentropy(
            true_class, pred_class, from_logits=False)

        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        no_obj_loss = tf.reduce_sum(no_obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        loss = xy_loss + wh_loss + obj_loss + 0.5 * no_obj_loss + class_loss

        self.xy_losses.append(xy_loss)
        self.wh_losses.append(wh_loss)
        self.obj_losses.append(obj_loss)
        self.no_obj_losses.append(no_obj_loss)
        self.class_losses.append(class_loss)
        self.count_batches += 1
        if self.count_batches == self.num_batches - 1:
            print('saving metrics')
            self.save_metrics()

        return loss

    def save_metrics(self):
        tf.summary.scalar('{}_xy_loss'.format(self.__name__),
                          step=self.epoch,
                          data=tf.reduce_mean(tf.concat(self.wh_losses,
                                                        axis=-1)))
        tf.summary.scalar('{}_wh_loss'.format(self.__name__),
                          step=self.epoch,
                          data=tf.reduce_mean(tf.concat(self.xy_losses,
                                                        axis=-1)))
        tf.summary.scalar('{}_obj_loss'.format(self.__name__),
                          step=self.epoch,
                          data=tf.reduce_mean(
                              tf.concat(self.obj_losses, axis=-1)))
        tf.summary.scalar('{}_no_obj_loss'.format(self.__name__),
                          step=self.epoch,
                          data=tf.reduce_mean(
                              tf.concat(self.no_obj_losses, axis=-1)))
        tf.summary.scalar('{}_class_loss'.format(self.__name__),
                          step=self.epoch,
                          data=tf.reduce_mean(
                              tf.concat(self.class_losses, axis=-1)))
        self.xy_losses = []
        self.wh_losses = []
        self.obj_losses = []
        self.no_obj_losses = []
        self.class_losses = []
        self.count_batches = 0
        self.epoch += 1

    def __str__(self):
        return self.__name__

    def __repr__(self):
        return f'{self.__name__} at {hex(id(self))}'


def make_loss(num_classes,
              anchors,
              masks,
              img_size,
              num_batches,
              ignore_iou_threshold=0.7,
              loss_name='yolo'):
    """helper to create the losses
    
    Arguments:
        num_classes {int} -- the number of classes
        anchors {np.ndarray} -- the anchors used for the detection
        masks {np.ndarray} -- the mask used to filter the anchors
        img_size {tuple} -- the shape of the images
    
    Keyword Arguments:
        ignore_iou_threshold {float} -- the value used to ignore predictions (default: {0.7})
        loss_name {str} -- the loss function used (default: {yolo}) (values: {yolo, focal})
    
    Returns:
        [type] -- [description]
    """
    loss_cls = YoloLoss
    if loss_name == 'yolo':
        loss_cls = YoloLoss
    elif loss_name == 'focal':
        loss_cls = FocalLoss

    loss_fns = [
        loss_cls(anchors[m], ignore_iou_threshold, img_size, num_classes,
                 f'yolo_loss{i}', num_batches) for i, m in enumerate(masks)
    ]
    return loss_fns
