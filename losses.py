import tensorflow as tf
from tensorflow.python.keras.optimizers import *

from tensorflow.python.keras.losses import binary_crossentropy, categorical_crossentropy, mean_squared_error
#push dev3 4


def categorical_focal_loss(gt, pr, gamma=2.0, alpha=0.25, class_indexes=None, **kwargs):
    r"""Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = - gt * alpha * ((1 - pr)^gamma) * log(pr)
    Args:
        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
        alpha: the same as weighting factor in balanced cross entropy, default 0.25
        gamma: focusing parameter for modulating factor (1-p), default 2.0
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
    """

    backend = kwargs['backend']
    gt, pr = gather_channels(gt, pr, indexes=class_indexes, **kwargs)

    # clip to prevent NaN's and Inf's
    pr = backend.clip(pr, backend.epsilon(), 1.0 - backend.epsilon())

    # Calculate focal loss
    loss = - gt * (alpha * backend.pow((1 - pr), gamma) * backend.log(pr))

    return backend.mean(loss)

def _gather_channels(x, indexes, **kwargs):
    """Slice tensor along channels axis by given indexes"""

    x = tf.transpose(x, (1, 0, 2, 3))
    x = gather_channels(x, indexes)
    x = tf.transpose(x, (1, 0, 2, 3))

    return x

def gather_channels(*xs, indexes=None, **kwargs):
    """Slice tensors along channels axis by given indexes"""
    if indexes is None:
        return xs
    elif isinstance(indexes, (int)):
        indexes = [indexes]
    xs = [_gather_channels(x, indexes=indexes, **kwargs) for x in xs]
    return xs

def focal_loss_weighted(weights):
    weights = weights

    def focal_loss_weighted_loss(y_true, y_pred):

        y_pred = tf.clip_by_value(y_pred,
                                  tf.keras.backend.epsilon.epsilon(), 1 -
                                  tf.keras.backend.epsilon.epsilon())
        # calc
        y_true, y_pred = gather_channels(y_true, y_pred, indexes=[int(i) for i in weights])
        loss = - y_true * (0.25 *
                           tf.math.pow((1 - y_pred), 2.0) * tf.math.log(y_pred))
        return loss

    return focal_loss_weighted_loss

def binary_cross_entropy_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred)

def categorical_focal_loss_fixed(y_true, y_pred):
    """
       Softmax version of focal loss.
              m
         FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
             c=1
         where m = number of classes, c = class and o = observation
       Parameters:
         alpha -- the same as weighing factor in balanced cross entropy
         gamma -- focusing parameter for modulating factor (1-p)
       Default value:
         gamma -- 2.0 as mentioned in the paper
         alpha -- 0.25 as mentioned in the paper
       References:
           Official paper: https://arxiv.org/pdf/1708.02002.pdf
           https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
       Usage:
        model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
       """
    """
    :param y_true: A tensor of the same shape as `y_pred`
    :param y_pred: A tensor resulting from a softmax
    :return: Output tensor.
    """
    # Scale predictions so that the class probas of each sample sum to 1
    y_pred /= tf.math.reduce_sum(y_pred, axis=-1, keepdims=True)

    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

    # Calculate Cross Entropy
    cross_entropy = -y_true * tf.math.log(y_pred)

    # Calculate Focal Loss
    loss = .25 * tf.math.pow(1 - y_pred, 2.) * cross_entropy

    # Sum the losses in mini_batch
    return tf.reduce_sum(loss, axis=1)


# def categorical_cross_entropy_weighted(weights):
    #weights = weights
def categorical_cross_entropy_weighted_loss(y_true, y_pred):
    #weights = [1 / 200, 1 / 80, 1 / 20, 1 / 6, 1 / 3, 1 / 4]
    weights = [1, 5]
    # background,finger crack,microcrack

    # scale predictions so that the class probas of each sample sum to 1
    # y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    # calc
    loss = y_true * tf.math.log(y_pred) * weights
    loss = -tf.reduce_sum(loss, -1)
    return 1 - loss

# return categorical_cross_entropy_weighted_loss


def categorical_cross_entropy(y_true, y_pred):
    return categorical_crossentropy(y_pred=y_pred, y_true=y_true)

def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def mix_focal_dice(y_true, y_pred):
    return dice_coeff_orig_loss(y_true, y_pred) + binary_focal_loss_fixed(y_true, y_pred)

def dice_coeff_orig(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    ret = (2. * tf.math.reduce_sum(y_true * y_pred) + 1.) / (tf.math.reduce_sum(y_true) + tf.math.reduce_sum(y_pred) + 1.)
    ret = ret / tf.cast(tf.shape(y_pred)[0], tf.float32) #divide by number of batches
    return ret

def dice_coeff_orig_loss(y_true, y_pred):
    ret = dice_coeff_orig(y_true, y_pred)
    return 1 - ret

def dice_coeff_and_binary_cross_entropy(y_true, y_pred):
    beta = 0.5
    dice_loss = - dice_coeff_orig(y_true, y_pred)
    binary = binary_cross_entropy_loss(y_true, y_pred)
    return beta*dice_loss + (1-beta)*binary

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
        possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
        recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
        predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+tf.keras.backend.epsilon()))

def tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1e-10):
    """ Tversky loss function.
    Parameters
    ----------
    y_true : keras tensor
        tensor containing target mask.
    y_pred : keras tensor
        tensor containing predicted mask.
    alpha : float
        real value, weight of '0' class.
    beta : float
        real value, weight of '1' class.
    smooth : float
        small real value used for avoiding division by zero error.
    Returns
    -------
    keras tensor
        tensor containing tversky loss.
    """
    y_true = tf.reshape(y_true, -1)
    y_pred = tf.reshape(y_pred, -1)
    truepos = tf.reduce_sum(y_true * y_pred)
    fp_and_fn = alpha * tf.reduce_sum(y_pred * (1 - y_true)) + beta * tf.reduce_sum((1 - y_pred) * y_true)
    answer = (truepos + smooth) / ((truepos + smooth) + fp_and_fn)
    return -answer

# def binary_focal_loss(gamma=2., alpha=.25):
#     """
#     Binary form of focal loss.
#       FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
#       where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
#     References:
#         https://arxiv.org/pdf/1708.02002.pdf
#     Usage:
#      model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
#     """
def binary_focal_loss_fixed(y_true, y_pred):
    gamma=2.
    alpha=.25
    """
    :param y_true: A tensor of the same shape as `y_pred`
    :param y_pred:  A tensor resulting from a sigmoid
    :return: Output tensor.
    """
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    epsilon = tf.keras.backend.epsilon()
    # clip to prevent NaN's and Inf's
    pt_1 = tf.clip_by_value(pt_1, epsilon, 1. - epsilon)
    pt_0 = tf.clip_by_value(pt_0, epsilon, 1. - epsilon)

    return -tf.reduce_sum(alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1)) \
           - tf.reduce_sum((1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1. - pt_0))

    # return binary_focal_loss_fixed

def weighted_dice_coef(y_true, y_pred):
    mean = 0.00009
    w_0 = 1/mean**2
    w_1 = 1/(1-mean)**2
    y_true_f_1 = tf.reshape(y_true, -1)
    y_pred_f_1 = tf.reshape(y_pred, -1)
    y_true_f_0 = tf.reshape(1-y_true, -1)
    y_pred_f_0 = tf.reshape(1-y_pred, -1)

    intersection_0 = tf.reduce_sum(y_true_f_0 * y_pred_f_0)
    intersection_1 = tf.reduce_sum(y_true_f_1 * y_pred_f_1)

    return 2 * (w_0 * intersection_0 + w_1 * intersection_1) / \
           ((w_0 * (tf.reduce_sum(y_true_f_0) + tf.reduce_sum(y_pred_f_0))) + (w_1 * (tf.reduce_sum(y_true_f_1) + tf.reduce_sum(y_pred_f_1))))

def weighted_dice_coef_loss(y_true, y_pred):
    return 1 - weighted_dice_coef(y_true, y_pred)

def binary_cross_entropy(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred)

#TODO: test it
def weighted_binary_cross_entropy(y_true, y_pred):

    # fore = np.where(y_true == 1, 1*3., 1)
    # back = np.where(y_true == 1, 1, 1*1.)
    # y_pred = y_pred * fore * back
    #
    #
    # return binary_cross_entropy(y_true, y_pred)
    # calculate the binary cross entropy

    # bin_crossentropy = binary_crossentropy(y_true, y_pred)

    weight_one = 3.
    # apply the weights
    y_true = tf.clip_by_value(y_true, tf.keras.backend.epsilon(), 1-tf.keras.backend.epsilon())
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1-tf.keras.backend.epsilon())
    logloss = -(y_true * tf.math.log(y_pred) * weight_one + (1 - y_true) * tf.math.log(1 - y_pred))
    return tf.reduce_mean( logloss, axis=-1)

def iou(y_true, y_pred):
    intersection = y_true * y_pred
    notTrue = 1 - y_true
    union = y_true + (notTrue * y_pred)

    return (tf.reduce_sum(intersection, axis=-1) + tf.keras.backend.epsilon()) / (tf.reduce_sum(union, axis=-1) + tf.keras.backend.epsilon())

def iou_loss(y_true, y_pred):
    return 1 - iou(y_true, y_pred)

def iou_nobacground(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    # calc
    y_true, y_pred = gather_channels(y_true, y_pred, indexes=[int(i) for i in [0,1,1,1]])

    return iou(y_true, y_pred)

def mse_loss(y_true, y_pred):

    return mean_squared_error(y_true, y_pred)


def ssim_loss(gt, y_pred, max_val=1.0):
    return 1 - tf.reduce_mean(tf.image.ssim(gt, y_pred, max_val=max_val))


