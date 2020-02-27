import keras.backend as K
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

class Metrics():
    def __init__(self):
        pass
        
    def mean_iou(y_true, y_pred):
        mean_iou, op = tf.metrics.mean_iou(y_true, y_pred, classes)
        return mean_iou

    def dice_coeff(y_true, y_pred):
        smooth = 1e-6
        y_true = tf.cast(y_true, dtype=tf.bool)
        y_pred = tf.cast(y_pred, dtype=tf.bool)
        intersection = tf.cast(tf.logical_and(y_true, y_pred), dtype=tf.int32)
        union = tf.cast(tf.logical_or(y_true, y_pred), dtype=tf.int32)
        score = tf.cast(tf.reduce_sum(2 * intersection)/tf.reduce_sum(union) + smooth, dtype=tf.float32)

        #intersection = tf.cast(tf.reduce_sum(y_true * y_pred), dtype=tf.float32)
        #score = ((2.0 * intersection) + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)
        return score

    def original_dice_coef(tp, fp, fn):
        smooth = 1e-6
        return (2 * tf.reduce_sum(tp) / (tf.reduce_sum(2*tp) + tf.reduce_sum(fp) + tf.reduce_sum(fn) + smooth))

    def dice_loss(y_true, y_pred):
        smooth = 1e-6
        intersection = tf.reduce_sum(y_true * y_pred)
        score = (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)
        loss = 1 - score
        return loss

    def original_dice_loss(tp, tn, fp, fn):
        smooth = 1
        return (2 * tf.reduce_sum(tp) / (tf.reduce_sum(2*tp) + tf.reduce_sum(fp) + tf.reduce_sum(fn) + smooth))

    def weighted_bce_loss(y_true, y_pred):
        loss = BETA * (y_true * (-tf.log(y_pred + eps))) + (1 - BETA) * ((1 - y_true) * (-tf.log((1 - y_pred) + eps)))
        return loss

    def bce_loss(y_true, y_pred):
        loss = (y_true * (-tf.log(y_pred + eps))) + ((1 - y_true) * (-tf.log((1 - y_pred) + eps)))
        return loss

    def get_confusion_matrix(y_pred, y_true):
        ones_like_actuals = tf.ones_like(y_true)
        zeros_like_actuals = tf.zeros_like(y_true)
        ones_like_predictions = tf.ones_like(y_pred)
        zeros_like_predictions = tf.zeros_like(y_pred)

        TP = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, ones_like_actuals), tf.equal(y_pred, ones_like_predictions)), dtype=tf.float32))
        TN = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, zeros_like_actuals), tf.equal(y_pred, zeros_like_predictions)), dtype=tf.float32))
        FP = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, zeros_like_actuals), tf.equal(y_pred, ones_like_predictions)), dtype=tf.float32))
        FN = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, ones_like_actuals), tf.equal(y_pred, zeros_like_predictions)), dtype=tf.float32))

        #TP = tf.cast(tf.count_nonzero(pred_int * labels, dtype=tf.int32), dtype=tf.float32)
        #TN = tf.cast(tf.count_nonzero((pred_int - 1) * (labels - 1), dtype=tf.int32), dtype=tf.float32)
        #FP = tf.cast(tf.count_nonzero(pred_int * (labels - 1), dtype=tf.int32), dtype=tf.float32)
        #FN = tf.cast(tf.count_nonzero((pred_int - 1) * labels, dtype=tf.int32), dtype=tf.float32)

        return TP, TN, FP, FN

    def get_accuracy_recall_precision_f1score(tp, tn, fp, fn):
        # accuracy ::= (TP + TN) / (TN + FN + TP + FP)
        accuracy = tf.cast(tf.divide(tp + tn, tp + tn + fp + fn, name="Accuracy"), dtype=tf.float32)
        # precision ::= TP / (TP + FP)
        precision = tf.cast(tf.divide(tp, tp + fp, name="Precision"), dtype=tf.float32)
        precision = tf.where(tf.is_nan(precision), 0., precision)
        # recall ::= TP / (TP + FN)
        recall = tf.cast(tf.divide(tp, tp + fn, name="Recall"), dtype=tf.float32)
        recall = tf.where(tf.is_nan(recall), 0., recall)
        # F1 score ::= 2 * precision * recall / (precision + recall)
        f1 = tf.cast(tf.divide((2 * precision * recall), (precision + recall), name="F1_score"), dtype=tf.float32)

        TPR = 1. * tp / (tp + fn)
        FPR = 1. * fp / (fp + tn)
        return accuracy, precision, recall, f1
