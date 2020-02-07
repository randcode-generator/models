# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Constructs model, inputs, and training environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import functools
import os

import tensorflow as tf

from object_detection import exporter as exporter_lib
from object_detection import inputs
from object_detection.builders import graph_rewriter_builder
from object_detection.builders import model_builder
from object_detection.builders import optimizer_builder
from object_detection.core import standard_fields as fields
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import ops
from object_detection.utils import shape_utils
from object_detection.utils import variables_helper

# A map of names to methods that help build the model.
MODEL_BUILD_UTIL_MAP = {
    'get_configs_from_pipeline_file':
        config_util.get_configs_from_pipeline_file,
    'create_pipeline_proto_from_configs':
        config_util.create_pipeline_proto_from_configs,
    'merge_external_params_with_configs':
        config_util.merge_external_params_with_configs,
    'create_train_input_fn':
        inputs.create_train_input_fn,
    'create_eval_input_fn':
        inputs.create_eval_input_fn,
    'create_predict_input_fn':
        inputs.create_predict_input_fn,
    'detection_model_fn_base': model_builder.build,
}

def unstack_batch(tensor_dict, unpad_groundtruth_tensors=True):
  """Unstacks all tensors in `tensor_dict` along 0th dimension.

  Unstacks tensor from the tensor dict along 0th dimension and returns a
  tensor_dict containing values that are lists of unstacked, unpadded tensors.

  Tensors in the `tensor_dict` are expected to be of one of the three shapes:
  1. [batch_size]
  2. [batch_size, height, width, channels]
  3. [batch_size, num_boxes, d1, d2, ... dn]

  When unpad_groundtruth_tensors is set to true, unstacked tensors of form 3
  above are sliced along the `num_boxes` dimension using the value in tensor
  field.InputDataFields.num_groundtruth_boxes.

  Note that this function has a static list of input data fields and has to be
  kept in sync with the InputDataFields defined in core/standard_fields.py

  Args:
    tensor_dict: A dictionary of batched groundtruth tensors.
    unpad_groundtruth_tensors: Whether to remove padding along `num_boxes`
      dimension of the groundtruth tensors.

  Returns:
    A dictionary where the keys are from fields.InputDataFields and values are
    a list of unstacked (optionally unpadded) tensors.

  Raises:
    ValueError: If unpad_tensors is True and `tensor_dict` does not contain
      `num_groundtruth_boxes` tensor.
  """
  unbatched_tensor_dict = {
      key: tf.unstack(tensor) for key, tensor in tensor_dict.items()
  }
  if unpad_groundtruth_tensors:
    if (fields.InputDataFields.num_groundtruth_boxes not in
        unbatched_tensor_dict):
      raise ValueError('`num_groundtruth_boxes` not found in tensor_dict. '
                       'Keys available: {}'.format(
                           unbatched_tensor_dict.keys()))
    unbatched_unpadded_tensor_dict = {}
    unpad_keys = set([
        # List of input data fields that are padded along the num_boxes
        # dimension. This list has to be kept in sync with InputDataFields in
        # standard_fields.py.
        fields.InputDataFields.groundtruth_instance_masks,
        fields.InputDataFields.groundtruth_classes,
        fields.InputDataFields.groundtruth_boxes,
        fields.InputDataFields.groundtruth_keypoints,
        fields.InputDataFields.groundtruth_group_of,
        fields.InputDataFields.groundtruth_difficult,
        fields.InputDataFields.groundtruth_is_crowd,
        fields.InputDataFields.groundtruth_area,
        fields.InputDataFields.groundtruth_weights
    ]).intersection(set(unbatched_tensor_dict.keys()))

    for key in unpad_keys:
      unpadded_tensor_list = []
      for num_gt, padded_tensor in zip(
          unbatched_tensor_dict[fields.InputDataFields.num_groundtruth_boxes],
          unbatched_tensor_dict[key]):
        tensor_shape = shape_utils.combined_static_and_dynamic_shape(
            padded_tensor)
        slice_begin = tf.zeros([len(tensor_shape)], dtype=tf.int32)
        slice_size = tf.stack(
            [num_gt] + [-1 if dim is None else dim for dim in tensor_shape[1:]])
        unpadded_tensor = tf.slice(padded_tensor, slice_begin, slice_size)
        unpadded_tensor_list.append(unpadded_tensor)
      unbatched_unpadded_tensor_dict[key] = unpadded_tensor_list
    unbatched_tensor_dict.update(unbatched_unpadded_tensor_dict)

  return unbatched_tensor_dict


def provide_groundtruth(model, labels):
  """Provides the labels to a model as groundtruth.

  This helper function extracts the corresponding boxes, classes,
  keypoints, weights, masks, etc. from the labels, and provides it
  as groundtruth to the models.

  Args:
    model: The detection model to provide groundtruth to.
    labels: The labels for the training or evaluation inputs.
  """
  gt_boxes_list = labels[fields.InputDataFields.groundtruth_boxes]
  gt_classes_list = labels[fields.InputDataFields.groundtruth_classes]
  gt_masks_list = None
  if fields.InputDataFields.groundtruth_instance_masks in labels:
    gt_masks_list = labels[
        fields.InputDataFields.groundtruth_instance_masks]
  gt_keypoints_list = None
  if fields.InputDataFields.groundtruth_keypoints in labels:
    gt_keypoints_list = labels[fields.InputDataFields.groundtruth_keypoints]
  gt_weights_list = None
  if fields.InputDataFields.groundtruth_weights in labels:
    gt_weights_list = labels[fields.InputDataFields.groundtruth_weights]
  gt_confidences_list = None
  if fields.InputDataFields.groundtruth_confidences in labels:
    gt_confidences_list = labels[
        fields.InputDataFields.groundtruth_confidences]
  gt_is_crowd_list = None
  if fields.InputDataFields.groundtruth_is_crowd in labels:
    gt_is_crowd_list = labels[fields.InputDataFields.groundtruth_is_crowd]
  model.provide_groundtruth(
      groundtruth_boxes_list=gt_boxes_list,
      groundtruth_classes_list=gt_classes_list,
      groundtruth_confidences_list=gt_confidences_list,
      groundtruth_masks_list=gt_masks_list,
      groundtruth_keypoints_list=gt_keypoints_list,
      groundtruth_weights_list=gt_weights_list,
      groundtruth_is_crowd_list=gt_is_crowd_list)


def create_model_fn(detection_model_fn, configs, hparams, use_tpu=False,
                    postprocess_on_cpu=False):
  """Creates a model function for `Estimator`.

  Args:
    detection_model_fn: Function that returns a `DetectionModel` instance.
    configs: Dictionary of pipeline config objects.
    hparams: `HParams` object.
    use_tpu: Boolean indicating whether model should be constructed for
        use on TPU.
    postprocess_on_cpu: When use_tpu and postprocess_on_cpu is true, postprocess
        is scheduled on the host cpu.

  Returns:
    `model_fn` for `Estimator`.
  """
  train_config = configs['train_config']

  def model_fn(features, labels, mode, params=None):
    """Constructs the object detection model.

    Args:
      features: Dictionary of feature tensors, returned from `input_fn`.
      labels: Dictionary of groundtruth tensors if mode is TRAIN or EVAL,
        otherwise None.
      mode: Mode key from tf.estimator.ModeKeys.
      params: Parameter dictionary passed from the estimator.

    Returns:
      An `EstimatorSpec` that encapsulates the model and its serving
        configurations.
    """
    if(mode == tf.estimator.ModeKeys.EVAL):
      return
    params = params or {}
    total_loss, train_op, detections, export_outputs = None, None, None, None
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    detection_model = detection_model_fn(
        is_training=is_training, add_summaries=(not use_tpu))
    
    if mode == tf.estimator.ModeKeys.TRAIN:
      labels = unstack_batch(
          labels,
          unpad_groundtruth_tensors=train_config.unpad_groundtruth_tensors)

    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
      provide_groundtruth(detection_model, labels)

    preprocessed_images = features[fields.InputDataFields.image]
    prediction_dict = detection_model.predict(
        preprocessed_images,
        features[fields.InputDataFields.true_image_shape])

    if mode == tf.estimator.ModeKeys.TRAIN:
      load_pretrained = hparams.load_pretrained if hparams else False
      if train_config.fine_tune_checkpoint and load_pretrained:
        if not train_config.fine_tune_checkpoint_type:
          # train_config.from_detection_checkpoint field is deprecated. For
          # backward compatibility, set train_config.fine_tune_checkpoint_type
          # based on train_config.from_detection_checkpoint.
          if train_config.from_detection_checkpoint:
            train_config.fine_tune_checkpoint_type = 'detection'
          else:
            train_config.fine_tune_checkpoint_type = 'classification'
        asg_map = detection_model.restore_map(
            fine_tune_checkpoint_type=train_config.fine_tune_checkpoint_type,
            load_all_detection_checkpoint_vars=(
                train_config.load_all_detection_checkpoint_vars))
        available_var_map = (
            variables_helper.get_variables_available_in_checkpoint(
                asg_map,
                train_config.fine_tune_checkpoint,
                include_global_step=False))
        tf.train.init_from_checkpoint(train_config.fine_tune_checkpoint,
                                      available_var_map)

    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
      losses_dict = detection_model.loss(
          prediction_dict, features[fields.InputDataFields.true_image_shape])
      losses = [loss_tensor for loss_tensor in losses_dict.values()]
      if train_config.add_regularization_loss:
        regularization_losses = detection_model.regularization_losses()
        if regularization_losses:
          regularization_loss = tf.add_n(
              regularization_losses, name='regularization_loss')
          losses.append(regularization_loss)
          losses_dict['Loss/regularization_loss'] = regularization_loss
      total_loss = tf.add_n(losses, name='total_loss')
      losses_dict['Loss/total_loss'] = total_loss

      # TODO(rathodv): Stop creating optimizer summary vars in EVAL mode once we
      # can write learning rate summaries on TPU without host calls.
      global_step = tf.train.get_or_create_global_step()
      training_optimizer, _ = optimizer_builder.build(
          train_config.optimizer)

      # Optionally freeze some layers by setting their gradients to be zero.
      trainable_variables = None
      include_variables = (
          train_config.update_trainable_variables
          if train_config.update_trainable_variables else None)
      exclude_variables = (
          train_config.freeze_variables
          if train_config.freeze_variables else None)
      trainable_variables = tf.contrib.framework.filter_variables(
          tf.trainable_variables(),
          include_patterns=include_variables,
          exclude_patterns=exclude_variables)

      clip_gradients_value = None
      if train_config.gradient_clipping_by_norm > 0:
        clip_gradients_value = train_config.gradient_clipping_by_norm

      train_op = tf.contrib.layers.optimize_loss(
          loss=total_loss,
          global_step=global_step,
          learning_rate=None,
          clip_gradients=clip_gradients_value,
          optimizer=training_optimizer,
          update_ops=detection_model.updates(),
          variables=trainable_variables,
          name='')  # Preventing scope prefix on all variables.

    eval_metric_ops = None
    scaffold = None

    if scaffold is None:
      keep_checkpoint_every_n_hours = (
          train_config.keep_checkpoint_every_n_hours)
      saver = tf.train.Saver(
          sharded=True,
          keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
          save_relative_paths=True)
      tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
      scaffold = tf.train.Scaffold(saver=saver)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=detections,
        loss=total_loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        export_outputs=export_outputs,
        scaffold=scaffold)

  return model_fn


def create_estimator_and_inputs(run_config,
                                hparams,
                                pipeline_config_path,
                                config_override=None,
                                train_steps=None,
                                sample_1_of_n_eval_examples=None,
                                sample_1_of_n_eval_on_train_examples=1,
                                model_fn_creator=create_model_fn,
                                use_tpu_estimator=False,
                                use_tpu=False,
                                num_shards=1,
                                params=None,
                                override_eval_num_epochs=True,
                                save_final_config=False,
                                postprocess_on_cpu=False,
                                export_to_tpu=None,
                                **kwargs):
  """Creates `Estimator`, input functions, and steps.

  Args:
    run_config: A `RunConfig`.
    hparams: A `HParams`.
    pipeline_config_path: A path to a pipeline config file.
    config_override: A pipeline_pb2.TrainEvalPipelineConfig text proto to
      override the config from `pipeline_config_path`.
    train_steps: Number of training steps. If None, the number of training steps
      is set from the `TrainConfig` proto.
    sample_1_of_n_eval_examples: Integer representing how often an eval example
      should be sampled. If 1, will sample all examples.
    sample_1_of_n_eval_on_train_examples: Similar to
      `sample_1_of_n_eval_examples`, except controls the sampling of training
      data for evaluation.
    model_fn_creator: A function that creates a `model_fn` for `Estimator`.
      Follows the signature:

      * Args:
        * `detection_model_fn`: Function that returns `DetectionModel` instance.
        * `configs`: Dictionary of pipeline config objects.
        * `hparams`: `HParams` object.
      * Returns:
        `model_fn` for `Estimator`.

    use_tpu_estimator: Whether a `TPUEstimator` should be returned. If False,
      an `Estimator` will be returned.
    use_tpu: Boolean, whether training and evaluation should run on TPU. Only
      used if `use_tpu_estimator` is True.
    num_shards: Number of shards (TPU cores). Only used if `use_tpu_estimator`
      is True.
    params: Parameter dictionary passed from the estimator. Only used if
      `use_tpu_estimator` is True.
    override_eval_num_epochs: Whether to overwrite the number of epochs to 1 for
      eval_input.
    save_final_config: Whether to save final config (obtained after applying
      overrides) to `estimator.model_dir`.
    postprocess_on_cpu: When use_tpu and postprocess_on_cpu are true,
      postprocess is scheduled on the host cpu.
    export_to_tpu: When use_tpu and export_to_tpu are true,
      `export_savedmodel()` exports a metagraph for serving on TPU besides the
      one on CPU.
    **kwargs: Additional keyword arguments for configuration override.

  Returns:
    A dictionary with the following fields:
    'estimator': An `Estimator` or `TPUEstimator`.
    'train_input_fn': A training input function.
    'eval_input_fns': A list of all evaluation input functions.
    'eval_input_names': A list of names for each evaluation input.
    'eval_on_train_input_fn': An evaluation-on-train input function.
    'predict_input_fn': A prediction input function.
    'train_steps': Number of training steps. Either directly from input or from
      configuration.
  """
  get_configs_from_pipeline_file = MODEL_BUILD_UTIL_MAP[
      'get_configs_from_pipeline_file']
  merge_external_params_with_configs = MODEL_BUILD_UTIL_MAP[
      'merge_external_params_with_configs']
  create_pipeline_proto_from_configs = MODEL_BUILD_UTIL_MAP[
      'create_pipeline_proto_from_configs']
  create_train_input_fn = MODEL_BUILD_UTIL_MAP['create_train_input_fn']
  create_eval_input_fn = MODEL_BUILD_UTIL_MAP['create_eval_input_fn']
  create_predict_input_fn = MODEL_BUILD_UTIL_MAP['create_predict_input_fn']
  detection_model_fn_base = MODEL_BUILD_UTIL_MAP['detection_model_fn_base']

  configs = get_configs_from_pipeline_file(
      pipeline_config_path, config_override=config_override)
  kwargs.update({
      'train_steps': train_steps,
      'use_bfloat16': configs['train_config'].use_bfloat16 and use_tpu
  })
  if sample_1_of_n_eval_examples >= 1:
    kwargs.update({
        'sample_1_of_n_eval_examples': sample_1_of_n_eval_examples
    })
  if override_eval_num_epochs:
    kwargs.update({'eval_num_epochs': 1})
    tf.logging.warning(
        'Forced number of epochs for all eval validations to be 1.')
  configs = merge_external_params_with_configs(
      configs, hparams, kwargs_dict=kwargs)
  model_config = configs['model']
  train_config = configs['train_config']
  train_input_config = configs['train_input_config']
  eval_config = configs['eval_config']
  eval_input_configs = configs['eval_input_configs']
  eval_on_train_input_config = copy.deepcopy(train_input_config)
  eval_on_train_input_config.sample_1_of_n_examples = (
      sample_1_of_n_eval_on_train_examples)
  if override_eval_num_epochs and eval_on_train_input_config.num_epochs != 1:
    tf.logging.warning('Expected number of evaluation epochs is 1, but '
                       'instead encountered `eval_on_train_input_config'
                       '.num_epochs` = '
                       '{}. Overwriting `num_epochs` to 1.'.format(
                           eval_on_train_input_config.num_epochs))
    eval_on_train_input_config.num_epochs = 1

  # update train_steps from config but only when non-zero value is provided
  if train_steps is None and train_config.num_steps != 0:
    train_steps = train_config.num_steps

  detection_model_fn = functools.partial(
      detection_model_fn_base, model_config=model_config)

  # Create the input functions for TRAIN/EVAL/PREDICT.
  train_input_fn = create_train_input_fn(
      train_config=train_config,
      train_input_config=train_input_config,
      model_config=model_config)
  eval_input_fns = [
      create_eval_input_fn(
          eval_config=eval_config,
          eval_input_config=eval_input_config,
          model_config=model_config) for eval_input_config in eval_input_configs
  ]
  eval_input_names = [
      eval_input_config.name for eval_input_config in eval_input_configs
  ]
  eval_on_train_input_fn = create_eval_input_fn(
      eval_config=eval_config,
      eval_input_config=eval_on_train_input_config,
      model_config=model_config)
  predict_input_fn = create_predict_input_fn(
      model_config=model_config, predict_input_config=eval_input_configs[0])

  # Read export_to_tpu from hparams if not passed.
  if export_to_tpu is None:
    export_to_tpu = hparams.get('export_to_tpu', False)
  tf.logging.info('create_estimator_and_inputs: use_tpu %s, export_to_tpu %s',
                  use_tpu, export_to_tpu)
  model_fn = model_fn_creator(detection_model_fn, configs, hparams, use_tpu,
                              postprocess_on_cpu)
  estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config)

  # Write the as-run pipeline config to disk.
  if run_config.is_chief and save_final_config:
    pipeline_config_final = create_pipeline_proto_from_configs(configs)
    config_util.save_pipeline_config(pipeline_config_final, estimator.model_dir)

  return dict(
      estimator=estimator,
      train_input_fn=train_input_fn,
      eval_input_fns=eval_input_fns,
      eval_input_names=eval_input_names,
      eval_on_train_input_fn=eval_on_train_input_fn,
      predict_input_fn=predict_input_fn,
      train_steps=train_steps)


def create_train_and_eval_specs(train_input_fn,
                                eval_input_fns,
                                eval_on_train_input_fn,
                                predict_input_fn,
                                train_steps,
                                eval_on_train_data=False,
                                final_exporter_name='Servo',
                                eval_spec_names=None):
  """Creates a `TrainSpec` and `EvalSpec`s.

  Args:
    train_input_fn: Function that produces features and labels on train data.
    eval_input_fns: A list of functions that produce features and labels on eval
      data.
    eval_on_train_input_fn: Function that produces features and labels for
      evaluation on train data.
    predict_input_fn: Function that produces features for inference.
    train_steps: Number of training steps.
    eval_on_train_data: Whether to evaluate model on training data. Default is
      False.
    final_exporter_name: String name given to `FinalExporter`.
    eval_spec_names: A list of string names for each `EvalSpec`.

  Returns:
    Tuple of `TrainSpec` and list of `EvalSpecs`. If `eval_on_train_data` is
    True, the last `EvalSpec` in the list will correspond to training data. The
    rest EvalSpecs in the list are evaluation datas.
  """
  train_spec = tf.estimator.TrainSpec(
      input_fn=train_input_fn, max_steps=train_steps)

  if eval_spec_names is None:
    eval_spec_names = [str(i) for i in range(len(eval_input_fns))]

  eval_specs = []
  for index, (eval_spec_name, eval_input_fn) in enumerate(
      zip(eval_spec_names, eval_input_fns)):
    # Uses final_exporter_name as exporter_name for the first eval spec for
    # backward compatibility.
    if index == 0:
      exporter_name = final_exporter_name
    else:
      exporter_name = '{}_{}'.format(final_exporter_name, eval_spec_name)
    exporter = tf.estimator.FinalExporter(
        name=exporter_name, serving_input_receiver_fn=predict_input_fn)
    eval_specs.append(
        tf.estimator.EvalSpec(
            name=eval_spec_name,
            input_fn=eval_input_fn,
            steps=None,
            exporters=exporter))

  if eval_on_train_data:
    eval_specs.append(
        tf.estimator.EvalSpec(
            name='eval_on_train', input_fn=eval_on_train_input_fn, steps=None))

  return train_spec, eval_specs
