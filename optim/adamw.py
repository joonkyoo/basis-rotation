# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from torch.optim.optimizer import required

from .optimizer import OptimizerWithWeightStashing


class AdamWWithWeightStashing(OptimizerWithWeightStashing):
    """AdamW optimizer with weight stashing for pipeline-parallel training."""

    def __init__(self, modules, master_parameters,
                 num_versions, lr=required, betas=(0.9, 0.999),
                 weight_decay=0,
                 verbose_freq=0, macrobatch=False,
                 clip_grad=None, save_dir=None, stash_to_cpu=False, accumulation_steps=1):
        super(AdamWWithWeightStashing, self).__init__(
            optim_name='AdamW',
            modules=modules, master_parameters=master_parameters,
            num_versions=num_versions, lr=lr, betas=betas,
            weight_decay=weight_decay,
            verbose_freq=verbose_freq, macrobatch=macrobatch,
            clip_grad=clip_grad, save_dir=save_dir, stash_to_cpu=stash_to_cpu,
            accumulation_steps=accumulation_steps,
        )


class BasisRotationWithWeightStashing(OptimizerWithWeightStashing):
    """Adam with Basis Rotation (Algorithm 1) with weight stashing for pipeline-parallel training.

    Args:
        rotation_geometry: "bi" (bilateral) or "uni" (unilateral). See Algorithm 2.
        approx_source: "2nd" (second-order covariance) or "1st" (first-order gradient).
                       See Algorithm 2.
        subspace_update_frequency: How often to refresh the rotation basis (steps).
    """

    def __init__(self, modules, master_parameters,
                 num_versions, lr=required, betas=(0.9, 0.999),
                 weight_decay=0, subspace_update_frequency=10,
                 rotation_geometry='bi', approx_source='2nd',
                 verbose_freq=0, macrobatch=False,
                 clip_grad=None, save_dir=None, stash_to_cpu=False, accumulation_steps=1):
        super(BasisRotationWithWeightStashing, self).__init__(
            optim_name='BasisRotation',
            modules=modules, master_parameters=master_parameters,
            num_versions=num_versions, lr=lr, betas=betas,
            weight_decay=weight_decay,
            precondition_frequency=subspace_update_frequency,
            rotation_geometry=rotation_geometry,
            approx_source=approx_source,
            verbose_freq=verbose_freq, macrobatch=macrobatch,
            clip_grad=clip_grad, save_dir=save_dir, stash_to_cpu=stash_to_cpu,
            accumulation_steps=accumulation_steps,
        )
