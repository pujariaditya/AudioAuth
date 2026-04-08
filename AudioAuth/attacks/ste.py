"""
Straight-Through Estimator for Attack Pipeline

This module implements straight-through estimators (STE) to enable gradient flow
through binary decisions in the attack pipeline. STE allows binary operations
in the forward pass while passing gradients straight through in the backward pass,
circumventing the zero-gradient problem of hard thresholding.
"""

import torch
import torch.nn as nn
from typing import Optional, Union


class STEBinarize(torch.autograd.Function):
    """
    Straight-through estimator for binarization.

    Forward: Applies hard threshold to create binary output (> 0)
    Backward: Passes gradients straight through (identity gradient)
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: binarize input (1 if > 0, else 0).

        Args:
            input: Input tensor with binary values (0 or 1)

        Returns:
            Binary tensor with values 0 or 1
        """
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass: identity gradient (STE core idea).

        Args:
            grad_output: Gradient from the next layer

        Returns:
            Gradient w.r.t input, passed through unchanged
        """
        return grad_output


class STEStep(torch.autograd.Function):
    """
    Straight-through estimator for step function.

    Similar to STEBinarize but explicitly models as a step function.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: step function (0 if x <= 0, 1 if x > 0).

        Args:
            input: Input tensor

        Returns:
            Binary tensor with step function applied
        """
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass: identity gradient (STE core idea).

        Args:
            grad_output: Gradient from the next layer

        Returns:
            Gradient w.r.t input, passed through unchanged
        """
        return grad_output


class StraightThroughMask(nn.Module):
    """
    Module for creating binary masks with straight-through estimator.

    This module wraps the STE functionality in a nn.Module for easy integration
    with existing PyTorch models and attack pipelines.

    Automatically uses STE during training and hard decisions during evaluation.
    """

    def __init__(self):
        """Initialize the StraightThroughMask module."""
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Apply masking with appropriate gradient handling.

        During training: Uses STE for gradient flow
        During eval: Uses hard binary decisions

        Args:
            input: Input tensor with binary values (0 or 1)

        Returns:
            Binary mask with gradient support in training mode
        """
        if self.training:
            return STEBinarize.apply(input)
        else:
            return (input > 0).float()


class DifferentiableAttackSelector(nn.Module):
    """
    Differentiable attack selection using Gumbel-Softmax or STE.

    This module enables gradient flow through discrete attack selection decisions.

    Attributes:
        num_attacks: Number of possible attacks.
        temperature: Temperature for Gumbel-Softmax (lower = more discrete).
        use_gumbel: Whether to use Gumbel-Softmax (True) or STE (False).
    """

    def __init__(
        self,
        num_attacks: int,
        temperature: float = 1.0,
        use_gumbel: bool = False
    ):
        """
        Initialize the attack selector.

        Args:
            num_attacks: Number of possible attacks
            temperature: Temperature for Gumbel-Softmax (lower = more discrete)
            use_gumbel: Whether to use Gumbel-Softmax (True) or STE (False)
        """
        super().__init__()
        self.num_attacks = num_attacks
        self.temperature = temperature
        self.use_gumbel = use_gumbel

    def forward(
        self,
        attack_logits: torch.Tensor,
        hard: bool = True
    ) -> torch.Tensor:
        """
        Select attack with differentiable selection.

        Args:
            attack_logits: Logits for each attack [batch_size, num_attacks]
            hard: Whether to use hard (one-hot) or soft selection

        Returns:
            Attack selection tensor [batch_size, num_attacks]
        """
        if self.use_gumbel:
            if self.training:
                selection = torch.nn.functional.gumbel_softmax(
                    attack_logits,
                    tau=self.temperature,
                    hard=hard,
                    dim=-1
                )
            else:
                indices = attack_logits.argmax(dim=-1)
                selection = torch.nn.functional.one_hot(
                    indices,
                    num_classes=self.num_attacks
                ).float()
        else:
            if self.training and hard:
                probs = torch.softmax(attack_logits, dim=-1)
                indices = probs.argmax(dim=-1)
                one_hot = torch.nn.functional.one_hot(
                    indices,
                    num_classes=self.num_attacks
                ).float()
                # STE trick: hard selection forward, soft gradient backward
                selection = one_hot - probs.detach() + probs
            else:
                indices = attack_logits.argmax(dim=-1)
                selection = torch.nn.functional.one_hot(
                    indices,
                    num_classes=self.num_attacks
                ).float()

        return selection


class STEMaskGenerator(nn.Module):
    """
    Generate binary masks for attack regions using STE.

    This module generates masks that indicate which regions of audio
    are attacked, with automatic gradient flow during training.

    Attributes:
        mask_module: StraightThroughMask instance used for binarization.
    """

    def __init__(self):
        """Initialize mask generator."""
        super().__init__()
        self.mask_module = StraightThroughMask()

    def forward(
        self,
        attack_mask: torch.Tensor,
        segment_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate binary mask for attacked regions.

        Args:
            attack_mask: Binary mask of attack positions (0 or 1)
            segment_mask: Optional pre-computed segment mask

        Returns:
            Binary mask indicating attacked regions (0) vs clean regions (1)
        """
        attack_mask = self.mask_module(attack_mask)

        # Invert: 1 = watermark present, 0 = watermark absent
        watermark_mask = 1.0 - attack_mask

        if segment_mask is not None:
            watermark_mask = watermark_mask * segment_mask

        return watermark_mask


def create_ste_mask(
    input: torch.Tensor,
    training: bool = True
) -> torch.Tensor:
    """
    Convenience function to create STE mask.

    Args:
        input: Input tensor with binary values (0 or 1)
        training: Whether in training mode

    Returns:
        Binary mask with STE gradient support
    """
    if training:
        return STEBinarize.apply(input)
    else:
        return (input > 0).float()


def test_gradient_flow():
    """
    Test that gradients flow through STE operations.

    This function verifies that the STE implementation correctly
    passes gradients in the backward pass.
    """
    input_tensor = torch.tensor([0.0, 1.0, 0.0], requires_grad=True)

    binary = STEBinarize.apply(input_tensor)

    loss = binary.sum()

    loss.backward()

    assert input_tensor.grad is not None, "No gradients computed"
    assert not torch.allclose(input_tensor.grad, torch.zeros_like(input_tensor.grad)), \
        "Gradients are all zero"

    assert binary.shape == input_tensor.shape, "Output shape mismatch"
    assert torch.all((binary == 0) | (binary == 1)), "Output contains non-binary values"


__all__ = [
    'STEBinarize',
    'STEStep',
    'StraightThroughMask',
    'DifferentiableAttackSelector',
    'STEMaskGenerator',
    'create_ste_mask',
    'test_gradient_flow'
]


if __name__ == "__main__":
    test_gradient_flow()
