import torch
import torch.nn as nn

from ..attack import Attack


class FGSM(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.FGSM(model, eps=8/255)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8 / 255, k = 1):
        super().__init__("FGSM", model)
        self.eps = eps
        self.k = k;
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        images.requires_grad = True
        outputs = self.get_logits(images)

        # Calculate loss
        if self.targeted:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)

        c = 1e-3;
        print("ha")
        for i in range(self.k):
            noise = torch.randn_like(data)
            new_data = c*noise+data

            output2 = model(new_data)
            
            loss2 = F.nll_loss(output2, label)
            # changed loss to cost
            data_grad += (noise*(loss2-cost)/c)/k
        
        # Update adversarial images, remove? test later
        # grad = torch.autograd.grad(
        #     cost, images, retain_graph=False, create_graph=False
        # )[0]

        # changed grad here to data_grad
        adv_images = images + self.eps * data_grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images
