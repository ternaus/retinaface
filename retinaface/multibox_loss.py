from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from retinaface.box_utils import log_sum_exp, match


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function.

    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter.
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(
        self,
        num_classes: int,
        overlap_thresh: float,
        prior_for_matching: bool,
        bkg_label: int,
        neg_mining: bool,
        neg_pos: int,
        neg_overlap: float,
        encode_target: bool,
        priors: torch.Tensor,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]
        self.priors = priors

    def forward(
        self, predictions: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Multibox Loss.

        Args:
            predictions: A tuple containing locations predictions, confidence predictions,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size, num_priors, num_classes)
                loc shape: torch.size(batch_size, num_priors, 4)
                priors shape: torch.size(num_priors, 4)

            targets: Ground truth boxes and labels_gt for a batch,
                shape: [batch_size, num_objs, 5] (last box_index is the label).
        """
        locations_data, confidence_data, landmark_data = predictions

        priors = self.priors.to(targets[0].device)
        defaults = priors.data

        num_predicted_boxes = locations_data.size(0)
        num_priors = priors.size(0)

        # match priors (default boxes) and ground truth boxes
        boxes_t = torch.zeros(num_predicted_boxes, num_priors, 4).to(targets[0].device)
        landmarks_t = torch.zeros(num_predicted_boxes, num_priors, 10).to(targets[0].device)
        conf_t = torch.zeros(num_predicted_boxes, num_priors).to(targets[0].device).long()

        for box_index in range(num_predicted_boxes):
            box_gt = targets[box_index][:, :4].data
            landmarks_gt = targets[box_index][:, 4:14].data
            labels_gt = targets[box_index][:, 14].data

            match(
                self.threshold,
                box_gt,
                defaults,
                self.variance,
                labels_gt,
                landmarks_gt,
                boxes_t,
                conf_t,
                landmarks_t,
                box_index,
            )

        # landmark Loss (Smooth L1) Shape: [batch, num_priors, 10]
        positive_1 = conf_t > torch.zeros_like(conf_t)
        num_positive_landmarks = positive_1.long().sum(1, keepdim=True)
        n1 = max(num_positive_landmarks.data.sum().float(), 1)  # type: ignore
        pos_idx1 = positive_1.unsqueeze(positive_1.dim()).expand_as(landmark_data)
        landmarks_p = landmark_data[pos_idx1].view(-1, 10)
        landmarks_t = landmarks_t[pos_idx1].view(-1, 10)
        loss_landm = F.smooth_l1_loss(landmarks_p, landmarks_t, reduction="sum")

        positive = conf_t != torch.zeros_like(conf_t)
        conf_t[positive] = 1

        # Localization Loss (Smooth L1) Shape: [batch, num_priors, 4]
        pos_idx = positive.unsqueeze(positive.dim()).expand_as(locations_data)
        loc_p = locations_data[pos_idx].view(-1, 4)
        boxes_t = boxes_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, boxes_t, reduction="sum")

        # Compute max conf across batch for hard negative mining
        batch_conf = confidence_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[positive.view(-1, 1)] = 0  # filter out positive boxes for now
        loss_c = loss_c.view(num_predicted_boxes, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = positive.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=positive.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = positive.unsqueeze(2).expand_as(confidence_data)
        neg_idx = neg.unsqueeze(2).expand_as(confidence_data)
        conf_p = confidence_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(positive + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction="sum")

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        n = max(num_pos.data.sum().float(), 1)  # type: ignore

        return loss_l / n, loss_c / n, loss_landm / n1
