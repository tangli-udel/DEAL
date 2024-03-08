import torch
import torch.nn as nn

def normalize_heatmap(heatmap):
    min_val = heatmap.min()
    max_val = heatmap.max()
    return (heatmap - min_val) / (max_val - min_val + 1e-8)  # Adding a small value for numerical stability


class SeparationLoss(nn.Module):
    def __init__(self):
        super(SeparationLoss, self).__init__()

    def forward(self, heatmaps):
        # Normalize heatmaps
        heatmaps = [normalize_heatmap(h) for h in heatmaps]
        heatmaps = torch.stack(heatmaps, dim=0)

        h_exp1 = heatmaps[:, None]
        h_exp2 = heatmaps[None, :]

        overlaps = h_exp1 * h_exp2

        lower_triangle_indices = torch.tril_indices(row=heatmaps.size(0), col=heatmaps.size(0), offset=-1)
        total_overlap = overlaps[lower_triangle_indices[0], lower_triangle_indices[1]].sum()

        return total_overlap


class ConsistencyLoss(nn.Module):
    def __init__(self):
        super(ConsistencyLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, heatmaps, target_heatmap):
        # Normalize heatmaps and target_heatmap
        heatmaps = [normalize_heatmap(h) for h in heatmaps]
        heatmaps = torch.stack(heatmaps, dim=0)
        target_heatmap = normalize_heatmap(target_heatmap)

        summed_heatmaps = heatmaps.sum(dim=0)
        loss = self.mse_loss(summed_heatmaps, target_heatmap)

        return loss


class BatchSeparationLoss(nn.Module):
    def __init__(self):
        super(BatchSeparationLoss, self).__init__()

    def forward(self, heatmaps_list):
        batch_size = len(heatmaps_list)
        batch_loss = 0.0

        for i in range(batch_size):
            heatmaps = heatmaps_list[i]

            # Normalize heatmaps
            heatmaps = [normalize_heatmap(h) for h in heatmaps]
            heatmaps = torch.stack(heatmaps, dim=0)

            h_exp1 = heatmaps[:, None]
            h_exp2 = heatmaps[None, :]

            overlaps = h_exp1 * h_exp2

            lower_triangle_indices = torch.tril_indices(row=heatmaps.size(0), col=heatmaps.size(0), offset=-1)
            total_overlap = overlaps[lower_triangle_indices[0], lower_triangle_indices[1]].sum()

            batch_loss += total_overlap

        return batch_loss / batch_size


class BatchConsistencyLoss(nn.Module):
    def __init__(self):
        super(BatchConsistencyLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, heatmaps_list, target_heatmaps_list):
        # First, normalize all heatmaps and target heatmaps
        norm_heatmaps_list = [[normalize_heatmap(h) for h in heatmaps] for heatmaps in heatmaps_list]
        norm_target_heatmaps_list = [normalize_heatmap(target) for target in target_heatmaps_list]

        # Stack heatmaps and target heatmaps along a new batch dimension
        # Here we assume each group has the same number of heatmaps, say 'n'.
        all_heatmaps = torch.stack([torch.stack(norm_heatmaps, dim=0) for norm_heatmaps in norm_heatmaps_list], dim=0)
        all_target_heatmaps = torch.stack(norm_target_heatmaps_list, dim=0)

        # Sum heatmaps along the 'group' dimension (dim=1)
        summed_heatmaps = all_heatmaps.sum(dim=1)

        # Calculate MSE loss
        loss = self.mse_loss(summed_heatmaps, all_target_heatmaps)

        # Sum the loss over all pixels and average over the batch
        batch_loss = loss.sum() / len(heatmaps_list)

        return batch_loss

class SparsityLoss(nn.Module):
    def __init__(self):
        super(SparsityLoss, self).__init__()

    def forward(self, heatmaps):
        # Ensure the heatmaps are in tensor form
        if isinstance(heatmaps, list):
            heatmaps = torch.stack(heatmaps, dim=0)

        # Normalize each heatmap in the batch
        heatmaps = torch.stack([normalize_heatmap(h) for h in heatmaps], dim=0)

        l1_loss = torch.abs(heatmaps).sum()  # L1 norm encourages sparsity
        return l1_loss