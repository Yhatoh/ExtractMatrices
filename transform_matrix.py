import torch
import numpy as np

# this is for algorithm GPTQ to reconstruct the quantized matrix
def final_matrix(qzeros, scales, qweight, bits, group_size, wf):
    if bits in [2, 4, 8]:
        if wf.device != qzeros.device:
            wf = wf.to(qzeros.device)

        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(qzeros, 2).expand(-1, -1, 32 // bits),
            wf.unsqueeze(0),
        ).to(torch.int16 if bits == 8 else torch.int8)

        zeros = zeros + 1
        zeros = torch.bitwise_and(
            zeros, (2**bits) - 1
        )

        zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])

        scales = scales
        scales = scales.reshape(-1, 1, scales.shape[-1])

        weight = torch.bitwise_right_shift(
            torch.unsqueeze(qweight, 1).expand(-1, 32 // bits, -1),
            wf.unsqueeze(-1),
        ).to(torch.int16 if bits == 8 else torch.int8)
        weight = torch.bitwise_and(weight, (2**bits) - 1)
        weight = weight.reshape(-1, group_size, weight.shape[2])

        weight = scales * (weight - zeros)
        weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
    else:
        zeros = qzeros.reshape(qzeros.shape[0], qzeros.shape[1] // 3, 3, 1).expand(
            -1, -1, -1, 12
        )
        zeros = zeros >> wf.unsqueeze(0)
        zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | ((zeros[:, :, 1, 0] << 2) & 0x4)
        zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | ((zeros[:, :, 2, 0] << 1) & 0x6)
        zeros = zeros & 0x7
        zeros = torch.cat(
            [zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]],
            dim=2,
        )

        zeros = zeros + 1
        zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])

        scales = scales
        scales = scales.reshape(-1, 1, scales.shape[-1])

        weight = qweight.reshape(qweight.shape[0] // 3, 3, 1, qweight.shape[1]).expand(
            -1, -1, 12, -1
        )
        weight = (weight >> wf.unsqueeze(-1)) & 0x7
        weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
        weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
        weight = weight & 0x7
        weight = torch.cat([weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1)
        weight = weight.reshape(-1, group_size, weight.shape[2])

        weight = scales * (weight - zeros)
        weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
    return weight
