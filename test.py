from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import sys
import numpy as np
import inspect
import torch

# where you will save the matrices
save_dir = "/data/matrix/matrices/"
# where you will get the models
path = "/data/matrix/models"


# this is to get the sizeof of an objetc
def get_bytes(obj):
    return sys.getsizeof(obj)

# this is to get the bytes of a vector
def get_bytes_vector(m):
    ret = get_bytes(m)
    for i in range(m.shape[0]):
        ret += get_bytes(m[i])
    return ret

# this it to get the bytes of a matrix
def get_bytes_matrix(m):
    ret = get_bytes(m)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ret += get_bytes(m[i][j])
    return ret

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

# write the matrix in the file `name`
def write_matrices(name, matrices):
    #matrices = matrices.astype('f')
    n = matrices.shape[0]
    m = matrices.shape[1]

    with open(name, 'wb') as file:
        matrices.tofile(file)

# get the bytes used by some matrices of the llama model
def llama_bytes_matrices(model):
    ret_bytes = 0
    for i in range(1):
        qweight = model.model.layers[i].mlp.down_proj.qweight
        qzeros = model.model.layers[i].mlp.down_proj.qzeros
        scales = model.model.layers[i].mlp.down_proj.scales
        bits = model.model.layers[i].mlp.down_proj.bits
        wf = model.model.layers[i].mlp.down_proj.wf
        group_size = model.model.layers[i].mlp.down_proj.group_size

        qweight_bytes = get_bytes_matrix(qweight)
        qzeros_bytes = get_bytes_matrix(qzeros)
        scales_bytes = get_bytes_matrix(scales)
        bits_bytes = get_bytes(bits)
        wf_bytes = get_bytes(wf)

        ret_bytes += qweight_bytes + qzeros_bytes + scales_bytes + wf_bytes + bits_bytes

        qweight = model.model.layers[i].mlp.gate_proj.qweight
        qzeros = model.model.layers[i].mlp.gate_proj.qzeros
        scales = model.model.layers[i].mlp.gate_proj.scales
        bits = model.model.layers[i].mlp.gate_proj.bits
        wf = model.model.layers[i].mlp.gate_proj.wf
        group_size = model.model.layers[i].mlp.gate_proj.group_size

        qweight_bytes = get_bytes_matrix(qweight)
        qzeros_bytes = get_bytes_matrix(qzeros)
        scales_bytes = get_bytes_matrix(scales)
        bits_bytes = get_bytes(bits)
        wf_bytes = get_bytes(wf)

        ret_bytes += qweight_bytes + qzeros_bytes + scales_bytes + wf_bytes + bits_bytes

        qweight = model.model.layers[i].mlp.up_proj.qweight
        qzeros = model.model.layers[i].mlp.up_proj.qzeros
        scales = model.model.layers[i].mlp.up_proj.scales
        bits = model.model.layers[i].mlp.up_proj.bits
        wf = model.model.layers[i].mlp.up_proj.wf
        group_size = model.model.layers[i].mlp.up_proj.group_size

        qweight_bytes = get_bytes_matrix(qweight)
        qzeros_bytes = get_bytes_matrix(qzeros)
        scales_bytes = get_bytes_matrix(scales)
        bits_bytes = get_bytes(bits)
        wf_bytes = get_bytes(wf)

        ret_bytes += qweight_bytes + qzeros_bytes + scales_bytes + wf_bytes + bits_bytes

        qweight = model.model.layers[i].self_attn.k_proj.qweight
        qzeros = model.model.layers[i].self_attn.k_proj.qzeros
        scales = model.model.layers[i].self_attn.k_proj.scales
        bits = model.model.layers[i].self_attn.k_proj.bits
        wf = model.model.layers[i].self_attn.k_proj.wf
        group_size = model.model.layers[i].self_attn.k_proj.group_size

        qweight_bytes = get_bytes_matrix(qweight)
        qzeros_bytes = get_bytes_matrix(qzeros)
        scales_bytes = get_bytes_matrix(scales)
        bits_bytes = get_bytes(bits)
        wf_bytes = get_bytes(wf)

        ret_bytes += qweight_bytes + qzeros_bytes + scales_bytes + wf_bytes + bits_bytes

        qweight = model.model.layers[i].self_attn.o_proj.qweight
        qzeros = model.model.layers[i].self_attn.o_proj.qzeros
        scales = model.model.layers[i].self_attn.o_proj.scales
        bits = model.model.layers[i].self_attn.o_proj.bits
        wf = model.model.layers[i].self_attn.o_proj.wf
        group_size = model.model.layers[i].self_attn.o_proj.group_size

        qweight_bytes = get_bytes_matrix(qweight)
        qzeros_bytes = get_bytes_matrix(qzeros)
        scales_bytes = get_bytes_matrix(scales)
        bits_bytes = get_bytes(bits)
        wf_bytes = get_bytes(wf)

        ret_bytes += qweight_bytes + qzeros_bytes + scales_bytes + wf_bytes + bits_bytes

        qweight = model.model.layers[i].self_attn.q_proj.qweight
        qzeros = model.model.layers[i].self_attn.q_proj.qzeros
        scales = model.model.layers[i].self_attn.q_proj.scales
        bits = model.model.layers[i].self_attn.q_proj.bits
        wf = model.model.layers[i].self_attn.q_proj.wf
        group_size = model.model.layers[i].self_attn.q_proj.group_size

        qweight_bytes = get_bytes_matrix(qweight)
        qzeros_bytes = get_bytes_matrix(qzeros)
        scales_bytes = get_bytes_matrix(scales)
        bits_bytes = get_bytes(bits)
        wf_bytes = get_bytes(wf)

        ret_bytes += qweight_bytes + qzeros_bytes + scales_bytes + wf_bytes + bits_bytes

        qweight = model.model.layers[i].self_attn.v_proj.qweight
        qzeros = model.model.layers[i].self_attn.v_proj.qzeros
        scales = model.model.layers[i].self_attn.v_proj.scales
        bits = model.model.layers[i].self_attn.v_proj.bits
        wf = model.model.layers[i].self_attn.v_proj.wf
        group_size = model.model.layers[i].self_attn.v_proj.group_size

        qweight_bytes = get_bytes_matrix(qweight)
        qzeros_bytes = get_bytes_matrix(qzeros)
        scales_bytes = get_bytes_matrix(scales)
        bits_bytes = get_bytes(bits)
        wf_bytes = get_bytes(wf)

        ret_bytes += qweight_bytes + qzeros_bytes + scales_bytes + wf_bytes + bits_bytes
    return ret_bytes * len(model.model.layers)

# get the bytes used by some matrices of the mixtral model
def mixtral_bytes_matrices(model):
    ret_bytes = 0
    for i in range(1):
        for j in range(len(model.model.layers[i].block_sparse_moe.experts)):
            qweight = model.model.layers[i].block_sparse_moe.experts[j].w1.qweight
            qzeros = model.model.layers[i].block_sparse_moe.experts[j].w1.qzeros
            scales = model.model.layers[i].block_sparse_moe.experts[j].w1.scales
            bits = model.model.layers[i].block_sparse_moe.experts[j].w1.bits
            wf = model.model.layers[i].block_sparse_moe.experts[j].w1.wf
            group_size = model.model.layers[i].block_sparse_moe.experts[j].w1.group_size

            qweight_bytes = get_bytes_matrix(qweight)
            qzeros_bytes = get_bytes_matrix(qzeros)
            scales_bytes = get_bytes_matrix(scales)
            bits_bytes = get_bytes(bits)
            wf_bytes = get_bytes(wf)

            ret_bytes += qweight_bytes + qzeros_bytes + scales_bytes + wf_bytes + bits_bytes

            qweight = model.model.layers[i].block_sparse_moe.experts[j].w2.qweight
            qzeros = model.model.layers[i].block_sparse_moe.experts[j].w2.qzeros
            scales = model.model.layers[i].block_sparse_moe.experts[j].w2.scales
            bits = model.model.layers[i].block_sparse_moe.experts[j].w2.bits
            wf = model.model.layers[i].block_sparse_moe.experts[j].w2.wf
            group_size = model.model.layers[i].block_sparse_moe.experts[j].w2.group_size

            qweight_bytes = get_bytes_matrix(qweight)
            qzeros_bytes = get_bytes_matrix(qzeros)
            scales_bytes = get_bytes_matrix(scales)
            bits_bytes = get_bytes(bits)
            wf_bytes = get_bytes(wf)

            ret_bytes += qweight_bytes + qzeros_bytes + scales_bytes + wf_bytes + bits_bytes

            qweight = model.model.layers[i].block_sparse_moe.experts[j].w3.qweight
            qzeros = model.model.layers[i].block_sparse_moe.experts[j].w3.qzeros
            scales = model.model.layers[i].block_sparse_moe.experts[j].w3.scales
            bits = model.model.layers[i].block_sparse_moe.experts[j].w3.bits
            wf = model.model.layers[i].block_sparse_moe.experts[j].w3.wf
            group_size = model.model.layers[i].block_sparse_moe.experts[j].w3.group_size

            qweight_bytes = get_bytes_matrix(qweight)
            qzeros_bytes = get_bytes_matrix(qzeros)
            scales_bytes = get_bytes_matrix(scales)
            bits_bytes = get_bytes(bits)
            wf_bytes = get_bytes(wf)

            ret_bytes += qweight_bytes + qzeros_bytes + scales_bytes + wf_bytes + bits_bytes

        qweight = model.model.layers[i].self_attn.k_proj.qweight
        qzeros = model.model.layers[i].self_attn.k_proj.qzeros
        scales = model.model.layers[i].self_attn.k_proj.scales
        bits = model.model.layers[i].self_attn.k_proj.bits
        wf = model.model.layers[i].self_attn.k_proj.wf
        group_size = model.model.layers[i].self_attn.k_proj.group_size

        qweight_bytes = get_bytes_matrix(qweight)
        qzeros_bytes = get_bytes_matrix(qzeros)
        scales_bytes = get_bytes_matrix(scales)
        bits_bytes = get_bytes(bits)
        wf_bytes = get_bytes(wf)

        ret_bytes += qweight_bytes + qzeros_bytes + scales_bytes + wf_bytes + bits_bytes

        qweight = model.model.layers[i].self_attn.o_proj.qweight
        qzeros = model.model.layers[i].self_attn.o_proj.qzeros
        scales = model.model.layers[i].self_attn.o_proj.scales
        bits = model.model.layers[i].self_attn.o_proj.bits
        wf = model.model.layers[i].self_attn.o_proj.wf
        group_size = model.model.layers[i].self_attn.o_proj.group_size

        qweight_bytes = get_bytes_matrix(qweight)
        qzeros_bytes = get_bytes_matrix(qzeros)
        scales_bytes = get_bytes_matrix(scales)
        bits_bytes = get_bytes(bits)
        wf_bytes = get_bytes(wf)

        ret_bytes += qweight_bytes + qzeros_bytes + scales_bytes + wf_bytes + bits_bytes

        qweight = model.model.layers[i].self_attn.q_proj.qweight
        qzeros = model.model.layers[i].self_attn.q_proj.qzeros
        scales = model.model.layers[i].self_attn.q_proj.scales
        bits = model.model.layers[i].self_attn.q_proj.bits
        wf = model.model.layers[i].self_attn.q_proj.wf
        group_size = model.model.layers[i].self_attn.q_proj.group_size

        qweight_bytes = get_bytes_matrix(qweight)
        qzeros_bytes = get_bytes_matrix(qzeros)
        scales_bytes = get_bytes_matrix(scales)
        bits_bytes = get_bytes(bits)
        wf_bytes = get_bytes(wf)

        ret_bytes += qweight_bytes + qzeros_bytes + scales_bytes + wf_bytes + bits_bytes

        qweight = model.model.layers[i].self_attn.v_proj.qweight
        qzeros = model.model.layers[i].self_attn.v_proj.qzeros
        scales = model.model.layers[i].self_attn.v_proj.scales
        bits = model.model.layers[i].self_attn.v_proj.bits
        wf = model.model.layers[i].self_attn.v_proj.wf
        group_size = model.model.layers[i].self_attn.v_proj.group_size

        qweight_bytes = get_bytes_matrix(qweight)
        qzeros_bytes = get_bytes_matrix(qzeros)
        scales_bytes = get_bytes_matrix(scales)
        bits_bytes = get_bytes(bits)
        wf_bytes = get_bytes(wf)

        ret_bytes += qweight_bytes + qzeros_bytes + scales_bytes + wf_bytes + bits_bytes
    return ret_bytes * len(model.model.layers)

# get the quantized matrix that is used to predict and save it in a binary file
# this is for llama 2 model
def llama_GPTQ_matrices(model, name, quant):
    files = []
    shapes = []
    dir_file = save_dir + name + "/" + quant + "/"
    for i in range(len(model.model.layers)):
        qweight = model.model.layers[i].mlp.down_proj.qweight
        qzeros = model.model.layers[i].mlp.down_proj.qzeros
        scales = model.model.layers[i].mlp.down_proj.scales
        bits = model.model.layers[i].mlp.down_proj.bits
        wf = model.model.layers[i].mlp.down_proj.wf
        group_size = model.model.layers[i].mlp.down_proj.group_size

        ret = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()

        name_file = "mlp-down_proj-" + str(i)
        shapes.append("x".join([str(ret.shape[0]), str(ret.shape[1])]))
        files.append(name_file)
        
        write_matrices(dir_file + name_file, ret)

        qweight = model.model.layers[i].mlp.gate_proj.qweight
        qzeros = model.model.layers[i].mlp.gate_proj.qzeros
        scales = model.model.layers[i].mlp.gate_proj.scales
        bits = model.model.layers[i].mlp.gate_proj.bits
        wf = model.model.layers[i].mlp.gate_proj.wf
        group_size = model.model.layers[i].mlp.gate_proj.group_size

        ret = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()
        name_file = "mlp-gate_proj-" + str(i)
        shapes.append("x".join([str(ret.shape[0]), str(ret.shape[1])]))
        files.append(name_file)

        write_matrices(dir_file + name_file, ret)

        qweight = model.model.layers[i].mlp.up_proj.qweight
        qzeros = model.model.layers[i].mlp.up_proj.qzeros
        scales = model.model.layers[i].mlp.up_proj.scales
        bits = model.model.layers[i].mlp.up_proj.bits
        wf = model.model.layers[i].mlp.up_proj.wf
        group_size = model.model.layers[i].mlp.up_proj.group_size

        ret = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()
        name_file = "mlp-up_proj-" + str(i)
        shapes.append("x".join([str(ret.shape[0]), str(ret.shape[1])]))
        files.append(name_file)

        write_matrices(dir_file + name_file, ret)

        qweight = model.model.layers[i].self_attn.k_proj.qweight
        qzeros = model.model.layers[i].self_attn.k_proj.qzeros
        scales = model.model.layers[i].self_attn.k_proj.scales
        bits = model.model.layers[i].self_attn.k_proj.bits
        wf = model.model.layers[i].self_attn.k_proj.wf
        group_size = model.model.layers[i].self_attn.k_proj.group_size

        ret = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()

        name_file = "self_attn-k_proj-" + str(i)
        shapes.append("x".join([str(ret.shape[0]), str(ret.shape[1])]))
        files.append(name_file)

        write_matrices(dir_file + name_file, ret)

        qweight = model.model.layers[i].self_attn.o_proj.qweight
        qzeros = model.model.layers[i].self_attn.o_proj.qzeros
        scales = model.model.layers[i].self_attn.o_proj.scales
        bits = model.model.layers[i].self_attn.o_proj.bits
        wf = model.model.layers[i].self_attn.o_proj.wf
        group_size = model.model.layers[i].self_attn.o_proj.group_size

        ret = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()

        name_file = "self_attn-o_proj-" + str(i)
        shapes.append("x".join([str(ret.shape[0]), str(ret.shape[1])]))
        files.append(name_file)

        write_matrices(dir_file + name_file, ret)

        qweight = model.model.layers[i].self_attn.q_proj.qweight
        qzeros = model.model.layers[i].self_attn.q_proj.qzeros
        scales = model.model.layers[i].self_attn.q_proj.scales
        bits = model.model.layers[i].self_attn.q_proj.bits
        wf = model.model.layers[i].self_attn.q_proj.wf
        group_size = model.model.layers[i].self_attn.q_proj.group_size

        ret = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()

        name_file = "self_attn-q_proj-" + str(i)
        shapes.append("x".join([str(ret.shape[0]), str(ret.shape[1])]))
        files.append(name_file)

        write_matrices(dir_file + name_file, ret)

        qweight = model.model.layers[i].self_attn.v_proj.qweight
        qzeros = model.model.layers[i].self_attn.v_proj.qzeros
        scales = model.model.layers[i].self_attn.v_proj.scales
        bits = model.model.layers[i].self_attn.v_proj.bits
        wf = model.model.layers[i].self_attn.v_proj.wf
        group_size = model.model.layers[i].self_attn.v_proj.group_size

        ret = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()

        name_file = "self_attn-v_proj-" + str(i)
        shapes.append("x".join([str(ret.shape[0]), str(ret.shape[1])]))
        files.append(name_file)

        write_matrices(dir_file + name_file, ret)
    print("command: ", "--files " + ":".join(files), "--sizes " + ":".join(shapes))

# not quantized models
def llama_fp16_matrices(model, name, quant):
    files = []
    shapes = []
    dir_file = save_dir + name + "/" + quant + "/"
    for i in range(len(model.model.layers)):
        ret = model.model.layers[i].mlp.down_proj.weight

        name_file = "mlp-down_proj-" + str(i)
        shapes.append("x".join([str(ret.shape[0]), str(ret.shape[1])]))
        files.append(name_file)
        
        write_matrices(dir_file + name_file, ret)

        ret = model.model.layers[i].mlp.gate_proj.weight

        name_file = "mlp-gate_proj-" + str(i)
        shapes.append("x".join([str(ret.shape[0]), str(ret.shape[1])]))
        files.append(name_file)

        write_matrices(dir_file + name_file, ret)

        ret = model.model.layers[i].mlp.up_proj.weight

        name_file = "mlp-up_proj-" + str(i)
        shapes.append("x".join([str(ret.shape[0]), str(ret.shape[1])]))
        files.append(name_file)

        write_matrices(dir_file + name_file, ret)

        ret = model.model.layers[i].self_attn.k_proj.weight

        name_file = "self_attn-k_proj-" + str(i)
        shapes.append("x".join([str(ret.shape[0]), str(ret.shape[1])]))
        files.append(name_file)

        write_matrices(dir_file + name_file, ret)

        ret = model.model.layers[i].self_attn.o_proj.weight

        name_file = "self_attn-o_proj-" + str(i)
        shapes.append("x".join([str(ret.shape[0]), str(ret.shape[1])]))
        files.append(name_file)

        write_matrices(dir_file + name_file, ret)

        ret = model.model.layers[i].self_attn.q_proj.weight

        name_file = "self_attn-q_proj-" + str(i)
        shapes.append("x".join([str(ret.shape[0]), str(ret.shape[1])]))
        files.append(name_file)

        write_matrices(dir_file + name_file, ret)

        ret = model.model.layers[i].self_attn.v_proj.weight

        name_file = "self_attn-v_proj-" + str(i)
        shapes.append("x".join([str(ret.shape[0]), str(ret.shape[1])]))
        files.append(name_file)

        write_matrices(dir_file + name_file, ret)
    print("command: ", "--files " + ":".join(files), "--sizes " + ":".join(shapes))


# get the quantized matrix that is used to predict and save it in a binary file
# this is for mixtral model
def mixtral_GPTQ_matrices(model, name, quant):
    files = []
    shapes = []
    dir_file = save_dir + name + "/" + quant + "/"
    for i in range(len(model.model.layers)):
        for j in range(len(model.model.layers[i].block_sparse_moe.experts)):
            qweight = model.model.layers[i].block_sparse_moe.experts[j].w1.qweight
            qzeros = model.model.layers[i].block_sparse_moe.experts[j].w1.qzeros
            scales = model.model.layers[i].block_sparse_moe.experts[j].w1.scales
            bits = model.model.layers[i].block_sparse_moe.experts[j].w1.bits
            wf = model.model.layers[i].block_sparse_moe.experts[j].w1.wf
            group_size = model.model.layers[i].block_sparse_moe.experts[j].w1.group_size

            ret = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()
            name_file = "block_sparse-w1-" + str(j) + "-" + str(i)
            
            files.append(name_file)
            shapes.append("x".join([str(ret.shape[0]), str(ret.shape[1])]))

            write_matrices(dir_file + name_file, ret)

            qweight = model.model.layers[i].block_sparse_moe.experts[j].w2.qweight
            qzeros = model.model.layers[i].block_sparse_moe.experts[j].w2.qzeros
            scales = model.model.layers[i].block_sparse_moe.experts[j].w2.scales
            bits = model.model.layers[i].block_sparse_moe.experts[j].w2.bits
            wf = model.model.layers[i].block_sparse_moe.experts[j].w2.wf
            group_size = model.model.layers[i].block_sparse_moe.experts[j].w2.group_size

            ret = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()
            name_file = "block_sparse-w2-" + str(j) + "-" + str(i)

            files.append(name_file)
            shapes.append("x".join([str(ret.shape[0]), str(ret.shape[1])]))

            write_matrices(dir_file + name_file, ret)

            qweight = model.model.layers[i].block_sparse_moe.experts[j].w3.qweight
            qzeros = model.model.layers[i].block_sparse_moe.experts[j].w3.qzeros
            scales = model.model.layers[i].block_sparse_moe.experts[j].w3.scales
            bits = model.model.layers[i].block_sparse_moe.experts[j].w3.bits
            wf = model.model.layers[i].block_sparse_moe.experts[j].w3.wf
            group_size = model.model.layers[i].block_sparse_moe.experts[j].w3.group_size

            ret = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()
            name_file = "block_sparse-w3-" + str(j) + "-" + str(i)

            shapes.append("x".join([str(ret.shape[0]), str(ret.shape[1])]))
            files.append(name_file)

            write_matrices(dir_file + name_file, ret)

        qweight = model.model.layers[i].self_attn.k_proj.qweight
        qzeros = model.model.layers[i].self_attn.k_proj.qzeros
        scales = model.model.layers[i].self_attn.k_proj.scales
        bits = model.model.layers[i].self_attn.k_proj.bits
        wf = model.model.layers[i].self_attn.k_proj.wf
        group_size = model.model.layers[i].self_attn.k_proj.group_size

        ret = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()

        name_file = "self_attn-k_proj-" + str(i)
        shapes.append("x".join([str(ret.shape[0]), str(ret.shape[1])]))
        files.append(name_file)

        write_matrices(dir_file + name_file, ret)

        qweight = model.model.layers[i].self_attn.o_proj.qweight
        qzeros = model.model.layers[i].self_attn.o_proj.qzeros
        scales = model.model.layers[i].self_attn.o_proj.scales
        bits = model.model.layers[i].self_attn.o_proj.bits
        wf = model.model.layers[i].self_attn.o_proj.wf
        group_size = model.model.layers[i].self_attn.o_proj.group_size

        ret = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()

        name_file = "self_attn-o_proj-" + str(i)
        shapes.append("x".join([str(ret.shape[0]), str(ret.shape[1])]))
        files.append(name_file)

        write_matrices(dir_file + name_file, ret)

        qweight = model.model.layers[i].self_attn.q_proj.qweight
        qzeros = model.model.layers[i].self_attn.q_proj.qzeros
        scales = model.model.layers[i].self_attn.q_proj.scales
        bits = model.model.layers[i].self_attn.q_proj.bits
        wf = model.model.layers[i].self_attn.q_proj.wf
        group_size = model.model.layers[i].self_attn.q_proj.group_size

        ret = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()

        name_file = "self_attn-q_proj-" + str(i)
        shapes.append("x".join([str(ret.shape[0]), str(ret.shape[1])]))
        files.append(name_file)

        write_matrices(dir_file + name_file, ret)

        qweight = model.model.layers[i].self_attn.v_proj.qweight
        qzeros = model.model.layers[i].self_attn.v_proj.qzeros
        scales = model.model.layers[i].self_attn.v_proj.scales
        bits = model.model.layers[i].self_attn.v_proj.bits
        wf = model.model.layers[i].self_attn.v_proj.wf
        group_size = model.model.layers[i].self_attn.v_proj.group_size

        ret = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()

        name_file = "self_attn-v_proj-" + str(i)
        shapes.append("x".join([str(ret.shape[0]), str(ret.shape[1])]))
        files.append(name_file)

        write_matrices(dir_file + name_file, ret)
    print("command: ", "--files " + ":".join(files), "--sizes " + ":".join(shapes))

def llama_matrices_stacked(model, name, quant):
    files = []
    shapes = []
    dir_file = save_dir + name + "/" + quant + "/stacked/"
    for i in range(1):
        qweight = model.model.layers[i].mlp.down_proj.qweight
        qzeros = model.model.layers[i].mlp.down_proj.qzeros
        scales = model.model.layers[i].mlp.down_proj.scales
        bits = model.model.layers[i].mlp.down_proj.bits
        wf = model.model.layers[i].mlp.down_proj.wf
        group_size = model.model.layers[i].mlp.down_proj.group_size

        down_proj = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()

        qweight = model.model.layers[i].mlp.gate_proj.qweight
        qzeros = model.model.layers[i].mlp.gate_proj.qzeros
        scales = model.model.layers[i].mlp.gate_proj.scales
        bits = model.model.layers[i].mlp.gate_proj.bits
        wf = model.model.layers[i].mlp.gate_proj.wf
        group_size = model.model.layers[i].mlp.gate_proj.group_size

        gate_proj = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()

        qweight = model.model.layers[i].mlp.up_proj.qweight
        qzeros = model.model.layers[i].mlp.up_proj.qzeros
        scales = model.model.layers[i].mlp.up_proj.scales
        bits = model.model.layers[i].mlp.up_proj.bits
        wf = model.model.layers[i].mlp.up_proj.wf
        group_size = model.model.layers[i].mlp.up_proj.group_size

        up_proj = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()

        qweight = model.model.layers[i].self_attn.k_proj.qweight
        qzeros = model.model.layers[i].self_attn.k_proj.qzeros
        scales = model.model.layers[i].self_attn.k_proj.scales
        bits = model.model.layers[i].self_attn.k_proj.bits
        wf = model.model.layers[i].self_attn.k_proj.wf
        group_size = model.model.layers[i].self_attn.k_proj.group_size

        k_proj = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()

        qweight = model.model.layers[i].self_attn.o_proj.qweight
        qzeros = model.model.layers[i].self_attn.o_proj.qzeros
        scales = model.model.layers[i].self_attn.o_proj.scales
        bits = model.model.layers[i].self_attn.o_proj.bits
        wf = model.model.layers[i].self_attn.o_proj.wf
        group_size = model.model.layers[i].self_attn.o_proj.group_size

        o_proj = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()

        qweight = model.model.layers[i].self_attn.q_proj.qweight
        qzeros = model.model.layers[i].self_attn.q_proj.qzeros
        scales = model.model.layers[i].self_attn.q_proj.scales
        bits = model.model.layers[i].self_attn.q_proj.bits
        wf = model.model.layers[i].self_attn.q_proj.wf
        group_size = model.model.layers[i].self_attn.q_proj.group_size

        q_proj = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()

        qweight = model.model.layers[i].self_attn.v_proj.qweight
        qzeros = model.model.layers[i].self_attn.v_proj.qzeros
        scales = model.model.layers[i].self_attn.v_proj.scales
        bits = model.model.layers[i].self_attn.v_proj.bits
        wf = model.model.layers[i].self_attn.v_proj.wf
        group_size = model.model.layers[i].self_attn.v_proj.group_size

        v_proj = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()

    for i in range(1, len(model.model.layers)):
        qweight = model.model.layers[i].mlp.down_proj.qweight
        qzeros = model.model.layers[i].mlp.down_proj.qzeros
        scales = model.model.layers[i].mlp.down_proj.scales
        bits = model.model.layers[i].mlp.down_proj.bits
        wf = model.model.layers[i].mlp.down_proj.wf
        group_size = model.model.layers[i].mlp.down_proj.group_size

        down_proj = np.concatenate((down_proj, final_matrix(qzeros, scales, qweight, bits,
            group_size, wf).numpy()))

    
        qweight = model.model.layers[i].mlp.gate_proj.qweight
        qzeros = model.model.layers[i].mlp.gate_proj.qzeros
        scales = model.model.layers[i].mlp.gate_proj.scales
        bits = model.model.layers[i].mlp.gate_proj.bits
        wf = model.model.layers[i].mlp.gate_proj.wf
        group_size = model.model.layers[i].mlp.gate_proj.group_size

        gate_proj = np.concatenate((gate_proj, final_matrix(qzeros, scales, qweight, bits,
            group_size, wf).numpy()))

        qweight = model.model.layers[i].mlp.up_proj.qweight
        qzeros = model.model.layers[i].mlp.up_proj.qzeros
        scales = model.model.layers[i].mlp.up_proj.scales
        bits = model.model.layers[i].mlp.up_proj.bits
        wf = model.model.layers[i].mlp.up_proj.wf
        group_size = model.model.layers[i].mlp.up_proj.group_size

        up_proj = np.concatenate((up_proj, final_matrix(qzeros, scales, qweight, bits,
            group_size, wf).numpy()))

        qweight = model.model.layers[i].self_attn.k_proj.qweight
        qzeros = model.model.layers[i].self_attn.k_proj.qzeros
        scales = model.model.layers[i].self_attn.k_proj.scales
        bits = model.model.layers[i].self_attn.k_proj.bits
        wf = model.model.layers[i].self_attn.k_proj.wf
        group_size = model.model.layers[i].self_attn.k_proj.group_size

        k_proj = np.concatenate((k_proj, final_matrix(qzeros, scales, qweight, bits,
            group_size, wf).numpy()))

        qweight = model.model.layers[i].self_attn.o_proj.qweight
        qzeros = model.model.layers[i].self_attn.o_proj.qzeros
        scales = model.model.layers[i].self_attn.o_proj.scales
        bits = model.model.layers[i].self_attn.o_proj.bits
        wf = model.model.layers[i].self_attn.o_proj.wf
        group_size = model.model.layers[i].self_attn.o_proj.group_size

        o_proj = np.concatenate((o_proj, final_matrix(qzeros, scales, qweight, bits,
            group_size, wf).numpy()))

        qweight = model.model.layers[i].self_attn.q_proj.qweight
        qzeros = model.model.layers[i].self_attn.q_proj.qzeros
        scales = model.model.layers[i].self_attn.q_proj.scales
        bits = model.model.layers[i].self_attn.q_proj.bits
        wf = model.model.layers[i].self_attn.q_proj.wf
        group_size = model.model.layers[i].self_attn.q_proj.group_size

        q_proj = np.concatenate((q_proj, final_matrix(qzeros, scales, qweight, bits,
            group_size, wf).numpy()))

        qweight = model.model.layers[i].self_attn.v_proj.qweight
        qzeros = model.model.layers[i].self_attn.v_proj.qzeros
        scales = model.model.layers[i].self_attn.v_proj.scales
        bits = model.model.layers[i].self_attn.v_proj.bits
        wf = model.model.layers[i].self_attn.v_proj.wf
        group_size = model.model.layers[i].self_attn.v_proj.group_size

        v_proj = np.concatenate((v_proj, final_matrix(qzeros, scales, qweight, bits,
            group_size, wf).numpy()))

    name_file = "mlp-down_proj-" + str(i)
    shapes.append("x".join([str(down_proj.shape[0]), str(down_proj.shape[1])]))
    files.append(name_file)
    
    write_matrices(dir_file + name_file, down_proj)

    name_file = "mlp-gate_proj-" + str(i)
    shapes.append("x".join([str(gate_proj.shape[0]), str(gate_proj.shape[1])]))
    files.append(name_file)
    
    write_matrices(dir_file + name_file, gate_proj)

    name_file = "mlp-up_proj-" + str(i)
    shapes.append("x".join([str(up_proj.shape[0]), str(up_proj.shape[1])]))
    files.append(name_file)
    
    write_matrices(dir_file + name_file, up_proj)

    name_file = "self_attn-k_proj-" + str(i)
    shapes.append("x".join([str(k_proj.shape[0]), str(k_proj.shape[1])]))
    files.append(name_file)
    
    write_matrices(dir_file + name_file, k_proj)

    name_file = "self_attn-q_proj-" + str(i)
    shapes.append("x".join([str(q_proj.shape[0]), str(q_proj.shape[1])]))
    files.append(name_file)
    
    write_matrices(dir_file + name_file, q_proj)

    name_file = "self_attn-o_proj-" + str(i)
    shapes.append("x".join([str(o_proj.shape[0]), str(o_proj.shape[1])]))
    files.append(name_file)
    
    write_matrices(dir_file + name_file, o_proj)

    name_file = "self_attn-v_proj-" + str(i)
    shapes.append("x".join([str(v_proj.shape[0]), str(v_proj.shape[1])]))
    files.append(name_file)
    
    write_matrices(dir_file + name_file, v_proj)
    
    print("command: ", "--files " + ":".join(files), "--sizes " + ":".join(shapes))

def mixtral_matrices_stacked(model, name, quant):
    files = []
    shapes = []
    dir_file = save_dir + name + "/" + quant + "/stacked"
    for i in range(len(model.model.layers)):
        for j in range(len(model.model.layers[i].block_sparse_moe.experts)):
            qweight = model.model.layers[i].block_sparse_moe.experts[j].w1.qweight
            qzeros = model.model.layers[i].block_sparse_moe.experts[j].w1.qzeros
            scales = model.model.layers[i].block_sparse_moe.experts[j].w1.scales
            bits = model.model.layers[i].block_sparse_moe.experts[j].w1.bits
            wf = model.model.layers[i].block_sparse_moe.experts[j].w1.wf
            group_size = model.model.layers[i].block_sparse_moe.experts[j].w1.group_size

            ret = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()
            name_file = "block_sparse-w1-" + str(j) + "-" + str(i)
            
            files.append(name_file)
            shapes.append("x".join([str(ret.shape[0]), str(ret.shape[1])]))

            write_matrices(dir_file + name_file, ret)

            qweight = model.model.layers[i].block_sparse_moe.experts[j].w2.qweight
            qzeros = model.model.layers[i].block_sparse_moe.experts[j].w2.qzeros
            scales = model.model.layers[i].block_sparse_moe.experts[j].w2.scales
            bits = model.model.layers[i].block_sparse_moe.experts[j].w2.bits
            wf = model.model.layers[i].block_sparse_moe.experts[j].w2.wf
            group_size = model.model.layers[i].block_sparse_moe.experts[j].w2.group_size

            ret = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()
            name_file = "block_sparse-w2-" + str(j) + "-" + str(i)

            files.append(name_file)
            shapes.append("x".join([str(ret.shape[0]), str(ret.shape[1])]))

            write_matrices(dir_file + name_file, ret)

            qweight = model.model.layers[i].block_sparse_moe.experts[j].w3.qweight
            qzeros = model.model.layers[i].block_sparse_moe.experts[j].w3.qzeros
            scales = model.model.layers[i].block_sparse_moe.experts[j].w3.scales
            bits = model.model.layers[i].block_sparse_moe.experts[j].w3.bits
            wf = model.model.layers[i].block_sparse_moe.experts[j].w3.wf
            group_size = model.model.layers[i].block_sparse_moe.experts[j].w3.group_size

            ret = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()
            name_file = "block_sparse-w3-" + str(j) + "-" + str(i)

            shapes.append("x".join([str(ret.shape[0]), str(ret.shape[1])]))
            files.append(name_file)

            write_matrices(dir_file + name_file, ret)

        qweight = model.model.layers[i].self_attn.k_proj.qweight
        qzeros = model.model.layers[i].self_attn.k_proj.qzeros
        scales = model.model.layers[i].self_attn.k_proj.scales
        bits = model.model.layers[i].self_attn.k_proj.bits
        wf = model.model.layers[i].self_attn.k_proj.wf
        group_size = model.model.layers[i].self_attn.k_proj.group_size

        ret = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()

        name_file = "self_attn-k_proj-" + str(i)
        shapes.append("x".join([str(ret.shape[0]), str(ret.shape[1])]))
        files.append(name_file)

        write_matrices(dir_file + name_file, ret)

        qweight = model.model.layers[i].self_attn.o_proj.qweight
        qzeros = model.model.layers[i].self_attn.o_proj.qzeros
        scales = model.model.layers[i].self_attn.o_proj.scales
        bits = model.model.layers[i].self_attn.o_proj.bits
        wf = model.model.layers[i].self_attn.o_proj.wf
        group_size = model.model.layers[i].self_attn.o_proj.group_size

        ret = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()

        name_file = "self_attn-o_proj-" + str(i)
        shapes.append("x".join([str(ret.shape[0]), str(ret.shape[1])]))
        files.append(name_file)

        write_matrices(dir_file + name_file, ret)

        qweight = model.model.layers[i].self_attn.q_proj.qweight
        qzeros = model.model.layers[i].self_attn.q_proj.qzeros
        scales = model.model.layers[i].self_attn.q_proj.scales
        bits = model.model.layers[i].self_attn.q_proj.bits
        wf = model.model.layers[i].self_attn.q_proj.wf
        group_size = model.model.layers[i].self_attn.q_proj.group_size

        ret = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()

        name_file = "self_attn-q_proj-" + str(i)
        shapes.append("x".join([str(ret.shape[0]), str(ret.shape[1])]))
        files.append(name_file)

        write_matrices(dir_file + name_file, ret)

        qweight = model.model.layers[i].self_attn.v_proj.qweight
        qzeros = model.model.layers[i].self_attn.v_proj.qzeros
        scales = model.model.layers[i].self_attn.v_proj.scales
        bits = model.model.layers[i].self_attn.v_proj.bits
        wf = model.model.layers[i].self_attn.v_proj.wf
        group_size = model.model.layers[i].self_attn.v_proj.group_size

        ret = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()

        name_file = "self_attn-v_proj-" + str(i)
        shapes.append("x".join([str(ret.shape[0]), str(ret.shape[1])]))
        files.append(name_file)

        write_matrices(dir_file + name_file, ret)
    print("command: ", "--files " + ":".join(files), "--sizes " + ":".join(shapes))

# this is for llama 2 model
def llama_divided_matrices(model, name, quant):
    dir_file = save_dir  + name + "/" + quant + "/original/"
    for i in range(1):
        name_file = "mlp-down_proj-" + str(i)
        write_matrices(dir_file + name_file + "-qweight", model.model.layers[i].mlp.down_proj.qweight.numpy())
        write_matrices(dir_file + name_file + "-qzeros", model.model.layers[i].mlp.down_proj.qzeros.numpy())
        write_matrices(dir_file + name_file + "-scales", model.model.layers[i].mlp.down_proj.scales.numpy())

        name_file = "mlp-gate_proj-" + str(i)
        write_matrices(dir_file + name_file + "-qweight", model.model.layers[i].mlp.gate_proj.qweight.numpy())
        write_matrices(dir_file + name_file + "-qzeros", model.model.layers[i].mlp.gate_proj.qzeros.numpy())
        write_matrices(dir_file + name_file + "-scales", model.model.layers[i].mlp.gate_proj.scales.numpy())

        name_file = "mlp-up_proj-" + str(i)
        write_matrices(dir_file + name_file + "-qweight", model.model.layers[i].mlp.up_proj.qweight.numpy())
        write_matrices(dir_file + name_file + "-qzeros", model.model.layers[i].mlp.up_proj.qzeros.numpy())
        write_matrices(dir_file + name_file + "-scales", model.model.layers[i].mlp.up_proj.scales.numpy())

        name_file = "self_attn-k_proj-" + str(i)
        write_matrices(dir_file + name_file + "-qweight", model.model.layers[i].self_attn.k_proj.qweight.numpy())
        write_matrices(dir_file + name_file + "-qzeros", model.model.layers[i].self_attn.k_proj.qzeros.numpy())
        write_matrices(dir_file + name_file + "-scales", model.model.layers[i].self_attn.k_proj.scales.numpy())

        name_file = "self_attn-o_proj-" + str(i)
        write_matrices(dir_file + name_file + "-qweight", model.model.layers[i].self_attn.o_proj.qweight.numpy())
        write_matrices(dir_file + name_file + "-qzeros", model.model.layers[i].self_attn.o_proj.qzeros.numpy())
        write_matrices(dir_file + name_file + "-scales", model.model.layers[i].self_attn.o_proj.scales.numpy())

        name_file = "self_attn-q_proj-" + str(i)
        write_matrices(dir_file + name_file + "-qweight", model.model.layers[i].self_attn.q_proj.qweight.numpy())
        write_matrices(dir_file + name_file + "-qzeros", model.model.layers[i].self_attn.q_proj.qzeros.numpy())
        write_matrices(dir_file + name_file + "-scales", model.model.layers[i].self_attn.q_proj.scales.numpy())

        name_file = "self_attn-v_proj-" + str(i)
        write_matrices(dir_file + name_file + "-qweight", model.model.layers[i].self_attn.v_proj.qweight.numpy())
        write_matrices(dir_file + name_file + "-qzeros", model.model.layers[i].self_attn.v_proj.qzeros.numpy())
        write_matrices(dir_file + name_file + "-scales", model.model.layers[i].self_attn.v_proj.scales.numpy())

def mixtral_divided_matrices(model, name, quant):
    dir_file = save_dir  + name + "/" + quant + "/original/"
    for i in range(1):
        for j in range(len(model.model.layers[i].block_sparse_moe.experts)):
            name_file = "block_sparse-w1-" + str(j) + "-" + str(i)
            write_matrices(dir_file + name_file + "-qweight", model.model.layers[i].block_sparse_moe.experts[j].w1.qweight.numpy())
            write_matrices(dir_file + name_file + "-qzeros", model.model.layers[i].block_sparse_moe.experts[j].w1.qzeros.numpy())
            write_matrices(dir_file + name_file + "-scales", model.model.layers[i].block_sparse_moe.experts[j].w1.scales.numpy())

            name_file = "block_sparse-w2-" + str(j) + "-" + str(i)
            write_matrices(dir_file + name_file + "-qweight", model.model.layers[i].block_sparse_moe.experts[j].w2.qweight.numpy())
            write_matrices(dir_file + name_file + "-qzeros", model.model.layers[i].block_sparse_moe.experts[j].w2.qzeros.numpy())
            write_matrices(dir_file + name_file + "-scales", model.model.layers[i].block_sparse_moe.experts[j].w2.scales.numpy())

            name_file = "block_sparse-w3-" + str(j) + "-" + str(i)
            write_matrices(dir_file + name_file + "-qweight", model.model.layers[i].block_sparse_moe.experts[j].w3.qweight.numpy())
            write_matrices(dir_file + name_file + "-qzeros", model.model.layers[i].block_sparse_moe.experts[j].w3.qzeros.numpy())
            write_matrices(dir_file + name_file + "-scales", model.model.layers[i].block_sparse_moe.experts[j].w3.scales.numpy())

        name_file = "self_attn-k_proj-" + str(i)
        write_matrices(dir_file + name_file + "-qweight", model.model.layers[i].self_attn.k_proj.qweight.numpy())
        write_matrices(dir_file + name_file + "-qzeros", model.model.layers[i].self_attn.k_proj.qzeros.numpy())
        write_matrices(dir_file + name_file + "-scales", model.model.layers[i].self_attn.k_proj.scales.numpy())

        name_file = "self_attn-o_proj-" + str(i)
        write_matrices(dir_file + name_file + "-qweight", model.model.layers[i].self_attn.o_proj.qweight.numpy())
        write_matrices(dir_file + name_file + "-qzeros", model.model.layers[i].self_attn.o_proj.qzeros.numpy())
        write_matrices(dir_file + name_file + "-scales", model.model.layers[i].self_attn.o_proj.scales.numpy())

        name_file = "self_attn-q_proj-" + str(i)
        write_matrices(dir_file + name_file + "-qweight", model.model.layers[i].self_attn.q_proj.qweight.numpy())
        write_matrices(dir_file + name_file + "-qzeros", model.model.layers[i].self_attn.q_proj.qzeros.numpy())
        write_matrices(dir_file + name_file + "-scales", model.model.layers[i].self_attn.q_proj.scales.numpy())

        name_file = "self_attn-v_proj-" + str(i)
        write_matrices(dir_file + name_file + "-qweight", model.model.layers[i].self_attn.v_proj.qweight.numpy())
        write_matrices(dir_file + name_file + "-qzeros", model.model.layers[i].self_attn.v_proj.qzeros.numpy())
        write_matrices(dir_file + name_file + "-scales", model.model.layers[i].self_attn.v_proj.scales.numpy())

#quant = "8b-nonegs"
#name = "Mixtral-8x7B-Instruct-v0.1-GPTQ"
#model = AutoModelForCausalLM.from_pretrained(path + "/" + name + "/" + quant,
#                                             trust_remote_code=False,
#                                             revision="main")
#mixtral_divided_matrices(model, name, quant)
#quant = "4b-nonegs"
#name = "Mixtral-8x7B-Instruct-v0.1-GPTQ"
#model = AutoModelForCausalLM.from_pretrained(path + "/" + name + "/" + quant,
#                                             trust_remote_code=False,
#                                             revision="main")
#mixtral_divided_matrices(model, name, quant)
#quant = "3b-nonegs"
#name = "Mixtral-8x7B-Instruct-v0.1-GPTQ"
#model = AutoModelForCausalLM.from_pretrained(path + "/" + name + "/" + quant,
#                                             trust_remote_code=False,
#                                             revision="main")
#mixtral_divided_matrices(model, name, quant)
name = "Llama-2-13B-chat-GPTQ"
quant = "4b-64gs"
model = AutoModelForCausalLM.from_pretrained(path + "/" + name + "/" + quant,
                                             trust_remote_code=False,
                                             revision="main")

llama_fp16_matrices(model, name, quant)

name = "Llama-2-13B-chat-GPTQ"
quant = "8b-64gs"
model = AutoModelForCausalLM.from_pretrained(path + "/" + name + "/" + quant,
                                             trust_remote_code=False,
                                             revision="main")
llama_fp16_matrices(model, name, quant)


name = "Llama-2-13B-chat-GPTQ"
quant = "4b-128gs"
model = AutoModelForCausalLM.from_pretrained(path + "/" + name + "/" + quant,
                                             trust_remote_code=False,
                                             revision="main")
llama_fp16_matrices(model, name, quant)

name = "Llama-2-13B-chat-GPTQ"
quant = "8b-128gs"
model = AutoModelForCausalLM.from_pretrained(path + "/" + name + "/" + quant,
                                             trust_remote_code=False,
                                             revision="main")
llama_fp16_matrices(model, name, quant)
