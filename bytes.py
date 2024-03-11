import sys
import numpy as np

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


