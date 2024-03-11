import numpy as np
from transform_matrix import final_matrix

def llama_GPTQ_uniques(model):
    suma = 0
    for i in range(len(model.model.layers)):
        qweight = model.model.layers[i].mlp.down_proj.qweight
        qzeros = model.model.layers[i].mlp.down_proj.qzeros
        scales = model.model.layers[i].mlp.down_proj.scales
        bits = model.model.layers[i].mlp.down_proj.bits
        wf = model.model.layers[i].mlp.down_proj.wf
        group_size = model.model.layers[i].mlp.down_proj.group_size

        ret = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()
        suma += np.unique(ret).shape[0]

        qweight = model.model.layers[i].mlp.gate_proj.qweight
        qzeros = model.model.layers[i].mlp.gate_proj.qzeros
        scales = model.model.layers[i].mlp.gate_proj.scales
        bits = model.model.layers[i].mlp.gate_proj.bits
        wf = model.model.layers[i].mlp.gate_proj.wf
        group_size = model.model.layers[i].mlp.gate_proj.group_size

        ret = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()
        suma += np.unique(ret).shape[0]

        qweight = model.model.layers[i].mlp.up_proj.qweight
        qzeros = model.model.layers[i].mlp.up_proj.qzeros
        scales = model.model.layers[i].mlp.up_proj.scales
        bits = model.model.layers[i].mlp.up_proj.bits
        wf = model.model.layers[i].mlp.up_proj.wf
        group_size = model.model.layers[i].mlp.up_proj.group_size

        ret = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()
        suma += np.unique(ret).shape[0]

        qweight = model.model.layers[i].self_attn.k_proj.qweight
        qzeros = model.model.layers[i].self_attn.k_proj.qzeros
        scales = model.model.layers[i].self_attn.k_proj.scales
        bits = model.model.layers[i].self_attn.k_proj.bits
        wf = model.model.layers[i].self_attn.k_proj.wf
        group_size = model.model.layers[i].self_attn.k_proj.group_size

        ret = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()
        suma += np.unique(ret).shape[0]

        qweight = model.model.layers[i].self_attn.o_proj.qweight
        qzeros = model.model.layers[i].self_attn.o_proj.qzeros
        scales = model.model.layers[i].self_attn.o_proj.scales
        bits = model.model.layers[i].self_attn.o_proj.bits
        wf = model.model.layers[i].self_attn.o_proj.wf
        group_size = model.model.layers[i].self_attn.o_proj.group_size

        ret = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()
        suma += np.unique(ret).shape[0]

        qweight = model.model.layers[i].self_attn.q_proj.qweight
        qzeros = model.model.layers[i].self_attn.q_proj.qzeros
        scales = model.model.layers[i].self_attn.q_proj.scales
        bits = model.model.layers[i].self_attn.q_proj.bits
        wf = model.model.layers[i].self_attn.q_proj.wf
        group_size = model.model.layers[i].self_attn.q_proj.group_size

        ret = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()
        suma += np.unique(ret).shape[0]

        qweight = model.model.layers[i].self_attn.v_proj.qweight
        qzeros = model.model.layers[i].self_attn.v_proj.qzeros
        scales = model.model.layers[i].self_attn.v_proj.scales
        bits = model.model.layers[i].self_attn.v_proj.bits
        wf = model.model.layers[i].self_attn.v_proj.wf
        group_size = model.model.layers[i].self_attn.v_proj.group_size

        ret = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()
        suma += np.unique(ret).shape[0]
    return suma

def llama_fp16_uniques(model):
    suma = 0
    for i in range(len(model.model.layers)):
        ret = model.model.layers[i].mlp.down_proj.weight.detach().numpy()
        suma += np.unique(ret).shape[0]

        ret = model.model.layers[i].mlp.gate_proj.weight.detach().numpy()
        suma += np.unique(ret).shape[0]

        ret = model.model.layers[i].mlp.up_proj.weight.detach().numpy()
        suma += np.unique(ret).shape[0]

        ret = model.model.layers[i].self_attn.k_proj.weight.detach().numpy()
        suma += np.unique(ret).shape[0]

        ret = model.model.layers[i].self_attn.o_proj.weight.detach().numpy()
        suma += np.unique(ret).shape[0]

        ret = model.model.layers[i].self_attn.q_proj.weight.detach().numpy()
        suma += np.unique(ret).shape[0]

        ret = model.model.layers[i].self_attn.v_proj.weight.detach().numpy()
        suma += np.unique(ret).shape[0]
    return suma

def mixtral_GPTQ_uniques(model):
    suma = 0
    for i in range(len(model.model.layers)):
        for j in range(len(model.model.layers[i].block_sparse_moe.experts)):
            qweight = model.model.layers[i].block_sparse_moe.experts[j].w1.qweight
            qzeros = model.model.layers[i].block_sparse_moe.experts[j].w1.qzeros
            scales = model.model.layers[i].block_sparse_moe.experts[j].w1.scales
            bits = model.model.layers[i].block_sparse_moe.experts[j].w1.bits
            wf = model.model.layers[i].block_sparse_moe.experts[j].w1.wf
            group_size = model.model.layers[i].block_sparse_moe.experts[j].w1.group_size

            ret = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()
            suma += np.unique(ret).shape[0]

            qweight = model.model.layers[i].block_sparse_moe.experts[j].w2.qweight
            qzeros = model.model.layers[i].block_sparse_moe.experts[j].w2.qzeros
            scales = model.model.layers[i].block_sparse_moe.experts[j].w2.scales
            bits = model.model.layers[i].block_sparse_moe.experts[j].w2.bits
            wf = model.model.layers[i].block_sparse_moe.experts[j].w2.wf
            group_size = model.model.layers[i].block_sparse_moe.experts[j].w2.group_size

            ret = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()
            suma += np.unique(ret).shape[0]

            qweight = model.model.layers[i].block_sparse_moe.experts[j].w3.qweight
            qzeros = model.model.layers[i].block_sparse_moe.experts[j].w3.qzeros
            scales = model.model.layers[i].block_sparse_moe.experts[j].w3.scales
            bits = model.model.layers[i].block_sparse_moe.experts[j].w3.bits
            wf = model.model.layers[i].block_sparse_moe.experts[j].w3.wf
            group_size = model.model.layers[i].block_sparse_moe.experts[j].w3.group_size

            ret = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()
            suma += np.unique(ret).shape[0]

        qweight = model.model.layers[i].self_attn.k_proj.qweight
        qzeros = model.model.layers[i].self_attn.k_proj.qzeros
        scales = model.model.layers[i].self_attn.k_proj.scales
        bits = model.model.layers[i].self_attn.k_proj.bits
        wf = model.model.layers[i].self_attn.k_proj.wf
        group_size = model.model.layers[i].self_attn.k_proj.group_size

        ret = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()
        suma += np.unique(ret).shape[0]

        qweight = model.model.layers[i].self_attn.o_proj.qweight
        qzeros = model.model.layers[i].self_attn.o_proj.qzeros
        scales = model.model.layers[i].self_attn.o_proj.scales
        bits = model.model.layers[i].self_attn.o_proj.bits
        wf = model.model.layers[i].self_attn.o_proj.wf
        group_size = model.model.layers[i].self_attn.o_proj.group_size

        ret = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()
        suma += np.unique(ret).shape[0]

        qweight = model.model.layers[i].self_attn.q_proj.qweight
        qzeros = model.model.layers[i].self_attn.q_proj.qzeros
        scales = model.model.layers[i].self_attn.q_proj.scales
        bits = model.model.layers[i].self_attn.q_proj.bits
        wf = model.model.layers[i].self_attn.q_proj.wf
        group_size = model.model.layers[i].self_attn.q_proj.group_size

        ret = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()
        suma += np.unique(ret).shape[0]

        qweight = model.model.layers[i].self_attn.v_proj.qweight
        qzeros = model.model.layers[i].self_attn.v_proj.qzeros
        scales = model.model.layers[i].self_attn.v_proj.scales
        bits = model.model.layers[i].self_attn.v_proj.bits
        wf = model.model.layers[i].self_attn.v_proj.wf
        group_size = model.model.layers[i].self_attn.v_proj.group_size

        ret = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()
        suma += np.unique(ret).shape[0]
    return suma


