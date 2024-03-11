from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import sys
import numpy as np
import inspect
import torch

from bytes import get_bytes, get_bytes_vector, get_bytes_matrix
from transform_matrix import final_matrix
from write_matrix import write_matrices

# where you will save the matrices
save_dir = "/data/matrix/matrices/"
# where you will get the models
path = "/data/matrix/models"

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

#quant = "8b-nonegs"
#name = "Mixtral-8x7B-Instruct-v0.1-GPTQ"
#model = AutoModelForCausalLM.from_pretrained(path + "/" + name + "/" + quant,
#                                             trust_remote_code=False,
#                                             revision="main")
#print("8b-nonegs", mixtral_GPTQ_uniques(model))
#
#quant = "4b-nonegs"
#name = "Mixtral-8x7B-Instruct-v0.1-GPTQ"
#model = AutoModelForCausalLM.from_pretrained(path + "/" + name + "/" + quant,
#                                             trust_remote_code=False,
#                                             revision="main")
#
#print("4b-nonegs", mixtral_GPTQ_uniques(model))
#
#quant = "3b-nonegs"
#name = "Mixtral-8x7B-Instruct-v0.1-GPTQ"
#model = AutoModelForCausalLM.from_pretrained(path + "/" + name + "/" + quant,
#                                             trust_remote_code=False,
#                                             revision="main")
#print("3b-nonegs", mixtral_GPTQ_uniques(model))

#name = "Llama-2-7B-chat-GPTQ"
#quant = "4b-64gs"
#model = AutoModelForCausalLM.from_pretrained(path + "/" + name + "/" + quant,
#                                             trust_remote_code=False,
#                                             revision="main")
#
#llama_GPTQ_matrices(model, name, quant)
#llama_divided_matrices(model, name, quant)

#name = "Llama-2-7B-chat-GPTQ"
#quant = "4b-128gs"
#model = AutoModelForCausalLM.from_pretrained(path + "/" + name + "/" + quant,
#                                             trust_remote_code=False,
#                                             revision="main")

#llama_GPTQ_matrices(model, name, quant)
#llama_divided_matrices(model, name, quant)

#name = "Llama-2-13B-chat-GPTQ"
#quant = "8b-64gs"
#model = AutoModelForCausalLM.from_pretrained(path + "/" + name + "/" + quant,
#                                             trust_remote_code=False,
#                                             revision="main")
#
#llama_GPTQ_matrices(model, name, quant)
#
#name = "Llama-2-13B-chat-GPTQ"
#quant = "8b-128gs"
#model = AutoModelForCausalLM.from_pretrained(path + "/" + name + "/" + quant,
#                                             trust_remote_code=False,
#                                             revision="main")
#
#llama_GPTQ_matrices(model, name, quant)
#
#name = "Llama-2-7B-chat-GPTQ"
#quant = "fp16"
#model = AutoModelForCausalLM.from_pretrained(path + "/" + name + "/" + quant,
#                                             trust_remote_code=False,
#                                             revision="main")
#llama_fp16_matrices(model, name, quant)
#

#name = "Gemma2b"
#quant = "gemma-2b"
#model = AutoModelForCausalLM.from_pretrained(path + "/" + name + "/" + quant)
#llama_fp16_matrices(model, name, quant)

#name = "Gemma7b"
#quant = "gemma-7b"
#model = AutoModelForCausalLM.from_pretrained(path + "/" + name + "/" + quant)
#llama_fp16_matrices(model, name, quant)
