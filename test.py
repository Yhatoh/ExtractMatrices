from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import sys
import numpy as np
import inspect
import torch

from bytes import get_bytes, get_bytes_vector, get_bytes_matrix
from transform_matrix import final_matrix
from write_matrix import write_matrices, llama_matrices_stacked, llama_fp16_matrices, mixtral_fp16_matrices

# where you will save the matrices
save_dir = "/data/matrix/matrices/"
# where you will get the models
path = "/data/matrix/models"

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

quant = "fp16"
name = "Mixtral-8x7B-Instruct-v0.1-GPTQ"
model = AutoModelForCausalLM.from_pretrained(path + "/" + name + "/" + quant,
                                             trust_remote_code=False,
                                             revision="main")
mixtral_fp16_matrices(model, name, quant)
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
#quant = "4b-128gs"
#model = AutoModelForCausalLM.from_pretrained(path + "/" + name + "/" + quant,
#                                             trust_remote_code=False,
#                                             revision="main")
#
#llama_matrices_stacked(model, name, quant, 4)
#
#name = "Llama-2-13B-chat-GPTQ"
#quant = "8b-128gs"
#model = AutoModelForCausalLM.from_pretrained(path + "/" + name + "/" + quant,
#                                             trust_remote_code=False,
#                                             revision="main")
#llama_matrices_stacked(model, name, quant, 4)

#llama_GPTQ_matrices(model, name, quant)
#
#name = "Llama-2-7B-chat-GPTQ"
#quant = "fp16"
#model = AutoModelForCausalLM.from_pretrained(path + "/" + name + "/" + quant,
#                                             trust_remote_code=False,
#                                             revision="main")
#llama_fp16_matrices(model, name, quant)
#

name = "Gemma2b"
quant = "gemma-2b"
model = AutoModelForCausalLM.from_pretrained(path + "/" + name + "/" + quant)
llama_fp16_matrices(model, name, quant)

#name = "Gemma7b"
#quant = "gemma-7b"
#model = AutoModelForCausalLM.from_pretrained(path + "/" + name + "/" + quant)
#llama_fp16_matrices(model, name, quant)
