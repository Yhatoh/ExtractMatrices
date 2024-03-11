import numpy as np
from transform_matrix import final_matrix

# write the matrix in the file `name`
def write_matrices(name, matrices):
    #matrices = matrices.astype('f')
    n = matrices.shape[0]
    m = matrices.shape[1]

    with open(name, 'wb') as file:
        matrices.tofile(file)

#def write_matrices(name, matrices):
#    #matrices = matrices.astype('f')
#    n = matrices.shape[0]
#    m = matrices.shape[1]
#
#    np.savetxt(name + ".csv", matrices, delimiter=";")

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
        ret = model.model.layers[i].mlp.down_proj.weight.detach().numpy()

        name_file = "mlp-down_proj-" + str(i)
        shapes.append("x".join([str(ret.shape[0]), str(ret.shape[1])]))
        files.append(name_file)
        
        write_matrices(dir_file + name_file, ret)

        ret = model.model.layers[i].mlp.gate_proj.weight.detach().numpy()

        name_file = "mlp-gate_proj-" + str(i)
        shapes.append("x".join([str(ret.shape[0]), str(ret.shape[1])]))
        files.append(name_file)

        write_matrices(dir_file + name_file, ret)

        ret = model.model.layers[i].mlp.up_proj.weight.detach().numpy()

        name_file = "mlp-up_proj-" + str(i)
        shapes.append("x".join([str(ret.shape[0]), str(ret.shape[1])]))
        files.append(name_file)

        write_matrices(dir_file + name_file, ret)

        ret = model.model.layers[i].self_attn.k_proj.weight.detach().numpy()

        name_file = "self_attn-k_proj-" + str(i)
        shapes.append("x".join([str(ret.shape[0]), str(ret.shape[1])]))
        files.append(name_file)

        write_matrices(dir_file + name_file, ret)

        ret = model.model.layers[i].self_attn.o_proj.weight.detach().numpy()

        name_file = "self_attn-o_proj-" + str(i)
        shapes.append("x".join([str(ret.shape[0]), str(ret.shape[1])]))
        files.append(name_file)

        write_matrices(dir_file + name_file, ret)

        ret = model.model.layers[i].self_attn.q_proj.weight.detach().numpy()

        name_file = "self_attn-q_proj-" + str(i)
        shapes.append("x".join([str(ret.shape[0]), str(ret.shape[1])]))
        files.append(name_file)

        write_matrices(dir_file + name_file, ret)

        ret = model.model.layers[i].self_attn.v_proj.weight.detach().numpy()

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

def llama_matrices_stacked(model, name, quant, group):
    files = []
    shapes = []
    dir_file = save_dir + name + "/" + quant + "/stacked/"
    down_proj = None
    gate_proj = None
    up_proj = None
    k_proj = None
    o_proj = None
    q_proj = None
    v_proj = None
    count = 0
    for i in range(len(model.model.layers)):
        count += 1
        qweight = model.model.layers[i].mlp.down_proj.qweight
        qzeros = model.model.layers[i].mlp.down_proj.qzeros
        scales = model.model.layers[i].mlp.down_proj.scales
        bits = model.model.layers[i].mlp.down_proj.bits
        wf = model.model.layers[i].mlp.down_proj.wf
        group_size = model.model.layers[i].mlp.down_proj.group_size

        if down_proj is None:
            down_proj = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()
        else:
            down_proj = np.concatenate((down_proj, final_matrix(qzeros, scales, qweight, bits,
                group_size, wf).numpy()))
    
        qweight = model.model.layers[i].mlp.gate_proj.qweight
        qzeros = model.model.layers[i].mlp.gate_proj.qzeros
        scales = model.model.layers[i].mlp.gate_proj.scales
        bits = model.model.layers[i].mlp.gate_proj.bits
        wf = model.model.layers[i].mlp.gate_proj.wf
        group_size = model.model.layers[i].mlp.gate_proj.group_size

        if gate_proj is None:
            gate_proj = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()
        else:
            gate_proj = np.concatenate((gate_proj, final_matrix(qzeros, scales, qweight, bits,
                group_size, wf).numpy()))

        qweight = model.model.layers[i].mlp.up_proj.qweight
        qzeros = model.model.layers[i].mlp.up_proj.qzeros
        scales = model.model.layers[i].mlp.up_proj.scales
        bits = model.model.layers[i].mlp.up_proj.bits
        wf = model.model.layers[i].mlp.up_proj.wf
        group_size = model.model.layers[i].mlp.up_proj.group_size

        if up_proj is None:
            up_proj = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()
        else:
            up_proj = np.concatenate((up_proj, final_matrix(qzeros, scales, qweight, bits,
                group_size, wf).numpy()))

        qweight = model.model.layers[i].self_attn.k_proj.qweight
        qzeros = model.model.layers[i].self_attn.k_proj.qzeros
        scales = model.model.layers[i].self_attn.k_proj.scales
        bits = model.model.layers[i].self_attn.k_proj.bits
        wf = model.model.layers[i].self_attn.k_proj.wf
        group_size = model.model.layers[i].self_attn.k_proj.group_size

        if k_proj is None:
            k_proj = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()
        else:
            k_proj = np.concatenate((k_proj, final_matrix(qzeros, scales, qweight, bits,
                group_size, wf).numpy()))

        qweight = model.model.layers[i].self_attn.o_proj.qweight
        qzeros = model.model.layers[i].self_attn.o_proj.qzeros
        scales = model.model.layers[i].self_attn.o_proj.scales
        bits = model.model.layers[i].self_attn.o_proj.bits
        wf = model.model.layers[i].self_attn.o_proj.wf
        group_size = model.model.layers[i].self_attn.o_proj.group_size

        if o_proj is None:
            o_proj = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()
        else:
            o_proj = np.concatenate((o_proj, final_matrix(qzeros, scales, qweight, bits,
                group_size, wf).numpy()))

        qweight = model.model.layers[i].self_attn.q_proj.qweight
        qzeros = model.model.layers[i].self_attn.q_proj.qzeros
        scales = model.model.layers[i].self_attn.q_proj.scales
        bits = model.model.layers[i].self_attn.q_proj.bits
        wf = model.model.layers[i].self_attn.q_proj.wf
        group_size = model.model.layers[i].self_attn.q_proj.group_size

        if q_proj is None:
            q_proj = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()
        else:
            q_proj = np.concatenate((q_proj, final_matrix(qzeros, scales, qweight, bits,
                group_size, wf).numpy()))

        qweight = model.model.layers[i].self_attn.v_proj.qweight
        qzeros = model.model.layers[i].self_attn.v_proj.qzeros
        scales = model.model.layers[i].self_attn.v_proj.scales
        bits = model.model.layers[i].self_attn.v_proj.bits
        wf = model.model.layers[i].self_attn.v_proj.wf
        group_size = model.model.layers[i].self_attn.v_proj.group_size

        if v_proj is None:
            v_proj = final_matrix(qzeros, scales, qweight, bits, group_size, wf).numpy()
        else:
            v_proj = np.concatenate((v_proj, final_matrix(qzeros, scales, qweight, bits,
                group_size, wf).numpy()))

        if count == group:
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
            
            down_proj = None
            gate_proj = None
            up_proj = None
            k_proj = None
            o_proj = None
            q_proj = None
            v_proj = None
            count = 0
    if count != 0:
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
