import dgl
import time
import torch
time.sleep(10)
from ogb.nodeproppred import DglNodePropPredDataset
import gc
import numpy as np
from dgl.nn import SAGEConv
import torch.nn as nn
import dgl.nn as dglnn
import torch.nn.functional as F
from tqdm import trange
import random
import torch.multiprocessing as mp
import threading
import math
import gc
import pickle
from dgl.convert import create_block
from torch import cat
from dgl.transforms import to_block
from pathlib import Path
import os
NTYPE = "_TYPE"
NID = "_ID"
ETYPE = "_TYPE"
EID = "_ID"


class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GraphConv(in_feats, n_hidden, allow_zero_in_degree = True))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.GraphConv(n_hidden, n_hidden, allow_zero_in_degree = True))
        self.layers.append(dglnn.GraphConv(n_hidden, n_classes, allow_zero_in_degree = True))

    def forward(self, bipartites, x):
        for l, (layer, bipartite) in enumerate(zip(self.layers, bipartites)):
            x = layer(bipartite, x)
            if l != self.n_layers - 1:
                x = F.relu(x)
        return x




class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))

    def forward(self, bipartites, x):
        for l, (layer, bipartite) in enumerate(zip(self.layers, bipartites)):
            x = layer(bipartite, x)
            if l != self.n_layers - 1:
                x = F.relu(x)
        return x


class GAT(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, num_heads):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GATConv(in_feats, n_hidden, num_heads, allow_zero_in_degree = True))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.GATConv(n_hidden, n_hidden, num_heads, allow_zero_in_degree = True))
        self.layers.append(dglnn.GATConv(n_hidden, n_classes, num_heads, allow_zero_in_degree = True))

    def forward(self, bipartites, x):
        for l, (layer, bipartite) in enumerate(zip(self.layers, bipartites)):
            x = layer(bipartite, x)
            if l != self.n_layers - 1:
                x = F.relu(x)
        x = torch.squeeze(torch.mean(x, 1))
        return x
        



def measure(dataset_id, train, batch_size, num_trials, device, num_layers, model_type, num_heads, warm_up, util):
    if dataset_id == 0:
        graph_name = 'ogbn-papers100M'
    elif dataset_id == 1:
        graph_name = 'Friendster'
    print(graph_name, train, batch_size, num_trials, device, model_type, num_heads)
    print('loading the dataset ...')
    if dataset_id == 0:
        graph_name = 'ogbn-papers100M'
        if train:
            dataset = DglNodePropPredDataset(graph_name)
            graph, node_labels = dataset[0]
            node_labels = torch.nan_to_num(node_labels)
            graph.ndata.pop('feat')
            split_idx = dataset.get_idx_split()
            node_labels = node_labels.to(dtype = torch.int64)
            train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
            num_classes = torch.max(node_labels).to(dtype = torch.long)  + 1
            del graph
            del dataset
            gc.collect()
        else:
            num_classes = 172
        file_name = 'dataset/PA.pkl'
        with open(file_name, 'rb') as f:
            [graph_a, feats] = pickle.load(f)
        graph = graph_a.formats(formats = 'csc')
        del graph_a
        gc.collect()
        if not(graph.is_pinned()):
            graph.pin_memory_()
        num_nodes = graph.number_of_nodes()
        freqs = torch.zeros(num_nodes, dtype = torch.int16)
        if not(feats.is_pinned()):
            cudart = torch.cuda.cudart()
            rt = cudart.cudaHostRegister(feats.data_ptr(), feats.numel() * feats.element_size(), 0)       
    elif dataset_id == 1:
        bin_path = "dataset/friendster_dgl.bin"
        g_list, _ = dgl.load_graphs(bin_path)
        graph = g_list[0]
        graph.create_formats_()
        del g_list
        gc.collect()
        if not(graph.is_pinned()):
            graph.pin_memory_()
        feats = torch.empty((graph.number_of_nodes(), 256), dtype = torch.float32)
  

        num_nodes = graph.number_of_nodes()
        freqs = torch.zeros(num_nodes, dtype = torch.int16)
        graph_name = 'friendster'
        num_classes = 100
        if not(feats.is_pinned()):
            cudart = torch.cuda.cudart()
            rt = cudart.cudaHostRegister(feats.data_ptr(), feats.numel() * feats.element_size(), 0)
        if train:
            node_labels = torch.randint(0, num_classes - 1, (num_nodes,), dtype = torch.int64)
            train_idx_sz = int(num_nodes*0.01)
            train_idx = torch.randperm(num_nodes)[0:train_idx_sz]
    print('dataset loaded.')
    
    
    
    if model_type == 'SAGE':
        model = SAGE(feats.shape[1],256, num_classes, num_layers).to(device = device)
    elif model_type == 'GAT':
        model = GAT(feats.shape[1],256, num_classes, num_layers, num_heads).to(device = device)
    elif model_type == 'GCN':
        model = GCN(feats.shape[1],256, num_classes, num_layers).to(device = device)
    if train:
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
    else:
        model.eval()
    print('model initialized.')
    
    
    r1 = batch_size[0]
    r2 = batch_size[1]
    batch_sizes = (torch.FloatTensor(num_trials).uniform_(r1, r2)).to(dtype = torch.long)
    n = num_trials*r2
    if train:
        nids = torch.from_numpy(np.random.choice(train_idx.numpy(), size = n))
    else:
        nids = torch.from_numpy(np.random.choice(num_nodes, size = n))
    if train:
        if num_layers == 2:
            sampler = dgl.dataloading.MultiLayerNeighborSampler([25,10])
        elif num_layers == 3:
            sampler = dgl.dataloading.MultiLayerNeighborSampler([15,10,5])
    else:
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers)
    if train:
        torch.enable_grad()
    else:
        torch.no_grad()
    in_degs = graph.in_degrees()
    res = torch.zeros((9, num_trials - warm_up), dtype = torch.float)
    print('sampler initialized. starting ...')
    p = 0
    T_ref = time.time()
    mems = [[], [], [], [], []]
    j = 0
    print(nids)



    for i in trange(num_trials):
        seeds = nids[p:p+batch_sizes[i]].to(device)
        p += batch_sizes[i]
        sampler.tc = 0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        torch.cuda.reset_peak_memory_stats()
        minibatch = sampler.sample_blocks(graph, seeds)
        m0 = torch.cuda.max_memory_allocated()
        if minibatch[0].shape[0] > 8000000:
            continue
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        blks = minibatch[2]
        try:
            torch.cuda.reset_peak_memory_stats()
            input_features = dgl.utils.gather_pinned_tensor_rows(feats, minibatch[0].to(device = device))
            m1 = torch.cuda.max_memory_allocated()
        except:
            continue
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        try:
            torch.cuda.reset_peak_memory_stats()
            preds = model(blks, input_features)
            m2 = torch.cuda.max_memory_allocated()
        except:
            continue
        if train:
            loss = F.cross_entropy(preds, torch.squeeze(node_labels[seeds.cpu()].cuda()))
            opt.zero_grad()
            loss.backward()
            opt.step()            
        #out = torch.argmax(preds, dim = 1).detach()
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        if i >= warm_up:
            memsa = sampler.mems
            torch.cuda.empty_cache()
            freqs[minibatch[0].cpu()] = (freqs[minibatch[0].cpu()] + 1).cpu()   #access frequencies
            res[0, j] = (t1 - t0)                                               #sample
            res[1, j] = sampler.tc                                              #layout transition
            res[2, j] = t2 - t1                                                 #fetch
            res[3, j] = t3 - t2                                                 #forward
            res[4, j] = t3 - t0                                                 #total
            res[5, j] = minibatch[0].shape[0]                                   #num inputs
            res[6, j] = minibatch[2][0].number_of_edges()                       #num edges last hop
            res[7, j] = blks[-1].number_of_edges()                              #num edges first hop
            res[8, j] = blks[-2].number_of_edges()                              #num edges second hop
            util[j,0] = max(m0, m1, m2)                                         #batch memory util
            util[j,1] = time.time() - T_ref                                     #timestamp
            torch.cuda.reset_peak_memory_stats()
            mems[0].append(memsa[0])                                            #memory util sample
            mems[1].append(memsa[1])  
            mems[2].append(m1)                                                  #memory util fetch
            mems[3].append(m2)                                                  #memory util forward
            mems[4].append(time.time() - T_ref)                                 #timestamp
            j += 1
            #print(m0, m1, m2)
            #print(i, util[i,0].item()/1000/1000/1000)
    print(torch.sum(res[0:-2], dim = 1))
    #print('\n')
    del sampler
    #del input_features
    del blks
    #del out
    #del preds
    #del minibatch
    gc.collect()
    torch.cuda.empty_cache()
    return res, freqs, mems    
    
    


def run(config, result_file_name):
    warm_up = config[8]
    results = []
    with open(result_file_name, 'wb') as f:
        pickle.dump(results, f)
    gc.collect()
    util = torch.zeros((100000,2))
    res, freqs, mems = measure(config[0], config[1], config[2], config[3], config[4], config[5], config[6], config[7], warm_up, util)
    [n_frq, _] = torch.sort(freqs, descending = True)
    tot = torch.sum(n_frq)
    K = int(n_frq.shape[0]/100) - 1
    hr = torch.zeros(100)
    for i in range(100):
        n = K *(i+1)
        hr[i] =  torch.sum(n_frq[0:n])/tot      
    exp_result = {}
    exp_result['config'] = config
    exp_result['data'] = res
    exp_result['util'] = util
    exp_result['freqs'] = hr
    exp_result['mems'] = mems
    with open(result_file_name, 'rb') as f:
        results = pickle.load(f)
    results.append(exp_result)
    with open(result_file_name, 'wb') as f:
        pickle.dump(results, f)
    print('results saved.')
        
        


if __name__ == '__main__':
    exp_res_dir = "experimental_analysis_results"
    Path(exp_res_dir).mkdir(parents=True, exist_ok=True)
    warm_up = 100
    
    
    
    #dataset_id, train, batch_size, num_trials, device, num_layers, model, num_heads, warm_up
    config = [0, False, [16, 17], 1000, 'cuda:0', 3, 'SAGE', 1, 100]
    result_file_name = 'profile_SAGE_const_PA.pkl'
    res_path = os.path.join(exp_res_dir, result_file_name)
    p = mp.Process(target=run, args=(config, res_path))
    p.start()
    p.join()
    
    
    #config = [0, False, [1, 9], 1000, 'cuda:0', 3, 'SAGE', 1, 100]
    #result_file_name = 'profile_SAGE_var_PA.pkl'
    #res_path = os.path.join(exp_res_dir, result_file_name)
    #p = mp.Process(target=run, args=(config, res_path))
    #p.start()
    #p.join()    
        
        
    config = [0, False, [8, 9], 1000, 'cuda:0', 3, 'GAT', 1, 100]
    result_file_name = 'profile_GAT_const_PA.pkl'
    res_path = os.path.join(exp_res_dir, result_file_name)
    p = mp.Process(target=run, args=(config, res_path))
    p.start()
    p.join() 

        
    #config = [0, False, [1, 5], 1000, 'cuda:0', 3, 'GAT', 1, 100]
    #result_file_name = 'profile_GAT_var_PA.pkl'
    #res_path = os.path.join(exp_res_dir, result_file_name)
    #p = mp.Process(target=run, args=(config, res_path))
    #p.start()
    #p.join()  



    #config = [0, True, [1024, 1025], 5000, 'cuda:0', 3, 'SAGE', 1, 100]
    #result_file_name = 'profile_SAGE_const_train_PA.pkl'
    #res_path = os.path.join(exp_res_dir, result_file_name)
    #p = mp.Process(target=run, args=(config, res_path))
    #p.start()
    #p.join() 
    
    








    #config = [1, False, [8, 9], 1000, 'cuda:0', 3, 'SAGE', 1, 100]
    #result_file_name = 'profile_SAGE_const_FR.pkl'
    #res_path = os.path.join(exp_res_dir, result_file_name)
    #p = mp.Process(target=run, args=(config, res_path))
    #p.start()
    #p.join()
    
    
    #config = [1, False, [1, 9], 1000, 'cuda:0', 3, 'SAGE', 1, 100]
    #result_file_name = 'profile_SAGE_var_FR.pkl'
    #res_path = os.path.join(exp_res_dir, result_file_name)
    #p = mp.Process(target=run, args=(config, res_path))
    #p.start()
    #p.join()   
        
        
    #config = [1, False, [1, 5], 1000, 'cuda:0', 3, 'GAT', 1, 100]
    #result_file_name = 'profile_GAT_const_FR.pkl'
    #res_path = os.path.join(exp_res_dir, result_file_name)
    #p = mp.Process(target=run, args=(config, res_path))
    #p.start()
    #p.join() 


        
    #config = [1, False, [1, 5], 1000, 'cuda:0', 3, 'GAT', 1, 100]
    #result_file_name = 'profile_GAT_var_FR.pkl'
    #res_path = os.path.join(exp_res_dir, result_file_name)
    #p = mp.Process(target=run, args=(config, res_path))
    #p.start()
    #p.join()   



    #config = [1, True, [1024, 1025], 5000, 'cuda:0', 3, 'SAGE', 1, 100]
    #result_file_name = 'profile_SAGE_const_train_FR.pkl'
    #res_path = os.path.join(exp_res_dir, result_file_name)
    #p = mp.Process(target=run, args=(config, res_path))
    #p.start()
    #p.join()
