#### Deprecated #####

from typing import List, Optional, Union, Tuple, Iterable
import math
import warnings

import torch
from e3nn import o3
from e3nn.util.jit import compile_mode, script
# from einops import rearrange


from diffusion_edf.equiformer.tensor_product_rescale import LinearRS

from diffusion_edf import EXTRACTOR_INFO_TYPE, GNN_OUTPUT_TYPE, QUERY_TYPE, EDF_INFO_TYPE
from diffusion_edf.embedding import NodeEmbeddingNetwork
from diffusion_edf.block import EquiformerBlock
from diffusion_edf.connectivity import FpsPool, RadiusGraph, RadiusConnect
from diffusion_edf.radial_func import GaussianRadialBasisLayerFiniteCutoff
from diffusion_edf.utils import multiply_irreps, ParityInversionSh
from diffusion_edf.skip import ProjectIfMismatch
from diffusion_edf.extractor import EdfExtractorLight




class EdfUnet(torch.nn.Module):  
    def __init__(self,
        irreps: List[o3.Irreps],
        irreps_edge_attr: List[o3.Irreps], 
        num_heads: List[int], 
        fc_neurons: List[List[int]],
        radius: List[float],
        pool_ratio: List[float],
        n_layers: List[int],
        deterministic: bool = False,
        pool_method: Union[Optional[str], List[Optional[str]]] = 'fps',
        irreps_mlp_mid: Union[Union[o3.Irreps, int], List[Union[o3.Irreps, int]]] = 3,
        attn_type: Union[str, List[str]] = 'mlp',
        alpha_drop: Union[float, List[float]] = 0.1,
        proj_drop: Union[float, List[float]] = 0.1,
        drop_path_rate: Union[float, List[float]] = 0.0,
        n_layers_mid: int = 2):
        
        super().__init__()
        self.irreps: List[o3.Irreps] = [o3.Irreps(irrep) for irrep in irreps]
        self.irreps_edge_attr: List[o3.Irreps] = [o3.Irreps(irrep) for irrep in irreps_edge_attr]
        self.num_heads: List[int] = num_heads
        self.fc_neurons: List[List[int]] = fc_neurons
        self.radius: List[float] = radius
        self.pool_ratio: List[float] = pool_ratio
        self.n_layers: List[int] = n_layers
        self.deterministic: bool = deterministic
        self.n_layers_mid = n_layers_mid
        
        self.n_scales = len(self.irreps)
        if isinstance(pool_method, list):
            self.pool_method: List[Optional[str]] = pool_method
        else:
            self.pool_method: List[Optional[str]] = [pool_method for _ in range(self.n_scales)]
        if isinstance(irreps_mlp_mid, list):
            self.irreps_mlp_mid: List[Union[o3.Irreps, int]] = irreps_mlp_mid
        else:
            self.irreps_mlp_mid: List[Union[o3.Irreps, int]] = [irreps_mlp_mid for _ in range(self.n_scales)]
        if isinstance(attn_type, list):
            self.attn_type: List[str] = attn_type
        else:
            self.attn_type: List[str] = [attn_type for _ in range(self.n_scales)]
        if isinstance(alpha_drop, list):
            self.alpha_drop: List[float] = alpha_drop
        else:
            self.alpha_drop: List[float] = [alpha_drop for _ in range(self.n_scales)]
        if isinstance(proj_drop, list):
            self.proj_drop: List[float] = proj_drop
        else:
            self.proj_drop: List[float] = [proj_drop for _ in range(self.n_scales)]
        if isinstance(drop_path_rate, list):
            self.drop_path_rate: List[float] = drop_path_rate
        else:
            self.drop_path_rate: List[float] = [drop_path_rate for _ in range(self.n_scales)]

        assert self.n_scales == len(self.irreps)
        assert self.n_scales == len(self.irreps_edge_attr) 
        assert self.n_scales == len(self.num_heads) 
        assert self.n_scales == len(self.fc_neurons) 
        assert self.n_scales == len(self.radius) 
        assert self.n_scales == len(self.pool_ratio) 
        assert self.n_scales == len(self.n_layers) 
        assert self.n_scales == len(self.pool_method) 
        assert self.n_scales == len(self.irreps_mlp_mid) 
        assert self.n_scales == len(self.attn_type) 
        assert self.n_scales == len(self.alpha_drop) 
        assert self.n_scales == len(self.proj_drop) 
        assert self.n_scales == len(self.drop_path_rate)
        self.irreps_head: List[o3.Irreps] = [multiply_irreps(self.irreps[n], 1/self.num_heads[n], strict=True) for n in range(self.n_scales)]

        for n in range(self.n_scales):
            if self.pool_ratio[n] == 1.0:
                assert self.pool_method[n] is None
            else:
                assert self.pool_method[n] is not None
            assert self.n_layers[n] >= 1


        #### Down Block ####
        self.down_blocks = torch.nn.ModuleList()
        for n in range(self.n_scales):
            block = torch.nn.ModuleDict()
            if self.pool_method[n] == 'fps':
                block['pool'] = FpsPool(ratio=self.pool_ratio[n], random_start=not self.deterministic, r=self.radius[n], max_num_neighbors=1000)
                block['pool_proj'] = ProjectIfMismatch(irreps_in = self.irreps[max(n-1,0)], irreps_out = self.irreps[n])
            else:
                raise NotImplementedError
            block['radius_graph'] = RadiusGraph(r=self.radius[n], max_num_neighbors=1000)
            block['spherical_harmonics'] = o3.SphericalHarmonics(irreps_out = self.irreps_edge_attr[n], normalize = True, normalization='component')

            pool_layer = torch.nn.ModuleDict()
            pool_layer['radial'] = GaussianRadialBasisLayerFiniteCutoff(num_basis=self.fc_neurons[n][0], cutoff=0.99 * self.radius[n])
            pool_layer['gnn'] = EquiformerBlock(irreps_src = self.irreps[max(n-1,0)], 
                                                irreps_dst = self.irreps[n], 
                                                irreps_edge_attr = self.irreps_edge_attr[n], 
                                                irreps_head = self.irreps_head[n],
                                                num_heads = self.num_heads[n], 
                                                fc_neurons = self.fc_neurons[n],
                                                irreps_mlp_mid = self.irreps_mlp_mid[n],
                                                attn_type = self.attn_type[n],
                                                alpha_drop = self.alpha_drop[n],
                                                proj_drop = self.proj_drop[n],
                                                drop_path_rate = self.drop_path_rate[n],
                                                src_bias = False,
                                                dst_bias = True)
            block['pool_layer'] = pool_layer

            layer_stack = torch.nn.ModuleList()
            for _ in range(self.n_layers[n] - 1):
                layer = torch.nn.ModuleDict()
                layer['radial'] = GaussianRadialBasisLayerFiniteCutoff(num_basis=self.fc_neurons[n][0], cutoff=0.99 * self.radius[n])
                layer['gnn'] = EquiformerBlock(irreps_src = self.irreps[n], 
                                               irreps_dst = self.irreps[n], 
                                               irreps_edge_attr = self.irreps_edge_attr[n], 
                                               irreps_head = self.irreps_head[n],
                                               num_heads = self.num_heads[n], 
                                               fc_neurons = self.fc_neurons[n],
                                               irreps_mlp_mid = self.irreps_mlp_mid[n],
                                               attn_type = self.attn_type[n],
                                               alpha_drop = self.alpha_drop[n],
                                               proj_drop = self.proj_drop[n],
                                               drop_path_rate = self.drop_path_rate[n],
                                               src_bias = False,
                                               dst_bias = True)
                layer_stack.append(layer)
            block['layer_stack'] = layer_stack

            self.down_blocks.append(block)


        #### Mid Block ####
        self.mid_block = torch.nn.ModuleList()
        for i in range(self.n_layers_mid):
            layer = torch.nn.ModuleDict()
            layer['radial'] = GaussianRadialBasisLayerFiniteCutoff(num_basis=self.fc_neurons[-1][0], cutoff=0.99 * self.radius[-1])
            layer['gnn'] = EquiformerBlock(irreps_src = self.irreps[-1], 
                                            irreps_dst = self.irreps[-1], 
                                            irreps_edge_attr = self.irreps_edge_attr[-1], 
                                            irreps_head = self.irreps_head[-1],
                                            num_heads = self.num_heads[-1], 
                                            fc_neurons = self.fc_neurons[-1],
                                            irreps_mlp_mid = self.irreps_mlp_mid[-1],
                                            attn_type = self.attn_type[-1],
                                            alpha_drop = self.alpha_drop[-1],
                                            proj_drop = self.proj_drop[-1],
                                            drop_path_rate = self.drop_path_rate[-1],
                                            src_bias = False,
                                            dst_bias = True)
            self.mid_block.append(layer)

        #### Up Block ####
        self.up_blocks = torch.nn.ModuleList()
        for n in range(self.n_scales-1, -1, -1):
            block = torch.nn.ModuleDict()
            block['parity_inversion'] = ParityInversionSh(irreps = self.irreps_edge_attr[n])

            layer_stack = torch.nn.ModuleList()
            for _ in range(self.n_layers[n] - 1):
                layer = torch.nn.ModuleDict()
                layer['radial'] = GaussianRadialBasisLayerFiniteCutoff(num_basis=self.fc_neurons[n][0], cutoff=0.99 * self.radius[n])
                layer['gnn'] = EquiformerBlock(irreps_src = self.irreps[n], 
                                               irreps_dst = self.irreps[n], 
                                               irreps_edge_attr = self.irreps_edge_attr[n], 
                                               irreps_head = self.irreps_head[n],
                                               num_heads = self.num_heads[n], 
                                               fc_neurons = self.fc_neurons[n],
                                               irreps_mlp_mid = self.irreps_mlp_mid[n],
                                               attn_type = self.attn_type[n],
                                               alpha_drop = self.alpha_drop[n],
                                               proj_drop = self.proj_drop[n],
                                               drop_path_rate = self.drop_path_rate[n],
                                               src_bias = False,
                                               dst_bias = True)
                layer_stack.append(layer)
            block['layer_stack'] = layer_stack

            unpool_layer = torch.nn.ModuleDict()
            unpool_layer['radial'] = GaussianRadialBasisLayerFiniteCutoff(num_basis=self.fc_neurons[n][0], cutoff=0.99 * self.radius[n])
            unpool_layer['gnn'] = EquiformerBlock(irreps_src = self.irreps[n], 
                                                  irreps_dst = self.irreps[max(n-1,0)], 
                                                  irreps_edge_attr = self.irreps_edge_attr[n], 
                                                  irreps_head = self.irreps_head[max(n-1,0)],
                                                  num_heads = self.num_heads[n], 
                                                  fc_neurons = self.fc_neurons[n],
                                                  irreps_mlp_mid = self.irreps_mlp_mid[n],
                                                  attn_type = self.attn_type[n],
                                                  alpha_drop = self.alpha_drop[n],
                                                  proj_drop = self.proj_drop[n],
                                                  drop_path_rate = self.drop_path_rate[n],
                                                  src_bias = False,
                                                  dst_bias = True)
            block['unpool_layer'] = unpool_layer


            self.up_blocks.append(block)


        output_idx = [0, 1]
        for n_layers in self.n_layers[:-1]:
            output_idx.append(output_idx[-1] + n_layers)
        self.output_idx: Tuple[int] = tuple(output_idx)

        output_edge_idx = [0]
        for n_layers in self.n_layers[:-1]:
            output_edge_idx.append(output_edge_idx[-1] + n_layers)
        self.output_edge_idx: Tuple[int] = tuple(output_edge_idx)



        self.project_outputs = torch.nn.ModuleList()
        for n in range(self.n_scales + 1):
            self.project_outputs.append(ProjectIfMismatch(irreps_in=self.irreps[max(0,n-1)],
                                                          irreps_out=self.irreps[-1]))


    def forward(self, node_feature: torch.Tensor,
                node_coord: torch.Tensor,
                batch: torch.Tensor) -> GNN_OUTPUT_TYPE:

        ########### Downstream Block #############
        downstream_outputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        downstream_edges: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        downstream_outputs.append((node_feature, node_coord, batch))
        for n, block in enumerate(self.down_blocks):
            ##### Pooling #####
            pool_graph = block['pool'](node_coord_src = node_coord, 
                                       node_feature_src = node_feature, 
                                       batch_src = batch)
            node_feature_dst, node_coord_dst, edge_src, edge_dst, degree, batch_dst = pool_graph
            node_feature_dst = block['pool_proj'](node_feature_dst)
            edge_vec: torch.Tensor = node_coord.index_select(0, edge_src) - node_coord_dst.index_select(0, edge_dst)
            edge_length = edge_vec.norm(dim=1, p=2)
            edge_attr = block['spherical_harmonics'](edge_vec)

            edge_scalars = block['pool_layer']['radial'](edge_length)
            node_feature_dst = block['pool_layer']['gnn'](node_input_src = node_feature,
                                                          node_input_dst = node_feature_dst,
                                                          batch_dst = batch_dst,
                                                          edge_src = edge_src,
                                                          edge_dst = edge_dst,
                                                          edge_attr = edge_attr,
                                                          edge_scalars = edge_scalars)
            

            node_feature = node_feature_dst
            node_coord = node_coord_dst
            batch = batch_dst
            downstream_outputs.append((node_feature, node_coord, batch))
            downstream_edges.append((edge_src, edge_dst, edge_length, edge_attr))
            
            ##### Radius Graph #####
            radius_graph = block['radius_graph'](node_coord_src = node_coord, 
                                                 node_feature_src = node_feature, 
                                                 batch_src = batch)
            node_feature_dst, node_coord_dst, edge_src, edge_dst, degree, batch_dst = radius_graph
            edge_vec: torch.Tensor = node_coord.index_select(0, edge_src) - node_coord_dst.index_select(0, edge_dst)
            edge_length = edge_vec.norm(dim=1, p=2)
            edge_attr = block['spherical_harmonics'](edge_vec)

            for i, layer in enumerate(block['layer_stack']):
                edge_scalars = layer['radial'](edge_length)
                node_feature_dst = layer['gnn'](node_input_src = node_feature,
                                                node_input_dst = node_feature_dst,
                                                batch_dst = batch_dst,
                                                edge_src = edge_src,
                                                edge_dst = edge_dst,
                                                edge_attr = edge_attr,
                                                edge_scalars = edge_scalars)


                node_feature = node_feature_dst
                node_coord = node_coord_dst
                batch = batch_dst
                downstream_outputs.append((node_feature, node_coord, batch))
                downstream_edges.append((edge_src, edge_dst, edge_length, edge_attr))



        ########### Mid Block #############
        for n, layer in enumerate(self.mid_block):
            edge_scalars = layer['radial'](edge_length)
            node_feature_dst = layer['gnn'](node_input_src = node_feature,
                                            node_input_dst = node_feature_dst,
                                            batch_dst = batch,
                                            edge_src = edge_src,
                                            edge_dst = edge_dst,
                                            edge_attr = edge_attr,
                                            edge_scalars = edge_scalars)
            node_feature = node_feature_dst
            node_coord = node_coord_dst
            batch = batch_dst

        node_feature_dst, node_coord_dst, batch_dst = downstream_outputs.pop()
        node_feature = (node_feature + node_feature_dst) / math.sqrt(3) # Skip connection.


        ########### Upstream Block #############
        upstream_outputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        upstream_edges: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for n, block in enumerate(self.up_blocks):            
            ##### Radius Graph #####
            for i, layer in enumerate(block['layer_stack']):
                node_feature_dst, node_coord_dst, batch_dst = downstream_outputs.pop()
                edge_src, edge_dst, edge_length, edge_attr = downstream_edges.pop()
                edge_src, edge_dst, edge_attr = edge_dst, edge_src, block['parity_inversion'](edge_attr) # Swap source and destination.
                node_feature_dst = (node_feature + node_feature_dst) / math.sqrt(3) # Skip connection.

                edge_scalars = layer['radial'](edge_length)
                node_feature_dst = layer['gnn'](node_input_src = node_feature,
                                                node_input_dst = node_feature_dst,
                                                batch_dst = batch_dst,
                                                edge_src = edge_src,
                                                edge_dst = edge_dst,
                                                edge_attr = edge_attr,
                                                edge_scalars = edge_scalars)
                node_feature = node_feature_dst
                node_coord = node_coord_dst
                batch = batch_dst
                upstream_outputs.append((node_feature, node_coord, batch))
                upstream_edges.append((edge_src, edge_dst, edge_length, edge_attr))

            ##### Unpooling #####
            node_feature_dst, node_coord_dst, batch_dst = downstream_outputs.pop()
            edge_src, edge_dst, edge_length, edge_attr = downstream_edges.pop()
            edge_src, edge_dst, edge_attr = edge_dst, edge_src, block['parity_inversion'](edge_attr) # Swap source and destination.
            # node_feature_dst = (node_feature + node_feature_dst) / math.sqrt(2) # Cannot apply skip connection, as node number of input and output is different. Instead, directly use node_feature_dst from downstream output, therefore it serves as skip connection.

            edge_scalars = block['unpool_layer']['radial'](edge_length)
            node_feature_dst = block['unpool_layer']['gnn'](node_input_src = node_feature,
                                                            node_input_dst = node_feature_dst,
                                                            batch_dst = batch_dst,
                                                            edge_src = edge_src,
                                                            edge_dst = edge_dst,
                                                            edge_attr = edge_attr,
                                                            edge_scalars = edge_scalars)
            
            node_feature = node_feature_dst
            node_coord = node_coord_dst
            batch = batch_dst
            upstream_outputs.append((node_feature, node_coord, batch))
            upstream_edges.append((edge_src, edge_dst, edge_length, edge_attr))

        upstream_outputs, upstream_edges = upstream_outputs[::-1], upstream_edges[::-1]
        upstream_outputs, upstream_edges = [upstream_outputs[n] for n in self.output_idx], [upstream_edges[n] for n in self.output_edge_idx]
        
        node_features: List[torch.Tensor] = []
        node_coords: List[torch.Tensor] = []
        batchs: List[torch.Tensor] = []
        edge_srcs: List[torch.Tensor] = []
        edge_dsts: List[torch.Tensor] = []
        scale_slice: List[int] = [0,]

        N_nodes = 0
        for scale, projection in enumerate(self.project_outputs):
            (node_feature, node_coord, batch) = upstream_outputs[scale]

            N_nodes_this_scale = len(node_feature)
            assert N_nodes_this_scale == len(node_feature) == len(node_coord) == len(batch)

            node_features.append(projection(node_feature))
            node_coords.append(node_coord)
            batchs.append(batch)
            scale_slice.append(len(node_coord) + scale_slice[-1])


            if scale >= self.n_scales: # Last layer is self-connecting
                assert scale == self.n_scales # shouldn't be higher than self.n_scales
            else:
                (edge_src, edge_dst, _, _) = upstream_edges[scale]
                # print((edge_src.min().item(), edge_src.max().item()), (edge_dst.min().item(), edge_dst.max().item()))
                edge_dst = edge_dst + N_nodes
                edge_dsts.append(edge_dst)
                N_nodes = N_nodes + N_nodes_this_scale
                edge_src = edge_src + N_nodes    
                edge_srcs.append(edge_src)


        node_feature = torch.cat(node_features, dim=-2)
        node_coord = torch.cat(node_coords, dim=-2)
        batch = torch.cat(batchs, dim=-1)
        edge_src = torch.cat(edge_srcs, dim=-1)
        edge_dst = torch.cat(edge_dsts, dim=-1)
        
        return node_feature, node_coord, batch, scale_slice, edge_src, edge_dst
    






class EDF(torch.nn.Module):
    def __init__(self, 
                 irreps_input: o3.Irreps,
                 irreps_emb_init: o3.Irreps,
                 irreps_sh: o3.Irreps,
                 fc_neurons_init: List[int],
                 num_heads: int,
                 n_scales: int,
                 pool_ratio: float,
                 dim_mult: List[Union[float, int]], 
                 n_layers: int,
                 gnn_radius: Union[float, List[float]],
                 cutoff_radius: Union[float, List[float]],
                 alpha_drop: float = 0.1,
                 proj_drop: float = 0.1,
                 drop_path_rate: float = 0.0,
                 irreps_mlp_mid: int = 3,
                 deterministic: bool = False,
                 detach_extractor: bool = False,
                 compile_head: bool = False,
                 attn_type: str = 'mlp',
                 input_mean = torch.tensor([0.5, 0.5, 0.5]), 
                 input_std = torch.tensor([0.5, 0.5, 0.5]),
                 infinite = False,
                 ):
        super().__init__()
        self.irreps_input = o3.Irreps(irreps_input)
        assert dim_mult[0] == 1
        self.irreps: List[o3.Irreps] = [multiply_irreps(o3.Irreps(irreps_emb_init), dim_mult[n], strict=True) for n in range(n_scales)]
        self.irreps_emb = self.irreps[-1]
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.num_heads = num_heads
        self.n_scales = n_scales
        self.deterministic = deterministic
        self.fc_neurons = [[round(n_neurons * dim_mult[n]) for n_neurons in fc_neurons_init] for n in range(n_scales)]
        assert len(self.fc_neurons) == self.n_scales
        if isinstance(pool_ratio, Iterable):
            self.pool_ratio = pool_ratio
        else:
            self.pool_ratio = [pool_ratio for _ in range(n_scales)]
        self.n_layers = [n_layers for _ in range(n_scales)]
        if isinstance(gnn_radius, Iterable):
            self.gnn_radius = gnn_radius
        else:
            self.gnn_radius = [gnn_radius / math.sqrt(self.pool_ratio[n]**n) for n in range(n_scales)]
        for n in range(1, len(self.gnn_radius)):
            if not self.gnn_radius[n-1] < self.gnn_radius[n]:
                warnings.warn(f"gnn_radius[{n}] ({self.gnn_radius[n]}) is smaller than radius[{n-1}] ({self.gnn_radius[n-1]})")
        if isinstance(cutoff_radius, Iterable):
            self.cutoff_radius = cutoff_radius
        else:
            self.cutoff_radius = [cutoff_radius / math.sqrt(self.pool_ratio[n]**n) for n in range(n_scales)]
        for n in range(1, len(self.cutoff_radius)):
            if not self.cutoff_radius[n-1] < self.cutoff_radius[n]:
                warnings.warn(f"cutoff_radius[{n}] ({self.cutoff_radius[n]}) is smaller than radius[{n-1}] ({self.cutoff_radius[n-1]})")
        self.irreps_mlp_mid = irreps_mlp_mid
        self.attn_type = attn_type
        self.alpha_drop = alpha_drop
        self.proj_drop = proj_drop
        self.drop_path_rate = drop_path_rate
        self.infinite = infinite


        self.enc = NodeEmbeddingNetwork(irreps_input=self.irreps_input, irreps_node_emb=self.irreps[0], input_mean=input_mean, input_std=input_std)
        self.gnn = EdfUnet(
            irreps = self.irreps,
            irreps_edge_attr = [self.irreps_sh for _ in range(n_scales)],
            num_heads = [self.num_heads for _ in range(n_scales)],
            fc_neurons = self.fc_neurons,
            radius = self.gnn_radius,
            pool_ratio = self.pool_ratio,
            n_layers = self.n_layers,
            deterministic = deterministic,
            irreps_mlp_mid = self.irreps_mlp_mid,
            alpha_drop=self.alpha_drop,
            proj_drop=self.proj_drop,
            drop_path_rate=self.drop_path_rate,
            pool_method = 'fps',
            attn_type = self.attn_type,
            n_layers_mid = 2,
        )


        self.min_offset = 0.01 * self.cutoff_radius[0]
        self.detach_extractor = detach_extractor

        if self.detach_extractor:
            self.extractor = None
        else:
            self.extractor = EdfExtractorLight(
                irreps_inputs = [self.gnn.irreps[-1] for _ in range(n_scales)],
                irreps_emb = self.gnn.irreps[-1],
                irreps_edge_attr = self.gnn.irreps_edge_attr[-1],
                irreps_head = self.gnn.irreps_head[-1],
                num_heads = self.gnn.num_heads[-1],
                fc_neurons = self.gnn.fc_neurons[-1],
                n_layers = 1,
                cutoffs = self.cutoff_radius,
                offsets = [self.min_offset] + [max(self.min_offset, offset - 0.2*(cutoff - offset)) for offset, cutoff in zip(self.cutoff_radius[:-1], self.cutoff_radius[1:])],
                irreps_mlp_mid = self.irreps_mlp_mid,
                attn_type=self.attn_type,
                alpha_drop=self.alpha_drop, 
                proj_drop=self.proj_drop,
                drop_path_rate=self.drop_path_rate,
                infinite = self.infinite
            )
        if compile_head and not self.detach_extractor:
            self.extractor = torch.jit.script(self.extractor)


    def get_extractor(self) -> EdfExtractorLight:
        extr = EdfExtractorLight(irreps_inputs = self.gnn.irreps,
                            irreps_emb = self.gnn.irreps[-1],
                            irreps_edge_attr = self.gnn.irreps_edge_attr[-1],
                            irreps_head = self.gnn.irreps_head[-1],
                            num_heads = self.gnn.num_heads[-1],
                            fc_neurons = self.gnn.fc_neurons[-1],
                            n_layers = 1,
                            cutoffs = self.cutoff_radius,
                            offsets = [self.min_offset] + [max(self.min_offset, offset - 0.2*(cutoff - offset)) for offset, cutoff in zip(self.cutoff_radius[:-1], self.cutoff_radius[1:])],
                            irreps_mlp_mid = self.irreps_mlp_mid,
                            attn_type=self.attn_type,
                            alpha_drop=self.alpha_drop, 
                            proj_drop=self.proj_drop,
                            drop_path_rate=self.drop_path_rate,
                            infinite = self.infinite)
        return extr

    def get_gnn_outputs(self, node_feature: torch.Tensor, 
                        node_coord: torch.Tensor, 
                        batch: torch.Tensor) -> GNN_OUTPUT_TYPE: 
        node_emb = self.enc(node_feature)
        node_feature, node_coord, batch, scale_slice, edge_src, edge_dst = self.gnn(node_feature=node_emb,
                                                                                    node_coord=node_coord,
                                                                                    batch=batch)
        return node_feature, node_coord, batch, scale_slice, edge_src, edge_dst
    
    # def _reshape_extractor_inputs(self, query_coord: torch.Tensor,
    #                               query_batch: torch.Tensor,
    #                               gnn_outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
    #     assert query_coord.ndim == 2 and query_coord.shape[-1] == 3 and query_batch.ndim == 1
    #     Nq = len(query_coord)
    #     query_coord = query_coord.expand(self.n_scales, Nq, 3) # (Ns, Nq, 3)
    #     query_scale = torch.arange(self.n_scales, dtype=query_batch.dtype, device=query_batch.device)
    #     query_batch_n_scale = query_batch * query_scale.unsqueeze(-1) + query_scale.unsqueeze(-1) # (Ns, Nq)

    #     node_feature, node_coord, batch, scale, edge_src, edge_dst = gnn_outputs
    #     node_batch_n_scale = batch * scale + scale

    #     return query_coord, query_batch_n_scale, node_feature, node_coord, node_batch_n_scale
    
    # def get_scale_node(self, node_coord, node_batch_n_scale, n):
    #     this_scale_node_idx = (node_batch_n_scale % self.n_scales == n).nonzero().squeeze(-1)
    #     node_coord_this_scale = torch.index_select(node_coord, dim=-2, index = this_scale_node_idx)
    #     batch_this_scale = torch.index_select(node_batch_n_scale, dim=-1, index = this_scale_node_idx) // self.n_scales
            

    def forward(self, query_coord: torch.Tensor,
                query_batch: torch.Tensor,
                node_feature: torch.Tensor, 
                node_coord: torch.Tensor, 
                batch: torch.Tensor,
                info_mode: str = 'NONE') -> Tuple[torch.Tensor, Optional[EDF_INFO_TYPE]]:      
        gnn_outputs = self.get_gnn_outputs(node_feature=node_feature, node_coord=node_coord, batch=batch)
        node_feature, node_coord, batch, scale_slice, edge_src, edge_dst = gnn_outputs

        field_val, extractor_info = self.extractor(query_coord = query_coord, 
                                                   query_batch = query_batch,
                                                   node_feature = node_feature,
                                                   node_coord = node_coord,
                                                   node_batch = batch,
                                                   node_scale_slice = scale_slice)
        
        if info_mode == 'NONE':
            edf_info = None
        elif info_mode == 'NO_GRAD' or info_mode == 'REQUIRES_GRAD':
            (edge_src_field, edge_dst_field) = extractor_info
            if info_mode == 'NO_GRAD':
                gnn_outputs = (node_feature.detach(), 
                            node_coord.detach(), 
                            batch.detach(),
                            scale_slice,
                            edge_src.detach(), 
                            edge_dst.detach())
                extractor_info = (edge_src_field.detach(), edge_dst_field.detach())
            edf_info = (extractor_info, gnn_outputs)
        else:
            raise ValueError(f"Unknown info_mode: {info_mode}")
        
        return field_val, edf_info