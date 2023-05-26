from typing import List, Optional, Union, Tuple, Iterable, Dict
import math
import warnings

from beartype import beartype

import torch
from e3nn import o3
from torch_scatter import scatter_log_softmax

from diffusion_edf.equiformer.tensor_product_rescale import LinearRS
from diffusion_edf.gnn_data import FeaturedPoints
from diffusion_edf.block import EquiformerBlock
from diffusion_edf.connectivity import FpsPool, RadiusGraph, RadiusConnect
from diffusion_edf.radial_func import GaussianRadialBasisLayerFiniteCutoff
from diffusion_edf.utils import multiply_irreps, ParityInversionSh
from diffusion_edf.skip import ProjectIfMismatch

class UnetFeatureExtractor(torch.nn.Module):
    @beartype
    def __init__(self,
        irreps_input: Optional[Union[str, o3.Irreps]],
        irreps_output: Union[str, o3.Irreps],
        irreps_emb: List[Union[str, o3.Irreps]],
        irreps_edge_attr: List[Union[str, o3.Irreps]], 
        num_heads: List[int], 
        fc_neurons: List[List[int]],
        n_layers: List[int],
        pool_ratio: List[float],
        radius: List[Optional[float]],
        deterministic: bool = False,
        pool_method: Union[
            Optional[str], 
            List[Optional[str]]
            ] = 'fps',
        irreps_mlp_mid: Union[
            Union[str, o3.Irreps, int], 
            List[Union[str, o3.Irreps, int]]
            ] = 3,
        attn_type: Union[str, List[str]] = 'mlp',
        alpha_drop: Union[float, List[float]] = 0.1,
        proj_drop: Union[float, List[float]] = 0.1,
        drop_path_rate: Union[float, List[float]] = 0.0,
        n_layers_midstream: int = 2,
        n_scales: Optional[int] = None,
        output_scalespace: Optional[List[int]] = None):

        self.log_num_points = math.log(10)
        
        super().__init__()

        self.irreps_output: o3.Irreps = o3.Irreps(irreps_output)
        self.irreps_emb: List[o3.Irreps] = [o3.Irreps(irrep) for irrep in irreps_emb]
        self.irreps_edge_attr: List[o3.Irreps] = [o3.Irreps(irrep) for irrep in irreps_edge_attr]
        self.num_heads: List[int] = num_heads
        self.fc_neurons: List[List[int]] = fc_neurons
        self.pool_ratio: List[float] = pool_ratio
        self.n_layers: List[int] = n_layers
        self.deterministic: bool = deterministic
        self.n_layers_midstream: int = n_layers_midstream

        if irreps_input is None:
            self.irreps_input: o3.Irreps = self.irreps_emb[0]
            self.input_emb = None
        else:
            self.irreps_input: o3.Irreps = o3.Irreps(irreps_input)
            self.input_emb = LinearRS(self.irreps_input, 
                                      self.irreps_emb[0], 
                                      bias=True)

        if n_scales is None:
            self.n_scales: int = len(self.irreps_emb)
        else:
            self.n_scales: int = n_scales
        if output_scalespace is None:
            output_scalespace = [n for n in range(self.n_scales)]
        self.output_scalespace: List[int] = [self.n_scales+n if n<0 else n for n in output_scalespace]

        self.radius: List[float] = [radius[0]]
        for n, r in enumerate(radius[1:]):
            if r is None:
                self.radius.append(
                    self.radius[-1] / math.sqrt(self.pool_ratio[n-1])
                )
            else:
                self.radius.append(r)

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

        assert self.n_scales == len(self.irreps_emb)
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
        self.irreps_head: List[o3.Irreps] = [multiply_irreps(self.irreps_emb[n], 1/self.num_heads[n], strict=True) for n in range(self.n_scales)]

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
                block['pool_proj'] = ProjectIfMismatch(irreps_in = self.irreps_emb[max(n-1,0)], irreps_out = self.irreps_emb[n])
            else:
                raise NotImplementedError
            block['radius_graph'] = RadiusGraph(r=self.radius[n], max_num_neighbors=1000)
            block['spherical_harmonics'] = o3.SphericalHarmonics(irreps_out = self.irreps_edge_attr[n], normalize = True, normalization='component')

            pool_layer = torch.nn.ModuleDict()
            pool_layer['radial'] = GaussianRadialBasisLayerFiniteCutoff(num_basis=self.fc_neurons[n][0], cutoff=0.99 * self.radius[n])
            pool_layer['gnn'] = EquiformerBlock(irreps_src = self.irreps_emb[max(n-1,0)], 
                                                irreps_dst = self.irreps_emb[n], 
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
                layer['gnn'] = EquiformerBlock(irreps_src = self.irreps_emb[n], 
                                               irreps_dst = self.irreps_emb[n], 
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
        for i in range(self.n_layers_midstream):
            layer = torch.nn.ModuleDict()
            layer['radial'] = GaussianRadialBasisLayerFiniteCutoff(num_basis=self.fc_neurons[-1][0], cutoff=0.99 * self.radius[-1])
            layer['gnn'] = EquiformerBlock(irreps_src = self.irreps_emb[-1], 
                                            irreps_dst = self.irreps_emb[-1], 
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
                layer['gnn'] = EquiformerBlock(irreps_src = self.irreps_emb[n], 
                                               irreps_dst = self.irreps_emb[n], 
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
            unpool_layer['gnn'] = EquiformerBlock(irreps_src = self.irreps_emb[n], 
                                                  irreps_dst = self.irreps_emb[max(n-1,0)], 
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

        self.project_outputs = torch.nn.ModuleList()
        for n in range(self.n_scales):
            self.project_outputs.append(ProjectIfMismatch(irreps_in=self.irreps_emb[n],
                                                          irreps_out=self.irreps_output))

    #@beartype
    def forward(self, pcd: FeaturedPoints) -> List[FeaturedPoints]:

        node_coord: torch.Tensor = pcd.x    # (N, 3)
        node_feature: torch.Tensor = pcd.f  # (N, F_in)
        batch: torch.Tensor = pcd.b         # (N, )
        assert node_feature.ndim == 2
        assert node_coord.ndim == 2
        assert batch.ndim == 1
        assert len(node_feature) == len(node_coord) == len(batch)

        if self.input_emb is not None:
            node_feature = self.input_emb(node_feature) # (N, F)

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
            edge_length = torch.norm(edge_vec, dim=1, p=2)
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
                # upstream_outputs.append((node_feature, node_coord, batch))
                # upstream_edges.append((edge_src, edge_dst, edge_length, edge_attr))
            upstream_outputs.append((node_feature, node_coord, batch))
            upstream_edges.append((edge_src, edge_dst, edge_length, edge_attr))

            ##### Unpooling #####
            node_feature_dst, node_coord_dst, batch_dst = downstream_outputs.pop()
            edge_src, edge_dst, edge_length, edge_attr = downstream_edges.pop()
            edge_src, edge_dst, edge_attr = edge_dst, edge_src, block['parity_inversion'](edge_attr) # Swap source and destination.
            # node_feature_dst = (node_feature + node_feature_dst) / math.sqrt(2) # Cannot apply skip connection, as node number of input and output is different. Instead, directly use node_feature_dst from downstream output, therefore it serves as skip connection.

            if n == self.n_scales-1:
                pass
            else:
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
                # upstream_outputs.append((node_feature, node_coord, batch))
                # upstream_edges.append((edge_src, edge_dst, edge_length, edge_attr))

        upstream_outputs, upstream_edges = upstream_outputs[::-1], upstream_edges[::-1]

        pcds = []
        for scale, projection in enumerate(self.project_outputs):
            if scale not in self.output_scalespace:
                continue
            (node_feature, node_coord, batch) = upstream_outputs[scale]
            f = projection(node_feature)
            pcd = FeaturedPoints(
                x = node_coord,
                f = f,
                b = batch,
                w = None,
            )
            pcds.append(pcd)

        return pcds

    #     pcd_multiscale: List[FeaturedPoints] = []
    #     # edge_edges: List[GraphEdge] = []

    #     for scale, projection in enumerate(self.project_outputs):
    #         (node_feature, node_coord, batch) = upstream_outputs[scale]
    #         f = projection(node_feature)
    #         if self.point_weight_mlp is None:
    #             w = None
    #         else:
    #             w = self.point_weight_mlp(f[..., :self.point_weight_emb_dim]).squeeze(-1).contiguous()
    #             # w = torch.sigmoid(w)
    #             w = scatter_log_softmax(src=w, index=batch, dim=-1) + self.log_num_points
    #             w = torch.exp(w)
    #             f = f[..., self.point_weight_emb_dim:].contiguous()

    #         pcd = FeaturedPoints(
    #             x = node_coord,
    #             f = f,
    #             b = batch,
    #             w = w,
    #         )
    #         pcd_multiscale.append(pcd)

    #         if scale >= self.n_scales: # Last layer is self-connecting
    #             assert scale == self.n_scales # shouldn't be higher than self.n_scales
    #         else:
    #             pass
    #             # (edge_src, edge_dst, _, _) = upstream_edges[scale]
    #             # graph_edge = GraphEdge(edge_src=edge_src, edge_dst=edge_dst)
        
    #     return pcd_multiscale #, graph_edge
    
    # def forward(self, pcd: FeaturedPoints) -> FeaturedPoints:
    #     scale_space: int = self.output_scalespace-1
    #     assert scale_space >= 0
        
    #     return self.forward_multiscale(pcd=pcd)[scale_space]