#### Deprecated #####

from typing import List, Tuple, Optional, Union
import torch

GNN_OUTPUT_TYPE = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int], torch.Tensor, torch.Tensor] # node_feature, node_coord, batch, scale_slice, edge_src, edge_dst
EXTRACTOR_INFO_TYPE = Tuple[torch.Tensor, torch.Tensor]                                                  # field_val, edf_info
QUERY_TYPE = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]                               # query_weight, query_feature, query_coord, query_batch
EDF_INFO_TYPE = Tuple[EXTRACTOR_INFO_TYPE, GNN_OUTPUT_TYPE]
SE3_SCORE_TYPE = Tuple[torch.Tensor, torch.Tensor]     # Angular score, Linear score