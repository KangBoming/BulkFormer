"""Portable inference for the two released checkpoints."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import trange


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def verify_bundle(directory, name):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest['format_version'] != 2 or manifest['model_name'] != name:
        raise ValueError('Bundle model or format does not match the requested model')
    for filename, info in manifest['files'].items():
        if Path(filename).name != filename:
            raise ValueError('Manifest must contain flat filenames')
        file = directory/filename
        if file.stat().st_size != info['bytes'] or sha256(file) != info['sha256']:
            raise ValueError(f'Model resource failed integrity verification: {filename}')
    return manifest


def _load(directory, name, device):
    directory=Path(directory)
    manifest=verify_bundle(directory, name)
    payload=torch.load(directory/'model.pt', map_location='cpu', mmap=True, weights_only=True)
    config=json.loads((directory/'model_config.json').read_text())
    if payload['model_name'] != name or payload['config'] != config or payload['checkpoint_metadata'] != manifest['checkpoint']:
        raise ValueError('Checkpoint metadata and model resources do not agree')
    vocabulary=pd.read_csv(directory/'gene_vocab.csv')
    if len(vocabulary) != config['model']['num_genes'] or not vocabulary.ensembl_gene.is_unique:
        raise ValueError('Unexpected gene vocabulary')
    if name=='atlas':
        from .atlas import BulkFormerAtlas
        model=BulkFormerAtlas(**config['model'], gradient_checkpointing=False)
    else:
        from ._backbone import LatestConfig
        from ._graph import load_weighted_gene_graph
        from ._missingness import MissingnessCalibrationConfig
        from .latest import BulkFormerLatest, GlobalContextConfig
        graph=load_weighted_gene_graph(directory,len(vocabulary),validate_symmetry=False,validate_unique_edges=False)
        model=BulkFormerLatest(LatestConfig(**config['model']),graph,
            GlobalContextConfig(**config['global_expression_context']),
            graph_aggregation=config['graph_execution']['aggregation'],
            missingness_calibration=MissingnessCalibrationConfig(**config['missingness_calibration']['network']))
    model.load_state_dict(payload['model'],strict=True)
    model.eval().to(device)
    model.release_metadata=manifest['checkpoint']
    model.release_config=config
    return model,vocabulary


def load_atlas(model_dir='model/atlas', device='cuda'):
    model,vocabulary=_load(model_dir,'atlas',device)
    return model,vocabulary,pd.read_csv(Path(model_dir)/'normalization_vocab.csv')


def load_latest(model_dir='model/latest', device='cuda'):
    return _load(model_dir,'latest',device)


def _precision(device, precision):
    if precision not in {'bf16','fp32'}:
        raise ValueError('Use bf16 or fp32')
    if precision=='bf16':
        if device.type!='cuda' or not torch.cuda.is_bf16_supported():
            raise ValueError('BF16 requires a supported CUDA GPU; use precision="fp32" otherwise')
        return torch.autocast('cuda',dtype=torch.bfloat16)
    return nullcontext()


def _check_expression(expression, width, batch_size):
    values=np.asarray(expression,dtype=np.float32)
    if values.ndim!=2 or not len(values) or values.shape[1]!=width:
        raise ValueError(f'Expected a nonempty [samples, {width}] expression matrix')
    if not np.isfinite(values).all() or not ((values>=0)|(values==-10)).all():
        raise ValueError('Expression must be nonnegative, with -10 only for unavailable entries')
    if not isinstance(batch_size,int) or isinstance(batch_size,bool) or batch_size<=0:
        raise ValueError('batch_size must be a positive integer')
    return values


def _indices(indices, width, maximum=None):
    selected=np.asarray(indices)
    if selected.ndim!=1 or selected.dtype.kind not in 'iu' or not len(selected):
        raise ValueError('Provide a nonempty integer gene index vector')
    if (selected<0).any() or (selected>=width).any() or len(np.unique(selected))!=len(selected):
        raise ValueError('Gene indices must be distinct and within the model vocabulary')
    if maximum is not None and len(selected)>maximum:
        raise ValueError(f'This release supports at most {maximum} input genes')
    return selected.astype(np.int64, copy=False)


@torch.inference_mode()
def extract_atlas(model, expression, gene_indices, *, batch_size=2, precision='bf16', progress=True):
    """Extract native CLS vectors, before the contrastive projection; no masking or noise."""
    values=_check_expression(expression,model.num_genes,batch_size)
    selected=_indices(gene_indices,model.num_genes,4096)
    if (values[:,selected]<0).any():
        raise ValueError('Atlas input-panel genes must be observed in every sample')
    model.eval()
    device=next(model.parameters()).device
    ids=torch.as_tensor(np.r_[model.num_genes,selected],dtype=torch.long,device=device)
    results=[]
    for start in trange(0,len(values),batch_size,disable=not progress,desc='Atlas'):
        x=torch.as_tensor(values[start:start+batch_size,selected],device=device)
        x=torch.cat([x.new_zeros(len(x),1),x],dim=1)
        gene_ids=ids[None].expand(len(x),-1)
        with _precision(device,precision):
            hidden=model.encode(gene_ids,x,torch.ones_like(gene_ids,dtype=torch.bool))
        results.append(hidden[:,0].float().cpu().numpy())
    return np.concatenate(results)


@torch.inference_mode()
def extract_latest(model, expression, *, gene_indices=None, return_expression=False,
                   batch_size=1, precision='bf16', progress=True):
    """Encode the full transcriptome, then select contextual gene outputs.

    Sample pooling includes all observed genes, including measured zeros.
    Request gene_indices explicitly to avoid materializing all gene embeddings.
    Expression predictions are natural ln(TPM+1), and can be slightly negative.
    """
    values=_check_expression(expression,model.config.num_genes,batch_size)
    if (values==-10).all(axis=1).any():
        raise ValueError('Cannot encode a sample with no observed genes')
    selected=None if gene_indices is None else _indices(gene_indices,model.config.num_genes)
    model.eval()
    device=next(model.parameters()).device
    result={'sample_embeddings':[]}
    if selected is not None: result['gene_embeddings']=[]
    if return_expression: result['expression_prediction']=[]
    for start in trange(0,len(values),batch_size,disable=not progress,desc='Latest'):
        x=torch.as_tensor(values[start:start+batch_size],device=device)
        with _precision(device,precision):
            out=model(x,return_gene_embeddings=True)
        genes=out['gene_embeddings'].float()
        valid=x.ne(-10)
        pooled=(genes*valid.unsqueeze(-1)).sum(1)/valid.sum(1,keepdim=True)
        result['sample_embeddings'].append(pooled.cpu().numpy())
        if selected is not None: result['gene_embeddings'].append(genes[:,selected].cpu().numpy())
        if return_expression: result['expression_prediction'].append(out['expression_prediction'].float().cpu().numpy())
    return {key:np.concatenate(blocks) for key,blocks in result.items()}
