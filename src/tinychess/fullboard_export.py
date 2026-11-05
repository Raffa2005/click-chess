"""Export a user-downloaded Maia3 checkpoint; external upstream license applies."""
import argparse,json,pathlib,time
import numpy as np
import torch
from .fullboard import FullBoardRanker,OnnxFullBoardRanker,packed_history,token_tensor


class PolicyOnly(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,tokens,self_elo,opponent_elo):return self.model(tokens,self_elo,opponent_elo)[0]


class ExportRMSNorm(torch.nn.Module):
    """Portable decomposition of PyTorch RMSNorm for ONNX opset 18."""
    def __init__(self,norm):
        super().__init__();self.weight=norm.weight;self.eps=norm.eps
        self.axes=tuple(range(-len(norm.normalized_shape),0))
    def forward(self,x):
        eps=self.eps if self.eps is not None else torch.finfo(x.dtype).eps
        return x*torch.rsqrt(x.square().mean(self.axes,keepdim=True)+eps)*self.weight


def decompose_rmsnorm(module):
    for name,child in list(module.named_children()):
        if isinstance(child,torch.nn.RMSNorm):setattr(module,name,ExportRMSNorm(child))
        else:decompose_rmsnorm(child)


def main():
    import chess
    p=argparse.ArgumentParser();p.add_argument('--size',choices=['5m','23m'],default='5m');a=p.parse_args()
    torch.set_num_threads(1)
    root=pathlib.Path('research/fullboard/checkpoints');base=root/f'maia3-{a.size}'
    ranker=FullBoardRanker(str(base)+'.pt',a.size)
    model=PolicyOnly(ranker.model).eval();packed=packed_history(chess.Board())[None]
    x=token_tensor(packed);elo=torch.full((1,),2400.)
    with torch.inference_mode():
        before=model(x,elo,elo)
        decompose_rmsnorm(model)
        torch.testing.assert_close(model(x,elo,elo),before,atol=1e-4,rtol=1e-4)
        torch.onnx.export(model,(x,elo,elo),str(base)+'.onnx',opset_version=18,dynamo=False,
                          input_names=['tokens','self_elo','opponent_elo'],output_names=['policy_logits'],
                          dynamic_axes={n:{0:'batch'} for n in ['tokens','self_elo','opponent_elo','policy_logits']})
    from onnxruntime.quantization import quantize_dynamic,QuantType
    quantize_dynamic(str(base)+'.onnx',str(base)+'.int8.onnx',weight_type=QuantType.QInt8,per_channel=True,
                     op_types_to_quantize=['MatMul','Gemm'],extra_options={'MatMulConstBOnly':True})
    results={}
    for suffix in ['.onnx','.int8.onnx']:
        m=OnnxFullBoardRanker(str(base)+suffix);times=[]
        for i in range(110):
            start=time.perf_counter();out=m.policy_logits(packed);times.append((time.perf_counter()-start)*1000)
        maxerr=float(np.max(np.abs(out-ranker.policy_logits(packed))))
        # Exercise dynamic batch shapes, including a final partial batch.
        for n in [2,7,32]:
            z=m.policy_logits(np.repeat(packed,n,axis=0))
            if not np.allclose(z,np.repeat(out,n,axis=0),atol=.01,rtol=.005):raise AssertionError('Batch mismatch')
        results[suffix]={'bytes':pathlib.Path(str(base)+suffix).stat().st_size,'ms_p50_p95':np.quantile(times[10:],[.5,.95]).tolist(),'initial_position_max_logit_error':maxerr}
    pathlib.Path(f'research/fullboard/export-{a.size}.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
