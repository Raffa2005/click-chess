import fs from 'node:fs';
import os from 'node:os';
import {performance} from 'node:perf_hooks';
import {createHash} from 'node:crypto';
import {Chess} from 'chess.js';
import {OriginRanker} from './ranker.mjs';
const root='research/manual-weights/';
const frozen=JSON.parse(fs.readFileSync(root+'frozen.json'));
const sha=b=>createHash('sha256').update(b).digest('hex');
const paths={manual_relu:frozen.artifacts['manual-relu'].file,manual_linear:frozen.artifacts['manual-linear'].file,trained_int8:frozen.incumbent_file};
const models=Object.fromEntries(Object.entries(paths).map(([name,path])=>{
  const bytes=fs.readFileSync(path);
  const expected=name==='trained_int8'?frozen.incumbent_sha256:frozen.artifacts[name.replace('_','-')].sha256;
  if(sha(bytes)!==expected)throw Error('Frozen model changed');
  return [name,path.endsWith('.bin')?OriginRanker.fromBinary(bytes):new OriginRanker(JSON.parse(bytes))];
}));
const inputs=JSON.parse(fs.readFileSync('tests/fixtures.json')).map(f=>{
  const engine=new Chess(f.fen),legal=engine.moves({verbose:true}),history=f.history.map(m=>({from:m.slice(0,2),to:m.slice(2,4)}));
  return {f,engine,legal,history,sessions:Object.fromEntries(Object.entries(models).map(([k,m])=>[k,m.clickSession(f.fen,legal,f.destination,history)]))};
});
function bench(call,iterations=3000){
  const names=Object.keys(models),times=Object.fromEntries(names.map(n=>[n,[]]));
  for(let i=0;i<300;i++)for(const name of names)call(models[name],inputs[i%inputs.length],name);
  for(let i=0;i<iterations;i++)for(let k=0;k<names.length;k++){
    const name=names[(i+k)%names.length],v=inputs[i%inputs.length],start=performance.now();
    call(models[name],v,name);times[name].push((performance.now()-start)*1000);
  }
  return Object.fromEntries(names.map(name=>{const t=times[name].sort((a,b)=>a-b);return [name,{iterations,median_us:t[Math.floor(t.length*.5)],p95_us:t[Math.floor(t.length*.95)],p99_us:t[Math.floor(t.length*.99)]}];}));
}
const report={runtime:process.version,platform:os.platform(),architecture:os.arch(),cpu:os.cpus()[0]?.model||'unavailable',fixture_count:inputs.length,
  score_only_features_supplied:bench((m,v)=>m.scores(v.f.rich_features)),
  rank_with_features_legal_moves_supplied:bench((m,v)=>m.rank(v.f.fen,v.legal,v.f.destination,v.history)),
  full_chessjs_legal_generation_and_ranking:bench((m,v)=>m.rank(v.engine.fen(),v.engine.moves({verbose:true}),v.f.destination,v.history)),
  cached_second_click:bench((m,v,n)=>v.sessions[n].click(),10000),
  note:'Warm same-process interleaved Node CPU benchmark. Model order rotates each iteration. No Android/UI/cold-start timing claim. Both models use identical v2 features; structural feature computation is included except in score-only timings.'};
fs.writeFileSync(root+'latency.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
