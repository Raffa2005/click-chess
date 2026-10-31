import fs from 'node:fs';
import os from 'node:os';
import {performance} from 'node:perf_hooks';
import {Chess} from 'chess.js';
import {features,OriginRanker} from './ranker.mjs';
const fixtures=JSON.parse(fs.readFileSync('tests/fixtures.json','utf8'));
const ranker=new OriginRanker(JSON.parse(fs.readFileSync('models/origin-ranker.int8.json','utf8')));
const inputs=fixtures.map(f=>({f,engine:new Chess(f.fen),history:f.history.map(m=>({from:m.slice(0,2),to:m.slice(2,4)}))}));
for(const v of inputs){v.legal=v.engine.moves({verbose:true});v.session=ranker.clickSession(v.f.fen,v.legal,v.f.destination,v.history);}
function bench(fn,iterations=3000){
  for(let i=0;i<300;i++)fn(inputs[i%inputs.length]);
  const times=[];for(let i=0;i<iterations;i++){const t=performance.now();fn(inputs[i%inputs.length]);times.push((performance.now()-t)*1000);}
  times.sort((a,b)=>a-b);return {iterations,median_us:times[Math.floor(times.length*.5)],p95_us:times[Math.floor(times.length*.95)],p99_us:times[Math.floor(times.length*.99)]};
}
const report={runtime:process.version,platform:os.platform(),architecture:os.arch(),cpu:os.cpus()[0]?.model||'unavailable',fixture_count:inputs.length,
  rank_with_features_legal_moves_supplied:bench(v=>ranker.rank(v.f.fen,v.legal,v.f.destination,v.history)),
  full_chessjs_legal_generation_and_ranking:bench(v=>ranker.rank(v.engine.fen(),v.engine.moves({verbose:true}),v.f.destination,v.history)),
  cached_second_click:bench(v=>v.session.click(),10000),
  note:'Warm Node.js CPU benchmark; excludes UI rendering, cold model load, network, and history reconstruction. No claim about mobile/browser latency.'};
fs.writeFileSync('results/latency-node.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
