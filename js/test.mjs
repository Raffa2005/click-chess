import assert from 'node:assert/strict';
import fs from 'node:fs';
import {Chess} from 'chess.js';
import {features,OriginRanker,square} from './ranker.mjs';

const fixtures=JSON.parse(fs.readFileSync('tests/fixtures.json','utf8'));
let maxError=0;
for(const f of fixtures){
  const engine=new Chess(f.fen),legal=engine.moves({verbose:true}).filter(m=>m.to===f.destination),unique=new Map();
  for(const m of legal)if(!unique.has(m.from)||m.promotion==='q')unique.set(m.from,m);
  const moves=[...unique.values()].sort((a,b)=>square(a.from)-square(b.from));
  assert.deepEqual(moves.map(m=>square(m.from)),f.origins,'legal-origin mismatch');
  const history=f.history.map(m=>({from:m.slice(0,2),to:m.slice(2,4)}));
  const x=features(f.fen,moves,history);
  for(let i=0;i<x.length;i++)for(let j=0;j<x[i].length;j++){
    const delta=Math.abs(x[i][j]-f.features[i][j]);maxError=Math.max(maxError,delta);
    assert.ok(delta<1e-6,`Feature mismatch ${i},${j}: ${x[i][j]} vs ${f.features[i][j]} at ${f.fen}`);
  }
  if(f.rich_features){
    const rich=features(f.fen,moves,history,'history',2);
    assert.equal(rich[0].length,147);
    for(let i=0;i<rich.length;i++)for(let j=0;j<rich[i].length;j++)assert.ok(Math.abs(rich[i][j]-f.rich_features[i][j])<1e-6,`Rich feature mismatch ${i},${j}`);
  }
}
for(const suffix of ['', '.int8']){
const modelPath=`models/origin-ranker${suffix}.json`;
if(fs.existsSync(modelPath)){
  const ranker=new OriginRanker(JSON.parse(fs.readFileSync(modelPath,'utf8')));
  const engine=new Chess(),legal=engine.moves({verbose:true});
  assert.equal(ranker.rank(engine.fen(),legal,'e5').length,0);
  assert.equal(ranker.rank(engine.fen(),legal,'e4')[0].from,'e2');
  const p=ranker.probabilities64(engine.fen(),legal,'f3');assert.ok(Math.abs([...p].reduce((a,b)=>a+b,0)-1)<1e-6);
  const session=ranker.clickSession(engine.fen(),legal,'f3');assert.notEqual(session.click().from,session.click().from);
  assert.equal(engine.fen(),new Chess().fen());
  if(fs.existsSync(`tests/predictions${suffix}.json`)){
    const expected=JSON.parse(fs.readFileSync(`tests/predictions${suffix}.json`,'utf8'));
    for(const [i,f] of fixtures.entries()){
      const e=new Chess(f.fen),history=f.history.map(m=>({from:m.slice(0,2),to:m.slice(2,4)}));
      const result=ranker.rank(f.fen,e.moves({verbose:true}),f.destination,history);
      assert.deepEqual(result.map(r=>r.from),expected[i].map(r=>r.from));
      for(let j=0;j<result.length;j++)assert.ok(Math.abs(result[j].probability-expected[i][j].probability)<1e-4);
    }
  }
}
}
console.log(JSON.stringify({fixtures:fixtures.length,maxFeatureError:maxError,status:'passed'}));
if(fs.existsSync('tests/round2-fixtures.json')){
  const experiments=JSON.parse(fs.readFileSync('tests/round2-fixtures.json','utf8'));
  for(const experiment of experiments){
    const ranker=OriginRanker.fromBinary(Buffer.from(experiment.binary_base64,'base64'));
    for(const sample of experiment.cases){
      const f=fixtures[sample.index],engine=new Chess(f.fen),unique=new Map();
      for(const m of engine.moves({verbose:true}).filter(m=>m.to===f.destination))if(!unique.has(m.from)||m.promotion==='q')unique.set(m.from,m);
      const moves=[...unique.values()].sort((a,b)=>square(a.from)-square(b.from));
      const history=f.history.map(m=>({from:m.slice(0,2),to:m.slice(2,4)}));
      const x=features(f.fen,moves,history,'history',ranker.featureVersion),scores=ranker.scores(x);
      for(let i=0;i<scores.length;i++)assert.ok(Math.abs(scores[i]-sample.scores[i])<1e-5,'Experimental score mismatch');
      if(sample.features)for(let i=0;i<x.length;i++)for(let j=0;j<x[i].length;j++)assert.ok(Math.abs(x[i][j]-sample.features[i][j])<1e-6,'Pooled feature mismatch');
      const reversed=ranker.scores(x.slice().reverse()).reverse();
      for(let i=0;i<scores.length;i++)assert.ok(Math.abs(scores[i]-reversed[i])<1e-6,'Candidate order changed prediction');
    }
  }
  console.log(JSON.stringify({experimental_architectures:experiments.length,binary_parity:'passed'}));
}
if(fs.existsSync('models/origin-ranker.bin')){
  const binary=OriginRanker.fromBinary(fs.readFileSync('models/origin-ranker.bin'));
  const reference=new OriginRanker(JSON.parse(fs.readFileSync('models/origin-ranker.int8.json','utf8')));
  for(const f of fixtures){
    const engine=new Chess(f.fen),legal=engine.moves({verbose:true}),history=f.history.map(m=>({from:m.slice(0,2),to:m.slice(2,4)}));
    assert.deepEqual(binary.rank(f.fen,legal,f.destination,history),reference.rank(f.fen,legal,f.destination,history));
  }
  console.log(JSON.stringify({binary_fixtures:fixtures.length,status:'passed'}));
}
