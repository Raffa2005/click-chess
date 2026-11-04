import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {Chess} from 'chess.js';
import {OriginRanker,features,square} from './ranker.mjs';
const root='research/manual-weights/', frozen=JSON.parse(fs.readFileSync(root+'frozen.json'));
const fixtures=JSON.parse(fs.readFileSync('tests/fixtures.json'));
const reference=JSON.parse(fs.readFileSync(root+'parity.json'));
let maxError=0;
for(const [name,artifact] of Object.entries(frozen.artifacts)){
  const bytes=fs.readFileSync(artifact.file);
  assert.equal(createHash('sha256').update(bytes).digest('hex'),artifact.sha256);
  const model=new OriginRanker(JSON.parse(bytes));
  for(const [i,f] of fixtures.entries()){
    const chess=new Chess(f.fen),legal=chess.moves({verbose:true}),unique=new Map();
    for(const m of legal.filter(m=>m.to===f.destination))if(!unique.has(m.from)||m.promotion==='q')unique.set(m.from,m);
    const moves=[...unique.values()].sort((a,b)=>square(a.from)-square(b.from));
    const history=f.history.map(m=>({from:m.slice(0,2),to:m.slice(2,4)}));
    const x=features(f.fen,moves,history,'history',2),scores=model.scores(x);
    assert.equal(scores.length,reference[name][i].length);
    for(let k=0;k<scores.length;k++){const error=Math.abs(scores[k]-reference[name][i][k]);maxError=Math.max(maxError,error);assert.ok(error<1e-5);}
    const rows=model.rank(f.fen,legal,f.destination,history);
    const expected=reference[name][i].map((score,k)=>({score,from:moves[k].from})).sort((a,b)=>b.score-a.score);
    assert.deepEqual(rows.map(r=>r.from),expected.map(r=>r.from));
    assert.ok(rows.every(r=>r.moves.every(u=>legal.some(m=>m.from+m.to+(m.promotion||'')===u))));
  }
}
console.log(JSON.stringify({manual_models:2,fixtures:fixtures.length,max_score_error:maxError,status:'passed'}));
