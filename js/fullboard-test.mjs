import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {FullBoardRanker,historyTokens,rankPolicy} from './fullboard-ranker.mjs';

const fixtures=JSON.parse(readFileSync(new URL('../tests/fullboard-fixtures.json',import.meta.url)));
const policy=Float32Array.from({length:4352},(_,i)=>(i%97-48)/10);
let maxError=0;
for (const f of fixtures) {
  const tokens=historyTokens(f.fens);
  assert.equal(createHash('sha256').update(new Uint8Array(tokens.buffer)).digest('hex'),f.tokens_sha256);
  const rows=rankPolicy(policy,f.fens.at(-1),f.legal,f.destination);
  assert.deepEqual(rows.map(r=>r.from),f.expected.map(r=>r.from));
  rows.forEach((r,i)=>{maxError=Math.max(maxError,Math.abs(r.probability-f.expected[i].probability));});
}
assert.ok(maxError<1e-10);
let calls=0;
const ort={Tensor:class{constructor(type,data,dims){Object.assign(this,{type,data,dims});}}};
const session={async run(){calls++;return {policy_logits:{data:policy}};}};
const model=new FullBoardRanker(ort,session),f=fixtures[0];
await model.rank(f.fens,f.legal,'e4');assert.equal(calls,0);
await Promise.all([model.prepare(f.fens),model.prepare(f.fens)]);assert.equal(calls,1);
await model.rank(f.fens,f.legal,f.destination);assert.equal(calls,1);
const click=await model.clickSession(f.fens,f.legal,f.destination);
const first=click.click();const second=click.click();assert.notEqual(first.from,second.from);assert.equal(click.click().from,first.from);
assert.equal(calls,1);
model.clearCache();await model.prepare(f.fens);assert.equal(calls,2);
model.rating=2500;await model.prepare(f.fens);assert.equal(calls,3);
assert.throws(()=>new FullBoardRanker(ort,session,{temperature:0}));
console.log(JSON.stringify({fullboard_fixture_count:fixtures.length,token_parity:'exact',max_probability_error:maxError,cache:'passed'}));
