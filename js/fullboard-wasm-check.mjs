// Optional integration check: npm install --no-save onnxruntime-web@1.31.1
import {readFileSync,writeFileSync} from 'node:fs';
import assert from 'node:assert/strict';
import {FullBoardRanker} from './fullboard-ranker.mjs';
const ort=await import(process.env.CLICK_CHESS_ORT_MODULE || 'onnxruntime-web');
ort.env.wasm.numThreads=1;
const fixture=JSON.parse(readFileSync('research/fullboard/onnx-api-fixtures.json'));
const bytes=readFileSync('research/fullboard/checkpoints/maia3-5m.int8.onnx');
const ranker=await FullBoardRanker.create(ort,bytes,{temperature:1.014909056819889});
let error=0;const times=[],differences=[];
for (let i=0;i<fixture.length;i++) {
  const f=fixture[i];ranker.clearCache();const start=performance.now();
  const rows=await ranker.rank(f.fens,f.legal,f.destination);times.push(performance.now()-start);
  rows.forEach(r=>{error=Math.max(error,Math.abs(r.probability-f.expected.find(e=>e.from===r.from).probability));});
  if (rows.map(r=>r.from).join()!==f.expected.map(r=>r.from).join()) differences.push({case:i,actual:rows,expected:f.expected});
}
const result={runtime:ort.env.versions,host:'Node WASM; not an Android/WebView measurement',cases:fixture.length,max_probability_error:error,rank_ms:times,ranking_differences:differences,passed:error<.005};
writeFileSync('research/fullboard/wasm-check.json',JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify(result));
assert.ok(error<.005,`WASM probability error ${error}`);
