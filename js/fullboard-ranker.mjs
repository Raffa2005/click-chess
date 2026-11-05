// EXPERIMENTAL: INT8 WASM failed native numerical parity; see research/fullboard/README.md.
// Optional full-board adapter. Inject ONNX Runtime Web; external Maia3 terms apply
// to the separately downloaded/exported model. Standard chess, legal moves supplied.
const square = s => {
  if (!/^[a-h][1-8]$/.test(s)) throw new Error('Invalid square');
  return s.charCodeAt(0) - 97 + 8 * (Number(s[1]) - 1);
};

function packFen(fen) {
  const [position, turn] = fen.split(' ');
  if (!['w','b'].includes(turn)) throw new Error('Invalid side to move');
  const ranks = position.split('/');
  if (ranks.length !== 8) throw new Error('Invalid board');
  const packed = new Uint8Array(64), flip = turn === 'b' ? 56 : 0;
  for (let row = 0; row < 8; row++) {
    let file = 0;
    for (const c of ranks[row]) {
      if (/^[1-8]$/.test(c)) { file += Number(c); continue; }
      const type = 'pnbrqk'.indexOf(c.toLowerCase());
      if (type < 0 || file >= 8) throw new Error('Invalid piece placement');
      const own = (c === c.toUpperCase()) === (turn === 'w');
      packed[((7-row)*8 + file++) ^ flip] = type + 1 + (own ? 0 : 6);
    }
    if (file !== 8) throw new Error('Invalid rank');
  }
  return packed;
}

export function historyTokens(fens) {
  if (!fens.length) throw new Error('Current position required');
  const history = fens.slice(-8).map(packFen);
  while (history.length < 8) history.unshift(history[0]);
  const result = new Float32Array(64*96);
  for (let t = 0; t < 8; t++) for (let s = 0; s < 64; s++) {
    const piece = history[t][s];
    if (piece) result[s*96 + t*12 + piece-1] = 1;
  }
  return result;
}

function logsumexp(xs) {
  const top = Math.max(...xs);
  return top + Math.log(xs.reduce((s,x) => s + Math.exp(x-top), 0));
}

export function rankPolicy(policy, fen, legalMoves, destination, temperature=1) {
  if (!(temperature > 0) || !Number.isFinite(temperature)) throw new Error('Invalid temperature');
  const dest = square(destination), flip = fen.split(' ')[1] === 'b' ? 56 : 0;
  const groups = new Map();
  for (const m of legalMoves) {
    if (m.to !== destination) continue;
    const f = square(m.from) ^ flip, t = dest ^ flip;
    let index = f*64+t;
    if (m.promotion) {
      const p = 'qrbn'.indexOf(m.promotion);
      if (p < 0) throw new Error('Invalid promotion');
      index = 4096 + ((f%8)*8+t%8)*4+p;
    }
    if (!groups.has(m.from)) groups.set(m.from, {from:m.from, logits:[], moves:[]});
    const g = groups.get(m.from);
    const uci = m.from+m.to+(m.promotion || '');
    if (g.moves.includes(uci)) continue;
    g.logits.push(policy[index]);g.moves.push(uci);
  }
  const rows = [...groups.values()];
  if (!rows.length) return [];
  for (const r of rows) r.score = logsumexp(r.logits)/temperature;
  const normalizer = logsumexp(rows.map(r=>r.score));
  return rows.sort((a,b)=>b.score-a.score || square(a.from)-square(b.from))
    .map(r=>({from:r.from, probability:Math.exp(r.score-normalizer), moves:r.moves}));
}

export class FullBoardRanker {
  constructor(ort, session, {temperature=1, rating=2400}={}) {
    if (!(temperature > 0) || !Number.isFinite(temperature) || !Number.isFinite(rating)) throw new Error('Invalid configuration');
    this.ort=ort;this.session=session;this.temperature=temperature;this.rating=rating;this.cache=null;
  }
  static async create(ort, modelBytes, options={}) {
    const session=await ort.InferenceSession.create(modelBytes, {executionProviders:['wasm']});
    return new FullBoardRanker(ort,session,options);
  }
  clearCache() { this.cache=null; }
  prepare(fens) {
    const history=fens.slice(-8), rating=this.rating;
    const key=JSON.stringify([history,rating]);
    if (this.cache?.key===key) return this.cache.promise;
    const tokens=historyTokens(history), ort=this.ort;
    const promise=this.session.run({tokens:new ort.Tensor('float32',tokens,[1,64,96]),
      self_elo:new ort.Tensor('float32',new Float32Array([rating]),[1]),
      opponent_elo:new ort.Tensor('float32',new Float32Array([rating]),[1])})
      .then(out=>Float32Array.from(out.policy_logits.data));
    this.cache={key,promise};
    promise.catch(()=>{if(this.cache?.promise===promise)this.cache=null;});
    return promise;
  }
  async rank(fens, legalMoves, destination) {
    const moves=legalMoves.filter(m=>m.to===destination), origins=new Set(moves.map(m=>m.from));
    if (!origins.size) return [];
    if (origins.size===1) return [{from:moves[0].from,probability:1,moves:[...new Set(moves.map(m=>m.from+m.to+(m.promotion||'')))]}];
    return rankPolicy(await this.prepare(fens),fens.at(-1),moves,destination,this.temperature);
  }
  async clickSession(fens,legalMoves,destination) {
    const ranking=await this.rank(fens,legalMoves,destination);let index=0;
    return {click:()=>ranking.length ? ranking[index++%ranking.length] : null};
  }
}
