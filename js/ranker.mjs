/** Dependency-free inference. Caller provides legal standard-chess moves.
 * Square indexing: a1=0,...,h8=63. No model-generated move is ever executed.
 */
const values=[0,1,3,3.25,5,9,12], letters=' pnbrqk';
export const square=s=>typeof s==='number'?s:(s.charCodeAt(0)-97)+8*(Number(s[1])-1);
export const squareName=s=>String.fromCharCode(97+s%8)+(1+Math.floor(s/8));
const dist=(a,b)=>Math.max(Math.abs(a%8-b%8),Math.abs((a>>3)-(b>>3)));

function parse(fen){
  const [position,turn,rights,ep,half,full]=fen.split(' '),board=Array(64).fill(0);
  position.split('/').forEach((row,i)=>{let f=0;for(const c of row){if(/[1-8]/.test(c))f+=Number(c);else board[(7-i)*8+f++]=(c===c.toUpperCase()?1:-1)*letters.indexOf(c.toLowerCase());}});
  return {board,side:turn==='w'?1:-1,rights,ep:ep==='-'?null:square(ep),half:Number(half),ply:2*(Number(full)-1)+(turn==='b'?1:0)};
}

function attacks(b,s){
  const p=b[s];if(!p)return [];
  const type=Math.abs(p),color=Math.sign(p),f=s%8,r=s>>3,out=[];
  const add=(x,y)=>{if(x>=0&&x<8&&y>=0&&y<8)out.push(x+8*y);};
  if(type===1){add(f-1,r+color);add(f+1,r+color);}
  else if(type===2){for(const [dx,dy] of [[1,2],[2,1],[2,-1],[1,-2],[-1,-2],[-2,-1],[-2,1],[-1,2]])add(f+dx,r+dy);}
  else if(type===6){for(let dx=-1;dx<=1;dx++)for(let dy=-1;dy<=1;dy++)if(dx||dy)add(f+dx,r+dy);}
  else {
    const dirs=type===3?[[1,1],[1,-1],[-1,1],[-1,-1]]:type===4?[[1,0],[-1,0],[0,1],[0,-1]]:[[1,1],[1,-1],[-1,1],[-1,-1],[1,0],[-1,0],[0,1],[0,-1]];
    for(const [dx,dy] of dirs){let x=f+dx,y=r+dy;while(x>=0&&x<8&&y>=0&&y<8){const t=x+8*y;out.push(t);if(b[t])break;x+=dx;y+=dy;}}
  }return out;
}
function attackers(b,color,s){const out=[];for(let i=0;i<64;i++)if(Math.sign(b[i])===color&&attacks(b,i).includes(s))out.push(i);return out;}
function attackInfo(b,c,s){const a=attackers(b,c,s);return [Math.min(a.length,8)/8,a.length?Math.min(...a.map(t=>values[Math.abs(b[t])]))/12:0];}
function activity(b,s,c){const a=attacks(b,s),enemy=a.filter(t=>Math.sign(b[t])===-c),friend=a.filter(t=>Math.sign(b[t])===c);return [a.length/32,enemy.length/8,enemy.length?Math.max(...enemy.map(t=>values[Math.abs(b[t])]))/12:0,friend.length/8];}
function pinned(b,s,c,k){
  let dx=s%8-k%8,dy=(s>>3)-(k>>3);if(!(dx===0||dy===0||Math.abs(dx)===Math.abs(dy)))return 0;
  dx=Math.sign(dx);dy=Math.sign(dy);if(!dx&&!dy)return 0;
  let x=k%8+dx,y=(k>>3)+dy,passed=false;
  while(x>=0&&x<8&&y>=0&&y<8){const t=x+8*y,p=b[t];if(t===s)passed=true;else if(p){return passed&&Math.sign(p)===-c&&(Math.abs(p)===5||Math.abs(p)===(dx&&dy?3:4))?1:0;}x+=dx;y+=dy;}return 0;
}
function moveInfo(position,m){
  const f=square(m.from),t=square(m.to),type=Math.abs(position.board[f]),flags=m.flags||'';
  return {f,t,promotion:m.promotion?letters.indexOf(m.promotion):0,
    castle:flags.includes('k')||flags.includes('q')||(type===6&&Math.abs(f-t)===2),
    ep:flags.includes('e')||(type===1&&t===position.ep&&!position.board[t]&&f%8!==t%8)};
}

export function features(fen,moves,history=[],representation='history'){
  const pos=parse(fen),{board:b,side:c,rights}=pos,flip=c===1?0:56;
  const ownking=b.indexOf(6*c),enemyking=b.indexOf(-6*c);
  if(ownking<0||enemyking<0)throw new Error('Both kings required');
  const counts=[];for(const color of [c,-c])for(let p=1;p<=6;p++)counts.push(b.filter(v=>v===p*color).length/8);
  const castling=[...(c===1?'KQkq':'kqKQ')].map(s=>Number(rights.includes(s)));
  if(representation==='planes'){
    const base=Array(903).fill(0);
    b.forEach((p,s)=>{if(p)base[((Math.sign(p)===c?0:6)+Math.abs(p)-1)*64+(s^flip)]=1;});
    base.splice(768,7,...castling,pos.ep===null?-1:(pos.ep^flip)/63,Math.min(pos.ply,200)/200,Number(attackers(b,-c,ownking).length>0));
    return moves.map(m=>{const x=base.slice();x[775+(square(m.from)^flip)]=1;x[839+(square(m.to)^flip)]=1;return x;});
  }
  const globals=[Math.min(pos.ply,200)/200,Math.min(pos.half,100)/100,Number(attackers(b,-c,ownking).length>0),...castling];
  return moves.map(m=>{
    const {f,t,promotion,castle,ep}=moveInfo(pos,m),cf=f^flip,ct=t^flip,ff=cf%8,fr=cf>>3,tf=ct%8,tr=ct>>3,type=Math.abs(b[f]);
    const row=[];for(let p=1;p<=6;p++)row.push(Number(p===type));
    for(const v of [ff,fr,tf,tr])for(let j=0;j<8;j++)row.push(Number(v===j));
    row.push((tf-ff)/7,(tr-fr)/7,Math.max(Math.abs(tf-ff),Math.abs(tr-fr))/7,(Math.abs(tf-ff)+Math.abs(tr-fr))/14,(Math.abs(ff-3.5)+Math.abs(fr-3.5))/7,(Math.abs(tf-3.5)+Math.abs(tr-3.5))/7,...counts,...globals);
    const captured=Math.abs(b[t])||(ep?1:0);
    row.push(Number(captured>0),values[captured]/12,Number(promotion>0),Number(castle),Number(ep),values[type]/12,...attackInfo(b,-c,f),...attackInfo(b,c,f));
    const before=activity(b,f,c),pin=pinned(b,f,c,ownking);
    let same=0,adjacent=0;for(let s=0;s<64;s++)if(b[s]===c){if(s%8===tf)same++;if(Math.abs(s%8-tf)===1)adjacent++;}
    const after=b.slice();after[f]=0;after[t]=(promotion||type)*c;if(ep)after[t-8*c]=0;
    if(castle){const rookFrom=t>f?f+3:f-4,rookTo=t>f?f+1:f-1;after[rookFrom]=0;after[rookTo]=4*c;}
    row.push(...attackInfo(after,-c,t),...attackInfo(after,c,t),...before,...activity(after,t,c),Number(attackers(after,c,enemyking).length>0),dist(f,ownking)/7,dist(t,ownking)/7,dist(f,enemyking)/7,dist(t,enemyking)/7,pin,same/8,adjacent/16);
    if(representation==='history')for(let i=1;i<=2;i++){
      const h=history[history.length-i];if(!h){row.push(0,0,0,0,0,0,0);continue;}
      const hf=square(h.from),ht=square(h.to),pf=hf^flip,pt=ht^flip;
      row.push(1,(pf%8-ff)/7,((pf>>3)-fr)/7,(pt%8-tf)/7,((pt>>3)-tr)/7,Number(ht===f),Number(ht===t));
    }
    return row;
  });
}

export class OriginRanker {
  constructor(artifact){
    if(artifact.feature_version!==1)throw new Error('Unsupported feature version');
    this.representation=artifact.representation;
    const source=artifact.weights||artifact.weights_int8.map((w,i)=>w.map(row=>row.map(v=>v*artifact.weight_scales[i])));
    this.layers=source.map((w,i)=>({input:w.length,output:w[0].length,w:Float32Array.from(w.flat()),b:Float32Array.from(artifact.biases[i])}));
  }
  score(row){
    let x=row;
    for(let li=0;li<this.layers.length;li++){
      const l=this.layers[li];if(x.length!==l.input)throw new Error('Feature dimensions differ');
      const y=Float32Array.from(l.b);
      for(let i=0;i<l.input;i++)for(let j=0;j<l.output;j++)y[j]+=x[i]*l.w[i*l.output+j];
      if(li<this.layers.length-1)for(let j=0;j<l.output;j++)y[j]=Math.max(0,y[j]);x=y;
    }return x[0];
  }
  rank(fen,legalMoves,destination,history=[]){
    const target=square(destination),legal=legalMoves.filter(m=>square(m.to)===target),unique=new Map();
    for(const m of legal)if(!unique.has(square(m.from))||m.promotion==='q')unique.set(square(m.from),m);
    const moves=[...unique.entries()].sort((a,b)=>a[0]-b[0]).map(e=>e[1]);if(!moves.length)return [];
    let probs=[1];
    if(moves.length>1){const s=features(fen,moves,history,this.representation).map(x=>this.score(x)),max=Math.max(...s);probs=s.map(v=>Math.exp(v-max));const total=probs.reduce((a,b)=>a+b,0);probs=probs.map(v=>v/total);}
    return moves.map((m,i)=>({from:squareName(square(m.from)),probability:probs[i],moves:legal.filter(v=>square(v.from)===square(m.from)).map(v=>squareName(square(v.from))+squareName(square(v.to))+(v.promotion||''))})).sort((a,b)=>b.probability-a.probability);
  }
  probabilities64(fen,legalMoves,destination,history=[]){const p=new Float32Array(64);for(const r of this.rank(fen,legalMoves,destination,history))p[square(r.from)]=r.probability;return p;}
  clickSession(fen,legalMoves,destination,history=[]){const ranking=this.rank(fen,legalMoves,destination,history);let i=0;return {ranking,click:()=>ranking.length?ranking[i++%ranking.length]:null};}
}
