import hashlib,json
import os
from pathlib import Path
R=Path(os.environ.get('BATA_TERMINOLOGY_ROOT', Path(__file__).resolve().parents[1])).expanduser().resolve();O=R/'stage1_behavior_dataset_v5';SEED='20260905'
SRC={'bo2zh':R/'stage1_step15_rebuild_bo2zh_v1','zh2bo':R/'stage1_reverse_candidate_mining_zh2bo_v1'}
SYS={'bo2zh':'你是专业的藏汉翻译助手。请准确、完整地将藏文翻译成中文，只输出中文译文，不要解释。','zh2bo':'你是专业的汉藏翻译助手。请准确、完整地将中文翻译成藏文，只输出藏文译文，不要解释。'};INST={'bo2zh':'请将以下藏文翻译成中文，只输出译文。','zh2bo':'请将以下中文翻译成藏文，只输出译文。'}
def r(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x]
def w(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',encoding='utf-8') as f:
  for z in x:f.write(json.dumps(z,ensure_ascii=False,sort_keys=True)+'\n')
def key(x):return hashlib.sha256((SEED+x['record_id']).encode()).hexdigest()
def row(x,t):
 u=f"{INST[x['direction']]}\n请优先正确使用术语：{x['source_term']} -> {x['target_term']}\n原文：{x['source']}"
 return {'record_id':x['record_id'],'direction':x['direction'],'sample_type':t,'source':x['source'],'reference':x['reference'],'source_term':x['source_term'],'target_term':x['target_term'],'canonical_target_term':x['canonical_target_term'],'terminology_id':x['terminology_id'],'messages':[{'role':'system','content':SYS[x['direction']]},{'role':'user','content':u},{'role':'assistant','content':x['reference']}],**({'prompt_rescuable':True} if t=='prompt_rescuable' else {})}
def valid(x):return x.get('conflict_status')=='NO_CONFLICT' and x.get('terminology_id') and x.get('canonical_target_term')
def main():
 cats={}; reports={};allrows=[]
 for d,p in SRC.items():
  hard=r(p/'hard_pool.jsonl');prompt=r(p/'prompt_rescuable.jsonl');pres=r(p/'preservation.jsonl'); bad=sum(not valid(x) for x in hard+prompt+pres)
  rescue=[row(x,'rescue') for x in hard if valid(x)];pr=[row(x,'prompt_rescuable') for x in sorted(prompt,key=key) if valid(x)][:300];ps=[row(x,'preservation') for x in sorted(pres,key=key) if valid(x)][:200]
  for group in (rescue,pr,ps):
   assert len({x['record_id'] for x in group})==len(group)
  cats[d]=(rescue,pr,ps);allrows+=rescue+pr+ps
  reports[d]={'direction':d,'rescue_count':len(rescue),'prompt_rescuable_count':len(pr),'preservation_count':len(ps),'total_count':len(rescue)+len(pr)+len(ps),'conflict_removed':sum(x.get('conflict_status')!='NO_CONFLICT' for x in hard+prompt+pres),'unresolved_removed':bad-sum(x.get('conflict_status')!='NO_CONFLICT' for x in hard+prompt+pres)}
 # One terminology id has one bucket across directions and categories.
 bucket={x['terminology_id']:('dev' if int(hashlib.sha256((SEED+x['terminology_id']).encode()).hexdigest()[:8],16)%5==0 else 'train') for x in allrows}
 for d,(rescue,pr,ps) in cats.items():
  q=O/d;w(q/'rescue.jsonl',rescue);w(q/'prompt_rescuable.jsonl',pr);w(q/'preservation.jsonl',ps);items=rescue+pr+ps;train=[x for x in items if bucket[x['terminology_id']]=='train'];dev=[x for x in items if bucket[x['terminology_id']]=='dev'];assert not({x['record_id'] for x in train}&{x['record_id'] for x in dev});w(q/'train.jsonl',train);w(q/'dev.jsonl',dev);reports[d]|={'train_count':len(train),'dev_count':len(dev),'split_method':'global deterministic concept-level SHA256 bucket by terminology_id'}
 report={'stage':'Stage 1.4 Behavior Dataset Construction v5','directions':reports,'training_started':False};(O/'reports').mkdir(parents=True,exist_ok=True);(O/'reports'/'dataset_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');(O/'reports'/'split_report.json').write_text(json.dumps({'concept_key':'terminology_id','seed':SEED,'cross_direction_concept_leakage':False,'global_train_concepts':len({x['terminology_id'] for x in allrows if bucket[x['terminology_id']]=='train'}),'global_dev_concepts':len({x['terminology_id'] for x in allrows if bucket[x['terminology_id']]=='dev'})},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
