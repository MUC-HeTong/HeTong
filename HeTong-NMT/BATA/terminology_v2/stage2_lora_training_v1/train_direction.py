#!/usr/bin/env python3
import argparse, hashlib, json, math, os, platform, random, time
from pathlib import Path

ROOT=Path(os.environ.get('BATA_TERMINOLOGY_ROOT', Path(__file__).resolve().parents[1])).expanduser().resolve()
CKPT=Path(os.environ.get('BATA_BASE_MODEL', '')).expanduser().resolve(); SEED=20260811; IGNORE=-100
TARGETS=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']
CFG={'r':8,'lora_alpha':8,'lora_dropout':0.1,'bias':'none','task_type':'CAUSAL_LM','learning_rate':1e-5,'num_train_epochs':1,'per_device_train_batch_size':1,'gradient_accumulation_steps':8,'effective_batch_size':8,'warmup_ratio':.01,'scheduler':'linear','optimizer':'AdamW','weight_decay':.01,'max_seq_length':512,'precision':'fp16','gradient_checkpointing':True,'seed':SEED}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def rows(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x]
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def main():
 a=argparse.ArgumentParser();a.add_argument('--direction',choices=('bo2zh','zh2bo'),required=True);z=a.parse_args()
 if not os.environ.get('BATA_BASE_MODEL'): raise RuntimeError('Set BATA_BASE_MODEL to the local TC-Backbone checkpoint directory')
 if not CKPT.is_dir(): raise FileNotFoundError(f'TC-Backbone checkpoint not found: {CKPT}')
 out=ROOT/'stage2_lora_training_v1'/z.direction;out.mkdir(parents=True,exist_ok=True);data=ROOT/'stage1_behavior_dataset_v5'/z.direction/'train.jsonl';items=rows(data)
 import torch,transformers,peft,accelerate
 from transformers import AutoTokenizer,AutoModelForCausalLM,get_linear_schedule_with_warmup
 from peft import LoraConfig,TaskType,get_peft_model
 assert torch.cuda.is_available();random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 tok=AutoTokenizer.from_pretrained(CKPT,trust_remote_code=True)
 encoded=[]
 for x in items:
  m=x['messages'];assert [q['role'] for q in m]==['system','user','assistant'];p=tok.apply_chat_template(m[:2],tokenize=True,add_generation_prompt=True);f=tok.apply_chat_template(m,tokenize=True,add_generation_prompt=False);p=p['input_ids'] if hasattr(p,'keys') else p;f=f['input_ids'] if hasattr(f,'keys') else f;assert f[:len(p)]==p and len(f)<=512,(x['record_id'],len(f));encoded.append((f,[IGNORE]*len(p)+f[len(p):]))
 weight_files=sorted(CKPT.glob('*.safetensors'));base_hash=hashlib.sha256(''.join(sha(x) for x in weight_files).encode()).hexdigest();ids=[x['record_id'] for x in items];dataset_hash=sha(data);write(out/'training_config.json',CFG|{'direction':z.direction,'base_checkpoint':str(CKPT),'base_checkpoint_weights_sha256':base_hash,'dataset':str(data),'dataset_sha256':dataset_hash,'adapter_loaded':False});write(out/'environment.json',{'python':platform.python_version(),'python_executable':os.sys.executable,'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,'accelerate':accelerate.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'single_gpu_only':True})
 base=AutoModelForCausalLM.from_pretrained(CKPT,torch_dtype=torch.float16,trust_remote_code=True).to('cuda:0');base.config.use_cache=False;base.gradient_checkpointing_enable();getattr(base,'enable_input_require_grads',lambda:None)();model=get_peft_model(base,LoraConfig(r=8,lora_alpha=8,lora_dropout=.1,bias='none',task_type=TaskType.CAUSAL_LM,target_modules=TARGETS));trainable=[p for _,p in model.named_parameters() if p.requires_grad];assert all('lora_' in n for n,p in model.named_parameters() if p.requires_grad);opt=torch.optim.AdamW(trainable,lr=1e-5);order=list(range(len(items)));random.Random(SEED).shuffle(order);steps=math.ceil(len(order)/8);sched=get_linear_schedule_with_warmup(opt,int(steps*.01),steps);scaler=torch.amp.GradScaler('cuda',enabled=True);logs=[];losses=[];model.train();opt.zero_grad(set_to_none=True);start=time.time()
 for pos,idx in enumerate(order,1):
  ids0,labels=encoded[idx];ii=torch.tensor([ids0],device='cuda:0');ll=torch.tensor([labels],device='cuda:0')
  with torch.autocast('cuda',dtype=torch.float16):loss=model(input_ids=ii,attention_mask=torch.ones_like(ii),labels=ll,use_cache=False).loss
  v=float(loss.detach());assert math.isfinite(v),f'non-finite loss at {pos}';losses.append(v);scaler.scale(loss/8).backward()
  if pos%8==0 or pos==len(order):
   scaler.unscale_(opt);scaler.step(opt);scaler.update();sched.step();opt.zero_grad(set_to_none=True);logs.append({'global_step':len(logs)+1,'micro_steps_completed':pos,'learning_rate':sched.get_last_lr()[0],'training_loss':sum(losses[max(0,pos-8):pos])/min(8,pos),'elapsed_seconds':round(time.time()-start,2)})
 (out/'training_log.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in logs),encoding='utf-8');adapter=out/'adapter_epoch1';model.save_pretrained(adapter,safe_serialization=True);adapter_hash=sha(adapter/'adapter_model.safetensors');state={'direction':z.direction,'epoch':1,'global_step':len(logs),'expected_steps':steps,'train_samples':len(items),'record_id_sha256':hashlib.sha256('\n'.join(ids).encode()).hexdigest(),'dataset_sha256':dataset_hash,'base_checkpoint_weights_sha256':base_hash,'adapter_sha256':adapter_hash,'training_complete':True,'evaluation_performed':False};write(out/'trainer_state.json',state);write(out/'training_report.json',state|{'first_loss':losses[0],'last_loss':logs[-1]['training_loss'],'mean_loss':sum(losses)/len(losses),'adapter_path':str(adapter),'config':CFG});(out/'training_report.md').write_text(f'# Stage 2.1 {z.direction} Training\n\nCompleted one epoch, {len(logs)} optimizer steps, {len(items)} samples. No evaluation was performed.\n',encoding='utf-8')
if __name__=='__main__':main()
