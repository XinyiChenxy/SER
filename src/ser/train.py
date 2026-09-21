"""Train/evaluate CLI with speaker-independent split, TensorBoard, and final test metrics."""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from ser.data import scan_dataset,make_splits,load_audio,EMOTIONS
from ser.model import SERModel

LABELS=list(EMOTIONS.values()); TO_ID={x:i for i,x in enumerate(LABELS)}
class AudioSet(Dataset):
    def __init__(self,df,sr=16000,duration=4.0): self.df=df.reset_index(drop=True); self.sr=sr; self.duration=duration
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]; return load_audio(r.path,self.sr,self.duration).squeeze(0),TO_ID[r.emotion]

def evaluate(model,loader,device,criterion):
    model.eval(); losses=[]; ys=[]; ps=[]
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(device),y.to(device); logits=model(x); losses.append(criterion(logits,y).item()); ps.extend(logits.argmax(1).cpu().tolist()); ys.extend(y.cpu().tolist())
    return float(np.mean(losses)),ys,ps

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default=None); ap.add_argument('--output',default='outputs'); ap.add_argument('--epochs',type=int,default=50); ap.add_argument('--batch-size',type=int,default=32); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--smoke',action='store_true'); args=ap.parse_args()
    root=args.data or __import__('os').environ.get('DATA_DIR','/data/crema-d'); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    df,bad=scan_dataset(root); bad.to_csv(out/'invalid_files.csv',index=False)
    splits=make_splits(df,args.seed); splits.to_csv(out/'split_manifest.csv',index=False)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(f'{len(df)} valid, {len(bad)} invalid; device={device}')
    if args.smoke: args.epochs=1; args.batch_size=min(args.batch_size,4)
    loaders={s:DataLoader(AudioSet(splits[splits.split==s]),batch_size=args.batch_size,shuffle=(s=='train'),num_workers=0) for s in ['train','validation','test']}
    model=SERModel().to(device); criterion=nn.CrossEntropyLoss(); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',patience=3)
    writer=SummaryWriter(out/'runs'); best=-1.; stale=0; patience=8; start=time.time()
    for epoch in range(args.epochs):
        t=time.time(); model.train(); losses=[]; ys=[]; ps=[]
        for x,y in loaders['train']:
            x,y=x.to(device),y.to(device); optimizer.zero_grad(); logits=model(x); loss=criterion(logits,y); loss.backward(); optimizer.step(); losses.append(loss.item()); ps.extend(logits.argmax(1).detach().cpu().tolist()); ys.extend(y.cpu().tolist())
        tr_loss=float(np.mean(losses)); tr_acc=accuracy_score(ys,ps); tr_f1=f1_score(ys,ps,average='macro',zero_division=0)
        va_loss,vy,vp=evaluate(model,loaders['validation'],device,criterion); va_acc=accuracy_score(vy,vp); va_f1=f1_score(vy,vp,average='macro',zero_division=0); va_wf1=f1_score(vy,vp,average='weighted',zero_division=0)
        scheduler.step(va_f1); elapsed=time.time()-t
        for tag,val in [('loss/train',tr_loss),('accuracy/train',tr_acc),('macro_f1/train',tr_f1),('loss/validation',va_loss),('accuracy/validation',va_acc),('macro_f1/validation',va_f1),('weighted_f1/validation',va_wf1),('learning_rate',optimizer.param_groups[0]['lr']),('epoch_seconds',elapsed)]: writer.add_scalar(tag,val,epoch)
        print(f'epoch {epoch+1}: train loss {tr_loss:.4f} f1 {tr_f1:.3f}; val loss {va_loss:.4f} macro-f1 {va_f1:.3f}')
        if va_f1>best:
            best=va_f1; stale=0; torch.save({'model_state':model.state_dict(),'labels':LABELS,'sample_rate':16000,'duration':4.0,'seed':args.seed},out/'best.pt')
        else: stale+=1
        if stale>=patience: break
    writer.close(); ckpt=torch.load(out/'best.pt',map_location=device,weights_only=False); model.load_state_dict(ckpt['model_state'])
    te_loss,ty,tp=evaluate(model,loaders['test'],device,criterion); report=classification_report(ty,tp,labels=list(range(6)),target_names=LABELS,output_dict=True,zero_division=0); cm=confusion_matrix(ty,tp,labels=list(range(6))).tolist()
    result={'test_loss':te_loss,'accuracy':accuracy_score(ty,tp),'macro_f1':f1_score(ty,tp,average='macro',zero_division=0),'weighted_f1':f1_score(ty,tp,average='weighted',zero_division=0),'classification_report':report,'confusion_matrix':cm,'train_wall_seconds':time.time()-start,'device':str(device)}
    (out/'test_metrics.json').write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k not in ('classification_report','confusion_matrix')},indent=2))
if __name__=='__main__': main()
