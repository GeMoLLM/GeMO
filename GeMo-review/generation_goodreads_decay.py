import argparse
import json
import transformers
from tqdm import tqdm
import torch
import numpy as np
from transformers import LogitsProcessor, LogitsProcessorList

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='meta-llama/Llama-2-13b-chat-hf')
parser.add_argument('--output_path', type=str, default='BOLD_completions_debug')
parser.add_argument('--max_new_tokens', type=int, default=500)
parser.add_argument('--top_p', type=float, default=0.9)
parser.add_argument('--top_k', type=int, default=50)
parser.add_argument('--person', type=str, default='')
parser.add_argument('--decay', type=str, default='linear')
parser.add_argument('--end_temperature', type=float, default=1.0)
parser.add_argument('--period', type=int, default=20)
args = parser.parse_args()

class DecayingTemperatureWarper(LogitsProcessor):
    def __init__(self, temperature: float, decay: str, end_temperature=1.0, period=20):
        if not isinstance(temperature, float) or not (temperature > 0):
            raise ValueError(f"`temperature` has to be a strictly positive float, but is {temperature}")

        self.temperature = temperature
        self.end_temperature = end_temperature
        if decay == 'exponential':
            if period == 50:
                if self.end_temperature == 1.2:
                    self.mapping = {1: 10.0, 2: 9.162569278716443, 3: 8.404830627086241, 4: 7.719200341999117, 5: 7.098816405113626, 6: 6.537469805471174, 7: 6.0295423976274325, 8: 5.569950673364404, 9: 5.15409488423155, 10: 4.777813005717272, 11: 4.437339082308693, 12: 4.1292655365431, 13: 3.8505090648273788, 14: 3.5982797786993106, 15: 3.3700532826861367, 16: 3.1635454093061823, 17: 2.976689358352967, 18: 2.8076150116640646, 19: 2.654630216349962, 20: 2.5162038491591883, 21: 2.390950492482192, 22: 2.277616568626241, 23: 2.1750677935885383, 24: 2.0822778247606726, 25: 1.9983179889468303, 26: 1.9223479878903094, 27: 1.853607488286138, 28: 1.7914085121097978, 29: 1.7351285511019179, 30: 1.6842043364963835, 31: 1.6381262016372027, 32: 1.5964329810633087, 33: 1.5587073950096226, 34: 1.524571873130912, 35: 1.4936847756508693, 36: 1.4657369741164028, 37: 1.4404487575361746, 38: 1.4175670329389867, 39: 1.3968627923342571, 40: 1.3781288207230786, 41: 1.3611776222208607, 42: 1.345839543535499, 43: 1.3319610760202036, 44: 1.3194033193073682, 45: 1.308040591147002, 46: 1.2977591695365323, 47: 1.2884561545527755, 48: 1.2800384384949233, 49: 1.2724217740313764, 50: 1.2655299310241341}
                elif self.end_temperature == 1.5:
                    self.mapping = {1: 10.0, 2: 9.191118053305656, 3: 8.459211401162847, 4: 7.796954875794602, 5: 7.197720391302934, 6: 6.6555106075573836, 7: 6.164898906799225, 8: 5.720975082226981, 9: 5.319296194996383, 10: 4.9558421077950925, 11: 4.62697524995726, 12: 4.329404211433676, 13: 4.060150801253718, 14: 3.816520240789107, 15: 3.5960741935036546, 16: 3.3966063612616537, 17: 3.2161204029545707, 18: 3.052809954448244, 19: 2.905040549883486, 20: 2.771333263392398, 21: 2.650349907511208, 22: 2.5408796401503464, 23: 2.441826846079838, 24: 2.3522001716438314, 25: 2.2711026029600063, 26: 2.19772248830314, 27: 2.131325414821838, 28: 2.071246858287873, 29: 2.016885532314353, 30: 1.9676973704794614, 31: 1.9231900811268436, 32: 1.8829182203452413, 33: 1.8464787338161128, 34: 1.8135069229105398, 35: 1.7836727946627717, 36: 1.7566777590897074, 37: 1.732251640801987, 38: 1.7101499749978848, 39: 1.6901515607774076, 40: 1.6720562472893374, 41: 1.6556829305542404, 42: 1.6408677409149708, 43: 1.6274624029740603, 44: 1.615332751603708, 45: 1.6043573891760816, 46: 1.5944264705750597, 47: 1.5854406038293853, 48: 1.5773098553644145, 49: 1.5699528499166702, 50: 1.5632959561028568}
                else:
                    raise NotImplementedError(f"End temperature {self.end_temperature} at {args.period}, {decay} not implemented")
            else:                
                self.mapping = {1: 10.0, 2: 8.009207047642644, 3: 6.458775937413701, 4: 5.251298974669132, 5: 4.3109149705429815,
                                6: 3.578543171741711, 7: 3.0081714413358682, 8: 2.5639654910540064, 9: 2.2180175491295144, 10: 1.9485930210567788,
                                11: 1.738764987615089, 12: 1.5753507508603681, 13: 1.4480836153107755, 14: 1.348967870485498, 15: 1.2717764508008664,
                                16: 1.211659712704082, 17: 1.1648407499986075, 18: 1.1283781051809934, 19: 1.0999809688441808, 20: 1.0778652568280858}
        elif decay == 'linear':
            if period == 50:
                if self.end_temperature == 1.2:
                    self.mapping = {1: 10.0, 2: 9.824, 3: 9.648, 4: 9.472, 5: 9.296, 6: 9.12, 7: 8.943999999999999, 8: 8.768, 9: 8.592, 10: 8.416, 11: 8.24, 12: 8.064, 13: 7.888, 14: 7.712, 15: 7.536, 16: 7.359999999999999, 17: 7.183999999999999, 18: 7.007999999999999, 19: 6.832, 20: 6.656, 21: 6.4799999999999995, 22: 6.304, 23: 6.128, 24: 5.952, 25: 5.776, 26: 5.6, 27: 5.4239999999999995, 28: 5.247999999999999, 29: 5.071999999999999, 30: 4.896, 31: 4.72, 32: 4.544, 33: 4.367999999999999, 34: 4.191999999999999, 35: 4.015999999999999, 36: 3.839999999999999, 37: 3.6639999999999997, 38: 3.4879999999999995, 39: 3.3119999999999994, 40: 3.1359999999999992, 41: 2.959999999999999, 42: 2.783999999999999, 43: 2.6079999999999997, 44: 2.4319999999999995, 45: 2.2559999999999993, 46: 2.079999999999999, 47: 1.904, 48: 1.7279999999999998, 49: 1.5519999999999996, 50: 1.3759999999999994}
                elif self.end_temperature == 1.5:
                    self.mapping = {1: 10.0, 2: 9.83, 3: 9.66, 4: 9.49, 5: 9.32, 6: 9.15, 7: 8.98, 8: 8.81, 9: 8.64, 10: 8.47, 11: 8.3, 12: 8.129999999999999, 13: 7.96, 14: 7.79, 15: 7.619999999999999, 16: 7.449999999999999, 17: 7.279999999999999, 18: 7.109999999999999, 19: 6.9399999999999995, 20: 6.77, 21: 6.6, 22: 6.43, 23: 6.26, 24: 6.09, 25: 5.92, 26: 5.75, 27: 5.58, 28: 5.409999999999999, 29: 5.239999999999999, 30: 5.069999999999999, 31: 4.8999999999999995, 32: 4.7299999999999995, 33: 4.56, 34: 4.39, 35: 4.22, 36: 4.05, 37: 3.88, 38: 3.71, 39: 3.539999999999999, 40: 3.369999999999999, 41: 3.1999999999999993, 42: 3.0299999999999994, 43: 2.8599999999999994, 44: 2.6899999999999995, 45: 2.5199999999999996, 46: 2.3499999999999996, 47: 2.1799999999999997, 48: 2.01, 49: 1.8399999999999999, 50: 1.67}
                else:
                    raise NotImplementedError(f"End temperature {self.end_temperature} at {args.period}, {decay} not implemented")
            else:
                self.mapping = {1: 10.0, 2: 9.53, 3: 9.06, 4: 8.59, 5: 8.12, 6: 7.65, 7: 7.18, 8: 6.71, 9: 6.24, 10: 5.77, 11: 5.30, 
                                12: 4.83, 13: 4.36, 14: 3.89, 15: 3.42, 16: 2.95, 17: 2.49, 18: 2.01, 19: 1.54, 20: 1.0}                
        else:
            raise NotImplementedError(f"Decay type {decay} not implemented")

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        self.temperature = self.mapping.get(cur_len, self.end_temperature)
        
        return scores

filename = f"../review_data/goodreads/titles_1000-2500.npy"

titles = np.load(filename)

def template(title):
    return f'Write a comprehensive review of the book titled {title}: '

def template2(title):
    return f'Write a personalized review of the book titled {title}: '

def template3(title):
    return f"Write a book review for the book titled {title}, from the viewpoint of different personas, such as 'aspiring writer', 'history enthusiast', 'teenage sci-fi fan', or 'career-focused parent', etc. Be creative!"    

def template4(title, person):
    return f'Write a book review for the book titled {title} as if you are {person}: '

if args.person:
    texts = [template4(title, args.person) for title in titles]
else:
    texts = [template2(title) for title in titles]

print(f'total # samples = {len(texts)}')

batch_size = 8

model_path = args.model_path # '../lm-evaluation-harness/models/model_13b_replace'

tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = "[PAD]"
tokenizer.padding_side = "left"
print('load tokenizer done!')

if 'falcon' in model_path:
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda()
elif 'zephyr' in model_path:
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda()
else:
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, torch_dtype="auto").cuda()
    if '7b' in model_path:
        model = model.bfloat16().cuda()
print('load model done!')

logits_wrapper = LogitsProcessorList(
            [
                DecayingTemperatureWarper(10.0, 
                                          args.decay,
                                          end_temperature=args.end_temperature,
                                          period=args.period)
            ]
        )

all_generated_texts = []
n_batch = (len(texts) - 1) // batch_size + 1
cnt = 0
for i in tqdm(range(n_batch)):
    cur_texts = texts[i*batch_size : min((i+1)*batch_size, len(texts))]
    inputs = tokenizer(cur_texts, return_tensors="pt", padding=True, truncation=True, max_length=100)
    inputs.to('cuda:0')
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens, # 100,
            do_sample=True,
            top_p=args.top_p,
            top_k=args.top_k,
            logits_processor=logits_wrapper,
        )
    generated_texts = tokenizer.batch_decode(outputs[:,inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
    all_generated_texts += generated_texts
    
    if (i+1) % 20 == 0:
        d = {"completions": all_generated_texts}
        # with open("completions_replace.json", "w") as f:
        with open(f'{args.output_path}_{cnt}.json', "w") as f:
            json.dump(d, f)
            
        all_generated_texts = []
        cnt += 1
        # break
        
if all_generated_texts:
    d = {"completions": all_generated_texts}
    # with open("completions_replace.json", "w") as f:
    with open(f'{args.output_path}_{cnt}.json', "w") as f:
        json.dump(d, f)
        
    all_generated_texts = []