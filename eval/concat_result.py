import json
import os
from tqdm import tqdm
import sys

print('*'*100)

base_path = sys.argv[1] 

total_data_li,vis = [],[]
cnt=0
correct_cnt=0
for i in list(range(32)):
    try:
        file_name = f'{base_path}.part{i}'
        if not os.path.exists(file_name):
            continue

        data_li = json.load(open(file_name))['processed_results']
        cur_cnt = 0
        for dic in data_li:
            if dic not in total_data_li and type(dic['score'])==bool:
                total_data_li.append(dic)
                cnt += 1
                cur_cnt+=1
                if dic['score']==1:
                    correct_cnt+=1
        print(cur_cnt)
    except Exception as e:
        print(e)

with open(base_path,'w') as f:
    json.dump({'processed_results':total_data_li},f,indent=4)



task2cnt = {}
task2scoreli = {}
question_word_2_score_li = {}

for dic in total_data_li:
    if 'task' in dic['raw_data']:
        k = 'task'
        task = dic['raw_data'][k]
    elif "question_type" in dic['raw_data']:
        k = 'question_type'
        task = dic['raw_data'][k]
    elif "question_category" in dic['raw_data']:
        k ="question_category"
        task = dic['raw_data'][k]
    elif 'niah' in base_path.lower():
        task = 'caption' if 'choose the caption' in dic['raw_data']['question'] else 'qa'
    else:
        print('No task type, skip')
        continue
    
    if 'how long' in dic['raw_data']['question'].lower():
        q_word ='how long'
    elif 'how much'in dic['raw_data']['question'].lower():
        q_word ='how much'
    elif 'how many'in dic['raw_data']['question'].lower():
        q_word ='how many'
    else:
        q_word = dic['raw_data']['question'].split()[0].lower()

    if q_word not in question_word_2_score_li:
        question_word_2_score_li[q_word] = []
    question_word_2_score_li[q_word].append(dic['score'])

    if type(task)== str:
        task2cnt[task] = task2cnt.get(task,0) + 1
        task2scoreli[task] = task2scoreli.get(task,[])
        task2scoreli[task].append(dic['score'])

    elif type(dic['raw_data'][k])== list:
        for item in task:
            task2cnt[item] = task2cnt.get(item,0) + 1
            task2scoreli[item] = task2scoreli.get(item,[])
            task2scoreli[item].append(dic['score'])
            
print('*'*100)
score_li = []
for task, c in task2cnt.items():
    print(f'Task: {task}\tCount: {c}\tScore: {sum(task2scoreli[task])/len(task2scoreli[task])}')
    score_li.append(sum(task2scoreli[task])/len(task2scoreli[task]))

print('*'*100)

print('Average score across samples:', correct_cnt/cnt, cnt)
print('Average score across tasks:', sum(score_li)/ len(score_li))

print('*'*100)


