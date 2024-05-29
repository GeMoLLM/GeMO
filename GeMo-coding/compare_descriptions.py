import evaluate
import json
d1 = json.load(open('parsed_description_valid_solution_codeonly_1_0.json'))
d2 = json.load(open('parsed_description_valid_solution_codeonly_2_0.json'))


predictions = [d1['description']]
references = [d2['description']]
bleu = evaluate.load("bleu")
results = bleu.compute(predictions=predictions, references=references)
print(results)

results = bleu.compute(predictions=references, references=predictions)
print(results)

print(predictions)
print(references)