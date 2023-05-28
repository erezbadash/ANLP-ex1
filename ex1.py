
from dataclasses import dataclass, field
import datasets
import transformers
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModel, EvalPrediction, Trainer, AutoModelForSequenceClassification
from transformers import  TrainingArguments, pipeline
from evaluate import load
import sys
import numpy as np
import time
# import wandb

NUM_SEEDS = 1
NUM_TRAIN = 2
NUM_VAL = 3
NUM_TEST = 4
MODEL_NAMES  = ['bert-base-uncased', 'roberta-base', 'google/electra-base-generator']
# OUTPUT_DIR = './gdrive/MyDrive/anlp_ex1'
OUTPUT_DIR = './'
def read_training_args():
    seeds_number = int(sys.argv[NUM_SEEDS])
    train_samples = int(sys.argv[NUM_TRAIN])
    eval_samples  = int(sys.argv[NUM_VAL])
    test_samples = int(sys.argv[NUM_TEST])
    return seeds_number,train_samples,eval_samples,test_samples

def preprocess_function(examples,tokenizer):
    result = tokenizer(examples['sentence'], truncation=True)
    return result

def prepare_train_validtion_datasets(token_dataset,train_samples,eval_samples):
    if train_samples != -1:
        train_dataset = token_dataset['train'].select(range(train_samples))
    else:
        train_dataset = token_dataset['train']
    if eval_samples != -1:
        val_dataset = token_dataset['validation'].select(range(eval_samples))
    else:
        val_dataset = token_dataset['validation']
    return train_dataset ,val_dataset

def evaluate_model_name(model_name,seeds_number,train_samples,eval_samples,raw_dataset):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_accuracies = []
    best_accuracy = 0
    best_model = None
    token_dataset = raw_dataset.map(preprocess_function, fn_kwargs={'tokenizer': tokenizer}, batched=True)
    train_dataset, validation_dataset = prepare_train_validtion_datasets(token_dataset,train_samples,eval_samples)
    metric = load('glue', 'sst2')
    training_time = 0

    def compute_metrics(p: tuple):
        logits, labels = p
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)
    output_dir = OUTPUT_DIR + '/models/' + model_name + '/'
    for seed in range(seeds_number):
        # run = wandb.init(
        #     project="ANLP_ex1",
        #     name=model_name + ", seed number: " + str(seed),
        #     tags=[model_name, str(seed), "finetuning"],
        #     notes="finetune model: " + model_name + " and seed: " + str(seed),
        #     job_type='train'
        # )
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        training_args = TrainingArguments(seed=seed,output_dir=output_dir,save_strategy ='no',report_to ='wandb')
        trainer = Trainer(model=model,args=training_args,train_dataset=train_dataset,eval_dataset=validation_dataset,
            tokenizer=tokenizer,compute_metrics=compute_metrics)
        start = time.time()
        train_result = trainer.train()
        end = time.time()
        training_time += end - start
        metrics = trainer.evaluate()
        # run.finish()
        model_accuracies.append(metrics['eval_accuracy'])
        if metrics['eval_accuracy'] > best_accuracy:
            best_accuracy = metrics['eval_accuracy']
            best_model_trainer = trainer
    model_name_dict = dict()
    model_name_dict['mean'] = np.mean(model_accuracies)
    model_name_dict['std'] = np.std(model_accuracies)
    model_name_dict['best model trainer'] = best_model_trainer
    model_name_dict['best accuracy'] = best_accuracy
    model_name_dict['training time'] = training_time
    return model_name_dict

def get_test_dataset(raw_dataset,test_samples):
    if test_samples != -1:
        return raw_dataset['test']['sentence'][:test_samples]
    return raw_dataset['test']['sentence']

def create_predictions_file(model_name,test_dataset):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classification_pipeline = pipeline("text-classification", model=OUTPUT_DIR +'/models/best/', tokenizer=tokenizer)
    start = time.time()
    predictions = classification_pipeline(test_dataset)
    end = time.time()
    labels = [prediction['label'] for prediction in predictions]
    text_predictions = "\n".join([f"{sentence}###{label[-1]}" for sentence, label in zip(test_dataset, labels)])
    with open(OUTPUT_DIR + '/predictions.txt','a') as f:
        f.write(text_predictions)
        f.close()
    predictions_time = end -start
    return predictions_time

def main():
    seeds_number, train_samples, eval_samples, test_samples = read_training_args()
    raw_dataset = load_dataset("glue", "sst2")
    models_dict = dict()
    for model_name in MODEL_NAMES:
        models_dict[model_name] = evaluate_model_name(model_name, seeds_number, train_samples, eval_samples, raw_dataset)
    best_model_trainer = None
    total_training_time = 0 
    best_accuracy = 0
    best_model_name = ""
    for model_name in MODEL_NAMES:
        total_training_time += models_dict[model_name]['training time']
        if models_dict[model_name]['mean'] > best_accuracy:
            best_model_name = model_name
            best_model_trainer = models_dict[model_name]['best model trainer']
            best_accuracy = models_dict[model_name]['mean']
    best_model_trainer.save_model(OUTPUT_DIR + '/models/best/')
    test_dataset = get_test_dataset(raw_dataset,test_samples)
    prediction_time = create_predictions_file(best_model_name,test_dataset)
    with open(OUTPUT_DIR + '/res.txt', 'a') as f:
        for model_name in MODEL_NAMES:
            f.write(model_name + ',' + str(models_dict[model_name]['mean']) + ' +- ' + str(models_dict[model_name]['std']) + '\n')
        f.write("train time," + str(total_training_time) + '\n')
        f.write("predict time," + str(prediction_time) + '\n')
        f.close()

    
            
            



if __name__ == "__main__":
    main()