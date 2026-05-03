import yaml
from trainer import Trainer
from model import Model
from preprocess import DataProcessor
from sklearn.metrics import classification_report, f1_score

if __name__ == "__main__":    
    # options
    target_column = 'category'
    value_column = 'question'
    yaml_file = './experiments/exp_test_0/exp_config.yaml'
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
        
    dataset_path = config['dataset_path']
    path_to_save = config['model_params']['path']

    # process data
    data_processor = DataProcessor(dataset_path=dataset_path,
                                   target_column=target_column)
    
    data_processor.df = data_processor.df.rename(columns={'Show Number' : 'show_number',
                        ' Air Date' : 'air_date',
                        ' Round' : 'round',
                        ' Category' : 'category',
                        ' Value' : 'value',
                        ' Question' : 'question',
                        ' Answer' : 'answer'})
    
    data_processor.balance()
    data_processor.get_top_n_categories()
    X_train, X_test, y_train, y_test = data_processor.split(value_column=value_column)
    
    # train
    trainer = Trainer()
    model = trainer.fit(X_train, y_train)
    
    # save model
    model = Model(model=model)
    model.save(path=path_to_save)
    
    # predict by saved model
    y_pred = model.predict(X_test)
    print("Macro F1:", f1_score(y_test, y_pred, average="macro"))
    print("Weighted F1:", f1_score(y_test, y_pred, average="weighted"))
    print(classification_report(y_test, y_pred))