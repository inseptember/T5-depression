# fmt: off
import logging
from pathlib import Path

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import TextClassificationProcessor
from farm.evaluation.metrics import register_metrics
from farm.modeling.optimization import initialize_optimizer
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import Roberta
from farm.modeling.prediction_head import MultiLabelTextClassificationHead
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings
from sklearn.metrics import precision_score, recall_score, f1_score


def doc_classification_multilabel_roberta():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    ml_logger = MLFlowLogger(tracking_uri="https://public-mlflow.deepset.ai/")
    ml_logger.init_experiment(experiment_name="Public_FARM", run_name="Run_doc_classification")

    ##########################
    ########## Settings
    ##########################
    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=True)
    n_epochs = 25
    batch_size = 16

    evaluate_every = 500
    lang_model = "hfl/chinese-roberta-wwm-ext"
    # lang_model = "bert-base-chinese"
    do_lower_case = False # roberta is a cased model

    # 1.Create a tokenizer
    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=lang_model, do_lower_case=do_lower_case
    )

    # 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
    # Here we load Toxic Comments Data automaticaly if it is not available.

    # label_list = ['烦躁激惹', '疲倦乏力', '睡眠困难', '食欲纳差', 'cognitive', '动作缓慢', '烦躁不安', '易怒激惹', '体重减轻', '体重增加', '精神萎靡']
    label_list = ['疲倦乏力', '睡眠困难', '食欲纳差', '专注困难', '动作缓慢', '烦躁不安', '易怒激惹', '记忆力差', '体重减轻', '体重增加', '精神萎靡']
    metric = "prf"

    def prf(preds, labels):
        return  {
            'p': precision_score(labels, preds, average='micro'),
            'r' : recall_score(labels, preds, average='micro'),
            'f1' : f1_score(labels, preds, average='micro')
        }
    register_metrics('prf', prf)

    processor = TextClassificationProcessor(tokenizer=tokenizer,
                                            max_seq_len=256,
                                            data_dir=Path("./"),
                                            label_list=label_list,
                                            label_column_name="label",
                                            metric=metric,
                                            quote_char='"',
                                            delimiter=",",
                                            multilabel=True,
                                            text_column_name='text',
                                            train_filename=Path("train.tsv"),
                                            dev_filename=Path("val.tsv"),
                                            test_filename=Path("test.tsv")
                                            )

    # 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a few descriptive statistics of our datasets
    data_silo = DataSilo(
        processor=processor,
        batch_size=batch_size)

    # 4. Create an AdaptiveModel
    # a) which consists of a pretrained language model as a basis
    language_model = Roberta.load(lang_model)
    # b) and a prediction head on top that is suited for our task => Text classification
    prediction_head = MultiLabelTextClassificationHead(num_labels=len(label_list))

    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_sequence"],
        device=device)

    # 5. Create an optimizer
    model, optimizer, lr_schedule = initialize_optimizer(
        model=model,
        learning_rate=3e-5,
        device=device,
        n_batches=len(data_silo.loaders["train"]),
        n_epochs=n_epochs)

    # 6. Feed everything to the Trainer, which keeps care of growing our model into powerful plant and evaluates it from time to time
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=n_epochs,
        n_gpu=n_gpu,
        lr_schedule=lr_schedule,
        evaluate_every=evaluate_every,
        device=device)

    # 7. Let it grow
    trainer.train()

    # 8. Hooray! You have a model. Store it:
    save_dir = Path("./output/ml/roberta")
    model.save(save_dir)
    processor.save(save_dir)

    # 9. Load it & harvest your fruits (Inference)
    # basic_texts = [
    #     {"text": "You fucking bastards"},
    #     {"text": "What a lovely world"},
    # ]
    # model = Inferencer.load(save_dir)
    # result = model.run_inference(dicts=basic_texts)
    # print(result)
    # model.close_multiprocessing_pool()


if __name__ == "__main__":
    doc_classification_multilabel_roberta()

# fmt: on
