from simulate.generate_data_with_random import UserDocumentDataGenerator, ClickExposureDataGenerator
from src.utils import get_model
from src.trainer import Trainer


def run_simulate(user_document_params, click_exposure_params):
    user_document_generator = UserDocumentDataGenerator(**user_document_params)
    click_exposure_generator = ClickExposureDataGenerator(user_document=user_document_generator,
                                                          **click_exposure_params)

    user_document_data = user_document_generator.generate_data()
    relevance, exposure, click, exposure_labels, implicit_feedback = click_exposure_generator.generate_data()

    # model setup
    model = get_model(implicit_feedback, user_document_data, exposure_labels)

    # train
    trainer = Trainer(model)
    trainer.train(relevance[0].reshape(-1), user_document_data.reshape(-1, 1100))

    return trainer
