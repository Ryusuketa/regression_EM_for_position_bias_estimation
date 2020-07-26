from simulate.run_experiment import run_simulate
import toml
import pickle


if __name__ == '__main__':
    params = toml.load('./conf/params.cfg')
    user_document_params = params['UserDocumentDataGenerator']
    click_exposure_params = params['ClickExposureDataGenerator']

    results = run_simulate(user_document_params, click_exposure_params)
    with open('trained.pkl', 'rb') as f:
        pickle.dump(results, f)

    
