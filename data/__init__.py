import importlib

def create_dataLoader(opt):

    dataset_filename = f'data.{opt.dataset}'
    dataset_lib = importlib.import_module(dataset_filename)

    dataLoader = dataset_lib.Data(opt)
    return dataLoader